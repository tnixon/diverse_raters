"""Spark edge: parse downloaded files into Delta tables and build the curated splits.

Split assignment itself lives in ``splits.py`` and is applied here to distinct ids only.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import BooleanType, StringType

from dre.data import (
    ANNOTATOR_SPLITS,
    COMMENT_SPLITS,
    JIGSAW_ANNOTATIONS,
    JIGSAW_COMMENTS,
    JIGSAW_HELDOUT_ANNOTATORS,
    JIGSAW_SEARCH,
    JIGSAW_TRAIN,
    JIGSAW_VALIDATION,
    MHS_ANNOTATIONS,
)
from dre.data.config import DataConfig
from dre.data.splits import assign_split, is_heldout_annotator
from dre.settings import Settings

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

COMMENT_COLUMNS = {"id": "long", "comment_text": "string", "toxicity": "double"}
ANNOTATION_COLUMNS = {
    "id": "long",
    "worker_id": "string",
    "toxic": "int",
    "severe_toxic": "int",
    "obscene": "int",
    "threat": "int",
    "insult": "int",
    "identity_attack": "int",
    "sexual_explicit": "int",
}
# Comment-level split name -> curated table.
SPLIT_TABLES = {
    "train": JIGSAW_TRAIN,
    "validation": JIGSAW_VALIDATION,
    "search": JIGSAW_SEARCH,
}


def _read_csv_columns(spark: SparkSession, path: str, columns: dict[str, str]) -> DataFrame:
    # multiLine and escape are needed because comment text contains newlines and quotes.
    raw = (
        spark.read.option("header", True)
        .option("multiLine", True)
        .option("escape", '"')
        .csv(path)
    )
    missing = set(columns) - set(raw.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns {sorted(missing)}")
    return raw.select(*(F.col(name).cast(dtype).alias(name) for name, dtype in columns.items()))


def write_bronze_tables(
    spark: SparkSession, settings: Settings, config: DataConfig, jigsaw_dir: str, mhs_path: str
) -> dict[str, int]:
    """Parse the downloaded files into raw Delta tables. Returns row counts by table name."""
    sources = {
        JIGSAW_COMMENTS: _read_csv_columns(
            spark, f"{jigsaw_dir}/{config.kaggle.comments_file}", COMMENT_COLUMNS
        ),
        JIGSAW_ANNOTATIONS: _read_csv_columns(
            spark, f"{jigsaw_dir}/{config.kaggle.annotations_file}", ANNOTATION_COLUMNS
        ),
        MHS_ANNOTATIONS: spark.read.parquet(mhs_path),
    }
    counts = {}
    for table, frame in sources.items():
        name = settings.table_fqn(settings.schemas.raw, table)
        frame.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(name)
        counts[table] = spark.table(name).count()
        if counts[table] == 0:
            raise ValueError(f"{name} is empty after parsing")
    return counts


def _assign_distinct(
    frame: DataFrame, key_column: str, output_column: str, assign: Callable, dtype
) -> DataFrame:
    """Apply ``assign`` to each distinct key once, rather than to every row."""
    udf = pandas_udf(lambda keys: keys.astype(str).map(assign), dtype)
    return (
        frame.select(key_column)
        .distinct()
        .withColumn(output_column, udf(F.col(key_column)))
    )


def write_split_tables(spark: SparkSession, settings: Settings, config: DataConfig) -> None:
    """Assign every comment to a split and every annotator to held-out or not."""
    raw, curated = settings.schemas.raw, settings.schemas.curated

    def comment_split(key: str) -> str:
        return assign_split(key, config.comment_salt, config.comment_splits)

    def heldout(key: str) -> bool:
        return is_heldout_annotator(key, config.annotator_salt, config.heldout_annotator_fraction)

    comments = spark.table(settings.table_fqn(raw, JIGSAW_COMMENTS))
    annotations = spark.table(settings.table_fqn(raw, JIGSAW_ANNOTATIONS))
    comment_assignments = _assign_distinct(comments, "id", "split", comment_split, StringType())
    annotator_assignments = _assign_distinct(
        annotations, "worker_id", "heldout", heldout, BooleanType()
    )
    comment_assignments.write.mode("overwrite").saveAsTable(
        settings.table_fqn(curated, COMMENT_SPLITS)
    )
    annotator_assignments.write.mode("overwrite").saveAsTable(
        settings.table_fqn(curated, ANNOTATOR_SPLITS)
    )


def write_curated_tables(spark: SparkSession, settings: Settings) -> dict[str, int]:
    """Join text, labels and split assignments into one table per dataset.

    Held-out annotators' labels go only to the held-out table, on every comment split.
    """
    raw, curated = settings.schemas.raw, settings.schemas.curated
    comments = spark.table(settings.table_fqn(raw, JIGSAW_COMMENTS)).withColumnRenamed(
        "toxicity", "toxicity_fraction"
    )
    labelled = (
        spark.table(settings.table_fqn(raw, JIGSAW_ANNOTATIONS))
        .join(comments, "id")
        .join(spark.table(settings.table_fqn(curated, COMMENT_SPLITS)), "id")
        .join(spark.table(settings.table_fqn(curated, ANNOTATOR_SPLITS)), "worker_id")
    )
    outputs = {
        table: labelled.filter((F.col("split") == split) & ~F.col("heldout"))
        for split, table in SPLIT_TABLES.items()
    }
    outputs[JIGSAW_HELDOUT_ANNOTATORS] = labelled.filter(F.col("heldout"))

    counts = {}
    for table, frame in outputs.items():
        name = settings.table_fqn(curated, table)
        frame.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(name)
        counts[table] = spark.table(name).count()
    return counts


def check_no_leakage(spark: SparkSession, settings: Settings) -> None:
    """Raise if the split invariants are broken. Run after every rebuild."""
    curated = settings.schemas.curated

    def table(name: str) -> DataFrame:
        return spark.table(settings.table_fqn(curated, name))

    for name in (*SPLIT_TABLES.values(), JIGSAW_HELDOUT_ANNOTATORS):
        if table(name).limit(1).count() == 0:
            raise ValueError(f"curated table {name} is empty")

    comment_splits = table(COMMENT_SPLITS)
    if comment_splits.count() != comment_splits.select("id").distinct().count():
        raise ValueError("a comment is assigned to more than one split")

    heldout_workers = table(ANNOTATOR_SPLITS).filter("heldout").select("worker_id")
    for name in SPLIT_TABLES.values():
        leaked = table(name).join(heldout_workers, "worker_id").limit(1).count()
        if leaked:
            raise ValueError(f"held-out annotator labels leaked into {name}")

    names = list(SPLIT_TABLES.values())
    for position, first in enumerate(names):
        for second in names[position + 1 :]:
            shared = table(first).select("id").join(table(second).select("id"), "id").limit(1)
            if shared.count():
                raise ValueError(f"comments shared between {first} and {second}")
