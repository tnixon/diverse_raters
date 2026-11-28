"""Spark edge: parse downloaded files into Delta tables and build the curated splits.

Split logic itself lives in ``splits.py`` and is applied here to distinct keys only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import BooleanType, DataType, DoubleType, StringType

from dre.data import (
    ANNOTATOR_SPLITS,
    COMMENT_SPLITS,
    JIGSAW_ANNOTATIONS,
    JIGSAW_COMMENTS,
    JIGSAW_HELDOUT_ANNOTATORS,
    JIGSAW_SEARCH,
    JIGSAW_SEARCH_SOURCES,
    JIGSAW_TRAIN,
    JIGSAW_VALIDATION,
    MHS_ANNOTATIONS,
    SOURCE_JIGSAW,
)
from dre.data.config import DataConfig
from dre.data.profile import require_enough_heldout_labels
from dre.data.splits import assign_split, hash_unit, is_heldout_annotator, text_key
from dre.reload import BuildResult, should_build
from dre.settings import Settings

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

# The precomputed `toxicity` in all_data.csv aggregates every annotator, held-out ones included,
# so it is deliberately not read. Labels are rebuilt from individual annotations per table.
COMMENT_COLUMNS = {"id": "long", "comment_text": "string"}
ANNOTATION_COLUMNS = {
    "id": "long",
    "worker": "string",
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
FORBIDDEN_PRECOMPUTED_COLUMNS = {"toxicity", "toxicity_fraction"}


def _write_table(
    spark: SparkSession, name: str, build: Callable[[], DataFrame], reload: bool
) -> BuildResult:
    """Write the table if it is missing, or if ``reload`` asks for a rebuild.

    ``build`` runs only when the table is written and must validate before returning, because a
    table that exists is assumed to be complete and is skipped on later runs.
    """
    built = should_build(spark.catalog.tableExists(name), reload)
    if built:
        build().write.mode("overwrite").option("overwriteSchema", True).saveAsTable(name)
    return BuildResult(rows=spark.table(name).count(), built=built)


def _map_udf(function: Callable, return_type: DataType):
    @pandas_udf(return_type)
    def apply(values: pd.Series) -> pd.Series:
        return values.astype(str).map(function)

    return apply


def _read_csv_columns(spark: SparkSession, path: Path, columns: dict[str, str]) -> DataFrame:
    # multiLine and escape are needed because comment text contains newlines and quotes.
    raw = (
        spark.read.option("header", True)
        .option("multiLine", True)
        .option("escape", '"')
        .csv(str(path))
    )
    missing = set(columns) - set(raw.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns {sorted(missing)}")
    return raw.select(*(F.col(name).cast(dtype).alias(name) for name, dtype in columns.items()))


def _validated(frame: DataFrame, table: str, non_null: Sequence[str] = ()) -> DataFrame:
    """Return ``frame`` unchanged, or raise if it is empty or has nulls in ``non_null`` columns."""
    for column in non_null:
        if frame.filter(F.col(column).isNull()).limit(1).count():
            raise ValueError(f"{table}.{column} contains nulls")
    if frame.limit(1).count() == 0:
        raise ValueError(f"{table} would be empty")
    return frame


def write_bronze_tables(
    spark: SparkSession, settings: Settings, files: Mapping[str, Path], reload: bool
) -> dict[str, BuildResult]:
    """Parse the downloaded files into raw Delta tables, building only what is missing."""

    def jigsaw_comments() -> DataFrame:
        # all_data.csv contains one comment with empty text (id 392337) and no annotations;
        # drop it instead of failing ingestion.
        frame = _read_csv_columns(spark, files["jigsaw_comments"], COMMENT_COLUMNS).filter(
            F.col("comment_text").isNotNull()
        )
        return _validated(frame, JIGSAW_COMMENTS, non_null=["comment_text"])

    def jigsaw_annotations() -> DataFrame:
        # The Jigsaw CSV names the annotator column "worker"; the tables use "worker_id".
        frame = _read_csv_columns(
            spark, files["jigsaw_annotations"], ANNOTATION_COLUMNS
        ).withColumnRenamed("worker", "worker_id")
        return _validated(frame, JIGSAW_ANNOTATIONS, non_null=["worker_id"])

    def mhs_annotations() -> DataFrame:
        return _validated(spark.read.parquet(str(files["mhs"])), MHS_ANNOTATIONS)

    builders = {
        JIGSAW_COMMENTS: jigsaw_comments,
        JIGSAW_ANNOTATIONS: jigsaw_annotations,
        MHS_ANNOTATIONS: mhs_annotations,
    }
    return {
        table: _write_table(spark, settings.table_fqn(settings.schemas.raw, table), build, reload)
        for table, build in builders.items()
    }


def _keyed_comments(spark: SparkSession, settings: Settings) -> DataFrame:
    comments = spark.table(settings.table_fqn(settings.schemas.raw, JIGSAW_COMMENTS))
    return comments.withColumn("text_key", _map_udf(text_key, StringType())(F.col("comment_text")))


def profile_inputs(spark: SparkSession, settings: Settings) -> tuple[pd.Series, pd.Series]:
    """Collect the small summaries ``profile.py`` needs, before any split exists.

    Returns labels per worker (indexed by worker id) and, for exact duplicate text, the number
    of texts by how many ids share them.
    """
    annotations = spark.table(settings.table_fqn(settings.schemas.raw, JIGSAW_ANNOTATIONS))
    labels_per_worker = (
        annotations.groupBy("worker_id").count().toPandas().set_index("worker_id")["count"]
    )
    texts_by_id_count = (
        _keyed_comments(spark, settings)
        .groupBy("text_key")
        .count()
        .groupBy("count")
        .agg(F.count("*").alias("texts"))
        .toPandas()
        .set_index("count")["texts"]
    )
    return labels_per_worker, texts_by_id_count


def write_split_tables(
    spark: SparkSession, settings: Settings, config: DataConfig, reload: bool
) -> dict[str, BuildResult]:
    """Assign every comment text to a split and every annotator to held-out or not."""
    # Checked even when the tables exist, so an unset rule fails before anything is skipped.
    fraction, minimum_labels = config.require_heldout_rule()
    curated = settings.schemas.curated
    raw = settings.schemas.raw

    def comment_splits() -> DataFrame:
        keyed = _keyed_comments(spark, settings)
        split_of = _map_udf(
            partial(assign_split, salt=config.comment_salt, fractions=config.comment_splits),
            StringType(),
        )
        assignments = keyed.select("text_key").distinct().withColumn("split", split_of("text_key"))
        return (
            keyed.select("id", "text_key")
            .join(assignments, "text_key")
            .select(F.lit(SOURCE_JIGSAW).alias("source"), "id", "text_key", "split")
        )

    def annotator_splits() -> DataFrame:
        annotations = spark.table(settings.table_fqn(raw, JIGSAW_ANNOTATIONS))
        heldout_of = _map_udf(
            partial(is_heldout_annotator, salt=config.annotator_salt, fraction=fraction),
            BooleanType(),
        )
        workers = (
            annotations.select("worker_id")
            .distinct()
            .withColumn("heldout", heldout_of("worker_id"))
        )
        heldout_labels = annotations.join(workers.filter("heldout"), "worker_id").count()
        require_enough_heldout_labels(heldout_labels, minimum_labels)
        return workers.select(F.lit(SOURCE_JIGSAW).alias("source"), "worker_id", "heldout")

    builders = {COMMENT_SPLITS: comment_splits, ANNOTATOR_SPLITS: annotator_splits}
    return {
        table: _write_table(spark, settings.table_fqn(curated, table), build, reload)
        for table, build in builders.items()
    }


def write_curated_tables(
    spark: SparkSession, settings: Settings, reload: bool
) -> dict[str, BuildResult]:
    """Join text, labels and split assignments into one table per dataset.

    Held-out annotators' labels go only to the held-out table, on every comment split. The
    per-comment label aggregates are computed inside each output table, so they only ever
    reflect the annotators that table is allowed to contain.
    """
    raw, curated = settings.schemas.raw, settings.schemas.curated
    comment_splits = (
        spark.table(settings.table_fqn(curated, COMMENT_SPLITS))
        .filter(F.col("source") == SOURCE_JIGSAW)
        .select("id", "text_key", "split")
    )
    annotator_splits = (
        spark.table(settings.table_fqn(curated, ANNOTATOR_SPLITS))
        .filter(F.col("source") == SOURCE_JIGSAW)
        .select("worker_id", "heldout")
    )
    labelled = (
        spark.table(settings.table_fqn(raw, JIGSAW_ANNOTATIONS))
        .join(spark.table(settings.table_fqn(raw, JIGSAW_COMMENTS)), "id")
        .join(comment_splits, "id")
        .join(annotator_splits, "worker_id")
    )
    per_comment = Window.partitionBy("id")

    def with_aggregates(frame: DataFrame) -> DataFrame:
        return frame.withColumn("toxicity_mean", F.avg("toxic").over(per_comment)).withColumn(
            "annotator_count", F.count("worker_id").over(per_comment)
        )

    outputs = {
        table: with_aggregates(labelled.filter((F.col("split") == split) & ~F.col("heldout")))
        for split, table in SPLIT_TABLES.items()
    }
    outputs[JIGSAW_HELDOUT_ANNOTATORS] = with_aggregates(labelled.filter(F.col("heldout")))

    return {
        table: _write_table(
            spark, settings.table_fqn(curated, table), partial(_validated, frame, table), reload
        )
        for table, frame in outputs.items()
    }


def write_search_sources(
    spark: SparkSession, settings: Settings, config: DataConfig, reload: bool
) -> BuildResult:
    """Choose the comments the search stage rewrites, from the search split only.

    Eligibility uses training-annotator labels (the search table holds no other). Selection is a
    seeded hash order, with one id per distinct text, so it is reproducible.
    """
    curated = settings.schemas.curated
    rule = config.search_sources

    def chosen_sources() -> DataFrame:
        eligible = (
            spark.table(settings.table_fqn(curated, JIGSAW_SEARCH))
            .select("id", "text_key", "comment_text", "toxicity_mean", "annotator_count")
            .distinct()
            .filter(
                (F.col("annotator_count") >= rule.min_annotators)
                & (F.col("toxicity_mean") >= rule.toxicity_threshold)
            )
        )
        one_id_per_text = eligible.groupBy("text_key").agg(F.min("id").alias("id"))
        hash_order = _map_udf(partial(hash_unit, salt=config.search_sources_salt), DoubleType())
        chosen = (
            eligible.join(one_id_per_text, ["text_key", "id"])
            .withColumn("selection_hash", hash_order("text_key"))
            .orderBy("selection_hash", "text_key")
            .limit(rule.count)
        )
        chosen_count = chosen.count()
        if chosen_count < rule.count:
            raise ValueError(
                f"only {chosen_count} eligible search comments, fewer than the {rule.count} "
                "required"
            )
        return chosen

    name = settings.table_fqn(curated, JIGSAW_SEARCH_SOURCES)
    return _write_table(spark, name, chosen_sources, reload)


def check_no_leakage(spark: SparkSession, settings: Settings) -> None:
    """Raise if the split invariants are broken. Run after every rebuild."""
    curated = settings.schemas.curated

    def table(name: str) -> DataFrame:
        return spark.table(settings.table_fqn(curated, name))

    datasets = (*SPLIT_TABLES.values(), JIGSAW_HELDOUT_ANNOTATORS, JIGSAW_SEARCH_SOURCES)
    for name in datasets:
        if table(name).limit(1).count() == 0:
            raise ValueError(f"curated table {name} is empty")
        leaked_columns = FORBIDDEN_PRECOMPUTED_COLUMNS & set(table(name).columns)
        if leaked_columns:
            raise ValueError(f"{name} carries all-annotator label columns {sorted(leaked_columns)}")

    comment_splits = table(COMMENT_SPLITS)
    if comment_splits.count() != comment_splits.select("source", "id").distinct().count():
        raise ValueError("a comment is assigned to more than one split")
    split_spanning_texts = (
        comment_splits.groupBy("text_key")
        .agg(F.countDistinct("split").alias("splits"))
        .filter("splits > 1")
    )
    if split_spanning_texts.limit(1).count():
        raise ValueError("identical comment text appears in more than one split")

    heldout_workers = table(ANNOTATOR_SPLITS).filter("heldout").select("worker_id")
    for name in SPLIT_TABLES.values():
        if table(name).join(heldout_workers, "worker_id").limit(1).count():
            raise ValueError(f"held-out annotator labels leaked into {name}")

    names = list(SPLIT_TABLES.values())
    for position, first in enumerate(names):
        for second in names[position + 1 :]:
            shared = table(first).select("id").join(table(second).select("id"), "id").limit(1)
            if shared.count():
                raise ValueError(f"comments shared between {first} and {second}")
