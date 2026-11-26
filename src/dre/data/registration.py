"""Register the curated tables as MLflow datasets, pinned to their current Delta version."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlflow
import mlflow.data

from dre.data import (
    JIGSAW_HELDOUT_ANNOTATORS,
    JIGSAW_SEARCH,
    JIGSAW_SEARCH_SOURCES,
    JIGSAW_TRAIN,
    JIGSAW_VALIDATION,
    MHS_ANNOTATIONS,
    RegisteredDataset,
)
from dre.settings import Settings

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

GIT_COMMIT_TAG = "mlflow.source.git.commit"


def registered_datasets(settings: Settings) -> tuple[RegisteredDataset, ...]:
    raw, curated = settings.schemas.raw, settings.schemas.curated
    return (
        RegisteredDataset(
            curated, JIGSAW_TRAIN, "training", "Rater training: training annotators, train split"
        ),
        RegisteredDataset(
            curated,
            JIGSAW_VALIDATION,
            "validation",
            "Matched-competence validation: training annotators, validation split",
        ),
        RegisteredDataset(
            curated,
            JIGSAW_SEARCH,
            "search",
            "Comments the search stage rewrites: training annotators, search split",
        ),
        RegisteredDataset(
            curated,
            JIGSAW_SEARCH_SOURCES,
            "search_sources",
            "Source comments for the search stage, chosen from the search split",
        ),
        RegisteredDataset(
            curated,
            JIGSAW_HELDOUT_ANNOTATORS,
            "heldout_annotators",
            "Trains the held-out rater model, a trusted scorer of rewrites; held-out labels only",
        ),
        RegisteredDataset(
            raw, MHS_ANNOTATIONS, "cross_dataset_eval", "Measuring Hate Speech, evaluation only"
        ),
    )


def delta_version(spark: SparkSession, table_name: str) -> int:
    latest = spark.sql(f"DESCRIBE HISTORY {table_name} LIMIT 1").collect()[0]
    return int(latest["version"])


def log_datasets(spark: SparkSession, settings: Settings) -> None:
    """Comment each table and log it as a dataset input of the active MLflow run."""
    for dataset in registered_datasets(settings):
        table_name = settings.table_fqn(dataset.schema, dataset.table)
        spark.sql(f"COMMENT ON TABLE {table_name} IS '{dataset.description}'")
        version = delta_version(spark, table_name)
        mlflow.log_input(
            mlflow.data.load_delta(table_name=table_name, version=str(version)),
            context=dataset.context,
        )


def require_git_commit(run_id: str) -> str:
    """Return the run's git commit tag, failing if MLflow did not record one."""
    commit = mlflow.get_run(run_id).data.tags.get(GIT_COMMIT_TAG)
    if not commit:
        raise RuntimeError(
            f"run {run_id} has no {GIT_COMMIT_TAG} tag; "
            "run the notebook from a Databricks Git folder"
        )
    return commit
