# Databricks notebook source
# MAGIC %md
# MAGIC # 02 Splits
# MAGIC Carves out the up-front splits, builds the curated tables, picks the search source
# MAGIC comments, and registers each table as an MLflow dataset. Run `01_ingest` first and set the
# MAGIC held-out annotator rule in `conf/data.yaml` from its profile.
# MAGIC
# MAGIC - Comments are split by the hash of their normalized text: train / validation / search.
# MAGIC - Held-out annotators' labels appear only in `jigsaw_heldout_annotators`. They train the
# MAGIC   held-out rater model, one of the trusted scorers of rewrites.
# MAGIC - Label aggregates are rebuilt from individual annotations in each table; the Kaggle
# MAGIC   all-annotator `toxicity` score is never used.
# MAGIC
# MAGIC Safe to rerun after a failure. A table that already exists is kept, and only missing ones
# MAGIC are built. Set the `reload` widget to `true` to rebuild every table: do this after changing
# MAGIC `conf/data.yaml` or after rerunning `01_ingest` with new inputs, because existing tables are
# MAGIC not checked against the current config or inputs.

# COMMAND ----------

import os
import sys

# The repo is a Databricks Git folder: make src/ importable without installing.
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "src")))

from pathlib import Path

from dre.data.config import load_data_config
from dre.settings import Settings, configure_mlflow

settings = Settings.from_env()
config_path = Path(os.getcwd()).parent / "conf" / "data.yaml"
config = load_data_config(config_path)
configure_mlflow(settings)

dbutils.widgets.dropdown("reload", "false", ["false", "true"], "Rebuild existing tables")
reload = dbutils.widgets.get("reload") == "true"

# COMMAND ----------

from dre.data import JIGSAW_SEARCH_SOURCES
from dre.data.tables import (
    check_no_leakage,
    write_curated_tables,
    write_search_sources,
    write_split_tables,
)
from dre.reload import describe_results, row_counts

# Fails fast if the held-out rule is unset, or too small to train a rater.
split_results = write_split_tables(spark, settings, config, reload)
curated_results = write_curated_tables(spark, settings, reload)
curated_results[JIGSAW_SEARCH_SOURCES] = write_search_sources(spark, settings, config, reload)
check_no_leakage(spark, settings)
print(describe_results({**split_results, **curated_results}))

# COMMAND ----------

import mlflow

from dre.data.download import sha256_file, volume_files
from dre.data.registration import log_datasets, require_git_commit

fraction, minimum_labels = config.require_heldout_rule()

with mlflow.start_run(run_name="data_prep") as run:
    mlflow.log_params(
        {
            "seed": config.seed,
            "comment_splits": config.comment_splits,
            "heldout_annotator_fraction": fraction,
            "min_heldout_annotations": minimum_labels,
            "search_sources": config.search_sources.model_dump(),
            "mhs_revision": config.huggingface.revision,
            "reload": reload,
        }
    )
    files = volume_files(settings, config)
    mlflow.log_params({f"sha256.{name}": sha256_file(path) for name, path in files.items()})
    mlflow.log_artifact(str(config_path))
    mlflow.log_metrics(
        {f"rows.{table}": rows for table, rows in row_counts(curated_results).items()}
    )
    log_datasets(spark, settings)

print("git commit:", require_git_commit(run.info.run_id))
print("run_id:", run.info.run_id)
