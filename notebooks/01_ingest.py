# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# dependencies = [
#   "kaggle",
#   "huggingface_hub",
#   "pyyaml",
#   "kagglehub",
# ]
# ///
# MAGIC %md
# MAGIC # 01 Ingest and profile
# MAGIC Downloads Jigsaw Civil Comments (individual annotations) and Measuring Hate Speech into the
# MAGIC `raw.downloads` volume, parses them into Delta tables, and profiles the annotators and
# MAGIC duplicate text. No split exists after this notebook. Read the `data_profile` run, then set
# MAGIC `heldout_annotator_fraction` and `min_heldout_annotations` in `conf/data.yaml` before
# MAGIC running `02_splits`.
# MAGIC
# MAGIC Prerequisites: accept the Kaggle competition rules, and store the Kaggle API token in the
# MAGIC secret scope named in `conf/data.yaml`.
# MAGIC
# MAGIC Safe to rerun after a failure. A file or table that already exists is kept, and only
# MAGIC missing ones are built. Set the `reload` widget to `true` to download and rebuild
# MAGIC everything: do this after changing the source files or their pinned revisions. If you
# MAGIC reload here, rerun `02_splits` with `reload` too, because its tables are built from these.

# COMMAND ----------

# MAGIC %pip install kagglehub huggingface_hub pyyaml

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

dbutils.widgets.dropdown("reload", "false", ["false", "true"], "Re-download and rebuild existing")
reload = dbutils.widgets.get("reload") == "true"

# COMMAND ----------

from dre.data.download import (
    download_huggingface_file,
    download_kaggle_file,
    sha256_file,
    volume_files,
)

files = volume_files(settings, config)
kaggle_token = dbutils.secrets.get(config.kaggle.secret_scope, config.kaggle.token_key)

downloaded = {
    name: download_kaggle_file(config.kaggle.competition, files[name], kaggle_token, reload)
    for name in ("jigsaw_comments", "jigsaw_annotations")
}
downloaded["mhs"] = download_huggingface_file(
    config.huggingface.repo_id, config.huggingface.revision, files["mhs"], reload
)
print({name: "downloaded" if fetched else "kept" for name, fetched in downloaded.items()})
checksums = {name: sha256_file(path) for name, path in files.items()}
print(checksums)

# COMMAND ----------

from dre.data.tables import profile_inputs, write_bronze_tables
from dre.reload import describe_results, row_counts

bronze_results = write_bronze_tables(spark, settings, files, reload)
print(describe_results(bronze_results))
bronze_counts = row_counts(bronze_results)
labels_per_worker, texts_by_id_count = profile_inputs(spark, settings)

# COMMAND ----------

import mlflow

from dre.data.profile import duplicate_text_summary, heldout_label_share, worker_volume_summary

profile = {
    "bronze_row_counts": bronze_counts,
    "worker_volume": worker_volume_summary(labels_per_worker),
    "heldout_if_fraction": heldout_label_share(
        labels_per_worker, config.annotator_salt, config.candidate_heldout_fractions
    ),
    "duplicate_text": duplicate_text_summary(texts_by_id_count),
}

with mlflow.start_run(run_name="data_profile") as run:
    mlflow.log_params(
        {"seed": config.seed, "mhs_revision": config.huggingface.revision, "reload": reload}
    )
    mlflow.log_params({f"sha256.{name}": digest for name, digest in checksums.items()})
    mlflow.log_dict(profile, "profile.json")

print(profile)
print("run_id:", run.info.run_id)
