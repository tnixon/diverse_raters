# Databricks notebook source
# MAGIC %md
# MAGIC # 01 Ingest and profile
# MAGIC Downloads Jigsaw Civil Comments (individual annotations) and Measuring Hate Speech into the
# MAGIC `raw.downloads` volume, parses them into Delta tables, and profiles the annotators and
# MAGIC duplicate text. No split exists after this notebook. Read the `data_profile` run, then set
# MAGIC `heldout_annotator_fraction` and `min_heldout_annotations` in `conf/data.yaml` before
# MAGIC running `02_splits`.
# MAGIC
# MAGIC Prerequisites: accept the Kaggle competition rules, and store the Kaggle username and key
# MAGIC in the secret scope named in `conf/data.yaml`. Safe to rerun: downloads are skipped when
# MAGIC present.

# COMMAND ----------

# MAGIC %pip install kaggle huggingface_hub pyyaml

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

# COMMAND ----------

from dre.data.download import (
    download_huggingface_file,
    download_kaggle_file,
    sha256_file,
    volume_files,
)

files = volume_files(settings, config)
kaggle_user = dbutils.secrets.get(config.kaggle.secret_scope, config.kaggle.username_key)
kaggle_key = dbutils.secrets.get(config.kaggle.secret_scope, config.kaggle.api_key_key)

for file in (files["jigsaw_comments"], files["jigsaw_annotations"]):
    download_kaggle_file(config.kaggle.competition, file.name, file.parent, kaggle_user, kaggle_key)
download_huggingface_file(
    config.huggingface.repo_id,
    config.huggingface.filename,
    config.huggingface.revision,
    files["mhs"].parent,
)
checksums = {name: sha256_file(path) for name, path in files.items()}
print(checksums)

# COMMAND ----------

from dre.data.tables import profile_inputs, write_bronze_tables

bronze_counts = write_bronze_tables(spark, settings, files)
print(bronze_counts)
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
    mlflow.log_params({"seed": config.seed, "mhs_revision": config.huggingface.revision})
    mlflow.log_params({f"sha256.{name}": digest for name, digest in checksums.items()})
    mlflow.log_dict(profile, "profile.json")

print(profile)
print("run_id:", run.info.run_id)
