# Databricks notebook source
# MAGIC %md
# MAGIC # 01 Data
# MAGIC Downloads Jigsaw Civil Comments (individual annotations) and Measuring Hate Speech into the
# MAGIC `raw.downloads` volume, parses them into Delta tables, carves out the up-front splits
# MAGIC (comments: train/validation/search; annotators: held-out), and registers each curated
# MAGIC table as an MLflow dataset. Safe to rerun: downloads are skipped when present and splits
# MAGIC are deterministic.
# MAGIC
# MAGIC Prerequisites: accept the Kaggle competition rules, and store the Kaggle username and key
# MAGIC in the secret scope named in `conf/data.yaml`.

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

jigsaw_dir = Path(settings.volume_path) / "jigsaw"
mhs_dir = Path(settings.volume_path) / "mhs"

# COMMAND ----------

from dre.data.download import (
    download_huggingface_file,
    download_kaggle_file,
    sha256_file,
)

kaggle_user = dbutils.secrets.get(config.kaggle.secret_scope, config.kaggle.username_key)
kaggle_key = dbutils.secrets.get(config.kaggle.secret_scope, config.kaggle.api_key_key)

downloaded = {
    name: download_kaggle_file(
        config.kaggle.competition, file_name, jigsaw_dir, kaggle_user, kaggle_key
    )
    for name, file_name in {
        "jigsaw_comments": config.kaggle.comments_file,
        "jigsaw_annotations": config.kaggle.annotations_file,
    }.items()
}
downloaded["mhs"] = download_huggingface_file(
    config.huggingface.repo_id,
    config.huggingface.filename,
    config.huggingface.revision,
    mhs_dir,
)
checksums = {name: sha256_file(path) for name, path in downloaded.items()}
print(checksums)

# COMMAND ----------

from dre.data.tables import (
    check_no_leakage,
    write_bronze_tables,
    write_curated_tables,
    write_split_tables,
)

bronze_counts = write_bronze_tables(
    spark, settings, config, str(jigsaw_dir), str(downloaded["mhs"])
)
write_split_tables(spark, settings, config)
curated_counts = write_curated_tables(spark, settings)
check_no_leakage(spark, settings)
print(bronze_counts, curated_counts)

# COMMAND ----------

import mlflow

from dre.data.registration import log_datasets, log_file_checksums, require_git_commit

with mlflow.start_run(run_name="data_prep") as run:
    mlflow.log_params(
        {
            "seed": config.seed,
            "comment_splits": config.comment_splits,
            "heldout_annotator_fraction": config.heldout_annotator_fraction,
            "mhs_revision": config.huggingface.revision,
        }
    )
    mlflow.log_artifact(str(config_path))
    log_file_checksums(checksums)
    mlflow.log_metrics({f"rows.{table}": count for table, count in curated_counts.items()})
    log_datasets(spark, settings)

print("git commit:", require_git_commit(run.info.run_id))
print("run_id:", run.info.run_id)
