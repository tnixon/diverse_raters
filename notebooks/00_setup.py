# Databricks notebook source
# MAGIC %md
# MAGIC # 00 Setup
# MAGIC Idempotently creates the catalog, schemas, volume and MLflow experiment, then runs a
# MAGIC smoke test: one MLflow run that logs a parameter and a metric and registers a trivial
# MAGIC model as `diverse_raters.models.smoke_test`.

# COMMAND ----------

import os
import sys

# The repo is a Databricks Git folder: make src/ importable without installing.
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "src")))

from dre.settings import Settings, configure_mlflow

settings = Settings.from_env()
print(settings.catalog, settings.schemas.all(), settings.volume_fqn, settings.experiment)

# COMMAND ----------

# If your workspace has no default storage for catalogs, add: MANAGED LOCATION '<url>'
spark.sql(f"CREATE CATALOG IF NOT EXISTS {settings.catalog}")
for schema in settings.schemas.all():
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {settings.schema_fqn(schema)}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {settings.volume_fqn}")

# COMMAND ----------

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.models import infer_signature

configure_mlflow(settings)  # set_experiment creates the experiment if it is missing


class _Echo(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        return model_input


example = pd.DataFrame({"x": [1.0, 2.0]})

with mlflow.start_run(run_name="smoke_test") as run:
    mlflow.log_param("smoke", "ok")
    mlflow.log_metric("answer", 42.0)
    info = mlflow.pyfunc.log_model(
        name="model",
        python_model=_Echo(),
        signature=infer_signature(example, example),
        input_example=example,
        registered_model_name=settings.model_name("smoke_test"),
    )

print("run_id:", run.info.run_id)
print("registered:", settings.model_name("smoke_test"), info.registered_model_version)
