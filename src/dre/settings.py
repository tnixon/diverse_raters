"""Typed project settings and MLflow wiring.

Names of Unity Catalog objects and the MLflow experiment are fixed here. Workspace
credentials are optional: when unset, the Databricks SDK's default authentication
applies (the runtime's own credentials inside Databricks, a CLI profile or
``~/.databrickscfg`` outside it). Constructing settings never touches the network.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, SecretStr

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient

ENV_HOST = "DATABRICKS_HOST"
ENV_TOKEN = "DATABRICKS_TOKEN"
ENV_PROFILE = "DATABRICKS_CONFIG_PROFILE"
ENV_RUNTIME = "DATABRICKS_RUNTIME_VERSION"


class Schemas(BaseModel):
    """Schema names inside the project catalog."""

    model_config = ConfigDict(frozen=True)

    raw: str = "raw"
    curated: str = "curated"
    results: str = "results"
    models: str = "models"

    def all(self) -> tuple[str, ...]:
        return (self.raw, self.curated, self.results, self.models)


class Settings(BaseModel):
    """Project-wide names plus optional workspace credentials."""

    model_config = ConfigDict(frozen=True)

    catalog: str = "diverse_raters"
    schemas: Schemas = Schemas()
    volume: str = "downloads"  # lives in the `raw` schema
    experiment: str = "/Shared/diverse-raters"

    host: str | None = None
    token: SecretStr | None = None
    profile: str | None = None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if environ is None else environ
        token = env.get(ENV_TOKEN)
        return cls(
            host=env.get(ENV_HOST) or None,
            token=SecretStr(token) if token else None,
            profile=env.get(ENV_PROFILE) or None,
        )

    def schema_fqn(self, schema: str) -> str:
        return f"{self.catalog}.{schema}"

    @property
    def volume_fqn(self) -> str:
        return f"{self.catalog}.{self.schemas.raw}.{self.volume}"

    @property
    def volume_path(self) -> str:
        return f"/Volumes/{self.catalog}/{self.schemas.raw}/{self.volume}"

    def model_name(self, name: str) -> str:
        """Three-level Unity Catalog name for a registered model."""
        return f"{self.catalog}.{self.schemas.models}.{name}"

    def workspace_client(self) -> WorkspaceClient:
        """Build a WorkspaceClient, passing only the credentials that are set."""
        from databricks.sdk import WorkspaceClient

        kwargs: dict[str, str] = {}
        if self.host:
            kwargs["host"] = self.host
        if self.token:
            kwargs["token"] = self.token.get_secret_value()
        if self.profile:
            kwargs["profile"] = self.profile
        return WorkspaceClient(**kwargs)


def in_databricks(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return ENV_RUNTIME in env


def configure_mlflow(settings: Settings | None = None) -> Settings:
    """Point MLflow at Databricks, use the UC model registry, and select the experiment."""
    import mlflow

    settings = settings or Settings.from_env()
    if not in_databricks():
        # MLflow's Databricks client reads credentials from the environment.
        if settings.host:
            os.environ[ENV_HOST] = settings.host
        if settings.token:
            os.environ[ENV_TOKEN] = settings.token.get_secret_value()

    tracking_uri = f"databricks://{settings.profile}" if settings.profile else "databricks"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(settings.experiment)
    return settings
