import databricks.sdk
import pytest

from dre.settings import Settings, in_databricks


def test_default_names():
    s = Settings()
    assert s.catalog == "diverse_raters"
    assert s.schemas.all() == ("raw", "curated", "results", "models")
    assert s.volume_fqn == "diverse_raters.raw.downloads"
    assert s.volume_path == "/Volumes/diverse_raters/raw/downloads"
    assert s.experiment == "/Shared/diverse-raters"
    assert s.schema_fqn("results") == "diverse_raters.results"
    assert s.table_fqn("curated", "jigsaw_train") == "diverse_raters.curated.jigsaw_train"
    assert s.model_name("smoke_test") == "diverse_raters.models.smoke_test"


def test_from_env_reads_credentials():
    s = Settings.from_env(
        {
            "DATABRICKS_HOST": "https://example.invalid",
            "DATABRICKS_TOKEN": "s3cret",
            "DATABRICKS_CONFIG_PROFILE": "dev",
        }
    )
    assert s.host == "https://example.invalid"
    assert s.token is not None
    assert s.token.get_secret_value() == "s3cret"
    assert s.profile == "dev"
    assert "s3cret" not in repr(s)


def test_from_env_empty_means_sdk_default_auth():
    s = Settings.from_env({})
    assert s.host is None
    assert s.token is None
    assert s.profile is None


def test_from_env_uses_os_environ(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "https://example.invalid")
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    monkeypatch.delenv("DATABRICKS_CONFIG_PROFILE", raising=False)
    assert Settings.from_env().host == "https://example.invalid"


def test_in_databricks():
    assert in_databricks({"DATABRICKS_RUNTIME_VERSION": "15.4"})
    assert not in_databricks({})


def test_settings_construction_makes_no_client(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("WorkspaceClient must not be built eagerly")

    monkeypatch.setattr(databricks.sdk, "WorkspaceClient", boom)
    Settings.from_env({"DATABRICKS_HOST": "https://example.invalid"})
    with pytest.raises(AssertionError):
        Settings().workspace_client()
