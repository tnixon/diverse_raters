from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from dre.data.config import DataConfig, load_data_config

SHIPPED_CONFIG = Path(__file__).parent.parent / "conf" / "data.yaml"


def shipped_values() -> dict:
    return yaml.safe_load(SHIPPED_CONFIG.read_text())


def test_shipped_config_loads():
    config = load_data_config(SHIPPED_CONFIG)
    assert sum(config.comment_splits.values()) == pytest.approx(1.0)
    assert config.comment_salt != config.annotator_salt


@pytest.mark.parametrize(
    "fractions",
    [{"train": 0.5, "validation": 0.1}, {"train": 1.0}, {"train": 1.2, "validation": -0.2}, {}],
)
def test_bad_split_fractions_are_rejected(fractions):
    with pytest.raises(ValidationError):
        DataConfig.model_validate({**shipped_values(), "comment_splits": fractions})


def test_heldout_rule_is_unset_until_the_profile_is_read():
    config = load_data_config(SHIPPED_CONFIG)
    assert config.heldout_annotator_fraction is None
    with pytest.raises(ValueError, match="data_profile"):
        config.require_heldout_rule()


def test_heldout_rule_is_returned_once_set():
    values = {
        **shipped_values(),
        "heldout_annotator_fraction": 0.1,
        "min_heldout_annotations": 5000,
    }
    assert DataConfig.model_validate(values).require_heldout_rule() == (0.1, 5000)


def test_search_source_fields_are_validated():
    values = shipped_values()
    values["search_sources"] = {**values["search_sources"], "count": 0}
    with pytest.raises(ValidationError):
        DataConfig.model_validate(values)


def test_salts_are_distinct_per_purpose():
    config = load_data_config(SHIPPED_CONFIG)
    assert len({config.comment_salt, config.annotator_salt, config.search_sources_salt}) == 3


@pytest.mark.parametrize("fraction", [0, 1, -0.1])
def test_bad_heldout_fraction_is_rejected(fraction):
    with pytest.raises(ValidationError):
        DataConfig.model_validate({**shipped_values(), "heldout_annotator_fraction": fraction})


def test_missing_field_is_rejected():
    values = shipped_values()
    del values["kaggle"]
    with pytest.raises(ValidationError):
        DataConfig.model_validate(values)
