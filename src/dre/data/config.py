"""Typed configuration for the data stage, loaded from ``conf/data.yaml``."""

from __future__ import annotations

import math
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class KaggleConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    competition: str
    comments_file: str
    annotations_file: str
    secret_scope: str
    username_key: str
    api_key_key: str


class HuggingFaceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    repo_id: str
    filename: str
    revision: str


class DataConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    seed: int
    comment_splits: dict[str, float]
    heldout_annotator_fraction: float = Field(gt=0, lt=1)
    kaggle: KaggleConfig
    huggingface: HuggingFaceConfig

    @field_validator("comment_splits")
    @classmethod
    def _fractions_form_a_partition(cls, fractions: dict[str, float]) -> dict[str, float]:
        if not fractions:
            raise ValueError("comment_splits must not be empty")
        if any(not 0 < fraction < 1 for fraction in fractions.values()):
            raise ValueError(f"every split fraction must be in (0, 1), got {fractions}")
        if not math.isclose(sum(fractions.values()), 1.0, abs_tol=1e-9):
            raise ValueError(f"comment_splits must sum to 1, got {sum(fractions.values())}")
        return fractions

    @property
    def comment_salt(self) -> str:
        return f"{self.seed}:comments"

    @property
    def annotator_salt(self) -> str:
        return f"{self.seed}:annotators"


def load_data_config(path: str | Path) -> DataConfig:
    with open(path) as handle:
        return DataConfig.model_validate(yaml.safe_load(handle))
