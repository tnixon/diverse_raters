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


class SearchSourcesConfig(BaseModel):
    """How the comments the search stage rewrites are chosen from the search split."""

    model_config = ConfigDict(frozen=True)

    count: int = Field(gt=0)
    toxicity_threshold: float = Field(ge=0, le=1)
    min_annotators: int = Field(gt=0)


class DataConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    seed: int
    comment_splits: dict[str, float]
    # Both are unset until the annotator profile has been read; see require_heldout_rule.
    heldout_annotator_fraction: float | None = Field(default=None, gt=0, lt=1)
    min_heldout_annotations: int | None = Field(default=None, gt=0)
    candidate_heldout_fractions: list[float]
    search_sources: SearchSourcesConfig
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

    @field_validator("candidate_heldout_fractions")
    @classmethod
    def _candidates_are_fractions(cls, fractions: list[float]) -> list[float]:
        if not fractions or any(not 0 < fraction < 1 for fraction in fractions):
            raise ValueError(f"candidate_heldout_fractions must be in (0, 1), got {fractions}")
        return fractions

    @property
    def comment_salt(self) -> str:
        return f"{self.seed}:comments"

    @property
    def annotator_salt(self) -> str:
        return f"{self.seed}:annotators"

    @property
    def search_sources_salt(self) -> str:
        return f"{self.seed}:search_sources"

    def require_heldout_rule(self) -> tuple[float, int]:
        """Return (fraction, minimum labels), failing if the profile has not set them yet."""
        if self.heldout_annotator_fraction is None or self.min_heldout_annotations is None:
            raise ValueError(
                "set heldout_annotator_fraction and min_heldout_annotations in conf/data.yaml "
                "after reviewing the data_profile run"
            )
        return self.heldout_annotator_fraction, self.min_heldout_annotations


def load_data_config(path: str | Path) -> DataConfig:
    with open(path) as handle:
        return DataConfig.model_validate(yaml.safe_load(handle))
