"""Data stage: download, parse, up-front splits, and dataset registration."""

from dataclasses import dataclass

# Value of the `source` column in the split tables. A later source adds rows under a new value.
SOURCE_JIGSAW = "jigsaw"

# Bronze tables, parsed from the downloaded files (schema: raw).
JIGSAW_COMMENTS = "jigsaw_comments"
JIGSAW_ANNOTATIONS = "jigsaw_annotations"
MHS_ANNOTATIONS = "mhs_annotations"

# Split assignments (schema: curated).
COMMENT_SPLITS = "comment_splits"
ANNOTATOR_SPLITS = "annotator_splits"

# Curated datasets (schema: curated), one per MLflow dataset context.
JIGSAW_TRAIN = "jigsaw_train"
JIGSAW_VALIDATION = "jigsaw_validation"
JIGSAW_SEARCH = "jigsaw_search"
JIGSAW_SEARCH_SOURCES = "jigsaw_search_sources"
JIGSAW_HELDOUT_ANNOTATORS = "jigsaw_heldout_annotators"


@dataclass(frozen=True)
class RegisteredDataset:
    """A table registered as an MLflow dataset, and the context it is logged under."""

    schema: str
    table: str
    context: str
    description: str
