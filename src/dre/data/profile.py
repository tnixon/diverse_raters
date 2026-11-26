"""Pure summaries that inform split settings before any split exists.

Inputs are small pandas objects collected from Spark, so these run without Spark.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from dre.data.splits import is_heldout_annotator

_TOP_WORKER_SHARES = (0.01, 0.10, 0.25)
_VOLUME_QUANTILES = (0.5, 0.9, 0.99)


def worker_volume_summary(labels_per_worker: pd.Series) -> dict[str, float]:
    """Quantiles of labels per worker, and the share of all labels held by the busiest workers."""
    ordered = labels_per_worker.sort_values(ascending=False)
    total = float(ordered.sum())
    summary = {"workers": float(len(ordered)), "labels": total, "max": float(ordered.max())}
    for quantile in _VOLUME_QUANTILES:
        summary[f"p{round(quantile * 100)}"] = float(ordered.quantile(quantile))
    for share in _TOP_WORKER_SHARES:
        busiest = ordered.iloc[: math.ceil(len(ordered) * share)]
        label_share = float(busiest.sum()) / total
        summary[f"label_share_of_top_{round(share * 100)}pct_workers"] = label_share
    return summary


def heldout_label_share(
    labels_per_worker: pd.Series, salt: str, fractions: Sequence[float]
) -> dict[str, dict[str, float]]:
    """For each candidate fraction, the worker and label share the hash rule would hold out."""
    total = float(labels_per_worker.sum())
    shares = {}
    for fraction in fractions:
        heldout = np.array(
            [
                is_heldout_annotator(str(worker), salt, fraction)
                for worker in labels_per_worker.index
            ]
        )
        shares[f"{fraction:g}"] = {
            "worker_share": float(heldout.mean()),
            "label_share": float(labels_per_worker[heldout].sum()) / total,
            "labels": float(labels_per_worker[heldout].sum()),
        }
    return shares


def duplicate_text_summary(texts_by_id_count: pd.Series) -> dict[str, float]:
    """Summarize exact duplicate text.

    ``texts_by_id_count`` maps "number of ids sharing one normalized text" to "number of such
    texts".
    """
    ids_per_group = texts_by_id_count.index.to_series()
    ids_by_group = ids_per_group * texts_by_id_count
    duplicated = ids_per_group > 1
    total_ids = float(ids_by_group.sum())
    return {
        "texts": float(texts_by_id_count.sum()),
        "duplicated_texts": float(texts_by_id_count[duplicated].sum()),
        "ids_in_duplicated_texts": float(ids_by_group[duplicated].sum()),
        "share_of_ids_in_duplicated_texts": float(ids_by_group[duplicated].sum()) / total_ids,
    }


def require_enough_heldout_labels(label_count: int, minimum: int) -> None:
    if label_count < minimum:
        raise ValueError(
            f"held-out annotators have {label_count} labels, below the required {minimum}; "
            "raise heldout_annotator_fraction or lower min_heldout_annotations"
        )
