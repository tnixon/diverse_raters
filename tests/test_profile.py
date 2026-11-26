import pandas as pd
import pytest

from dre.data.profile import (
    duplicate_text_summary,
    heldout_label_share,
    require_enough_heldout_labels,
    worker_volume_summary,
)


def skewed_counts() -> pd.Series:
    # One heavy worker and 99 light ones.
    return pd.Series([1000] + [10] * 99, index=[f"w{i}" for i in range(100)])


def test_worker_volume_summary_shows_skew():
    summary = worker_volume_summary(skewed_counts())
    assert summary["workers"] == 100
    assert summary["labels"] == 1990
    assert summary["max"] == 1000
    assert summary["p50"] == 10
    assert summary["label_share_of_top_1pct_workers"] == pytest.approx(1000 / 1990)


def test_heldout_label_share_reports_workers_and_labels():
    counts = pd.Series(10, index=[str(i) for i in range(20_000)])
    shares = heldout_label_share(counts, "salt", [0.1, 0.2])
    assert set(shares) == {"0.1", "0.2"}
    assert shares["0.1"]["worker_share"] == pytest.approx(0.1, abs=0.01)
    assert shares["0.1"]["label_share"] == pytest.approx(shares["0.1"]["worker_share"])
    assert shares["0.2"]["labels"] > shares["0.1"]["labels"]


def test_heldout_label_share_follows_the_hash_rule_per_worker():
    counts = skewed_counts()
    first = heldout_label_share(counts, "salt", [0.3])
    assert first == heldout_label_share(counts, "salt", [0.3])


def test_duplicate_text_summary():
    # 90 texts with one id, 5 texts shared by two ids, 1 text shared by ten ids.
    texts_by_id_count = pd.Series({1: 90, 2: 5, 10: 1})
    summary = duplicate_text_summary(texts_by_id_count)
    assert summary["texts"] == 96
    assert summary["duplicated_texts"] == 6
    assert summary["ids_in_duplicated_texts"] == 20
    assert summary["share_of_ids_in_duplicated_texts"] == pytest.approx(20 / 110)


def test_no_duplicates_gives_zero_share():
    summary = duplicate_text_summary(pd.Series({1: 50}))
    assert summary["duplicated_texts"] == 0
    assert summary["share_of_ids_in_duplicated_texts"] == 0


def test_require_enough_heldout_labels():
    require_enough_heldout_labels(100, 100)
    with pytest.raises(ValueError, match="below the required"):
        require_enough_heldout_labels(99, 100)
