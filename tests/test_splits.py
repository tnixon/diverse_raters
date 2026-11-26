import pytest

from dre.data.splits import (
    assign_split,
    hash_unit,
    is_heldout_annotator,
    normalize_text,
    text_key,
)

FRACTIONS = {"train": 0.8, "validation": 0.1, "search": 0.1}
IDS = [str(i) for i in range(100_000)]


def test_hash_unit_is_deterministic_and_in_range():
    values = [hash_unit(key, "salt") for key in IDS[:1000]]
    assert values == [hash_unit(key, "salt") for key in IDS[:1000]]
    assert all(0 <= value < 1 for value in values)


def test_hash_unit_is_pinned_across_machines():
    # SHA-256 based, so this must never change; if it does, held-out sets have moved.
    assert hash_unit("12345", "salt") == pytest.approx(0.32669845889563315, abs=1e-12)


def test_different_salts_give_different_assignments():
    differing = sum(
        assign_split(key, "a", FRACTIONS) != assign_split(key, "b", FRACTIONS) for key in IDS[:2000]
    )
    assert differing > 500


def test_every_key_gets_one_known_split_with_expected_proportions():
    assignments = [assign_split(key, "salt", FRACTIONS) for key in IDS]
    assert set(assignments) == set(FRACTIONS)
    for name, fraction in FRACTIONS.items():
        assert assignments.count(name) / len(IDS) == pytest.approx(fraction, abs=0.01)


def test_assignment_follows_fraction_order():
    only_last = {"first": 1e-12, "last": 1 - 1e-12}
    assert {assign_split(key, "salt", only_last) for key in IDS[:1000]} == {"last"}


def test_heldout_annotator_fraction_is_respected():
    heldout = sum(is_heldout_annotator(key, "salt", 0.1) for key in IDS)
    assert heldout / len(IDS) == pytest.approx(0.1, abs=0.01)


def test_normalize_text_folds_case_whitespace_and_unicode():
    assert normalize_text("  Hello \n  WORLD\t") == "hello world"
    assert normalize_text("ＦＵＬＬ width") == "full width"


def test_identical_texts_share_a_key_and_therefore_a_split():
    variants = ["You are   right.", "you are right.\n", "YOU ARE RIGHT."]
    assert len({text_key(text) for text in variants}) == 1
    assert len({assign_split(text_key(text), "salt", FRACTIONS) for text in variants}) == 1


def test_different_texts_get_different_keys():
    assert text_key("one") != text_key("two")
