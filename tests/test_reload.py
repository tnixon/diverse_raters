import pytest

from dre.reload import BuildResult, describe_results, row_counts, should_build


@pytest.mark.parametrize(
    ("exists", "reload", "expected"),
    [(False, False, True), (False, True, True), (True, False, False), (True, True, True)],
)
def test_should_build(exists, reload, expected):
    assert should_build(exists, reload) is expected


def test_row_counts_and_description():
    results = {"a": BuildResult(rows=10, built=True), "b": BuildResult(rows=3, built=False)}
    assert row_counts(results) == {"a": 10, "b": 3}
    assert describe_results(results) == "a: built (10 rows)\nb: kept (3 rows)"
