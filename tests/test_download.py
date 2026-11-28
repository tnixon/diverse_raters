import hashlib

import pytest

from dre.data.download import ensure_file, sha256_file


def writer(name, content="new"):
    def fetch(staging):
        (staging / name).write_text(content)

    return fetch


def test_sha256_file(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"hello")
    assert sha256_file(path) == hashlib.sha256(b"hello").hexdigest()


def test_missing_file_is_fetched(tmp_path):
    destination = tmp_path / "nested" / "file.csv"
    assert ensure_file(destination, writer("file.csv"), reload=False) is True
    assert destination.read_text() == "new"


def test_existing_file_is_kept_without_reload(tmp_path):
    destination = tmp_path / "file.csv"
    destination.write_text("old")

    def must_not_run(staging):
        raise AssertionError("fetch must not run")

    assert ensure_file(destination, must_not_run, reload=False) is False
    assert destination.read_text() == "old"


def test_existing_file_is_replaced_with_reload(tmp_path):
    destination = tmp_path / "file.csv"
    destination.write_text("old")
    assert ensure_file(destination, writer("file.csv"), reload=True) is True
    assert destination.read_text() == "new"


def test_failed_fetch_leaves_no_file_and_the_rerun_fetches_again(tmp_path):
    destination = tmp_path / "file.csv"

    def dies_midway(staging):
        (staging / "file.csv").write_text("trunc")
        raise ConnectionError("dropped")

    with pytest.raises(ConnectionError):
        ensure_file(destination, dies_midway, reload=False)
    assert not destination.exists()
    assert ensure_file(destination, writer("file.csv"), reload=False) is True
    assert destination.read_text() == "new"


def test_failed_reload_keeps_the_previous_good_file(tmp_path):
    destination = tmp_path / "file.csv"
    destination.write_text("old")

    def fails(staging):
        raise ConnectionError("dropped")

    with pytest.raises(ConnectionError):
        ensure_file(destination, fails, reload=True)
    assert destination.read_text() == "old"


def test_fetch_that_produces_nothing_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        ensure_file(tmp_path / "file.csv", lambda staging: None, reload=False)


def test_staging_directory_is_removed(tmp_path):
    ensure_file(tmp_path / "file.csv", writer("file.csv"), reload=False)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["file.csv"]
