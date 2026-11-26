import hashlib

import pytest

from dre.data.download import fetch_if_missing, sha256_file


def test_sha256_file(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"hello")
    assert sha256_file(path) == hashlib.sha256(b"hello").hexdigest()


def test_fetch_runs_when_missing_and_skips_when_present(tmp_path):
    destination = tmp_path / "nested" / "file.csv"
    calls = []

    def fake_fetch(path):
        calls.append(path)
        path.write_text("x")

    assert fetch_if_missing(destination, fake_fetch) == destination
    assert fetch_if_missing(destination, fake_fetch) == destination
    assert calls == [destination]


def test_fetch_that_produces_nothing_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        fetch_if_missing(tmp_path / "missing.csv", lambda path: None)
