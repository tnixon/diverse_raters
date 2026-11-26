"""Downloads into a Unity Catalog volume. This is an I/O edge: it touches the network."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from pathlib import Path

from dre.data.config import DataConfig
from dre.settings import Settings

_CHUNK_BYTES = 1024 * 1024


def volume_files(settings: Settings, config: DataConfig) -> dict[str, Path]:
    """Where each downloaded file lives in the volume, keyed by a short name."""
    volume = Path(settings.volume_path)
    return {
        "jigsaw_comments": volume / "jigsaw" / config.kaggle.comments_file,
        "jigsaw_annotations": volume / "jigsaw" / config.kaggle.annotations_file,
        "mhs": volume / "mhs" / config.huggingface.filename,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_if_missing(destination: Path, fetch: Callable[[Path], None]) -> Path:
    """Run ``fetch(destination)`` unless the file exists, and check that it produced the file."""
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    fetch(destination)
    if not destination.exists():
        raise FileNotFoundError(f"download did not produce {destination}")
    return destination


def download_kaggle_file(competition: str, file_name: str, directory: Path, api_token: str) -> Path:
    def fetch(destination: Path) -> None:
        # kagglehub has no token argument; this env var is its non-interactive way in.
        os.environ["KAGGLE_API_TOKEN"] = api_token
        import kagglehub

        kagglehub.competition_download(competition, path=file_name, output_dir=str(directory))

    return fetch_if_missing(directory / file_name, fetch)


def download_huggingface_file(repo_id: str, filename: str, revision: str, directory: Path) -> Path:
    def fetch(destination: Path) -> None:
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            repo_type="dataset",
            local_dir=str(directory),
        )

    return fetch_if_missing(directory / filename, fetch)
