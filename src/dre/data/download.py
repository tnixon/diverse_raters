"""Downloads into a Unity Catalog volume. This is an I/O edge: it touches the network."""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Callable
from pathlib import Path

from dre.data.config import DataConfig
from dre.reload import should_build
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


def ensure_file(destination: Path, fetch: Callable[[Path], None], reload: bool) -> bool:
    """Download ``destination`` if it is missing, or if ``reload`` asks for a fresh copy.

    ``fetch(staging_dir)`` must write a file named like ``destination`` into the staging
    directory. The finished file is then moved into place, so a download that fails part-way
    never leaves a truncated file behind for a later run to mistake for a complete one, and a
    failed reload keeps the previous good copy. Returns whether a download happened.
    """
    if not should_build(destination.exists(), reload):
        return False
    staging = destination.parent / f".{destination.name}.staging"
    shutil.rmtree(staging, ignore_errors=True)  # leftover from an earlier failed run
    staging.mkdir(parents=True)
    try:
        fetch(staging)
        staged = staging / destination.name
        if not staged.exists():
            raise FileNotFoundError(f"download did not produce {destination.name}")
        shutil.move(staged, destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return True


def download_kaggle_file(competition: str, destination: Path, api_token: str, reload: bool) -> bool:
    def fetch(staging: Path) -> None:
        # kagglehub has no token argument; this env var is its non-interactive way in.
        os.environ["KAGGLE_API_TOKEN"] = api_token
        import kagglehub

        kagglehub.competition_download(competition, path=destination.name, output_dir=str(staging))

    return ensure_file(destination, fetch, reload)


def download_huggingface_file(repo_id: str, revision: str, destination: Path, reload: bool) -> bool:
    def fetch(staging: Path) -> None:
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id=repo_id,
            filename=destination.name,
            revision=revision,
            repo_type="dataset",
            local_dir=str(staging),
        )

    return ensure_file(destination, fetch, reload)
