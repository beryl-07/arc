"""Environment and corpus preparation helpers."""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
from pathlib import Path

from .config import PipelineConfig


def ensure_directories(config: PipelineConfig) -> None:
    config.db_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.local_embed_dir.mkdir(parents=True, exist_ok=True)


def mount_google_drive(config: PipelineConfig) -> None:
    """Mount Google Drive when the code is running inside Colab."""

    if not config.use_drive:
        return
    if Path("/content/drive/MyDrive").exists():
        return
    try:
        from google.colab import drive  # type: ignore
    except ImportError:
        return
    try:
        drive.mount("/content/drive")
    except AttributeError as exc:
        raise RuntimeError(
            "Google Drive mounting must run inside a Colab notebook cell, not "
            "inside a plain Python subprocess. Mount Drive in the notebook first "
            "or run the CLI without --mount-drive after Drive is mounted."
        ) from exc


def ensure_medrag_repo(config: PipelineConfig) -> Path:
    """Clone MedRAG if it is not already present."""

    if config.medrag_repo_dir.exists():
        return config.medrag_repo_dir
    config.medrag_repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "https://github.com/Teddy-XiongGZ/MedRAG.git", str(config.medrag_repo_dir)],
        check=True,
    )
    return config.medrag_repo_dir


def prepare_runtime(config: PipelineConfig, mount_drive: bool = False) -> None:
    if mount_drive:
        mount_google_drive(config)
    ensure_directories(config)
    ensure_medrag_repo(config)


def prepare_statpearls(config: PipelineConfig) -> None:
    """Work around MedRAG StatPearls chunking paths by running from root_dir."""

    statpearls_dir = config.db_dir / "statpearls"
    raw_dir = statpearls_dir / "statpearls_NBK430685"
    tar_path = statpearls_dir / "statpearls_NBK430685.tar.gz"
    chunk_dir = statpearls_dir / "chunk"
    statpearls_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists() or not glob.glob(str(raw_dir / "*.nxml")):
        if not tar_path.exists():
            downloader = shutil.which("wget")
            if downloader is None:
                raise RuntimeError("wget is required to download the StatPearls archive.")
            subprocess.run(
                [
                    downloader,
                    "-q",
                    "https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz",
                    "-P",
                    str(statpearls_dir),
                ],
                check=True,
            )
        subprocess.run(["tar", "-xzf", str(tar_path), "-C", str(statpearls_dir)], check=True)

    n_nxml = len(glob.glob(str(raw_dir / "*.nxml")))
    if n_nxml == 0:
        raise RuntimeError(f"No .nxml files found after extraction in {raw_dir}")

    if not chunk_dir.exists() or not glob.glob(str(chunk_dir / "*.jsonl")):
        script = config.medrag_repo_dir / "src" / "data" / "statpearls.py"
        if not script.exists():
            raise RuntimeError(f"Missing MedRAG StatPearls script: {script}")
        subprocess.run(["python", str(script)], cwd=str(config.root_dir), check=True)

    n_chunks = len(glob.glob(str(chunk_dir / "*.jsonl")))
    if n_chunks == 0:
        raise RuntimeError(f"StatPearls chunking did not create JSONL files in {chunk_dir}")


def disk_usage(path: Path) -> str:
    result = subprocess.run(["du", "-sh", str(path)], check=False, capture_output=True, text=True)
    return result.stdout.strip()
