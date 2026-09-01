"""Environment and corpus preparation helpers."""

from __future__ import annotations

import glob
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .config import PipelineConfig


LOG = logging.getLogger(__name__)


def ensure_directories(config: PipelineConfig) -> None:
    LOG.info("Ensuring directories exist")
    config.db_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.local_embed_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("Directories ready: db_dir=%s output_dir=%s local_embed_dir=%s", config.db_dir, config.output_dir, config.local_embed_dir)


def mount_google_drive(config: PipelineConfig) -> None:
    """Mount Google Drive when the code is running inside Colab."""

    if not config.use_drive:
        LOG.info("USE_DRIVE is false; skipping Google Drive mount")
        return
    if Path("/content/drive/MyDrive").exists():
        LOG.info("Google Drive already appears mounted")
        return
    try:
        from google.colab import drive  # type: ignore
    except ImportError:
        LOG.info("google.colab is unavailable; skipping Google Drive mount")
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
        LOG.info("MedRAG repository already exists: %s", config.medrag_repo_dir)
        return config.medrag_repo_dir
    LOG.info("Cloning MedRAG repository into %s", config.medrag_repo_dir)
    config.medrag_repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "https://github.com/Teddy-XiongGZ/MedRAG.git", str(config.medrag_repo_dir)],
        check=True,
    )
    return config.medrag_repo_dir


def is_statpearls_ready(config: PipelineConfig) -> bool:
    chunk_dir = config.db_dir / "statpearls" / "chunk"
    return bool(glob.glob(str(chunk_dir / "*.jsonl")))


def prepare_runtime(config: PipelineConfig, mount_drive: bool = False) -> None:
    LOG.info("Runtime preparation started")
    if mount_drive:
        mount_google_drive(config)
    ensure_directories(config)
    ensure_medrag_repo(config)
    LOG.info("Runtime preparation finished")


def prepare_statpearls(config: PipelineConfig) -> None:
    """Work around MedRAG StatPearls chunking paths by running from root_dir."""

    LOG.info("StatPearls preparation started")
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
            LOG.info("Downloading StatPearls archive into %s", statpearls_dir)
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
        LOG.info("Extracting StatPearls archive: %s", tar_path)
        subprocess.run(["tar", "-xzf", str(tar_path), "-C", str(statpearls_dir)], check=True)

    n_nxml = len(glob.glob(str(raw_dir / "*.nxml")))
    LOG.info("StatPearls raw files found: %s", n_nxml)
    if n_nxml == 0:
        raise RuntimeError(f"No .nxml files found after extraction in {raw_dir}")

    if not chunk_dir.exists() or not glob.glob(str(chunk_dir / "*.jsonl")):
        script = config.medrag_repo_dir / "src" / "data" / "statpearls.py"
        if not script.exists():
            raise RuntimeError(f"Missing MedRAG StatPearls script: {script}")
        LOG.info("Chunking StatPearls with script=%s cwd=%s", script, config.root_dir)
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(config.root_dir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "MedRAG StatPearls chunking failed.\n"
                f"Command: {sys.executable} {script}\n"
                f"Working directory: {config.root_dir}\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

    n_chunks = len(glob.glob(str(chunk_dir / "*.jsonl")))
    LOG.info("StatPearls chunk files found: %s", n_chunks)
    if n_chunks == 0:
        raise RuntimeError(f"StatPearls chunking did not create JSONL files in {chunk_dir}")
    LOG.info("StatPearls preparation finished")


def disk_usage(path: Path) -> str:
    result = subprocess.run(["du", "-sh", str(path)], check=False, capture_output=True, text=True)
    return result.stdout.strip()
