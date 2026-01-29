"""
Utilities for managing S3 folders.
"""

import shutil
import subprocess
from pathlib import Path

import s3fs  # type: ignore[import-untyped]


def is_s3_path(path):
    return isinstance(path, str) and path.startswith("s3://")


def sync_s3_folder(*, local_dir: str | Path, s3_path: str) -> str:
    """
    Upload the contents of a local directory to S3
    Args:
        local_dir: a relative or absolute directory path in the local file system
        s3_path: the S3 URI (s3://bucket-name/folder-path)
    """
    # s3 = s3fs.S3FileSystem()
    # s3.put(local_dir, s3_path, recursive=True)
    result = subprocess.run(
        ["aws", "s3", "sync", str(local_dir), str(s3_path)],  # Local directory  # S3 URI
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("Error uploading to S3:", result.stderr)
    else:
        print("Upload successful.")

    return s3_path


def upload_s3_folder(*, local_dir: str | Path, s3_path: str) -> str:
    """
    Upload the contents of a local directory to S3 without sync semantics (no deletions).
    Args:
        local_dir: a relative or absolute directory path in the local file system
        s3_path: the S3 URI (s3://bucket-name/folder-path)
    """
    result = subprocess.run(
        ["aws", "s3", "cp", str(local_dir), str(s3_path), "--recursive"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Error uploading to S3:", result.stderr)
    else:
        print("Upload successful.")
    return s3_path


def download_s3_folder(*, s3_path: str, local_dir: str | Path) -> str:
    """
    Download the contents of a folder directory
    Args:
        s3_path: the S3 URI (s3://bucket-name/folder-path)
        local_dir: a relative or absolute directory path in the local file system
    """
    s3 = s3fs.S3FileSystem()
    local_dir = Path(local_dir)
    # Track whether the folder existed before this call
    existed_before = local_dir.exists()
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        s3.get(s3_path, local_dir, recursive=True)
    except Exception:
        if not existed_before and local_dir.exists():
            shutil.rmtree(local_dir, ignore_errors=True)
        raise

    return str(local_dir)


def get_local_object(path: str | Path, local_dir: str | Path):

    if is_s3_path(path):
        Path(local_dir).mkdir(exist_ok=True, parents=True)
        print(f"Downloading from S3: {path}")
        if not path.endswith("/"):
            path += "/"
        local_model_path = download_s3_folder(s3_path=path, local_dir=local_dir)
        return local_model_path
    else:
        return path
