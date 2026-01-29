"""
Utility for loading datasets from various sources.
"""

import logging
import os
import warnings
from pathlib import Path

from datasets import Dataset as HFDataset  # type: ignore[import-untyped]
from datasets import DatasetDict as HFDatasetDict
from datasets import load_dataset, load_from_disk

from .s3_folders_manager import download_s3_folder

logger = logging.getLogger(__name__)


def hf_dataset_loader(
    data_path: str | Path,
    *,
    local_cache_dir: str | Path | None = None,
    force_download: bool = False,
    split: str | None = None,
    name: str | None = None,
    data_files: str | dict[str, str] | None = None,
    **kwargs,
) -> HFDataset | HFDatasetDict:
    """
    Load data from HuggingFace Hub, local path, or S3, automatically detecting the source.
    First tries to download from S3, then local cache, then loads from HuggingFace.

    Args:
        data_path: Path to the dataset (HuggingFace repo_id, local path, or S3 URI)
        local_cache_dir: Directory to store downloaded S3 data (default: ./.cache/data)
        split: Which split of the data to load (e.g., 'train', 'validation', 'test')
        name: Name of the dataset configuration to load
        data_files: Path(s) to the data files to load
        **kwargs: Additional arguments to pass to load_dataset

    Returns:
        Dataset or DatasetDict object depending on whether a split is specified
    """
    if local_cache_dir is None:
        local_cache_dir = os.path.join(".cache", "data")
    local_cache_dir = Path(local_cache_dir)
    data_path = str(data_path)

    if data_path.startswith("s3://"):
        s3_dirname = Path(data_path.rstrip("/")).name
        cache_subdir = local_cache_dir / s3_dirname
        if cache_subdir.exists() and not force_download:
            warnings.warn(
                f"Cache directory '{cache_subdir}' already exists. "
                "If you want to replace data use force_download = True argument.",
                UserWarning,
                stacklevel=2,
            )
            local_path = str(cache_subdir)
        else:
            logger.info(f"Downloading dataset from s3 path: {data_path}...")
            local_path = download_s3_folder(s3_path=data_path, local_dir=cache_subdir)
            logger.info(f"Saved dataset in the local cache dir: {data_path}...")

        return load_from_disk(str(local_path), **kwargs)

    elif os.path.exists(data_path):
        logger.info(f"Loading dataset from local path: {data_path}...")
        return load_from_disk(data_path, **kwargs)

    # Handle HuggingFace Hub paths (default assumption if not S3 or local)
    else:
        logger.info("You are trying to download dataset from HuggingFace Hub.")
        return load_dataset(data_path, split=split, name=name, data_files=data_files, **kwargs)
