"""
Download — Dataset download and organization helper.

Downloads the garbage classification dataset from Kaggle and
organizes it into the 7 target classes used by this project.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.helpers import PROJECT_ROOT, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def download_from_kaggle(
    dataset_id: str,
    download_dir: str,
) -> Path:
    """
    Download a dataset from Kaggle using the Kaggle API.

    Args:
        dataset_id: Kaggle dataset identifier (e.g., 'mostafaabla/garbage-classification').
        download_dir: Directory to download the dataset into.

    Returns:
        Path to the downloaded (and extracted) dataset directory.

    Raises:
        ImportError: If kaggle package is not installed.
        RuntimeError: If download fails.
    """
    download_path = PROJECT_ROOT / download_dir
    download_path.mkdir(parents=True, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        logger.info(f"Downloading dataset '{dataset_id}' from Kaggle...")
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            dataset_id,
            path=str(download_path),
            unzip=True,
        )
        logger.info(f"Dataset downloaded and extracted to: {download_path}")
        return download_path

    except ImportError:
        logger.error(
            "Kaggle API not installed. Install with: pip install kaggle\n"
            "Then set up your API key at ~/.kaggle/kaggle.json"
        )
        raise
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise RuntimeError(f"Dataset download failed: {e}")


def organize_dataset(
    raw_dir: str,
    processed_dir: str,
    class_mapping: Dict[str, str],
    target_classes: List[str],
) -> Dict[str, int]:
    """
    Organize raw dataset into the 7 target classes.

    Maps the 12-class dataset to our 7 target classes by copying images
    into the appropriate directories.

    Args:
        raw_dir: Path to raw downloaded dataset.
        processed_dir: Path for organized output.
        class_mapping: Mapping from source class names to target class names.
        target_classes: List of target class names.

    Returns:
        Dictionary with per-class image counts.
    """
    raw_path = PROJECT_ROOT / raw_dir
    processed_path = PROJECT_ROOT / processed_dir

    logger.info(f"Organizing dataset from {raw_path} → {processed_path}")

    # Create target class directories
    for cls in target_classes:
        (processed_path / cls).mkdir(parents=True, exist_ok=True)

    # Find the actual data directory (may be nested)
    data_root = _find_data_root(raw_path)
    if data_root is None:
        raise FileNotFoundError(
            f"Could not find image class folders in {raw_path}. "
            "Please ensure the dataset is properly extracted."
        )

    logger.info(f"Found data root: {data_root}")

    # Copy and organize images
    class_counts: Dict[str, int] = {cls: 0 for cls in target_classes}
    skipped = 0

    for source_folder in sorted(data_root.iterdir()):
        if not source_folder.is_dir():
            continue

        source_class = source_folder.name.lower().replace(" ", "-")
        target_class = class_mapping.get(source_class)

        if target_class is None:
            logger.warning(
                f"Unknown source class '{source_class}', skipping folder."
            )
            skipped += 1
            continue

        if target_class not in target_classes:
            logger.warning(
                f"Target class '{target_class}' not in target classes, skipping."
            )
            skipped += 1
            continue

        target_dir = processed_path / target_class
        image_count = 0

        for img_file in source_folder.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".bmp", ".webp"
            }:
                # Create unique filename to avoid collisions when merging classes
                new_name = f"{source_class}_{img_file.name}"
                dest = target_dir / new_name

                if not dest.exists():
                    shutil.copy2(str(img_file), str(dest))
                    image_count += 1

        class_counts[target_class] += image_count
        logger.info(
            f"  {source_class} → {target_class}: {image_count} images"
        )

    total = sum(class_counts.values())
    logger.info(f"Dataset organized: {total} total images across {len(target_classes)} classes")

    if skipped > 0:
        logger.warning(f"Skipped {skipped} unknown source folders")

    # Print class distribution
    for cls, count in sorted(class_counts.items()):
        logger.info(f"  {cls}: {count} images ({count/max(total,1)*100:.1f}%)")

    return class_counts


def _find_data_root(path: Path) -> Optional[Path]:
    """
    Recursively find the root directory containing class folders.

    The data root is the directory that contains subdirectories with image files.

    Args:
        path: Starting path to search.

    Returns:
        Path to the data root, or None if not found.
    """
    # Check if current directory contains class subdirectories with images
    subdirs = [d for d in path.iterdir() if d.is_dir()]

    if not subdirs:
        return None

    # Check if subdirectories contain image files
    has_images = False
    for subdir in subdirs:
        images = list(subdir.glob("*.jpg")) + list(subdir.glob("*.png")) + list(subdir.glob("*.jpeg"))
        if images:
            has_images = True
            break

    if has_images:
        return path

    # Recurse into subdirectories
    for subdir in subdirs:
        if subdir.name.startswith("."):
            continue
        result = _find_data_root(subdir)
        if result is not None:
            return result

    return None


def create_sample_dataset(processed_dir: str, target_classes: List[str]) -> None:
    """
    Create a small sample dataset with placeholder images for testing/demo.

    This allows the Streamlit app to run in demo mode without downloading
    the full dataset.

    Args:
        processed_dir: Path for the processed dataset.
        target_classes: List of class names.
    """
    from PIL import Image
    import numpy as np

    processed_path = PROJECT_ROOT / processed_dir
    logger.info(f"Creating sample dataset at {processed_path}")

    # Class colors for sample images
    class_colors = {
        "cardboard": (196, 154, 108),
        "glass": (135, 206, 235),
        "metal": (184, 184, 184),
        "organic": (139, 69, 19),
        "paper": (245, 245, 220),
        "plastic": (255, 107, 107),
        "trash": (105, 105, 105),
    }

    for cls in target_classes:
        cls_dir = processed_path / cls
        cls_dir.mkdir(parents=True, exist_ok=True)

        color = class_colors.get(cls, (128, 128, 128))

        # Create 10 sample images per class
        for i in range(10):
            img_array = np.full((224, 224, 3), color, dtype=np.uint8)
            # Add some noise for variation
            noise = np.random.randint(-20, 20, img_array.shape, dtype=np.int16)
            img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            img = Image.fromarray(img_array)
            img.save(str(cls_dir / f"sample_{cls}_{i:03d}.jpg"))

    logger.info(f"Created sample dataset with {len(target_classes) * 10} images")


def setup_dataset(config: Optional[Dict] = None) -> Dict[str, int]:
    """
    Full dataset setup pipeline: download, extract, and organize.

    Args:
        config: Configuration dictionary. If None, loads from default path.

    Returns:
        Dictionary with per-class image counts.
    """
    if config is None:
        config = load_config()

    dataset_cfg = config["dataset"]
    raw_dir = dataset_cfg["raw_dir"]
    processed_dir = dataset_cfg["processed_dir"]
    target_classes = dataset_cfg["classes"]
    class_mapping = dataset_cfg["class_mapping"]

    processed_path = PROJECT_ROOT / processed_dir

    # Check if dataset is already organized
    if processed_path.exists():
        existing_classes = [
            d.name for d in processed_path.iterdir()
            if d.is_dir() and d.name in target_classes
        ]
        if len(existing_classes) == len(target_classes):
            # Count existing images
            class_counts = {}
            for cls in target_classes:
                cls_dir = processed_path / cls
                count = len(list(cls_dir.glob("*.*")))
                class_counts[cls] = count

            total = sum(class_counts.values())
            if total > 50:  # More than just sample data
                logger.info(
                    f"Dataset already organized: {total} images. Skipping download."
                )
                return class_counts

    # Try downloading from Kaggle
    raw_path = PROJECT_ROOT / raw_dir
    if not raw_path.exists() or not any(raw_path.iterdir()):
        try:
            download_from_kaggle(
                dataset_id=dataset_cfg["kaggle_dataset"],
                download_dir=raw_dir,
            )
        except (ImportError, RuntimeError) as e:
            logger.warning(f"Kaggle download failed: {e}")
            logger.info("Creating sample dataset for demo mode...")
            create_sample_dataset(processed_dir, target_classes)
            return {cls: 10 for cls in target_classes}

    # Organize dataset
    class_counts = organize_dataset(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        class_mapping=class_mapping,
        target_classes=target_classes,
    )

    return class_counts


if __name__ == "__main__":
    """Run dataset setup from command line."""
    counts = setup_dataset()
    print("\nDataset Setup Complete!")
    print("=" * 40)
    for cls, count in sorted(counts.items()):
        print(f"  {cls}: {count} images")
    print(f"  Total: {sum(counts.values())} images")
