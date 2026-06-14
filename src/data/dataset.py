"""
Dataset — Data loading, splitting, augmentation, and generator creation.

Handles the full data pipeline from organized images to Keras generators
ready for model training, validation, and testing.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.utils.helpers import PROJECT_ROOT, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GarbageDataset:
    """
    Manages the garbage classification dataset pipeline.

    Handles loading image paths, stratified splitting into train/val/test,
    creating Keras ImageDataGenerators with augmentation, and computing
    class weights for imbalance handling.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the dataset manager.

        Args:
            config: Configuration dictionary. Loads default if None.
        """
        if config is None:
            config = load_config()

        self.config = config
        self.dataset_cfg = config["dataset"]
        self.image_cfg = config["image"]
        self.aug_cfg = config.get("augmentation", {})

        self.processed_dir = PROJECT_ROOT / self.dataset_cfg["processed_dir"]
        self.classes = sorted(self.dataset_cfg["classes"])
        self.num_classes = len(self.classes)
        self.image_size = (self.image_cfg["width"], self.image_cfg["height"])

        # Data holders
        self.image_paths: List[str] = []
        self.labels: List[int] = []
        self.class_to_idx: Dict[str, int] = {
            cls: idx for idx, cls in enumerate(self.classes)
        }
        self.idx_to_class: Dict[int, str] = {
            idx: cls for cls, idx in self.class_to_idx.items()
        }

        # Split holders
        self.train_paths: List[str] = []
        self.train_labels: List[int] = []
        self.val_paths: List[str] = []
        self.val_labels: List[int] = []
        self.test_paths: List[str] = []
        self.test_labels: List[int] = []

        logger.info(
            f"GarbageDataset initialized with {self.num_classes} classes: {self.classes}"
        )

    def load_data(self) -> Tuple[List[str], List[int]]:
        """
        Load all image paths and labels from the processed dataset directory.

        Returns:
            Tuple of (image_paths, labels).

        Raises:
            FileNotFoundError: If processed directory doesn't exist.
        """
        if not self.processed_dir.exists():
            raise FileNotFoundError(
                f"Processed dataset directory not found: {self.processed_dir}. "
                "Run the download script first."
            )

        self.image_paths = []
        self.labels = []

        for cls_name in self.classes:
            cls_dir = self.processed_dir / cls_name
            if not cls_dir.exists():
                logger.warning(f"Class directory not found: {cls_dir}")
                continue

            for img_file in cls_dir.iterdir():
                if img_file.is_file() and img_file.suffix.lower() in {
                    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
                }:
                    self.image_paths.append(str(img_file))
                    self.labels.append(self.class_to_idx[cls_name])

        logger.info(f"Loaded {len(self.image_paths)} images from {self.num_classes} classes")
        return self.image_paths, self.labels

    def split_data(
        self,
        train_ratio: Optional[float] = None,
        val_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None,
        seed: int = 42,
    ) -> Dict[str, int]:
        """
        Split data into train, validation, and test sets with stratification.

        Args:
            train_ratio: Fraction for training. Defaults to config value.
            val_ratio: Fraction for validation. Defaults to config value.
            test_ratio: Fraction for testing. Defaults to config value.
            seed: Random seed for reproducibility.

        Returns:
            Dictionary with split sizes.
        """
        if not self.image_paths:
            self.load_data()

        train_ratio = train_ratio or self.dataset_cfg["train_ratio"]
        val_ratio = val_ratio or self.dataset_cfg["val_ratio"]
        test_ratio = test_ratio or self.dataset_cfg["test_ratio"]

        # First split: train+val vs test
        train_val_paths, self.test_paths, train_val_labels, self.test_labels = (
            train_test_split(
                self.image_paths,
                self.labels,
                test_size=test_ratio,
                stratify=self.labels,
                random_state=seed,
            )
        )

        # Second split: train vs val
        relative_val_ratio = val_ratio / (train_ratio + val_ratio)
        self.train_paths, self.val_paths, self.train_labels, self.val_labels = (
            train_test_split(
                train_val_paths,
                train_val_labels,
                test_size=relative_val_ratio,
                stratify=train_val_labels,
                random_state=seed,
            )
        )

        split_info = {
            "train": len(self.train_paths),
            "val": len(self.val_paths),
            "test": len(self.test_paths),
            "total": len(self.image_paths),
        }

        logger.info(
            f"Data split — Train: {split_info['train']}, "
            f"Val: {split_info['val']}, Test: {split_info['test']}"
        )

        return split_info

    def create_generators(
        self,
        batch_size: Optional[int] = None,
    ) -> Tuple[Any, Any, Any]:
        """
        Create Keras ImageDataGenerators for train, validation, and test.

        Training generator includes data augmentation; validation and test
        generators only apply rescaling.

        Args:
            batch_size: Batch size for generators. Defaults to config value.

        Returns:
            Tuple of (train_generator, val_generator, test_generator).
        """
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        import pandas as pd

        if not self.train_paths:
            self.split_data()

        batch_size = batch_size or self.config["training"]["phase1"]["batch_size"]

        # Training generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            rotation_range=self.aug_cfg.get("rotation_range", 20),
            width_shift_range=self.aug_cfg.get("width_shift_range", 0.2),
            height_shift_range=self.aug_cfg.get("height_shift_range", 0.2),
            shear_range=self.aug_cfg.get("shear_range", 0.15),
            zoom_range=self.aug_cfg.get("zoom_range", 0.15),
            horizontal_flip=self.aug_cfg.get("horizontal_flip", True),
            fill_mode=self.aug_cfg.get("fill_mode", "nearest"),
            brightness_range=self.aug_cfg.get("brightness_range", [0.8, 1.2]),
        )

        # Validation and test generators (only rescaling)
        val_test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

        # Create DataFrames for flow_from_dataframe
        train_df = pd.DataFrame({
            "filename": self.train_paths,
            "class": [self.idx_to_class[l] for l in self.train_labels],
        })

        val_df = pd.DataFrame({
            "filename": self.val_paths,
            "class": [self.idx_to_class[l] for l in self.val_labels],
        })

        test_df = pd.DataFrame({
            "filename": self.test_paths,
            "class": [self.idx_to_class[l] for l in self.test_labels],
        })

        # Common generator params
        gen_params = dict(
            x_col="filename",
            y_col="class",
            target_size=self.image_size,
            batch_size=batch_size,
            classes=self.classes,
            class_mode="categorical",
        )

        train_gen = train_datagen.flow_from_dataframe(
            train_df, **gen_params, shuffle=True
        )

        val_gen = val_test_datagen.flow_from_dataframe(
            val_df, **gen_params, shuffle=False
        )

        test_gen = val_test_datagen.flow_from_dataframe(
            test_df, **gen_params, shuffle=False
        )

        logger.info(
            f"Created generators — batch_size={batch_size}, "
            f"image_size={self.image_size}, augmentation=True (train)"
        )

        return train_gen, val_gen, test_gen

    def compute_class_weights(self) -> Dict[int, float]:
        """
        Compute class weights to handle class imbalance.

        Uses inverse frequency weighting via sklearn.

        Returns:
            Dictionary mapping class index to weight.
        """
        if not self.train_labels:
            self.split_data()

        labels_array = np.array(self.train_labels)
        unique_classes = np.unique(labels_array)

        weights = compute_class_weight(
            class_weight="balanced",
            classes=unique_classes,
            y=labels_array,
        )

        class_weights = {int(cls): float(w) for cls, w in zip(unique_classes, weights)}

        logger.info("Computed class weights:")
        for idx, weight in sorted(class_weights.items()):
            cls_name = self.idx_to_class[idx]
            count = int(np.sum(labels_array == idx))
            logger.info(f"  {cls_name}: weight={weight:.4f} (n={count})")

        return class_weights

    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive dataset statistics.

        Returns:
            Dictionary with per-class counts, total images, and split info.
        """
        if not self.image_paths:
            self.load_data()

        labels_array = np.array(self.labels)
        stats: Dict[str, Any] = {
            "total_images": len(self.image_paths),
            "num_classes": self.num_classes,
            "classes": self.classes,
            "per_class": {},
            "image_size": self.image_size,
        }

        for idx, cls_name in self.idx_to_class.items():
            count = int(np.sum(labels_array == idx))
            stats["per_class"][cls_name] = {
                "count": count,
                "percentage": round(count / max(len(self.labels), 1) * 100, 1),
            }

        # Add split info if available
        if self.train_paths:
            stats["splits"] = {
                "train": len(self.train_paths),
                "val": len(self.val_paths),
                "test": len(self.test_paths),
            }

        return stats
