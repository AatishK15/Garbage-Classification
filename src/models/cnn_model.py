"""
CNN Model — Custom Convolutional Neural Network for garbage classification.

Serves as the performance baseline before applying transfer learning.
A moderately deep CNN with BatchNorm, MaxPooling, and Dropout.
"""

from typing import Any, Dict, List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models

from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_custom_cnn(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 7,
    filters: Optional[List[int]] = None,
    kernel_size: int = 3,
    pool_size: int = 2,
    dense_units: int = 256,
    dropout_rate: float = 0.5,
) -> tf.keras.Model:
    """
    Build a custom CNN model for image classification.

    Architecture:
        4 Conv2D blocks (each with Conv → BatchNorm → ReLU → MaxPool → Dropout)
        → GlobalAveragePooling2D → Dense → Dropout → Output (softmax)

    Args:
        input_shape: Input image shape (height, width, channels).
        num_classes: Number of output classes.
        filters: List of filter counts for each conv block.
        kernel_size: Convolution kernel size.
        pool_size: Max pooling window size.
        dense_units: Number of units in the dense layer.
        dropout_rate: Dropout rate for regularization.

    Returns:
        Compiled Keras Model.
    """
    if filters is None:
        filters = [32, 64, 128, 256]

    logger.info(
        f"Building Custom CNN — input_shape={input_shape}, "
        f"num_classes={num_classes}, filters={filters}"
    )

    model = models.Sequential(name="Custom_CNN")

    # Input layer
    model.add(layers.InputLayer(input_shape=input_shape))

    # Convolutional blocks
    for i, num_filters in enumerate(filters):
        # First conv in block
        model.add(
            layers.Conv2D(
                num_filters,
                (kernel_size, kernel_size),
                padding="same",
                name=f"conv{i+1}_1",
            )
        )
        model.add(layers.BatchNormalization(name=f"bn{i+1}_1"))
        model.add(layers.Activation("relu", name=f"relu{i+1}_1"))

        # Second conv in block (deeper blocks)
        if i >= 1:
            model.add(
                layers.Conv2D(
                    num_filters,
                    (kernel_size, kernel_size),
                    padding="same",
                    name=f"conv{i+1}_2",
                )
            )
            model.add(layers.BatchNormalization(name=f"bn{i+1}_2"))
            model.add(layers.Activation("relu", name=f"relu{i+1}_2"))

        model.add(layers.MaxPooling2D((pool_size, pool_size), name=f"pool{i+1}"))
        model.add(layers.Dropout(0.25, name=f"drop{i+1}"))

    # Global Average Pooling
    model.add(layers.GlobalAveragePooling2D(name="global_avg_pool"))

    # Dense classification head
    model.add(layers.Dense(dense_units, name="dense1"))
    model.add(layers.BatchNormalization(name="bn_dense"))
    model.add(layers.Activation("relu", name="relu_dense"))
    model.add(layers.Dropout(dropout_rate, name="drop_dense"))

    # Output layer
    model.add(layers.Dense(num_classes, activation="softmax", name="output"))

    # Log model summary
    total_params = model.count_params()
    logger.info(f"Custom CNN built — Total parameters: {total_params:,}")

    return model


def build_custom_cnn_from_config(
    config: Dict[str, Any],
    num_classes: int = 7,
) -> tf.keras.Model:
    """
    Build custom CNN using parameters from configuration.

    Args:
        config: Configuration dictionary.
        num_classes: Number of output classes.

    Returns:
        Keras Model.
    """
    cnn_cfg = config.get("model", {}).get("custom_cnn", {})
    image_cfg = config.get("image", {})

    input_shape = (
        image_cfg.get("height", 224),
        image_cfg.get("width", 224),
        image_cfg.get("channels", 3),
    )

    return build_custom_cnn(
        input_shape=input_shape,
        num_classes=num_classes,
        filters=cnn_cfg.get("filters", [32, 64, 128, 256]),
        kernel_size=cnn_cfg.get("kernel_size", 3),
        pool_size=cnn_cfg.get("pool_size", 2),
        dense_units=cnn_cfg.get("dense_units", 256),
        dropout_rate=cnn_cfg.get("dropout_rate", 0.5),
    )
