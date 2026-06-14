"""
Transfer Model — Transfer learning architectures for garbage classification.

Supports MobileNetV2, ResNet50, and EfficientNetB0 with ImageNet pre-trained
weights and a custom classification head.
"""

from typing import Any, Dict, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, Model

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Supported base model architectures
SUPPORTED_MODELS = {
    "mobilenetv2": tf.keras.applications.MobileNetV2,
    "resnet50": tf.keras.applications.ResNet50,
    "efficientnet": tf.keras.applications.EfficientNetB0,
}

# Default input preprocessing functions per architecture
PREPROCESS_FUNCTIONS = {
    "mobilenetv2": tf.keras.applications.mobilenet_v2.preprocess_input,
    "resnet50": tf.keras.applications.resnet50.preprocess_input,
    "efficientnet": tf.keras.applications.efficientnet.preprocess_input,
}


def build_transfer_model(
    base_model_name: str = "mobilenetv2",
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 7,
    freeze_base: bool = True,
    dense_units: int = 256,
    dropout_rate: float = 0.3,
    use_batch_norm: bool = True,
) -> tf.keras.Model:
    """
    Build a transfer learning model with a pre-trained base and custom head.

    Architecture:
        Pre-trained Base (ImageNet) → GlobalAveragePooling2D
        → BatchNorm → Dense(256, relu) → Dropout → Dense(num_classes, softmax)

    Args:
        base_model_name: Name of the base architecture
                         ('mobilenetv2', 'resnet50', 'efficientnet').
        input_shape: Input image shape (height, width, channels).
        num_classes: Number of output classes.
        freeze_base: If True, freeze all base model layers initially.
        dense_units: Number of units in the dense layer.
        dropout_rate: Dropout rate for regularization.
        use_batch_norm: Whether to add BatchNormalization.

    Returns:
        Keras Model (not compiled).

    Raises:
        ValueError: If base_model_name is not supported.
    """
    base_model_name = base_model_name.lower()

    if base_model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{base_model_name}'. "
            f"Supported: {list(SUPPORTED_MODELS.keys())}"
        )

    logger.info(
        f"Building {base_model_name} transfer model — "
        f"input_shape={input_shape}, num_classes={num_classes}, "
        f"freeze_base={freeze_base}"
    )

    # Create base model with ImageNet weights
    base_model_cls = SUPPORTED_MODELS[base_model_name]
    base_model = base_model_cls(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )

    # Freeze/unfreeze base model layers
    base_model.trainable = not freeze_base

    if freeze_base:
        logger.info(
            f"Froze {len(base_model.layers)} base model layers "
            f"(Phase 1: feature extraction)"
        )

    # Build the full model
    inputs = tf.keras.Input(shape=input_shape, name="input_image")

    # Apply base model preprocessing
    if base_model_name in PREPROCESS_FUNCTIONS:
        x = PREPROCESS_FUNCTIONS[base_model_name](inputs)
    else:
        x = inputs

    # Base model feature extraction
    x = base_model(x, training=False)

    # Classification head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    if use_batch_norm:
        x = layers.BatchNormalization(name="bn_head")(x)

    x = layers.Dense(dense_units, activation="relu", name="dense_head")(x)
    x = layers.Dropout(dropout_rate, name="dropout_head")(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    # Create model
    model_name = f"Transfer_{base_model_name.capitalize()}"
    model = Model(inputs=inputs, outputs=outputs, name=model_name)

    # Log parameters
    trainable = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    non_trainable = sum(
        tf.keras.backend.count_params(w) for w in model.non_trainable_weights
    )
    logger.info(
        f"{model_name} built — "
        f"Trainable: {trainable:,}, Non-trainable: {non_trainable:,}, "
        f"Total: {trainable + non_trainable:,}"
    )

    return model


def unfreeze_model(
    model: tf.keras.Model,
    unfreeze_top_percent: float = 0.3,
) -> tf.keras.Model:
    """
    Unfreeze the top layers of the base model for fine-tuning (Phase 2).

    Args:
        model: The transfer learning model.
        unfreeze_top_percent: Fraction of base model layers to unfreeze
                              from the top (0.3 = top 30%).

    Returns:
        The model with partially unfrozen base layers.
    """
    # Find the base model layer (it's typically the second layer)
    base_model = None
    for layer in model.layers:
        if hasattr(layer, "layers") and len(layer.layers) > 10:
            base_model = layer
            break

    if base_model is None:
        logger.warning("Could not find base model layer for unfreezing.")
        return model

    # Make base model trainable
    base_model.trainable = True

    # Calculate how many layers to keep frozen
    total_layers = len(base_model.layers)
    freeze_until = int(total_layers * (1 - unfreeze_top_percent))

    # Freeze bottom layers, unfreeze top layers
    for i, layer in enumerate(base_model.layers):
        if i < freeze_until:
            layer.trainable = False
        else:
            layer.trainable = True

    unfrozen = total_layers - freeze_until
    logger.info(
        f"Fine-tuning: Unfroze top {unfrozen}/{total_layers} layers "
        f"({unfreeze_top_percent*100:.0f}%)"
    )

    return model


def build_transfer_model_from_config(
    config: Dict[str, Any],
    model_name: str = "mobilenetv2",
    num_classes: int = 7,
) -> tf.keras.Model:
    """
    Build a transfer learning model using configuration parameters.

    Args:
        config: Configuration dictionary.
        model_name: Name of the base architecture.
        num_classes: Number of output classes.

    Returns:
        Keras Model.
    """
    image_cfg = config.get("image", {})
    transfer_cfg = config.get("model", {}).get("transfer", {})

    input_shape = (
        image_cfg.get("height", 224),
        image_cfg.get("width", 224),
        image_cfg.get("channels", 3),
    )

    return build_transfer_model(
        base_model_name=model_name,
        input_shape=input_shape,
        num_classes=num_classes,
        freeze_base=transfer_cfg.get("freeze_base", True),
        dense_units=transfer_cfg.get("dense_units", 256),
        dropout_rate=transfer_cfg.get("dropout_rate", 0.3),
        use_batch_norm=transfer_cfg.get("batch_norm", True),
    )


def get_supported_models() -> list:
    """Return list of supported transfer learning model names."""
    return list(SUPPORTED_MODELS.keys())
