"""
Model Factory — Unified model creation interface.

Provides a factory pattern for creating any supported model architecture
from a single entry point, simplifying model switching and comparison.
"""

from typing import Any, Dict, Optional

import tensorflow as tf

from src.models.cnn_model import build_custom_cnn_from_config
from src.models.transfer_model import (
    build_transfer_model_from_config,
    get_supported_models,
)
from src.utils.helpers import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# All available model types
AVAILABLE_MODELS = ["custom_cnn"] + get_supported_models()


def create_model(
    model_name: str,
    config: Optional[Dict[str, Any]] = None,
    num_classes: int = 7,
    compile_model: bool = True,
) -> tf.keras.Model:
    """
    Create a model by name using the factory pattern.

    Args:
        model_name: Name of the model architecture.
                    Options: 'custom_cnn', 'mobilenetv2', 'resnet50', 'efficientnet'.
        config: Configuration dictionary. Loads default if None.
        num_classes: Number of output classes.
        compile_model: Whether to compile the model with optimizer and loss.

    Returns:
        Keras Model (compiled if requested).

    Raises:
        ValueError: If model_name is not recognized.
    """
    if config is None:
        config = load_config()

    model_name = model_name.lower().strip()
    logger.info(f"Creating model: '{model_name}'")

    if model_name == "custom_cnn":
        model = build_custom_cnn_from_config(config, num_classes=num_classes)
    elif model_name in get_supported_models():
        model = build_transfer_model_from_config(
            config, model_name=model_name, num_classes=num_classes
        )
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {AVAILABLE_MODELS}"
        )

    if compile_model:
        training_cfg = config.get("training", {}).get("phase1", {})
        lr = training_cfg.get("learning_rate", 0.001)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        logger.info(f"Model compiled with Adam(lr={lr}), categorical_crossentropy")

    return model


def list_available_models() -> list:
    """
    List all available model architectures.

    Returns:
        List of model name strings.
    """
    return AVAILABLE_MODELS
