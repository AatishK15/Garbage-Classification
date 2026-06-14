"""
Helpers — Miscellaneous utility functions.

Provides configuration loading, seed setting, device info,
and other shared helper functions.
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Project root directory (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML config file.
                     Defaults to 'config/config.yaml' relative to project root.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
        yaml.YAMLError: If config file has invalid YAML syntax.
    """
    if config_path is None:
        config_path = str(PROJECT_ROOT / "config" / "config.yaml")

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    logger.info(f"Loading configuration from: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError("Configuration file is empty.")

    return config


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # Ensure deterministic operations where possible
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
    except ImportError:
        logger.warning("TensorFlow not available; skipping TF seed.")

    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set to {seed} for reproducibility.")


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available compute devices (CPU/GPU).

    Returns:
        Dictionary with device information.
    """
    info: Dict[str, Any] = {
        "cpu_count": os.cpu_count(),
        "gpu_available": False,
        "gpu_devices": [],
        "gpu_memory": [],
    }

    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        info["gpu_available"] = len(gpus) > 0
        info["gpu_devices"] = [gpu.name for gpu in gpus]

        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"Enabled memory growth for GPU: {gpu.name}")
                except RuntimeError:
                    pass  # Memory growth must be set before GPU initialization

            # Get GPU memory info
            for gpu in gpus:
                try:
                    mem_info = tf.config.experimental.get_memory_info(gpu.name)
                    info["gpu_memory"].append(
                        {
                            "device": gpu.name,
                            "current_mb": mem_info.get("current", 0) / 1e6,
                            "peak_mb": mem_info.get("peak", 0) / 1e6,
                        }
                    )
                except Exception:
                    pass

        logger.info(
            f"Device info: CPU cores={info['cpu_count']}, "
            f"GPU available={info['gpu_available']}, "
            f"GPU count={len(info['gpu_devices'])}"
        )

    except ImportError:
        logger.warning("TensorFlow not available; cannot detect GPU.")

    return info


def ensure_directories(config: Dict[str, Any]) -> None:
    """
    Create all required output directories from configuration.

    Args:
        config: Configuration dictionary.
    """
    paths = config.get("paths", {})
    dirs_to_create = [
        paths.get("models_dir", "models/saved"),
        paths.get("plots_dir", "outputs/plots"),
        paths.get("reports_dir", "outputs/reports"),
        paths.get("logs_dir", "outputs/logs"),
    ]

    # Also ensure predictions directory
    predictions_db = paths.get("predictions_db", "data/predictions/history.db")
    dirs_to_create.append(str(Path(predictions_db).parent))

    # Ensure data directories
    dataset_cfg = config.get("dataset", {})
    dirs_to_create.append(dataset_cfg.get("raw_dir", "data/raw"))
    dirs_to_create.append(dataset_cfg.get("processed_dir", "data/processed"))

    for dir_path in dirs_to_create:
        full_path = PROJECT_ROOT / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory: {full_path}")

    logger.info(f"Created {len(dirs_to_create)} output directories.")


def get_class_info() -> Dict[str, Dict[str, str]]:
    """
    Get display information for each waste class including
    icons, colors, and disposal instructions.

    Returns:
        Dictionary mapping class name to display info.
    """
    return {
        "cardboard": {
            "icon": "📦",
            "color": "#C49A6C",
            "instruction": "Flatten and place in recycling bin. Remove any tape or labels.",
            "bin": "♻️ Recycling Bin (Blue)",
        },
        "glass": {
            "icon": "🫙",
            "color": "#87CEEB",
            "instruction": "Rinse and place in glass recycling. Do not mix colors if sorted.",
            "bin": "♻️ Glass Recycling Bin (Green)",
        },
        "metal": {
            "icon": "🥫",
            "color": "#B8B8B8",
            "instruction": "Rinse cans and containers. Crush if possible to save space.",
            "bin": "♻️ Recycling Bin (Blue)",
        },
        "organic": {
            "icon": "🍂",
            "color": "#8B4513",
            "instruction": "Place in compost bin. Includes food scraps, yard waste, and biodegradables.",
            "bin": "🟤 Compost Bin (Brown/Green)",
        },
        "paper": {
            "icon": "📄",
            "color": "#F5F5DC",
            "instruction": "Keep dry and clean. Remove staples and plastic windows from envelopes.",
            "bin": "♻️ Recycling Bin (Blue)",
        },
        "plastic": {
            "icon": "🧴",
            "color": "#FF6B6B",
            "instruction": "Check recycling number. Rinse containers. Remove caps if different material.",
            "bin": "♻️ Recycling Bin (Blue/Yellow)",
        },
        "trash": {
            "icon": "🗑️",
            "color": "#696969",
            "instruction": "Non-recyclable waste. Place in general waste bin.",
            "bin": "⬛ General Waste Bin (Black)",
        },
    }


def format_confidence(confidence: float) -> str:
    """
    Format a confidence score as a percentage string.

    Args:
        confidence: Float between 0 and 1.

    Returns:
        Formatted percentage string (e.g., '95.2%').
    """
    return f"{confidence * 100:.1f}%"


def get_model_summary_string(model) -> str:
    """
    Capture model.summary() output as a string.

    Args:
        model: A Keras model.

    Returns:
        String representation of model summary.
    """
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return "\n".join(lines)
