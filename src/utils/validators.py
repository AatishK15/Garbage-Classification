"""
Validators — Input validation and security utilities.

Validates uploaded images for type, size, and integrity to prevent
malicious uploads and ensure data quality.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Union

from PIL import Image

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Allowed image extensions and MIME types
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/webp",
}

# Maximum file size in bytes (10 MB)
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

# Minimum and maximum image dimensions
MIN_IMAGE_DIM = 32
MAX_IMAGE_DIM = 10000


class ImageValidationError(Exception):
    """Custom exception for image validation failures."""
    pass


def validate_file_extension(file_path: str) -> bool:
    """
    Check if the file has an allowed image extension.

    Args:
        file_path: Path to the file.

    Returns:
        True if extension is valid.

    Raises:
        ImageValidationError: If extension is not allowed.
    """
    ext = Path(file_path).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ImageValidationError(
            f"Invalid file extension '{ext}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    return True


def validate_file_size(
    file_path: Optional[str] = None,
    file_bytes: Optional[bytes] = None,
    max_size_bytes: int = MAX_FILE_SIZE_BYTES,
) -> bool:
    """
    Check if the file size is within the allowed limit.

    Args:
        file_path: Path to the file (used if file_bytes is None).
        file_bytes: Raw file bytes.
        max_size_bytes: Maximum allowed file size in bytes.

    Returns:
        True if file size is valid.

    Raises:
        ImageValidationError: If file exceeds size limit.
    """
    if file_bytes is not None:
        size = len(file_bytes)
    elif file_path is not None:
        size = os.path.getsize(file_path)
    else:
        raise ImageValidationError("Either file_path or file_bytes must be provided.")

    max_mb = max_size_bytes / (1024 * 1024)
    if size > max_size_bytes:
        actual_mb = size / (1024 * 1024)
        raise ImageValidationError(
            f"File size ({actual_mb:.1f} MB) exceeds limit ({max_mb:.1f} MB)."
        )
    return True


def validate_image_integrity(
    file_path: Optional[str] = None,
    image: Optional[Image.Image] = None,
) -> Tuple[int, int]:
    """
    Verify the file is a valid, non-corrupted image and check dimensions.

    Args:
        file_path: Path to the image file.
        image: PIL Image object (used if file_path is None).

    Returns:
        Tuple of (width, height) of the image.

    Raises:
        ImageValidationError: If image is corrupted or dimensions are invalid.
    """
    try:
        if image is None:
            if file_path is None:
                raise ImageValidationError(
                    "Either file_path or image must be provided."
                )
            image = Image.open(file_path)
            image.verify()  # Verify image integrity
            # Re-open after verify (verify closes the file)
            image = Image.open(file_path)

        width, height = image.size

        if width < MIN_IMAGE_DIM or height < MIN_IMAGE_DIM:
            raise ImageValidationError(
                f"Image dimensions ({width}x{height}) too small. "
                f"Minimum: {MIN_IMAGE_DIM}x{MIN_IMAGE_DIM}."
            )

        if width > MAX_IMAGE_DIM or height > MAX_IMAGE_DIM:
            raise ImageValidationError(
                f"Image dimensions ({width}x{height}) too large. "
                f"Maximum: {MAX_IMAGE_DIM}x{MAX_IMAGE_DIM}."
            )

        return width, height

    except ImageValidationError:
        raise
    except Exception as e:
        raise ImageValidationError(f"Invalid or corrupted image: {str(e)}")


def validate_image(
    file_path: Optional[str] = None,
    file_bytes: Optional[bytes] = None,
    max_size_bytes: int = MAX_FILE_SIZE_BYTES,
) -> bool:
    """
    Perform full validation on an uploaded image.

    Checks extension, file size, and image integrity.

    Args:
        file_path: Path to the image file.
        file_bytes: Raw file bytes (for in-memory validation).
        max_size_bytes: Maximum allowed file size.

    Returns:
        True if all validations pass.

    Raises:
        ImageValidationError: If any validation check fails.
    """
    logger.info("Validating image...")

    # Validate file size
    validate_file_size(file_path=file_path, file_bytes=file_bytes, max_size_bytes=max_size_bytes)

    if file_path:
        # Validate extension
        validate_file_extension(file_path)
        # Validate integrity
        validate_image_integrity(file_path=file_path)
    elif file_bytes:
        # For bytes, validate by trying to open as image
        import io
        try:
            image = Image.open(io.BytesIO(file_bytes))
            validate_image_integrity(image=image)
        except ImageValidationError:
            raise
        except Exception as e:
            raise ImageValidationError(f"Cannot read image from bytes: {str(e)}")

    logger.info("Image validation passed ✓")
    return True


def validate_config(config: dict) -> bool:
    """
    Validate the configuration dictionary for required keys and value ranges.

    Args:
        config: Configuration dictionary loaded from YAML.

    Returns:
        True if configuration is valid.

    Raises:
        ValueError: If configuration is invalid.
    """
    required_sections = ["dataset", "image", "model", "training", "paths"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: '{section}'")

    # Validate image dimensions
    img_cfg = config["image"]
    if img_cfg.get("width", 0) <= 0 or img_cfg.get("height", 0) <= 0:
        raise ValueError("Image width and height must be positive integers.")

    # Validate split ratios sum to 1.0
    ds_cfg = config["dataset"]
    total = ds_cfg.get("train_ratio", 0) + ds_cfg.get("val_ratio", 0) + ds_cfg.get("test_ratio", 0)
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Dataset split ratios must sum to 1.0, got {total:.2f}")

    # Validate learning rates are positive
    for phase in ["phase1", "phase2"]:
        lr = config["training"].get(phase, {}).get("learning_rate", 0)
        if lr <= 0:
            raise ValueError(f"Learning rate for {phase} must be positive, got {lr}")

    logger.info("Configuration validation passed ✓")
    return True
