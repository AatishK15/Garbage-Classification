"""
Tests — Basic tests for the predictor and utility modules.

Run with: python -m pytest tests/ -v
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestHelpers:
    """Tests for the helpers module."""

    def test_load_config(self):
        """Test that configuration loads successfully."""
        from src.utils.helpers import load_config

        config = load_config()
        assert config is not None
        assert "dataset" in config
        assert "image" in config
        assert "model" in config
        assert "training" in config

    def test_load_config_values(self):
        """Test that configuration has expected values."""
        from src.utils.helpers import load_config

        config = load_config()
        assert config["image"]["width"] == 224
        assert config["image"]["height"] == 224
        assert config["image"]["channels"] == 3
        assert len(config["dataset"]["classes"]) == 7

    def test_get_class_info(self):
        """Test class info returns all 7 categories."""
        from src.utils.helpers import get_class_info

        info = get_class_info()
        assert len(info) == 7
        expected_classes = {"cardboard", "glass", "metal", "organic", "paper", "plastic", "trash"}
        assert set(info.keys()) == expected_classes

        for cls, data in info.items():
            assert "icon" in data
            assert "color" in data
            assert "instruction" in data
            assert "bin" in data

    def test_format_confidence(self):
        """Test confidence formatting."""
        from src.utils.helpers import format_confidence

        assert format_confidence(0.95) == "95.0%"
        assert format_confidence(0.0) == "0.0%"
        assert format_confidence(1.0) == "100.0%"
        assert format_confidence(0.123) == "12.3%"

    def test_set_seed(self):
        """Test seed setting for reproducibility."""
        from src.utils.helpers import set_seed

        set_seed(42)
        a = np.random.rand(5)
        set_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)


class TestValidators:
    """Tests for the validators module."""

    def test_validate_file_extension_valid(self):
        """Test valid file extensions."""
        from src.utils.validators import validate_file_extension

        assert validate_file_extension("test.jpg") is True
        assert validate_file_extension("test.jpeg") is True
        assert validate_file_extension("test.png") is True
        assert validate_file_extension("test.bmp") is True
        assert validate_file_extension("test.webp") is True

    def test_validate_file_extension_invalid(self):
        """Test invalid file extensions."""
        from src.utils.validators import ImageValidationError, validate_file_extension

        with pytest.raises(ImageValidationError):
            validate_file_extension("test.pdf")
        with pytest.raises(ImageValidationError):
            validate_file_extension("test.txt")
        with pytest.raises(ImageValidationError):
            validate_file_extension("test.exe")

    def test_validate_file_size(self):
        """Test file size validation."""
        from src.utils.validators import ImageValidationError, validate_file_size

        # Small data should pass
        small_data = b"x" * 1000
        assert validate_file_size(file_bytes=small_data) is True

        # Large data should fail
        large_data = b"x" * (11 * 1024 * 1024)  # 11 MB
        with pytest.raises(ImageValidationError):
            validate_file_size(file_bytes=large_data)

    def test_validate_config_valid(self):
        """Test valid configuration validation."""
        from src.utils.helpers import load_config
        from src.utils.validators import validate_config

        config = load_config()
        assert validate_config(config) is True

    def test_validate_config_missing_section(self):
        """Test config validation with missing section."""
        from src.utils.validators import validate_config

        with pytest.raises(ValueError, match="Missing required config section"):
            validate_config({"image": {}, "model": {}, "training": {}})


class TestImageProcessing:
    """Tests for image preprocessing."""

    def test_preprocess_dimensions(self):
        """Test that preprocessing produces correct dimensions."""
        from src.inference.predictor import GarbagePredictor

        predictor = GarbagePredictor(config=None)

        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        processed = predictor.preprocess_image(dummy_image)

        assert processed.shape == (1, 224, 224, 3)
        assert processed.dtype == np.float32
        assert processed.min() >= 0.0
        assert processed.max() <= 1.0

    def test_preprocess_grayscale(self):
        """Test preprocessing with grayscale input."""
        from src.inference.predictor import GarbagePredictor

        predictor = GarbagePredictor(config=None)

        gray_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        processed = predictor.preprocess_image(gray_image)

        assert processed.shape == (1, 224, 224, 3)

    def test_preprocess_rgba(self):
        """Test preprocessing with RGBA input."""
        from src.inference.predictor import GarbagePredictor

        predictor = GarbagePredictor(config=None)

        rgba_image = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        processed = predictor.preprocess_image(rgba_image)

        assert processed.shape == (1, 224, 224, 3)


class TestModelFactory:
    """Tests for the model factory."""

    def test_list_available_models(self):
        """Test listing available models."""
        from src.models.model_factory import list_available_models

        models = list_available_models()
        assert "custom_cnn" in models
        assert "mobilenetv2" in models
        assert "resnet50" in models
        assert "efficientnet" in models
        assert len(models) == 4

    def test_create_invalid_model(self):
        """Test that invalid model name raises error."""
        from src.models.model_factory import create_model

        with pytest.raises(ValueError, match="Unknown model"):
            create_model("nonexistent_model")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
