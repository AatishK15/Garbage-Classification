"""
Predictor — Single image prediction pipeline for garbage classification.

Handles image loading, preprocessing, model inference, and result formatting
for both file-based and bytes-based inputs (Streamlit uploads).
"""

import io
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from src.utils.helpers import PROJECT_ROOT, format_confidence, get_class_info, load_config
from src.utils.logger import get_logger
from src.utils.validators import ImageValidationError, validate_image

logger = get_logger(__name__)


class GarbagePredictor:
    """
    Handles garbage image classification with a trained model.

    Supports prediction from file paths, raw bytes, and PIL Images.
    Returns structured results with confidence scores and disposal instructions.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the predictor.

        Args:
            model_path: Path to the trained model (.h5 file).
                        If None, uses the default model path from config.
            config: Configuration dictionary. Loads default if None.
        """
        if config is None:
            config = load_config()

        self.config = config
        self.image_cfg = config.get("image", {})
        self.inference_cfg = config.get("inference", {})

        self.image_size = (
            self.image_cfg.get("height", 224),
            self.image_cfg.get("width", 224),
        )
        self.classes = sorted(config.get("dataset", {}).get("classes", []))
        self.class_info = get_class_info()
        self.confidence_threshold = self.inference_cfg.get("confidence_threshold", 0.5)
        self.top_k = self.inference_cfg.get("top_k", 3)

        self.model = None
        self.model_path = model_path

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        """
        Load a trained Keras model from disk.

        Args:
            model_path: Path to the .h5 or SavedModel file.

        Raises:
            FileNotFoundError: If model file doesn't exist.
            RuntimeError: If model loading fails.
        """
        import tensorflow as tf

        model_path = Path(model_path)
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            logger.info(f"Loading model from: {model_path}")
            self.model = tf.keras.models.load_model(str(model_path))
            self.model_path = str(model_path)
            logger.info("Model loaded successfully ✓")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def preprocess_image(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Preprocess an image for model inference.

        Resizes to target dimensions and normalizes pixel values to [0, 1].

        Args:
            image: Input image as numpy array (H, W, C) in RGB format.

        Returns:
            Preprocessed image array with batch dimension (1, H, W, C).
        """
        # Resize
        image_resized = cv2.resize(
            image, (self.image_size[1], self.image_size[0])
        )

        # Ensure 3 channels
        if len(image_resized.shape) == 2:
            image_resized = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)
        elif image_resized.shape[2] == 4:
            image_resized = cv2.cvtColor(image_resized, cv2.COLOR_RGBA2RGB)

        # Normalize to [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0

        # Add batch dimension
        image_batch = np.expand_dims(image_normalized, axis=0)

        return image_batch

    def predict(
        self,
        image_path: str,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """
        Predict the garbage class for an image file.

        Args:
            image_path: Path to the image file.
            validate: Whether to validate the image before prediction.

        Returns:
            Prediction result dictionary with class, confidence, and details.

        Raises:
            RuntimeError: If model is not loaded.
            ImageValidationError: If image validation fails.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if validate:
            validate_image(file_path=image_path)

        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ImageValidationError(f"Failed to read image: {image_path}")

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess and predict
        return self._run_prediction(image_rgb, source=str(image_path))

    def predict_from_bytes(
        self,
        file_bytes: bytes,
        filename: str = "uploaded_image",
    ) -> Dict[str, Any]:
        """
        Predict from raw image bytes (e.g., Streamlit file upload).

        Args:
            file_bytes: Raw image file bytes.
            filename: Original filename for logging.

        Returns:
            Prediction result dictionary.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Validate
        validate_image(file_bytes=file_bytes)

        # Decode image from bytes
        pil_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        image_rgb = np.array(pil_image)

        return self._run_prediction(image_rgb, source=filename)

    def predict_from_pil(
        self,
        pil_image: Image.Image,
    ) -> Dict[str, Any]:
        """
        Predict from a PIL Image object.

        Args:
            pil_image: PIL Image.

        Returns:
            Prediction result dictionary.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        image_rgb = np.array(pil_image.convert("RGB"))
        return self._run_prediction(image_rgb, source="pil_image")

    def _run_prediction(
        self,
        image_rgb: np.ndarray,
        source: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Core prediction logic — preprocess, infer, and format results.

        Args:
            image_rgb: RGB image as numpy array.
            source: Source identifier for logging.

        Returns:
            Structured prediction result.
        """
        start_time = time.time()

        # Preprocess
        processed = self.preprocess_image(image_rgb)

        # Predict
        predictions = self.model.predict(processed, verbose=0)
        probabilities = predictions[0]

        inference_time = (time.time() - start_time) * 1000  # ms

        # Get top-k predictions
        top_indices = np.argsort(probabilities)[::-1][: self.top_k]
        top_predictions = []

        for idx in top_indices:
            cls_name = self.classes[idx]
            confidence = float(probabilities[idx])
            info = self.class_info.get(cls_name, {})

            top_predictions.append({
                "class": cls_name,
                "confidence": confidence,
                "confidence_pct": format_confidence(confidence),
                "icon": info.get("icon", "❓"),
                "color": info.get("color", "#808080"),
                "instruction": info.get("instruction", ""),
                "bin": info.get("bin", ""),
            })

        # Primary prediction
        predicted_class = self.classes[top_indices[0]]
        confidence = float(probabilities[top_indices[0]])
        primary_info = self.class_info.get(predicted_class, {})

        result = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "confidence_pct": format_confidence(confidence),
            "icon": primary_info.get("icon", "❓"),
            "color": primary_info.get("color", "#808080"),
            "instruction": primary_info.get("instruction", ""),
            "bin": primary_info.get("bin", ""),
            "is_confident": confidence >= self.confidence_threshold,
            "top_predictions": top_predictions,
            "all_probabilities": {
                cls: float(prob)
                for cls, prob in zip(self.classes, probabilities)
            },
            "inference_time_ms": round(inference_time, 1),
            "source": source,
        }

        logger.info(
            f"Prediction: {predicted_class} ({format_confidence(confidence)}) "
            f"in {inference_time:.1f}ms — source: {source}"
        )

        return result

    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.model is not None
