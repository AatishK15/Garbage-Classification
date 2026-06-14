"""
Predict — CLI entry point for single image prediction.

Usage:
    python scripts/predict.py --image path/to/image.jpg
    python scripts/predict.py --image path/to/image.jpg --model-path models/saved/best.h5
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictor import GarbagePredictor
from src.utils.helpers import load_config
from src.utils.logger import setup_logger

logger = setup_logger("predict")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict garbage class from image")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the image file",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to the trained model (.h5 file)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # Find model path
    model_path = args.model_path
    if model_path is None:
        # Look for any saved model
        models_dir = PROJECT_ROOT / config["paths"]["models_dir"]
        h5_files = list(models_dir.glob("*_final.h5"))
        if h5_files:
            model_path = str(h5_files[0])
            logger.info(f"Auto-detected model: {model_path}")
        else:
            print("❌ No trained model found. Train a model first with:")
            print("   python scripts/train.py --model mobilenetv2")
            sys.exit(1)

    # Create predictor
    predictor = GarbagePredictor(model_path=model_path, config=config)

    # Predict
    result = predictor.predict(args.image)

    if args.json:
        # JSON output
        output = {
            "predicted_class": result["predicted_class"],
            "confidence": result["confidence"],
            "top_predictions": result["top_predictions"],
            "inference_time_ms": result["inference_time_ms"],
        }
        print(json.dumps(output, indent=2))
    else:
        # Human-readable output
        print(f"\n{'='*50}")
        print(f"  🗑️  Garbage Classification Result")
        print(f"{'='*50}")
        print(f"  Image: {args.image}")
        print(f"  Predicted: {result['icon']} {result['predicted_class'].upper()}")
        print(f"  Confidence: {result['confidence_pct']}")
        print(f"  Inference Time: {result['inference_time_ms']}ms")
        print(f"\n  📋 Disposal: {result['instruction']}")
        print(f"  🗑️  Bin: {result['bin']}")
        print(f"\n  Top {len(result['top_predictions'])} Predictions:")
        for pred in result["top_predictions"]:
            bar = "█" * int(pred["confidence"] * 30)
            print(f"    {pred['icon']} {pred['class']:12s} {pred['confidence_pct']:>7s} {bar}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
