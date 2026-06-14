"""
Evaluate — CLI entry point for model evaluation.

Usage:
    python scripts/evaluate.py --model-path models/saved/mobilenetv2_final.h5
    python scripts/evaluate.py --model-path models/saved/custom_cnn_final.h5
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import GarbageDataset
from src.training.evaluator import ModelEvaluator
from src.utils.helpers import ensure_directories, load_config, set_seed
from src.utils.logger import setup_logger

logger = setup_logger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model (.h5 file)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    ensure_directories(config)

    import tensorflow as tf

    # Load model
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    logger.info(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    # Create data pipeline
    dataset = GarbageDataset(config)
    dataset.load_data()
    dataset.split_data(seed=config.get("seed", 42))
    _, _, test_gen = dataset.create_generators(batch_size=args.batch_size)

    # Evaluate
    model_name = model_path.stem
    evaluator = ModelEvaluator(config)

    # Load history if available
    history_path = (
        PROJECT_ROOT / config["paths"]["logs_dir"] / f"{model_name}_history.json"
    )
    history = {}
    if history_path.exists():
        import json
        with open(history_path) as f:
            history = json.load(f)

    results = evaluator.generate_full_report(
        model=model,
        test_gen=test_gen,
        history=history,
        model_name=model_name,
    )

    print(f"\n✅ Accuracy: {results['accuracy']:.4f}")
    print(f"✅ F1-Score: {results['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
