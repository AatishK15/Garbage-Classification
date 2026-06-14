"""
Train — CLI entry point for model training.

Usage:
    python scripts/train.py --model mobilenetv2 --epochs 50
    python scripts/train.py --model custom_cnn --quick-test
    python scripts/train.py --model all  # Train all models for comparison
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import GarbageDataset
from src.data.download import setup_dataset
from src.models.model_factory import create_model, list_available_models
from src.training.evaluator import ModelEvaluator
from src.training.trainer import ModelTrainer
from src.utils.helpers import ensure_directories, get_device_info, load_config, set_seed
from src.utils.logger import setup_logger

logger = setup_logger("train", log_file="outputs/logs/training.log")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train garbage classification models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available models: {', '.join(list_available_models())}",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenetv2",
        help="Model architecture to train (default: mobilenetv2)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test run with 2 epochs and small subset",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip dataset download (assumes data is already present)",
    )

    return parser.parse_args()


def train_single_model(
    model_name: str,
    config: dict,
    dataset: GarbageDataset,
    train_gen,
    val_gen,
    test_gen,
    class_weights: dict,
    epochs_override: int = None,
) -> dict:
    """
    Train and evaluate a single model.

    Args:
        model_name: Architecture name.
        config: Configuration dictionary.
        dataset: GarbageDataset instance.
        train_gen: Training generator.
        val_gen: Validation generator.
        test_gen: Test generator.
        class_weights: Class weights dictionary.
        epochs_override: Override epochs count.

    Returns:
        Evaluation results dictionary.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING MODEL: {model_name.upper()}")
    logger.info(f"{'='*60}")

    # Create model
    model = create_model(
        model_name=model_name,
        config=config,
        num_classes=dataset.num_classes,
    )

    # Create trainer
    trainer = ModelTrainer(config)

    # Train
    if model_name == "custom_cnn":
        history = trainer.train_custom_cnn(
            model=model,
            train_gen=train_gen,
            val_gen=val_gen,
            class_weights=class_weights,
            model_name=model_name,
        )
    else:
        history = trainer.train_transfer_learning(
            model=model,
            train_gen=train_gen,
            val_gen=val_gen,
            class_weights=class_weights,
            model_name=model_name,
        )

    # Evaluate
    evaluator = ModelEvaluator(config)
    results = evaluator.generate_full_report(
        model=model,
        test_gen=test_gen,
        history=history,
        model_name=model_name,
    )

    return results


def main():
    """Main training pipeline."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Quick test overrides
    if args.quick_test:
        logger.info("🚀 Quick test mode: 2 epochs, reduced data")
        config["training"]["phase1"]["epochs"] = 2
        config["training"]["phase2"]["epochs"] = 2

    if args.epochs:
        config["training"]["phase1"]["epochs"] = min(args.epochs, 10)
        config["training"]["phase2"]["epochs"] = args.epochs

    if args.batch_size:
        config["training"]["phase1"]["batch_size"] = args.batch_size
        config["training"]["phase2"]["batch_size"] = args.batch_size

    # Setup
    set_seed(config.get("seed", 42))
    ensure_directories(config)
    device_info = get_device_info()

    logger.info(f"GPU available: {device_info['gpu_available']}")

    # Setup dataset
    if not args.skip_download:
        logger.info("Setting up dataset...")
        setup_dataset(config)

    # Create data pipeline
    logger.info("Creating data pipeline...")
    dataset = GarbageDataset(config)
    dataset.load_data()
    split_info = dataset.split_data(seed=config.get("seed", 42))
    class_weights = dataset.compute_class_weights()

    batch_size = args.batch_size or config["training"]["phase1"]["batch_size"]
    train_gen, val_gen, test_gen = dataset.create_generators(batch_size=batch_size)

    # Plot class distribution
    evaluator = ModelEvaluator(config)
    stats = dataset.get_dataset_stats()
    class_counts = {cls: info["count"] for cls, info in stats["per_class"].items()}
    evaluator.plot_class_distribution(class_counts)

    # Train models
    model_name = args.model.lower()
    all_results = []

    if model_name == "all":
        # Train all models for comparison
        for name in list_available_models():
            try:
                results = train_single_model(
                    model_name=name,
                    config=config,
                    dataset=dataset,
                    train_gen=train_gen,
                    val_gen=val_gen,
                    test_gen=test_gen,
                    class_weights=class_weights,
                    epochs_override=args.epochs,
                )
                all_results.append(results)
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                continue

        # Compare models
        if len(all_results) > 1:
            evaluator.compare_models(all_results)
    else:
        results = train_single_model(
            model_name=model_name,
            config=config,
            dataset=dataset,
            train_gen=train_gen,
            val_gen=val_gen,
            test_gen=test_gen,
            class_weights=class_weights,
            epochs_override=args.epochs,
        )
        all_results.append(results)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    for r in all_results:
        logger.info(
            f"  {r['model_name']}: "
            f"Accuracy={r['accuracy']:.4f}, F1={r['f1_macro']:.4f}"
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
