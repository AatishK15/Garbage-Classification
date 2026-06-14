"""
Evaluator — Model evaluation with metrics, visualizations, and reports.

Computes accuracy, precision, recall, F1-score, generates confusion matrices,
training curves, and classification reports for model performance analysis.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils.helpers import PROJECT_ROOT
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Evaluates trained models and generates performance visualizations.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the evaluator.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.paths_cfg = config.get("paths", {})
        self.classes = sorted(config.get("dataset", {}).get("classes", []))

        self.plots_dir = PROJECT_ROOT / self.paths_cfg.get("plots_dir", "outputs/plots")
        self.reports_dir = PROJECT_ROOT / self.paths_cfg.get("reports_dir", "outputs/reports")
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Set consistent plot style
        plt.style.use("seaborn-v0_8-darkgrid")
        sns.set_palette("husl")

    def evaluate(
        self,
        model: Any,
        test_gen: Any,
        model_name: str = "model",
    ) -> Dict[str, Any]:
        """
        Evaluate model on test set and compute all metrics.

        Args:
            model: Trained Keras model.
            test_gen: Test data generator.
            model_name: Name for saving reports.

        Returns:
            Dictionary with all evaluation metrics.
        """
        logger.info(f"Evaluating model '{model_name}' on test set...")

        # Get predictions
        test_gen.reset()
        y_pred_probs = model.predict(test_gen, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = test_gen.classes[:len(y_pred)]

        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # Per-class metrics
        report_dict = classification_report(
            y_true, y_pred,
            target_names=self.classes,
            output_dict=True,
            zero_division=0,
        )
        report_text = classification_report(
            y_true, y_pred,
            target_names=self.classes,
            zero_division=0,
        )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        results = {
            "model_name": model_name,
            "accuracy": float(accuracy),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
            "classification_report": report_dict,
            "confusion_matrix": cm.tolist(),
            "num_test_samples": len(y_true),
            "per_class": {},
        }

        # Extract per-class metrics
        for cls_name in self.classes:
            if cls_name in report_dict:
                results["per_class"][cls_name] = {
                    "precision": report_dict[cls_name]["precision"],
                    "recall": report_dict[cls_name]["recall"],
                    "f1-score": report_dict[cls_name]["f1-score"],
                    "support": report_dict[cls_name]["support"],
                }

        # Log results
        logger.info(f"\n{'='*60}")
        logger.info(f"Model: {model_name}")
        logger.info(f"{'='*60}")
        logger.info(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall:    {recall:.4f}")
        logger.info(f"F1-Score:  {f1:.4f}")
        logger.info(f"{'='*60}")
        logger.info(f"\n{report_text}")

        # Save report
        self._save_report(results, model_name)

        return results

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "model",
        normalize: bool = True,
    ) -> str:
        """
        Plot and save a confusion matrix heatmap.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            model_name: Name for the saved plot.
            normalize: Whether to show percentages.

        Returns:
            Path to the saved plot.
        """
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm_display = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            fmt = ".1%"
            title = f"Confusion Matrix — {model_name} (Normalized)"
        else:
            cm_display = cm
            fmt = "d"
            title = f"Confusion Matrix — {model_name}"

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm_display,
            annot=True,
            fmt=fmt,
            cmap="YlGnBu",
            xticklabels=self.classes,
            yticklabels=self.classes,
            ax=ax,
            linewidths=0.5,
            square=True,
        )
        ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        save_path = self.plots_dir / f"{model_name}_confusion_matrix.png"
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Confusion matrix saved to: {save_path}")
        return str(save_path)

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        model_name: str = "model",
    ) -> str:
        """
        Plot training and validation accuracy/loss curves.

        Args:
            history: Training history dictionary.
            model_name: Name for the saved plot.

        Returns:
            Path to the saved plot.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        epochs = range(1, len(history.get("accuracy", history.get("loss", []))) + 1)

        # Accuracy curves
        if "accuracy" in history:
            axes[0].plot(epochs, history["accuracy"], "b-o", label="Training Accuracy", markersize=3)
        if "val_accuracy" in history:
            axes[0].plot(epochs, history["val_accuracy"], "r-o", label="Validation Accuracy", markersize=3)
        axes[0].set_title(f"Model Accuracy — {model_name}", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Epoch", fontsize=11)
        axes[0].set_ylabel("Accuracy", fontsize=11)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Loss curves
        if "loss" in history:
            axes[1].plot(epochs, history["loss"], "b-o", label="Training Loss", markersize=3)
        if "val_loss" in history:
            axes[1].plot(epochs, history["val_loss"], "r-o", label="Validation Loss", markersize=3)
        axes[1].set_title(f"Model Loss — {model_name}", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Epoch", fontsize=11)
        axes[1].set_ylabel("Loss", fontsize=11)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = self.plots_dir / f"{model_name}_training_curves.png"
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Training curves saved to: {save_path}")
        return str(save_path)

    def plot_class_distribution(
        self,
        class_counts: Dict[str, int],
        title: str = "Dataset Class Distribution",
    ) -> str:
        """
        Plot the dataset class distribution as a bar chart.

        Args:
            class_counts: Dictionary mapping class name to count.
            title: Plot title.

        Returns:
            Path to the saved plot.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        colors = sns.color_palette("husl", len(classes))

        bars = ax.bar(classes, counts, color=colors, edgecolor="white", linewidth=0.5)

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                str(count),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Waste Category", fontsize=12)
        ax.set_ylabel("Number of Images", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        save_path = self.plots_dir / "class_distribution.png"
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Class distribution plot saved to: {save_path}")
        return str(save_path)

    def compare_models(
        self,
        results_list: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Compare multiple models side by side.

        Args:
            results_list: List of evaluation result dictionaries.

        Returns:
            DataFrame with model comparison.
        """
        comparison = []
        for result in results_list:
            comparison.append({
                "Model": result["model_name"],
                "Accuracy": f"{result['accuracy']:.4f}",
                "Precision": f"{result['precision_macro']:.4f}",
                "Recall": f"{result['recall_macro']:.4f}",
                "F1-Score": f"{result['f1_macro']:.4f}",
                "Test Samples": result["num_test_samples"],
            })

        df = pd.DataFrame(comparison)

        # Save comparison table
        save_path = self.reports_dir / "model_comparison.csv"
        df.to_csv(str(save_path), index=False)
        logger.info(f"Model comparison saved to: {save_path}")

        # Log comparison
        logger.info("\n" + "=" * 70)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 70)
        logger.info(f"\n{df.to_string(index=False)}")

        # Plot comparison
        self._plot_model_comparison(results_list)

        return df

    def _plot_model_comparison(self, results_list: List[Dict[str, Any]]) -> None:
        """Plot model comparison bar chart."""
        if not results_list:
            return

        models = [r["model_name"] for r in results_list]
        metrics = {
            "Accuracy": [r["accuracy"] for r in results_list],
            "Precision": [r["precision_macro"] for r in results_list],
            "Recall": [r["recall_macro"] for r in results_list],
            "F1-Score": [r["f1_macro"] for r in results_list],
        }

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(models))
        width = 0.2
        multiplier = 0

        for metric_name, values in metrics.items():
            offset = width * multiplier
            bars = ax.bar(x + offset, values, width, label=metric_name)
            ax.bar_label(bars, fmt="%.3f", fontsize=8)
            multiplier += 1

        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.legend(loc="lower right", fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        save_path = self.plots_dir / "model_comparison.png"
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _save_report(self, results: Dict[str, Any], model_name: str) -> None:
        """Save evaluation results as JSON."""
        # Convert non-serializable types
        serializable = {
            k: v for k, v in results.items()
            if k != "confusion_matrix"
        }
        serializable["confusion_matrix"] = results.get("confusion_matrix", [])

        report_path = self.reports_dir / f"{model_name}_evaluation.json"
        with open(report_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        logger.info(f"Evaluation report saved to: {report_path}")

    def generate_full_report(
        self,
        model: Any,
        test_gen: Any,
        history: Dict[str, List[float]],
        model_name: str = "model",
    ) -> Dict[str, Any]:
        """
        Generate a complete evaluation report with all visualizations.

        Args:
            model: Trained Keras model.
            test_gen: Test data generator.
            history: Training history dictionary.
            model_name: Name for reports and plots.

        Returns:
            Complete evaluation results.
        """
        logger.info(f"Generating full evaluation report for '{model_name}'...")

        # Evaluate
        results = self.evaluate(model, test_gen, model_name)

        # Get predictions for confusion matrix
        test_gen.reset()
        y_pred_probs = model.predict(test_gen, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = test_gen.classes[:len(y_pred)]

        # Generate plots
        results["plots"] = {}
        results["plots"]["confusion_matrix"] = self.plot_confusion_matrix(
            y_true, y_pred, model_name
        )
        results["plots"]["training_curves"] = self.plot_training_curves(
            history, model_name
        )

        logger.info(f"Full report generated for '{model_name}'")
        return results
