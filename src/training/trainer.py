"""
Trainer — Model training pipeline with callbacks and two-phase training.

Handles the full training workflow including callback setup, two-phase
transfer learning training, history saving, and progress logging.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

from src.models.transfer_model import unfreeze_model
from src.utils.helpers import PROJECT_ROOT
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """
    Manages model training with callbacks, checkpointing, and
    two-phase transfer learning support.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trainer.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.training_cfg = config.get("training", {})
        self.paths_cfg = config.get("paths", {})

        # Ensure output directories exist
        self.models_dir = PROJECT_ROOT / self.paths_cfg.get("models_dir", "models/saved")
        self.logs_dir = PROJECT_ROOT / self.paths_cfg.get("logs_dir", "outputs/logs")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.history: Dict[str, List[float]] = {}
        self.best_val_accuracy: float = 0.0
        self.training_time: float = 0.0

    def _create_callbacks(
        self,
        phase: str = "phase1",
        model_name: str = "model",
    ) -> List[tf.keras.callbacks.Callback]:
        """
        Create training callbacks based on configuration.

        Args:
            phase: Training phase ('phase1' or 'phase2').
            model_name: Name prefix for saved files.

        Returns:
            List of Keras callbacks.
        """
        callbacks = []

        # Early Stopping
        es_cfg = self.training_cfg.get("early_stopping", {})
        callbacks.append(
            EarlyStopping(
                monitor=es_cfg.get("monitor", "val_loss"),
                patience=es_cfg.get("patience", 10),
                restore_best_weights=es_cfg.get("restore_best_weights", True),
                verbose=1,
            )
        )

        # Model Checkpoint
        ckpt_cfg = self.training_cfg.get("checkpoint", {})
        checkpoint_path = self.models_dir / f"{model_name}_{phase}_best.h5"
        callbacks.append(
            ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor=ckpt_cfg.get("monitor", "val_accuracy"),
                save_best_only=ckpt_cfg.get("save_best_only", True),
                mode="max",
                verbose=1,
            )
        )

        # Reduce Learning Rate on Plateau
        lr_cfg = self.training_cfg.get("reduce_lr", {})
        callbacks.append(
            ReduceLROnPlateau(
                monitor=lr_cfg.get("monitor", "val_loss"),
                factor=lr_cfg.get("factor", 0.2),
                patience=lr_cfg.get("patience", 5),
                min_lr=lr_cfg.get("min_lr", 1e-6),
                verbose=1,
            )
        )

        # CSV Logger
        csv_path = self.logs_dir / f"{model_name}_{phase}_training.csv"
        callbacks.append(CSVLogger(str(csv_path), append=False))

        # TensorBoard
        tb_dir = self.logs_dir / "tensorboard" / f"{model_name}_{phase}"
        callbacks.append(
            TensorBoard(
                log_dir=str(tb_dir),
                histogram_freq=0,
                write_graph=True,
            )
        )

        logger.info(
            f"Created {len(callbacks)} callbacks for {phase}: "
            f"EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard"
        )

        return callbacks

    def train(
        self,
        model: tf.keras.Model,
        train_gen: Any,
        val_gen: Any,
        class_weights: Optional[Dict[int, float]] = None,
        model_name: str = "model",
        epochs: Optional[int] = None,
        phase: str = "phase1",
    ) -> Dict[str, List[float]]:
        """
        Train the model for a single phase.

        Args:
            model: Keras model to train.
            train_gen: Training data generator.
            val_gen: Validation data generator.
            class_weights: Class weights for imbalance handling.
            model_name: Name prefix for saved files.
            epochs: Number of epochs (overrides config).
            phase: Training phase name.

        Returns:
            Training history dictionary.
        """
        phase_cfg = self.training_cfg.get(phase, {})
        epochs = epochs or phase_cfg.get("epochs", 20)

        use_weights = self.training_cfg.get("use_class_weights", True) and class_weights
        logger.info(
            f"Starting {phase} training — {epochs} epochs, "
            f"class_weights={'enabled' if use_weights else 'disabled'}"
        )

        callbacks = self._create_callbacks(phase=phase, model_name=model_name)

        start_time = time.time()

        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            class_weight=class_weights if use_weights else None,
            callbacks=callbacks,
            verbose=1,
        )

        elapsed = time.time() - start_time
        self.training_time += elapsed

        # Extract history
        hist_dict = {key: [float(v) for v in values] for key, values in history.history.items()}

        # Merge with existing history (for multi-phase training)
        for key, values in hist_dict.items():
            if key in self.history:
                self.history[key].extend(values)
            else:
                self.history[key] = values

        # Log best results
        best_val_acc = max(hist_dict.get("val_accuracy", [0]))
        best_epoch = hist_dict.get("val_accuracy", [0]).index(best_val_acc) + 1
        self.best_val_accuracy = max(self.best_val_accuracy, best_val_acc)

        logger.info(
            f"{phase} completed in {elapsed:.1f}s — "
            f"Best val_accuracy: {best_val_acc:.4f} (epoch {best_epoch})"
        )

        return hist_dict

    def train_transfer_learning(
        self,
        model: tf.keras.Model,
        train_gen: Any,
        val_gen: Any,
        class_weights: Optional[Dict[int, float]] = None,
        model_name: str = "transfer_model",
    ) -> Dict[str, List[float]]:
        """
        Two-phase transfer learning training.

        Phase 1: Train only the top classification layers (base frozen).
        Phase 2: Unfreeze top layers of base model and fine-tune with low LR.

        Args:
            model: Keras transfer learning model.
            train_gen: Training data generator.
            val_gen: Validation data generator.
            class_weights: Class weights for imbalance handling.
            model_name: Name prefix for saved files.

        Returns:
            Combined training history.
        """
        logger.info("=" * 60)
        logger.info("PHASE 1: Feature Extraction (frozen base)")
        logger.info("=" * 60)

        # Phase 1: Train top layers
        phase1_cfg = self.training_cfg.get("phase1", {})
        self.train(
            model=model,
            train_gen=train_gen,
            val_gen=val_gen,
            class_weights=class_weights,
            model_name=model_name,
            epochs=phase1_cfg.get("epochs", 10),
            phase="phase1",
        )

        logger.info("=" * 60)
        logger.info("PHASE 2: Fine-tuning (partially unfrozen base)")
        logger.info("=" * 60)

        # Unfreeze top layers of base model
        transfer_cfg = self.config.get("model", {}).get("transfer", {})
        unfreeze_percent = transfer_cfg.get("unfreeze_top_percent", 0.3)
        model = unfreeze_model(model, unfreeze_top_percent=unfreeze_percent)

        # Re-compile with lower learning rate
        phase2_cfg = self.training_cfg.get("phase2", {})
        lr = phase2_cfg.get("learning_rate", 0.0001)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        logger.info(f"Re-compiled with Adam(lr={lr}) for fine-tuning")

        # Phase 2: Fine-tune
        self.train(
            model=model,
            train_gen=train_gen,
            val_gen=val_gen,
            class_weights=class_weights,
            model_name=model_name,
            epochs=phase2_cfg.get("epochs", 40),
            phase="phase2",
        )

        # Save final model
        final_path = self.models_dir / f"{model_name}_final.h5"
        model.save(str(final_path))
        logger.info(f"Final model saved to: {final_path}")

        # Save complete training history
        self._save_history(model_name)

        logger.info(
            f"Total training time: {self.training_time:.1f}s | "
            f"Best val_accuracy: {self.best_val_accuracy:.4f}"
        )

        return self.history

    def train_custom_cnn(
        self,
        model: tf.keras.Model,
        train_gen: Any,
        val_gen: Any,
        class_weights: Optional[Dict[int, float]] = None,
        model_name: str = "custom_cnn",
    ) -> Dict[str, List[float]]:
        """
        Single-phase training for custom CNN model.

        Args:
            model: Custom CNN Keras model.
            train_gen: Training data generator.
            val_gen: Validation data generator.
            class_weights: Class weights for imbalance handling.
            model_name: Name prefix for saved files.

        Returns:
            Training history.
        """
        logger.info("=" * 60)
        logger.info("Training Custom CNN (single phase)")
        logger.info("=" * 60)

        # Use phase1 config for single-phase training
        phase1_cfg = self.training_cfg.get("phase1", {})
        total_epochs = (
            phase1_cfg.get("epochs", 10)
            + self.training_cfg.get("phase2", {}).get("epochs", 40)
        )

        self.train(
            model=model,
            train_gen=train_gen,
            val_gen=val_gen,
            class_weights=class_weights,
            model_name=model_name,
            epochs=total_epochs,
            phase="phase1",
        )

        # Save final model
        final_path = self.models_dir / f"{model_name}_final.h5"
        model.save(str(final_path))
        logger.info(f"Final model saved to: {final_path}")

        self._save_history(model_name)

        return self.history

    def _save_history(self, model_name: str) -> None:
        """Save training history as JSON."""
        history_path = self.logs_dir / f"{model_name}_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved to: {history_path}")
