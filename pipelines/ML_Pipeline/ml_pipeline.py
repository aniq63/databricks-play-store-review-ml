"""
ML Pipeline — End-to-end ML workflow orchestrator.

    [Gold Layer data]
            |
        MlSchema         — create ml schema + save features/train/test tables
            |
        TrainingPipeline — train 3 models (parent-child MLflow runs)
            |
        ModelRegistry    — register best model, save to Volume, predict

Usage:
    from pipelines.ML_Pipeline.ml_pipeline import MLPipeline
    MLPipeline(spark).run()
"""

import sys
import pandas as pd
from pyspark.sql import SparkSession

from src.logger import logging
from src.exception import MyException
from src.ml.ml_schema import MlSchema
from src.ml.training_pipeline import TrainingPipeline
from src.ml.model_registry import ModelRegistry
import warnings
warnings.filterwarnings("ignore")



class MLPipeline:
    """
    Runs the full ML pipeline:
        MlSchema -> TrainingPipeline -> ModelRegistry

    After run(), Unity Catalog ml schema contains:
        [1] ml_features       — full gold data
        [2] ml_train_data     — 80% training split
        [3] ml_test_data      — 20% test split
        [4] ml_predictions    — predictions with sentiment labels
        [5] models/ (Volume)  — best model .pkl
        [6] Registered model  — play_store_sentiment_model
    """

    def __init__(self, spark: SparkSession = None):
        """
        Initialize ML pipeline.

        Args:
            spark: SparkSession. Created automatically if not provided.
        """
        try:
            self.spark = spark or SparkSession.builder.getOrCreate()
            logging.info("MLPipeline initialized.")
        except Exception as e:
            logging.error(f"Failed to initialize MLPipeline: {e}")
            raise MyException(e, sys)

    # ── Stages ────────────────────────────────────────────────────────────────

    def _run_schema(self) -> None:
        """Setup ML schema: create tables for features, train, and test data."""
        logging.info("=" * 55)
        logging.info("STAGE 1 — ML Schema Setup")
        logging.info("=" * 55)
        try:
            MlSchema(self.spark).run()
            logging.info("ML schema stage complete.")
        except Exception as e:
            logging.error(f"ML schema stage failed: {e}")
            raise MyException(e, sys)

    def _run_training(self) -> list[dict]:
        """Train 3 models using parent-child MLflow runs."""
        logging.info("=" * 55)
        logging.info("STAGE 2 — Model Training")
        logging.info("=" * 55)
        try:
            results = TrainingPipeline(self.spark).run()
            logging.info("Training stage complete.")
            return results
        except Exception as e:
            logging.error(f"Training stage failed: {e}")
            raise MyException(e, sys)

    def _run_registry(self) -> pd.DataFrame:
        """Register best model, save to Volume, generate predictions."""
        logging.info("=" * 55)
        logging.info("STAGE 3 — Model Registry & Predictions")
        logging.info("=" * 55)
        try:
            predictions = ModelRegistry(self.spark).run()
            logging.info("Registry stage complete.")
            return predictions
        except Exception as e:
            logging.error(f"Registry stage failed: {e}")
            raise MyException(e, sys)

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """
        Run the full ML pipeline: Schema -> Training -> Registry.

        Returns:
            Pandas DataFrame with test predictions
            (content, actual_sentiment, predicted_sentiment).
        """
        try:
            logging.info("*" * 55)
            logging.info("   ML PIPELINE STARTED")
            logging.info("*" * 55)

            # Stage 1: Setup ML schema and save data tables
            self._run_schema()

            # Stage 2: Train all models with MLflow tracking
            training_results = self._run_training()

            # Stage 3: Register best model and generate predictions
            predictions_df = self._run_registry()

            # Final summary
            logging.info("*" * 55)
            logging.info("   ML PIPELINE COMPLETE")
            logging.info("*" * 55)
            logging.info("Training leaderboard (best per model):")
            for rank, r in enumerate(training_results, 1):
                bc  = r["best_child"]
                hps = ", ".join(f"{k}={v}" for k, v in bc["hp_params"].items())
                logging.info(
                    f"  #{rank} {r['model_name']:<22} "
                    f"F1={bc['f1']:.4f}  [{hps}]"
                )

            return predictions_df

        except Exception as e:
            logging.error(f"MLPipeline failed: {e}")
            raise MyException(e, sys)
