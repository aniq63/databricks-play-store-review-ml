"""
Model Registry — Find best model, register it, generate predictions.

What this module does:
    1. Search all MLflow runs for the best model (highest F1 + Accuracy)
    2. Register that model to Databricks Unity Catalog
    3. Save model .pkl locally (models/ directory)
    4. Save model .pkl to Unity Catalog Volume (play_store_reviews.ml.models)
    5. Generate predictions with actual sentiment labels
    6. Save predictions as Delta table in ML schema

ML Schema — after full pipeline:
    Tables:
        ml_train_data     <- training split
        ml_test_data      <- test split
        ml_predictions    <- predictions with sentiment labels
    Volume:
        models/           <- model .pkl file
    Registered Model:
        play_store_reviews.ml.play_store_sentiment_model
"""

import os
import sys
import joblib
import pandas as pd

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from pyspark.sql import SparkSession

from src.logger import logging
from src.exception import MyException
from src.ml.mlflow_connection import MLflowConnection
from src.config import (
    CATALOG,
    ML_SCHEMA,
    ML_TEST_TABLE,
    ML_PREDICTIONS_TABLE,
    REGISTERED_MODEL,
    MODELS_DIR,
    WRITE_FORMAT,
    WRITE_MODE,
    input_feature,
    target_feature,
)
import warnings
warnings.filterwarnings("ignore")

# ── Reverse label map: integer -> sentiment string ────────────────────────────
REVERSE_LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}


# ── Model Registry Class ──────────────────────────────────────────────────────

class ModelRegistry:
    """
    Selects best MLflow model, registers it, makes predictions, and saves
    all results to Unity Catalog ML schema.
    """

    def __init__(self, spark: SparkSession = None):
        """
        Initialize ModelRegistry.

        Args:
            spark: SparkSession. Created automatically if not provided.
        """
        try:
            self.spark         = spark or SparkSession.builder.getOrCreate()
            self.experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")
            self.client        = MlflowClient()
            os.makedirs(MODELS_DIR, exist_ok=True)
            logging.info("ModelRegistry initialized.")
        except Exception as e:
            logging.error(f"Failed to initialize ModelRegistry: {e}")
            raise MyException(e, sys)

    # ── Step 1: Find Best Model ────────────────────────────────────────────────

    def find_best_run(self) -> tuple[str, str, float, float]:
        """
        Search all MLflow child runs and return the one with the highest
        combined F1 + Accuracy scores.

        Returns:
            Tuple of (run_id, model_name, f1, accuracy)

        Raises:
            ValueError: if no runs are found.
        """
        logging.info("Searching MLflow runs for best model...")
        try:
            # Search all finished runs in the experiment
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string="status = 'FINISHED'",
                order_by=["metrics.f1_weighted DESC", "metrics.accuracy DESC"],
            )

            if runs.empty:
                raise ValueError(
                    "No finished MLflow runs found. Run training_pipeline.py first."
                )

            # Skip parent runs (they don't have f1_weighted themselves)
            child_runs = runs.dropna(subset=["metrics.f1_weighted"])

            if child_runs.empty:
                raise ValueError("No child runs with f1_weighted metric found.")

            best = child_runs.iloc[0]
            run_id     = best["run_id"]
            model_name = best.get("params.model", "unknown")
            f1         = best["metrics.f1_weighted"]
            accuracy   = best["metrics.accuracy"]

            logging.info(f"Best run found:")
            logging.info(f"  Model     : {model_name}")
            logging.info(f"  F1        : {f1:.4f}")
            logging.info(f"  Accuracy  : {accuracy:.4f}")
            logging.info(f"  Run ID    : {run_id}")

            return run_id, model_name, f1, accuracy

        except Exception as e:
            logging.error(f"Failed to find best run: {e}")
            raise MyException(e, sys)

    # ── Step 2: Register Model to Unity Catalog ───────────────────────────────

    def register_model(self, run_id: str, model_name: str) -> str:
        """
        Register the best run's model to Databricks Unity Catalog.

        UC model name: play_store_reviews.ml.play_store_sentiment_model

        Args:
            run_id:     Best MLflow run ID.
            model_name: Short model family name (e.g. LogisticRegression).

        Returns:
            Registered model version string.
        """
        logging.info(f"Registering '{model_name}' to Unity Catalog...")
        try:
            model_uri    = f"runs:/{run_id}/model"
            uc_full_name = f"{CATALOG}.{ML_SCHEMA}.{REGISTERED_MODEL}"

            # Instruct MLflow to use Unity Catalog for Model Registry
            mlflow.set_registry_uri("databricks-uc")

            registered = mlflow.register_model(
                model_uri=model_uri,
                name=uc_full_name,
            )
            version = registered.version
            logging.info(f"  Registered -> {uc_full_name} (version {version})")
            return version

        except Exception as e:
            logging.error(f"Model registration failed: {e}")
            raise MyException(e, sys)

    # ── Step 3: Save Model Locally ────────────────────────────────────────────

    def save_model_locally(self, run_id: str, model_name: str) -> str:
        """
        Download the best model from MLflow and save it as a .pkl file locally.

        Args:
            run_id:     MLflow run ID.
            model_name: Model family name for the filename.

        Returns:
            Local file path of the saved model.
        """
        logging.info(f"Saving model artifact to {MODELS_DIR}/...")
        try:
            model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
            save_path = os.path.join(MODELS_DIR, f"{model_name}_best.pkl")
            joblib.dump(model, save_path)
            logging.info(f"  Saved locally -> {save_path}")
            return save_path

        except Exception as e:
            logging.error(f"Failed to save model locally: {e}")
            raise MyException(e, sys)

    # ── Step 4: Save Model to Unity Catalog Volume ────────────────────────────

    def save_model_to_catalog(self, local_path: str, model_name: str) -> str:
        """
        Save the model .pkl file to a Unity Catalog Volume.

        On Databricks, UC Volumes are accessible as a local FUSE path:
            /Volumes/<catalog>/<schema>/<volume>/

        Creates the Volume DDL (if not exists), then copies the file.

        Args:
            local_path: Path to the local .pkl file.
            model_name: Model name used in the filename.

        Returns:
            UC Volume path where the model was saved (or local_path on error).
        """
        import shutil

        volume_name    = "models"
        uc_volume_path = f"/Volumes/{CATALOG}/{ML_SCHEMA}/{volume_name}"
        filename       = f"{model_name}_best.pkl"
        uc_model_path  = f"{uc_volume_path}/{filename}"

        logging.info(f"Saving model to UC Volume: {uc_model_path}...")
        try:
            # Create Volume in ML schema (DDL is safe to run locally via Databricks Connect)
            self.spark.sql(
                f"CREATE VOLUME IF NOT EXISTS "
                f"`{CATALOG}`.`{ML_SCHEMA}`.`{volume_name}`"
            )
            logging.info(f"  Volume '{CATALOG}.{ML_SCHEMA}.{volume_name}' ready.")

            # Copy local .pkl -> UC Volume FUSE mount
            # On Databricks Connect: /Volumes/ paths are mounted automatically
            os.makedirs(uc_volume_path, exist_ok=True)
            shutil.copy2(local_path, uc_model_path)
            logging.info(f"  Model saved -> {uc_model_path}")
            return uc_model_path

        except Exception as e:
            # Volume copy may fail in purely local environments — log and continue
            logging.warning(
                f"  Could not save to UC Volume (may not be mounted locally): {e}"
            )
            logging.info(f"  Model is still available locally at: {local_path}")
            return local_path

    # ── Step 5: Generate Predictions on Test Data ─────────────────────────────

    def generate_predictions(self, run_id: str) -> pd.DataFrame:
        """
        Load the best model and generate predictions on the test set.

        Test data is loaded from play_store_reviews.ml.ml_test_data
        (saved by training_pipeline.py).

        Predicted labels are mapped back to actual sentiment strings
        (Positive / Negative / Neutral) — not numeric codes.

        Args:
            run_id: MLflow run ID of the best model.

        Returns:
            Pandas DataFrame with columns:
                content, actual_sentiment, predicted_sentiment
        """
        logging.info("Generating predictions on test set...")
        try:
            # Load test data from catalog
            test_table = f"`{CATALOG}`.`{ML_SCHEMA}`.`{ML_TEST_TABLE}`"
            test_df    = self.spark.read.table(test_table).toPandas()
            logging.info(f"  Loaded {len(test_df):,} test rows from {test_table}")

            # Load best model from MLflow
            model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

            # Predict numeric labels
            preds_numeric = model.predict(test_df[input_feature])

            # Predicted labels are numeric -> convert to sentiment strings
            preds_sentiment = [
                REVERSE_LABEL_MAP.get(int(p), "Unknown") for p in preds_numeric
            ]
            
            # Actual labels already come as strings ("Positive", "Negative", etc.) from ml_test_data
            actual_sentiment = test_df[target_feature].tolist()

            result_df = pd.DataFrame({
                "content":             test_df[input_feature].values,
                "actual_sentiment":    actual_sentiment,
                "predicted_sentiment": preds_sentiment,
            })

            logging.info(
                f"  Predictions generated | {len(result_df):,} rows"
            )
            return result_df

        except Exception as e:
            logging.error(f"Failed to generate predictions: {e}")
            raise MyException(e, sys)

    # ── Step 5: Save Predictions to ML Schema ─────────────────────────────────

    def save_predictions_to_catalog(self, predictions_df: pd.DataFrame) -> None:
        """
        Save predictions as a Delta table in Unity Catalog.

        Saved to: play_store_reviews.ml.ml_predictions

        Args:
            predictions_df: DataFrame with actual and predicted sentiments.
        """
        logging.info("Saving predictions to Unity Catalog...")
        try:
            pred_table = f"`{CATALOG}`.`{ML_SCHEMA}`.`{ML_PREDICTIONS_TABLE}`"
            (
                self.spark.createDataFrame(predictions_df)
                .write.format(WRITE_FORMAT)
                .mode(WRITE_MODE)
                .option("overwriteSchema", "true")
                .saveAsTable(pred_table)
            )
            logging.info(
                f"  Saved {len(predictions_df):,} predictions -> {pred_table}"
            )
        except Exception as e:
            logging.error(f"Failed to save predictions to catalog: {e}")
            raise MyException(e, sys)

    # ── Pipeline Entry Point ──────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """
        Full model registry pipeline:
            1. Connect to Databricks MLflow
            2. Search all runs -> find best by F1 + Accuracy
            3. Register best model to Unity Catalog
            4. Save model artifact locally (.pkl)
            5. Generate predictions on test set
            6. Save predictions to ml.ml_predictions table

        After running, Unity Catalog ml schema contains:
            ml_train_data    <- training split
            ml_test_data     <- test split
            ml_predictions   <- predictions with actual sentiment labels
            Registered model: play_store_reviews.ml.play_store_sentiment_model

        Returns:
            DataFrame with predictions (content, actual_sentiment, predicted_sentiment).
        """
        try:
            logging.info("=" * 60)
            logging.info("MODEL REGISTRY PIPELINE STARTED")
            logging.info("=" * 60)

            # Connect MLflow
            MLflowConnection().connect()

            # Find best run
            run_id, model_name, f1, accuracy = self.find_best_run()

            # Register model to UC
            version = self.register_model(run_id, model_name)

            # Save model locally
            local_path = self.save_model_locally(run_id, model_name)

            # Save model to UC Volume (play_store_reviews.ml.models/)
            uc_model_path = self.save_model_to_catalog(local_path, model_name)

            # Generate predictions
            predictions_df = self.generate_predictions(run_id)

            # Save predictions to catalog
            self.save_predictions_to_catalog(predictions_df)

            # Summary
            logging.info("-" * 60)
            logging.info("MODEL REGISTRY COMPLETE — Summary:")
            logging.info("-" * 60)
            logging.info(f"  Best Model    : {model_name}")
            logging.info(f"  F1 Score      : {f1:.4f}")
            logging.info(f"  Accuracy      : {accuracy:.4f}")
            logging.info(f"  UC Registered : {CATALOG}.{ML_SCHEMA}.{REGISTERED_MODEL} v{version}")
            logging.info(f"  Local Model   : {local_path}")
            logging.info(f"  UC Volume     : {uc_model_path}")
            logging.info(f"  Predictions   : {CATALOG}.{ML_SCHEMA}.{ML_PREDICTIONS_TABLE}")
            logging.info("-" * 60)
            logging.info("  ML Schema now contains:")
            logging.info(f"   [1] {CATALOG}.{ML_SCHEMA}.ml_train_data      (table)")
            logging.info(f"   [2] {CATALOG}.{ML_SCHEMA}.ml_test_data       (table)")
            logging.info(f"   [3] {CATALOG}.{ML_SCHEMA}.{ML_PREDICTIONS_TABLE}  (table)")
            logging.info(f"   [4] {CATALOG}.{ML_SCHEMA}.models/            (volume)")
            logging.info(f"   [5] {CATALOG}.{ML_SCHEMA}.{REGISTERED_MODEL} (model)")
            logging.info("=" * 60)

            return predictions_df

        except Exception as e:
            logging.error(f"ModelRegistry.run() failed: {e}")
            raise MyException(e, sys)
