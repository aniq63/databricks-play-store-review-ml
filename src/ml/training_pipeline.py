"""
Training Pipeline — Model-as-Parent, Hyperparameter-as-Child MLflow Structure

Architecture (visible in Databricks MLflow UI):

    Parent Run: LogisticRegression
    ├── Child Run: C=0.01
    ├── Child Run: C=0.1
    └── Child Run: C=1.0    <-- best child logged on parent

    Parent Run: RandomForest
    ├── Child Run: n_estimators=50,  max_depth=None
    └── Child Run: n_estimators=100, max_depth=10

    Parent Run: GradientBoosting
    ├── Child Run: lr=0.01
    ├── Child Run: lr=0.1
    └── Child Run: lr=0.5

Each parent run stores the best F1 found across all its children.
"""

import os
import sys
import pandas as pd
from mlflow.models import infer_signature

import mlflow
import mlflow.sklearn

from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src.logger import logging
from src.exception import MyException
from src.ml.mlflow_connection import MLflowConnection
from src.config import (
    CATALOG,
    ML_SCHEMA,
    ML_FEATURES_TABLE,
    input_feature,
    target_feature,
    test_size,
    MODELS_DIR,
)
import warnings
warnings.filterwarnings("ignore")


# ── Custom MLflow Wrapper ─────────────────────────────────────────────────────

class TextModelWrapper(mlflow.pyfunc.PythonModel):
    """
    Custom MLflow wrapper to ensure Databricks Model Serving correctly handles
    batch JSON payloads and extracts the 1D text Series for the TfidfVectorizer.
    """
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        import pandas as pd
        # Databricks serving endpoint will pass a DataFrame.
        # We must extract the 1D text Series/List for TfidfVectorizer.
        
        if isinstance(model_input, pd.DataFrame):
            # Extract the first column as a list of strings
            input_data = model_input.iloc[:, 0].astype(str).tolist()
        elif isinstance(model_input, pd.Series):
            input_data = model_input.astype(str).tolist()
        else:
            input_data = model_input
            
        return self.model.predict(input_data)


# ── Label maps ────────────────────────────────────────────────────────────────

LABEL_MAP         = {"Negative": 0, "Neutral": 1, "Positive": 2}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


# ── Hyperparameter grids ──────────────────────────────────────────────────────
# Each entry is a (classifier_instance, param_dict_for_logging) pair.

MODEL_GRIDS: dict[str, list[tuple]] = {

    "LogisticRegression": [
        (LogisticRegression(C=0.01, max_iter=1000, random_state=42),
         {"C": 0.01, "max_iter": 1000}),
        (LogisticRegression(C=0.1,  max_iter=1000, random_state=42),
         {"C": 0.1,  "max_iter": 1000}),
        (LogisticRegression(C=1.0,  max_iter=1000, random_state=42),
         {"C": 1.0,  "max_iter": 1000}),
        (LogisticRegression(C=10.0, max_iter=1000, random_state=42),
         {"C": 10.0, "max_iter": 1000}),
    ],

    "RandomForest": [
        (RandomForestClassifier(n_estimators=50,  max_depth=None, random_state=42, n_jobs=-1),
         {"n_estimators": 50,  "max_depth": "None"}),
        (RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1),
         {"n_estimators": 100, "max_depth": "None"}),
        (RandomForestClassifier(n_estimators=100, max_depth=10,   random_state=42, n_jobs=-1),
         {"n_estimators": 100, "max_depth": 10}),
    ],

    "GradientBoosting": [
        (GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, random_state=42),
         {"learning_rate": 0.01, "n_estimators": 100}),
        (GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,  random_state=42),
         {"learning_rate": 0.1,  "n_estimators": 100}),
        (GradientBoostingClassifier(n_estimators=100, learning_rate=0.5,  random_state=42),
         {"learning_rate": 0.5,  "n_estimators": 100}),
    ],
}


# ── Training Pipeline ─────────────────────────────────────────────────────────

class TrainingPipeline:
    """
    Trains 3 classifiers with hyperparameter tuning via nested MLflow runs.

    Structure:
        Each MODEL = 1 Parent Run
        Each HYPERPARAMETER VARIANT = 1 Child Run (nested)
    """

    def __init__(self, spark: SparkSession = None):
        """
        Initialize TrainingPipeline.

        Args:
            spark: SparkSession. Created automatically if not provided.
        """
        try:
            self.spark         = spark or SparkSession.builder.getOrCreate()
            self.experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")
            os.makedirs(MODELS_DIR, exist_ok=True)
            logging.info("TrainingPipeline initialized.")
        except Exception as e:
            logging.error(f"Failed to initialize TrainingPipeline: {e}")
            raise MyException(e, sys)

    # ── Data loading ──────────────────────────────────────────────────────────

    def load_data(self) -> pd.DataFrame:
        """Load and return ML features from Unity Catalog as a Pandas DataFrame."""
        logging.info("Loading ML features from catalog...")
        try:
            table   = f"`{CATALOG}`.`{ML_SCHEMA}`.`{ML_FEATURES_TABLE}`"
            spark_df = self.spark.read.table(table)
            pdf = (
                spark_df
                .select(input_feature, target_feature)
                .toPandas()
                .dropna(subset=[input_feature, target_feature])
            )
            logging.info(f"Loaded {len(pdf):,} rows from {table}")
            return pdf
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            raise MyException(e, sys)

    def prepare_data(self, df: pd.DataFrame):
        """Encode labels and split into train/test. Returns (X_train, X_test, y_train, y_test)."""
        logging.info(f"Preparing data (test_size={test_size})...")
        try:
            df = df.copy()
            df[target_feature] = df[target_feature].map(LABEL_MAP)

            X = df[input_feature]
            y = df[target_feature]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            logging.info(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")
            return X_train, X_test, y_train, y_test
        except Exception as e:

            logging.error(f"Data preparation failed: {e}")
            raise MyException(e, sys)

    # ── Child run: one hyperparameter variant ─────────────────────────────────

    def _run_child(
        self,
        model_name: str,
        clf,
        hp_params: dict,
        X_train: pd.Series,
        X_test: pd.Series,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> dict:
        """
        Train one classifier variant as a CHILD run nested under its model parent.

        Args:
            model_name: Parent model name (e.g. "LogisticRegression").
            clf:        Sklearn classifier with hyperparams already set.
            hp_params:  Dict of hyperparameters (logged to MLflow).
            X/y train/test: Feature and label series.

        Returns:
            Dict with run_id, f1, accuracy, and trained pipeline.
        """
        child_label = ", ".join(f"{k}={v}" for k, v in hp_params.items())
        logging.info(f"    -> Child run: {model_name}({child_label})")
        try:
            # Build TF-IDF + classifier pipeline
            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(max_features=10_000, ngram_range=(1, 2))),
                ("clf",   clf),
            ])
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)

            f1        = f1_score(y_test, preds, average="weighted")
            accuracy  = accuracy_score(y_test, preds)
            precision = precision_score(y_test, preds, average="weighted", zero_division=0)
            recall    = recall_score(y_test, preds, average="weighted", zero_division=0)

            with mlflow.start_run(
                run_name=f"{model_name} | {child_label}",
                experiment_id=self.experiment_id,
                nested=True,
            ) as child_run:

                # Log hyperparams + shared params
                mlflow.log_params({
                    "model":        model_name,
                    "vectorizer":   "TF-IDF",
                    "max_features": 10_000,
                    "ngram_range":  "(1,2)",
                    **{str(k): str(v) for k, v in hp_params.items()},
                })

                # Log evaluation metrics
                mlflow.log_metrics({
                    "f1_weighted": round(f1, 4),
                    "accuracy":    round(accuracy, 4),
                    "precision":   round(precision, 4),
                    "recall":      round(recall, 4),
                })

                # Log model artifact with signature using custom PythonModel wrapper
                # Construct a strict DataFrame signature
                input_example = pd.DataFrame({input_feature: X_train.iloc[:3].values})
                signature     = infer_signature(input_example, pipeline.predict(X_train.iloc[:3]))
                
                # Wrap the sklearn pipeline in our custom class
                wrapped_model = TextModelWrapper(model=pipeline)

                # Use log_model from pyfunc, NOT sklearn, so our predict() wrapper is used
                # Also include code_paths so Databricks Serving gets the 'src' module
                mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=wrapped_model,
                    signature=signature,
                    input_example=input_example,
                    code_paths=["src"],
                )

                child_run_id = child_run.info.run_id
                logging.info(
                    f"       F1={f1:.4f} | Acc={accuracy:.4f} | run_id={child_run_id}"
                )

            return {
                "run_id":    child_run_id,
                "f1":        f1,
                "accuracy":  accuracy,
                "hp_params": hp_params,
                "pipeline":  pipeline,
            }

        except Exception as e:
            logging.error(f"Child run failed for {model_name}({child_label}): {e}")
            raise MyException(e, sys)

    # ── Parent run: one model with all hyperparameter children ────────────────

    def _run_model_parent(
        self,
        model_name: str,
        variants: list[tuple],
        X_train: pd.Series,
        X_test: pd.Series,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> dict:
        """
        Launch a PARENT run for one model and train all hyperparameter
        variants as CHILD runs inside it.

        Args:
            model_name: Name of the classifier (e.g. "RandomForest").
            variants:   List of (classifier, hp_params) tuples.
            X/y train/test: Feature and label series.

        Returns:
            Dict with model_name, best child run info, and all children results.
        """
        logging.info(f"  Parent run: {model_name} ({len(variants)} variants)")
        try:
            with mlflow.start_run(
                run_name=model_name,
                experiment_id=self.experiment_id,
                tags={
                    "model_family":  model_name,
                    "tuning_type":   "grid_search",
                    "num_variants":  str(len(variants)),
                    "dataset":       f"{CATALOG}.{ML_SCHEMA}.{ML_FEATURES_TABLE}",
                },
            ) as parent_run:

                parent_run_id = parent_run.info.run_id

                # Train each hyperparameter variant as a child
                children = []
                for clf, hp_params in variants:
                    result = self._run_child(
                        model_name=model_name,
                        clf=clf,
                        hp_params=hp_params,
                        X_train=X_train,
                        X_test=X_test,
                        y_train=y_train,
                        y_test=y_test,
                    )
                    children.append(result)

                # Find best child by F1
                children.sort(key=lambda x: x["f1"], reverse=True)
                best_child = children[0]

                # Log best metrics on parent so it's visible in leaderboard
                mlflow.log_metrics({
                    "best_f1":       round(best_child["f1"], 4),
                    "best_accuracy": round(best_child["accuracy"], 4),
                })
                best_hp_str = ", ".join(
                    f"{k}={v}" for k, v in best_child["hp_params"].items()
                )
                mlflow.set_tag("best_hyperparams", best_hp_str)
                mlflow.set_tag("best_child_run_id", best_child["run_id"])

                logging.info(
                    f"  {model_name} done | Best: {best_hp_str} "
                    f"(F1={best_child['f1']:.4f})"
                )

            return {
                "model_name":    model_name,
                "parent_run_id": parent_run_id,
                "best_child":    best_child,
                "all_children":  children,
            }

        except Exception as e:
            logging.error(f"Parent run failed for {model_name}: {e}")
            raise MyException(e, sys)

    # ── Pipeline entry point ──────────────────────────────────────────────────

    def run(self) -> list[dict]:
        """
        Full training pipeline:
            1. Connect to Databricks MLflow
            2. Load features from ML schema
            3. Prepare and split data
            4. For each model: open a parent run
               For each hyperparameter set: open a child run
            5. Log best params/metrics on every parent
            6. Return all results sorted by best F1

        Returns:
            List of model result dicts sorted by best child F1 (desc).
        """
        try:
            logging.info("=" * 60)
            logging.info("TRAINING PIPELINE STARTED")
            logging.info("=" * 60)

            # Connect MLflow to Databricks
            MLflowConnection().connect()

            # Load and prepare data
            raw_df = self.load_data()
            X_train, X_test, y_train, y_test = self.prepare_data(raw_df)

            # Run each model as parent + children
            model_results = []
            for model_name, variants in MODEL_GRIDS.items():
                result = self._run_model_parent(
                    model_name=model_name,
                    variants=variants,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                )
                model_results.append(result)

            # Sort models by best child F1
            model_results.sort(
                key=lambda x: x["best_child"]["f1"], reverse=True
            )

            # Final leaderboard output
            logging.info("-" * 60)
            logging.info("TRAINING COMPLETE — Model Leaderboard (best variant per model):")
            logging.info("-" * 60)
            for rank, r in enumerate(model_results, 1):
                bc  = r["best_child"]
                hps = ", ".join(f"{k}={v}" for k, v in bc["hp_params"].items())
                logging.info(
                    f"  #{rank}  {r['model_name']:<22} "
                    f"F1={bc['f1']:.4f}  Acc={bc['accuracy']:.4f}  [{hps}]"
                )
            logging.info("=" * 60)

            return model_results

        except Exception as e:
            logging.error(f"TrainingPipeline.run() failed: {e}")
            raise MyException(e, sys)
