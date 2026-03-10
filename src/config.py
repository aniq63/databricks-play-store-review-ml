"""
Global Configuration & Constants

Shared across all datawarehouse layers (bronze, silver, gold).
All env variables and catalog settings live here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Supabase ──────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "")
SOURCE_TABLE = "raw_reviews"

# ── Databricks Unity Catalog ──────────────────────────────────────────────────
CATALOG = "play_store_reviews"

BRONZE_SCHEMA = "bronze"
BRONZE_TABLE  = "bronze_reviews"

SILVER_SCHEMA = "silver"
SILVER_TABLE  = "silver_reviews"

GOLD_SCHEMA   = "gold"
GOLD_TABLE    = "gold_reviews"

# ── ML Schema — 4 tables in play_store_reviews.ml ────────────────────────────
ML_SCHEMA            = "ml"
ML_FEATURES_TABLE    = "ml_features"       # gold data loaded for training
ML_TRAIN_TABLE       = "ml_train_data"     # training split
ML_TEST_TABLE        = "ml_test_data"      # test split
ML_PREDICTIONS_TABLE = "ml_predictions"    # best model predictions (sentiments)

# ── ML Columns ────────────────────────────────────────────────────────────────
input_feature  = "content"
target_feature = "sentiment"
test_size      = 0.2

# ── JDBC ──────────────────────────────────────────────────────────────────────
JDBC_DRIVER       = "org.postgresql.Driver"
PREPARE_THRESHOLD = "0"       # Required for Supabase transaction pooler

# ── Delta write settings ──────────────────────────────────────────────────────
WRITE_FORMAT = "delta"
WRITE_MODE   = "overwrite"

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT = "play_store_sentiment"
REGISTERED_MODEL  = "play_store_sentiment_model"
MODELS_DIR        = "models"

