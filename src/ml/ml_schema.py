"""
ML Schema Setup — Creates ML schema and populates all 4 base tables.

Tables created in play_store_reviews.ml:
    [1] ml_features   <- full gold data (all rows)
    [2] ml_train_data <- 80% training split  (separate table)
    [3] ml_test_data  <- 20% test split      (separate table)

Usage:
    MlSchema(spark).run()
"""

import sys
import pandas as pd

from pyspark.sql import SparkSession, DataFrame
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import MyException
from src.config import (
    CATALOG,
    ML_SCHEMA,
    ML_FEATURES_TABLE,
    ML_TRAIN_TABLE,
    ML_TEST_TABLE,
    GOLD_SCHEMA,
    GOLD_TABLE,
    WRITE_FORMAT,
    WRITE_MODE,
    input_feature,
    target_feature,
    test_size,
)
import warnings
warnings.filterwarnings("ignore")


class MlSchema:
    """
    Sets up the ML schema in Unity Catalog.

    Responsibilities:
        1. Create the ML schema if it doesn't exist.
        2. Load full gold data -> save as ml_features.
        3. Split into train/test -> save as separate tables:
               ml_train_data  (80%)
               ml_test_data   (20%)
    """

    def __init__(self, spark: SparkSession = None):
        self.spark = spark or SparkSession.builder.getOrCreate()
        logging.info("MlSchema initialized.")

    # ── Step 1: Schema creation ───────────────────────────────────────────────

    def create_ml_schema(self) -> None:
        """Create the ML schema in Unity Catalog if it doesn't exist."""
        logging.info("Creating ML schema...")
        try:
            self.spark.sql(
                f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{ML_SCHEMA}`"
            )
            logging.info(f"  Schema '{CATALOG}.{ML_SCHEMA}' is ready.")
        except Exception as e:
            logging.error(f"Failed to create ML schema: {e}")
            raise MyException(e, sys)

    # ── Step 2: Load gold data ────────────────────────────────────────────────

    def load_gold_data(self) -> DataFrame:
        """Read cleaned gold data from Unity Catalog."""
        logging.info("Loading gold data...")
        try:
            gold_table = f"`{CATALOG}`.`{GOLD_SCHEMA}`.`{GOLD_TABLE}`"
            logging.info(f"  Reading from {gold_table}...")
            df = self.spark.read.table(gold_table)
            logging.info(f"  Loaded {df.count():,} rows.")
            return df
        except Exception as e:
            logging.error(f"Failed to load gold data: {e}")
            raise MyException(e, sys)

    # ── Step 3: Save full features table ─────────────────────────────────────

    def write_ml_features(self, df: DataFrame) -> None:
        """Save full gold data as the ml_features table."""
        logging.info("Writing ml_features table...")
        try:
            table = f"`{CATALOG}`.`{ML_SCHEMA}`.`{ML_FEATURES_TABLE}`"
            (
                df.write
                .format(WRITE_FORMAT)
                .mode(WRITE_MODE)
                .option("overwriteSchema", "true")
                .saveAsTable(table)
            )
            logging.info(f"  Saved -> {table}")
        except Exception as e:
            logging.error(f"Failed to write ml_features: {e}")
            raise MyException(e, sys)

    # ── Step 4: Split and save train / test tables ────────────────────────────

    def split_and_save(self, df: DataFrame) -> None:
        """
        Split gold data into train/test and save as two separate tables.

        Split: 80% train / 20% test (stratified by sentiment, random_state=42)

        Saved to:
            play_store_reviews.ml.ml_train_data
            play_store_reviews.ml.ml_test_data
        """
        logging.info(f"Splitting into train/test (test_size={test_size})...")
        try:
            # Convert to Pandas for sklearn split
            pdf = (
                df.select(input_feature, target_feature)
                .toPandas()
                .dropna(subset=[input_feature, target_feature])
            )

            train_pdf, test_pdf = train_test_split(
                pdf,
                test_size=test_size,
                random_state=42,
                stratify=pdf[target_feature],
            )

            logging.info(
                f"  Train: {len(train_pdf):,} rows | Test: {len(test_pdf):,} rows"
            )

            # Save each split as its own Delta table
            splits = {
                ML_TRAIN_TABLE: train_pdf,
                ML_TEST_TABLE:  test_pdf,
            }
            for table_name, split_pdf in splits.items():
                table = f"`{CATALOG}`.`{ML_SCHEMA}`.`{table_name}`"
                (
                    self.spark.createDataFrame(split_pdf)
                    .write.format(WRITE_FORMAT)
                    .mode(WRITE_MODE)
                    .option("overwriteSchema", "true")
                    .saveAsTable(table)
                )
                logging.info(f"  Saved {len(split_pdf):,} rows -> {table}")

        except Exception as e:
            logging.error(f"Failed to split and save tables: {e}")
            raise MyException(e, sys)

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self) -> DataFrame:
        """
        Full ML schema setup:
            1. Create schema
            2. Load gold data
            3. Save ml_features (full data)
            4. Split -> save ml_train_data + ml_test_data (separate tables)

        Returns:
            Spark DataFrame of ml_features for downstream use.
        """
        try:
            logging.info("=" * 55)
            logging.info("ML SCHEMA SETUP STARTED")
            logging.info("=" * 55)

            self.create_ml_schema()
            df_gold = self.load_gold_data()
            self.write_ml_features(df_gold)
            self.split_and_save(df_gold)

            logging.info("-" * 55)
            logging.info("ML SCHEMA SETUP COMPLETE — Tables created:")
            logging.info(f"  [1] {CATALOG}.{ML_SCHEMA}.{ML_FEATURES_TABLE}")
            logging.info(f"  [2] {CATALOG}.{ML_SCHEMA}.{ML_TRAIN_TABLE}")
            logging.info(f"  [3] {CATALOG}.{ML_SCHEMA}.{ML_TEST_TABLE}")
            logging.info("=" * 55)

            return self.spark.read.table(
                f"`{CATALOG}`.`{ML_SCHEMA}`.`{ML_FEATURES_TABLE}`"
            )

        except Exception as e:
            logging.error(f"ML Schema setup failed: {e}")
            raise MyException(e, sys)
