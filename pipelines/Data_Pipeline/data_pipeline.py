"""
Data Pipeline — Medallion Architecture orchestrator.

    [Supabase raw_reviews]
            |
        BronzeLayer   — ingest raw data + metadata
            |
        SilverLayer   — clean, deduplicate, type-cast
            |
        GoldLayer     — NLP-cleaned, ML-ready output

Usage:
    from pipelines.Data_Pipeline.data_pipeline import DataWarehousePipeline
    DataWarehousePipeline(spark).run()
"""

import sys
from pyspark.sql import SparkSession, DataFrame

from src.logger import logging
from src.exception import MyException
from src.datawarehouse.bronze.bronze import BronzeLayer
from src.datawarehouse.silver.silver import SilverLayer
from src.datawarehouse.gold.gold import GoldLayer


class DataWarehousePipeline:
    """
    Runs the full Medallion pipeline:
        Bronze -> Silver -> Gold
    """

    def __init__(self, spark: SparkSession = None):
        """
        Initialize pipeline.

        Args:
            spark: SparkSession. Created automatically if not provided.
        """
        try:
            self.spark = spark or SparkSession.builder.getOrCreate()
            logging.info("DataWarehousePipeline initialized.")
        except Exception as e:
            logging.error(f"Failed to initialize DataWarehousePipeline: {e}")
            raise MyException(e, sys)

    # ── Stages ────────────────────────────────────────────────────────────────

    def _run_bronze(self) -> DataFrame:
        """Ingest raw data from Supabase into the Bronze Delta table."""
        logging.info("=" * 55)
        logging.info("STAGE 1 — Bronze Layer")
        logging.info("=" * 55)
        try:
            df = BronzeLayer(self.spark).run()
            logging.info("Bronze stage complete.")
            return df
        except Exception as e:
            logging.error(f"Bronze stage failed: {e}")
            raise MyException(e, sys)

    def _run_silver(self) -> DataFrame:
        """Clean, deduplicate, and type-cast into the Silver Delta table."""
        logging.info("=" * 55)
        logging.info("STAGE 2 — Silver Layer")
        logging.info("=" * 55)
        try:
            df = SilverLayer(self.spark).run()
            logging.info("Silver stage complete.")
            return df
        except Exception as e:
            logging.error(f"Silver stage failed: {e}")
            raise MyException(e, sys)

    def _run_gold(self) -> DataFrame:
        """Apply NLP cleaning and produce ML-ready data in Gold Delta table."""
        logging.info("=" * 55)
        logging.info("STAGE 3 — Gold Layer")
        logging.info("=" * 55)
        try:
            df = GoldLayer(self.spark).run()
            logging.info("Gold stage complete.")
            return df
        except Exception as e:
            logging.error(f"Gold stage failed: {e}")
            raise MyException(e, sys)

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self) -> DataFrame:
        """
        Run the full pipeline: Bronze -> Silver -> Gold.

        Returns:
            Spark DataFrame of the Gold layer (ML-ready).
        """
        try:
            logging.info("*" * 55)
            logging.info("   DATA PIPELINE STARTED")
            logging.info("*" * 55)

            self._run_bronze()
            self._run_silver()
            df_gold = self._run_gold()

            logging.info("*" * 55)
            logging.info("   DATA PIPELINE COMPLETE")
            logging.info("*" * 55)

            return df_gold

        except Exception as e:
            logging.error(f"DataWarehousePipeline failed: {e}")
            raise MyException(e, sys)
