import re
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from src.logger import logging
from src.config import (
    CATALOG,
    SILVER_SCHEMA, SILVER_TABLE,
    GOLD_SCHEMA,   GOLD_TABLE,
    WRITE_FORMAT,  WRITE_MODE,
)



# ── Gold Layer ────────────────────────────────────────────────────────────────

class GoldLayer:
    """
    Reads silver Delta table → cleans 'content' column → writes to gold catalog.
    Gold data is clean and ready for ML, AI and analytics.
    """

    def __init__(self, spark: SparkSession = None):
        self.spark = spark or SparkSession.builder.getOrCreate()
        logging.info("GoldLayer initialised.")

    def _read_silver(self) -> DataFrame:
        """Load data from the silver Delta table."""
        full_table = f"`{CATALOG}`.`{SILVER_SCHEMA}`.`{SILVER_TABLE}`"
        logging.info(f"Reading from {full_table} …")
        df = self.spark.read.table(full_table)
        logging.info(f"Rows loaded from silver: {df.count():,}")
        return df

    def _clean_content(self, df: DataFrame) -> DataFrame:
        """Clean 'content' using native Spark SQL functions (no UDF — cluster-safe)."""
        logging.info("Cleaning 'content' column …")
        return (
            df
            # Cast to string
            .withColumn("content", F.col("content").cast("string"))
            # Remove HTML tags  e.g. <br>, <b>text</b>
            .withColumn("content", F.regexp_replace(F.col("content"), r"<.*?>", ""))
            # Remove URLs
            .withColumn("content", F.regexp_replace(F.col("content"), r"http\S+|www\S+", ""))
            # Collapse extra whitespace
            .withColumn("content", F.regexp_replace(F.col("content"), r"\s+", " "))
            # Strip leading/trailing spaces
            .withColumn("content", F.trim(F.col("content")))
        )


    def _write_gold(self, df: DataFrame) -> DataFrame:
        """Write cleaned DataFrame to gold Delta table."""
        full_table = f"`{CATALOG}`.`{GOLD_SCHEMA}`.`{GOLD_TABLE}`"
        logging.info(f"Writing to {full_table} …")
        (
            df.write
            .format(WRITE_FORMAT)
            .mode(WRITE_MODE)
            .option("overwriteSchema", "true")
            .saveAsTable(full_table)
        )
        logging.info(f"Gold table '{full_table}' saved successfully.")
        return self.spark.read.table(full_table)

    def run(self) -> DataFrame:
        df_silver = self._read_silver()
        df_clean  = self._clean_content(df_silver)
        df_gold   = self._write_gold(df_clean)
        logging.info(f"Gold pipeline complete. Final row count: {df_gold.count():,}")
        return df_gold
