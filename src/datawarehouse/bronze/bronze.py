import os
from datetime import datetime
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from src.logger import logging
from src.config import (
    DATABASE_URL,
    SOURCE_TABLE,
    CATALOG,
    BRONZE_SCHEMA  as SCHEMA,
    BRONZE_TABLE   as TABLE,
    JDBC_DRIVER,
    PREPARE_THRESHOLD,
    WRITE_MODE,
    WRITE_FORMAT,
)


class BronzeLayer:
    """
    Reads raw_reviews from Supabase PostgreSQL and writes it to
    the Databricks bronze Delta table with ingestion metadata.
    """

    def __init__(self, spark: SparkSession = None):
        self.spark     = spark or SparkSession.builder.getOrCreate()
        self._db_url, self._db_props = self._build_jdbc_config()
        logging.info("BronzeLayer initialised.")


    def _build_jdbc_config(self) -> tuple[str, dict]:
        """Parse DATABASE_URL into JDBC URL + connection props."""
        raw = DATABASE_URL.replace("postgres://", "postgresql://", 1)

        if not raw.startswith("postgresql://"):
            raise ValueError(f"Unsupported DATABASE_URL format: {raw[:30]}…")

        rest             = raw[len("postgresql://"):]
        creds, hostpart  = rest.split("@", 1)
        user, password   = creds.split(":", 1)
        jdbc_url         = f"jdbc:postgresql://{hostpart}"

        props = {
            "user":             user,
            "password":         password,
            "driver":           JDBC_DRIVER,
            "prepareThreshold": PREPARE_THRESHOLD,
        }

        logging.info(f"JDBC config built for host: {hostpart}")
        return jdbc_url, props


    def _read_supabase(self) -> DataFrame:
        """Read SOURCE_TABLE from Supabase via JDBC."""
        logging.info(f"Reading '{SOURCE_TABLE}' from Supabase …")
        df = (
            self.spark.read
            .format("jdbc")
            .option("url",     self._db_url)
            .option("dbtable", f'"{SOURCE_TABLE}"')
            .options(**self._db_props)
            .load()
        )
        logging.info(f"Rows read from Supabase: {df.count():,}")
        return df


    def _add_metadata(self, df: DataFrame) -> DataFrame:
        """Add ingestion metadata columns."""
        logging.info("Adding metadata columns …")
        return (
            df
            .withColumn("_ingested_at", F.current_timestamp())
            .withColumn("_source",      F.lit(f"supabase.{SOURCE_TABLE}"))
            .withColumn("_batch_date",  F.lit(datetime.utcnow().strftime("%Y-%m-%d")))
        )


    def _write_bronze(self, df: DataFrame) -> DataFrame:
        """Write to bronze Delta table and return it as a Spark DataFrame."""
        full_table = f"`{CATALOG}`.`{SCHEMA}`.`{TABLE}`"
        logging.info(f"Writing to {full_table} …")
        (
            df.write
            .format(WRITE_FORMAT)
            .mode(WRITE_MODE)
            .option("overwriteSchema", "true")
            .saveAsTable(full_table)
        )
        logging.info(f"Bronze table '{full_table}' saved successfully.")
        return self.spark.read.table(full_table)


    def run(self) -> DataFrame:
        df_raw    = self._read_supabase()
        df_bronze = self._add_metadata(df_raw)
        df_saved  = self._write_bronze(df_bronze)
        logging.info(f"Bronze pipeline complete. Final row count: {df_saved.count():,}")
        return df_saved
