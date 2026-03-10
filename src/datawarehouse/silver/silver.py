from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from src.logger import logging
from src.config import CATALOG, BRONZE_SCHEMA, BRONZE_TABLE, SILVER_SCHEMA, SILVER_TABLE, WRITE_FORMAT, WRITE_MODE


class SilverLayer:
    """
    Reads the bronze Delta table and returns it as a Spark DataFrame.
    Add your own transformations in the run() method.
    """

    def __init__(self, spark: SparkSession = None):
        self.spark = spark or SparkSession.builder.getOrCreate()
        logging.info("SilverLayer initialised.")

    def _read_bronze(self) -> DataFrame:
        """Load data from the bronze Delta table."""
        full_table = f"`{CATALOG}`.`{BRONZE_SCHEMA}`.`{BRONZE_TABLE}`"
        logging.info(f"Reading from {full_table} …")
        df = self.spark.read.table(full_table)
        logging.info(f"Rows loaded from bronze: {df.count():,}")
        return df

    def data_preprocessing(self, df: DataFrame) -> DataFrame:
        """
        Drop some useless columns and do some preprocessing
        """

        logging.info("Preprocessing data ...")

        # Select only required columns
        df = df.select("content", "score")
    
        # Create sentiment column based on score
        df = df.withColumn(
            "sentiment",
            F.when(F.col("score") <= 2, "Negative")
             .when(F.col("score") == 3, "Neutral")
             .when(F.col("score") >= 4, "Positive")
             .otherwise("Unknown")
        )

        # Drop score column
        df = df.drop("score")

        logging.info(f"Rows after preprocessing: {df.count():,}")
        logging.info("Data preprocessing completed.")
    
        return df

    def _write_silver(self, df: DataFrame) -> DataFrame:
        """Write to Silver Delta table and return it as a Spark DataFrame."""
        full_table = f"`{CATALOG}`.`{SILVER_SCHEMA}`.`{SILVER_TABLE}`"
        logging.info(f"Writing to {full_table} …")
        (
            df.write
            .format(WRITE_FORMAT)
            .mode(WRITE_MODE)
            .option("overwriteSchema", "true")
            .saveAsTable(full_table)
        )
        logging.info(f"Silver table '{full_table}' saved successfully.")
        return self.spark.read.table(full_table)

    
    def run(self) -> DataFrame:
        df_bronze = self._read_bronze()
        df_silver = self.data_preprocessing(df_bronze)
        df_saved  = self._write_silver(df_silver)
        return df_saved
