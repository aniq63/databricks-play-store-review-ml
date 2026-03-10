
from pyspark.sql import SparkSession

from pipelines.Data_Pipeline.data_pipeline import DataWarehousePipeline
from pipelines.ML_Pipeline.ml_pipeline import MLPipeline
import warnings
warnings.filterwarnings("ignore")


# SparkSession is managed by Databricks Connect
spark = SparkSession.builder.getOrCreate()


if __name__ == "__main__":

    # ── Stage 1: Data Pipeline ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STAGE 1 — DATA PIPELINE  (Bronze -> Silver -> Gold)")
    print("=" * 60)

    DataWarehousePipeline(spark).run()

    # ── Stage 2: ML Pipeline ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STAGE 2 — ML PIPELINE  (Schema -> Train -> Register)")
    print("=" * 60)

    predictions_df = MLPipeline(spark).run()

    # ── Preview predictions ────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("  Sample Predictions (first 10 rows):")
    print("-" * 60)
    print(predictions_df.head(10).to_string(index=False))
    print("\n  All done! Check Databricks for:")
    print("  - MLflow Experiment:  Parent + Child runs per model")
    print("  - Unity Catalog ml:   4 tables + 1 volume + registered model")
