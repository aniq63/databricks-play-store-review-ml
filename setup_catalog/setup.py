CATALOG_NAME = "play_store_reviews"


spark.sql(f"CREATE CATALOG IF NOT EXISTS `{CATALOG_NAME}`")
print(f"Catalog '{CATALOG_NAME}' is ready.")

schemas = ["bronze", "silver", "gold"]

for schema in schemas:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{CATALOG_NAME}`.`{schema}`")
    print(f"Schema '{CATALOG_NAME}.{schema}' is ready.")


print(spark.sql(f"SHOW SCHEMAS IN `{CATALOG_NAME}`"))
