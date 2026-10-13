from pyspark.sql import functions as F
from pyspark.sql.window import Window

# 1. Filter high-similarity pairs
df_pairs = (
    df_test_inner
    .filter(F.col("cosine_distance") > 0.99)
    .select(
        "external_id_1", "par_1",
        "external_id_2", "par_2",
        "cosine_distance"
    )
)

# 2. Build external_id -> set of items
df_items = (
    df_par_with_items
    .groupBy("external_id")
    .agg(F.collect_set("DSC_ITEM").alias("items"))
)

# 3. Attach item sets for both sides
df = (
    df_pairs
    .join(
        df_items.alias("i1"),
        F.col("external_id_1") == F.col("i1.external_id"),
        "left"
    )
    .join(
        df_items.alias("i2"),
        F.col("external_id_2") == F.col("i2.external_id"),
        "left"
    )
    .select(
        "external_id_1", "par_1",
        "external_id_2", "par_2",
        "cosine_distance",
        F.coalesce(F.col("i1.items"), F.array()).alias("items_1"),
        F.coalesce(F.col("i2.items"), F.array()).alias("items_2")
    )
)

# 4. Compute intersection size, union size, and pct_match
df_result = (
    df
    .withColumn(
        "intersection_size",
        F.size(F.array_intersect("items_1", "items_2"))
    )
    .withColumn(
        "union_size",
        F.size(F.array_union("items_1", "items_2"))
    )
    .withColumn(
        "pct_match",
        F.when(
            F.col("union_size") > 0,
            F.col("intersection_size") / F.col("union_size")
        ).otherwise(0.0)
    )
    .select(
        "external_id_1", "par_1",
        "external_id_2", "par_2",
        "cosine_distance",
        "intersection_size",
        "union_size",
        "pct_match",
        "items_1",
        "items_2"
    )
)

df_result.show(20, truncate=False)
