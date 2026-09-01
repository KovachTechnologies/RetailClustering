# 1. parse embeddings columns
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, IntegerType, ArrayType, DoubleType, StringType
)

vec_schema = StructType([
    StructField("type", IntegerType()),
    StructField("size", IntegerType()),
    StructField("indices", ArrayType(IntegerType())),
    StructField("values", ArrayType(DoubleType())),  # if parse fails, use ArrayType(StringType())
])

def parse_embedding(df, col="embedding"):
    raw = F.col(col)
    parsed = F.from_json(raw.cast("string"), vec_schema)
    vals = parsed["values"].cast("array<float>")
    # zero / empty vectors make cosine undefined
    norm = F.sqrt(F.aggregate(vals, F.lit(0.0), lambda acc, x: acc + x * x))
    return (
        df
        .withColumn("vec", vals)
        .withColumn("vec_norm", norm)
        .filter(F.col("vec").isNotNull() & (F.col("vec_norm") > 0))
    )

cc = parse_embedding(
    spark.table("scientists.dscoe_prcng_dev.creditcard_embeddings"),
    "embedding",
).select("record_id", "vec")

cu = parse_embedding(
    spark.table("scientists.dscoe_prcng_dev.customer_embeddings"),
    "embedding",
).select("customer_record_id", "identity_id", "vec")

# 2. cosine distance

cc.createOrReplaceTempView("cc")
cu.createOrReplaceTempView("cu")

df_matches = spark.sql("""
SELECT
  cu.customer_record_id,
  cu.identity_id,
  cc.record_id,
  CAST(1.0 - vector_cosine_similarity(cc.vec, cu.vec) AS DOUBLE) AS cosine_distance
FROM cc
INNER JOIN cu
  EXACT NEAREST 1 BY SIMILARITY vector_cosine_similarity(cc.vec, cu.vec)
""")


