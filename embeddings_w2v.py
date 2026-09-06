# Word2Vec baseline — compare against the Arctic notebook.
# Spark ML Word2Vec has no pretrained checkpoint. fit() only learns token
# vectors from this corpus (maxIter=1). minCount=1 keeps rare names;
# the original minCount=20 dropped them.

import os

from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from pyspark.sql.window import Window
from pyspark.ml.feature import RegexTokenizer, Word2Vec
from pyspark.ml.functions import vector_to_array

dbutils.widgets.text("cc_source", "cc_pre", "Credit-card source table")
dbutils.widgets.text("cu_source", "re_pre", "Customer source table")
dbutils.widgets.text("idr_run_date", "2026-08-31", "idr_run_date filter (if column exists)")
dbutils.widgets.text("out_candidates", "", "Optional output table (blank = session only)")

IDR_RUN_DATE = dbutils.widgets.get("idr_run_date")
CC_PRE_TABLE = dbutils.widgets.get("cc_source").strip()
RE_PRE_TABLE = dbutils.widgets.get("cu_source").strip()
OUT_CANDIDATES = dbutils.widgets.get("out_candidates").strip()

CC_ID_COL = "record_id"
CU_ID_COL = "customer_record_id"

CC_PII_FIELDS = [
    "physical_address", "full_name", "title", "first_name", "middle_name",
    "last_name", "suffix", "city", "zip_code", "date_of_birth",
]
RE_PII_FIELDS = [
    "address_all", "full_name", "title", "first_name", "middle_name",
    "last_name", "suffix", "city", "zip_code", "date_of_birth",
]

EMBEDDING_DIM = 768
N_PARTS = 40
SAMPLE_CC = 0.05
SAMPLE_CU = 0.05
SAMPLE_SEED = 42
MIN_NONEMPTY_FIELDS = 4
SIM_THRESHOLD = 0.78
TOP_K_PER_CC = 5

# ---------------------------------------------------------------------------
# 1. Text from PII values only (same function name as original)
# ---------------------------------------------------------------------------
def _nz(v):
    return "" if v is None else str(v).strip()


def create_json(address, full_name, title, first, middle, last, suffix, city, zip_code, dob):
    # Not JSON. JSON keys were identical on every row and dominated the embedding.
    parts = [
        _nz(full_name), _nz(title), _nz(first), _nz(middle), _nz(last), _nz(suffix),
        _nz(address), _nz(city), _nz(zip_code), _nz(dob),
    ]
    return " ".join(p for p in parts if p)


combine_names_udf = F.udf(create_json, StringType())


def nonempty_count(fields):
    return sum(F.when(F.col(c).isNotNull() & (F.trim(F.col(c).cast("string")) != ""), 1).otherwise(0) for c in fields)


def zip5(colname):
    return F.regexp_extract(F.upper(F.coalesce(F.col(colname).cast("string"), F.lit(""))), r"(\d{5})", 1)


def last3(colname):
    return F.substring(F.regexp_replace(F.upper(F.coalesce(F.col(colname).cast("string"), F.lit(""))), r"[^A-Z0-9]", ""), 1, 3)


df_cc = spark.table(CC_PRE_TABLE)
df_cu = spark.table(RE_PRE_TABLE)
if "idr_run_date" in df_cc.columns:
    df_cc = df_cc.filter(F.col("idr_run_date") == F.lit(IDR_RUN_DATE))
if "idr_run_date" in df_cu.columns:
    df_cu = df_cu.filter(F.col("idr_run_date") == F.lit(IDR_RUN_DATE))

df_cc = (
    df_cc.withColumn("date_of_birth", F.col("date_of_birth").cast("string"))
    .filter(F.col(CC_ID_COL).isNotNull())
    .filter(nonempty_count(CC_PII_FIELDS) >= MIN_NONEMPTY_FIELDS)
    .dropDuplicates([CC_ID_COL])
    .withColumn(
        "json",
        combine_names_udf(
            F.col("physical_address"), F.col("full_name"), F.col("title"),
            F.col("first_name"), F.col("middle_name"), F.col("last_name"),
            F.col("suffix"), F.col("city"), F.col("zip_code"), F.col("date_of_birth"),
        ),
    )
    .withColumn("block_zip", zip5("zip_code"))
    .withColumn("block_ln", last3("last_name"))
)

df_cu = (
    df_cu.withColumn("date_of_birth", F.col("date_of_birth").cast("string"))
    .filter(F.col(CU_ID_COL).isNotNull())
    .filter(nonempty_count(RE_PII_FIELDS) >= MIN_NONEMPTY_FIELDS)
    .dropDuplicates([CU_ID_COL])
    .withColumn(
        "json",
        combine_names_udf(
            F.col("address_all"), F.col("full_name"), F.col("title"),
            F.col("first_name"), F.col("middle_name"), F.col("last_name"),
            F.col("suffix"), F.col("city"), F.col("zip_code"), F.col("date_of_birth"),
        ),
    )
    .withColumn("block_zip", zip5("zip_code"))
    .withColumn("block_ln", last3("last_name"))
)

# ---------------------------------------------------------------------------
# 2. Spark Word2Vec — native Spark, runs on serverless workers (no `sc`)
# ---------------------------------------------------------------------------
cc_slim = df_cc.select(
    F.col(CC_ID_COL).cast("string").alias("record_id"),
    F.col("json").cast("string").alias("json"),
    F.col("block_zip").cast("string").alias("block_zip"),
    F.col("block_ln").cast("string").alias("block_ln"),
    F.col("first_name").cast("string").alias("cc_first_name"),
    F.col("last_name").cast("string").alias("cc_last_name"),
    F.col("physical_address").cast("string").alias("cc_address"),
)
cu_slim = df_cu.select(
    F.col(CU_ID_COL).cast("string").alias("customer_record_id"),
    F.col("json").cast("string").alias("json"),
    F.col("block_zip").cast("string").alias("block_zip"),
    F.col("block_ln").cast("string").alias("block_ln"),
    F.col("first_name").cast("string").alias("cu_first_name"),
    F.col("last_name").cast("string").alias("cu_last_name"),
    F.col("address_all").cast("string").alias("cu_address"),
)

if SAMPLE_CC < 1.0:
    cc_slim = cc_slim.sample(False, SAMPLE_CC, SAMPLE_SEED)
if SAMPLE_CU < 1.0:
    cu_slim = cu_slim.sample(False, SAMPLE_CU, SAMPLE_SEED)

tok = (
    RegexTokenizer(inputCol="json", outputCol="tokens", pattern="\\w+")
    .setToLowercase(True)
    .setMinTokenLength(2)
)
df_tok1 = tok.transform(cc_slim.fillna({"json": ""}))
df_tok2 = tok.transform(cu_slim.fillna({"json": ""}))

train_df = (
    df_tok1.select("tokens")
    .union(df_tok2.select("tokens"))
    .repartition(N_PARTS)
)

w2v = (
    Word2Vec()
    .setVectorSize(EMBEDDING_DIM)
    .setMinCount(1)
    .setWindowSize(5)
    .setNumPartitions(N_PARTS)
    .setMaxIter(1)
    .setInputCol("tokens")
    .setOutputCol("embedding")
)
print("Fitting Word2Vec on tokenized PII values …")
w2v_model = w2v.fit(train_df)


def l2_normalize(df):
    vals = vector_to_array(F.col("embedding")).cast("array<float>")
    norm = F.sqrt(F.aggregate(vals, F.lit(0.0), lambda acc, x: acc + x * x))
    unit = F.transform(vals, lambda x: x / norm)
    return (
        df.withColumn("vec", unit)
        .filter(F.col("embedding").isNotNull() & (norm > 0))
        .drop("embedding", "tokens")
    )


cc = l2_normalize(w2v_model.transform(df_tok1))
cu = l2_normalize(w2v_model.transform(df_tok2))
cc.createOrReplaceTempView("cc")
cu.createOrReplaceTempView("cu")
print("Word2Vec rows cc/cu:", cc.count(), cu.count())

# ---------------------------------------------------------------------------
# 3. Match: cosine SIMILARITY, high = similar, blocked join
# ---------------------------------------------------------------------------
dot = F.aggregate(
    F.arrays_zip(F.col("cc.vec"), F.col("cu.vec")),
    F.lit(0.0),
    lambda acc, x: acc + x["0"].cast("double") * x["1"].cast("double"),
).alias("cosine_similarity")

pairs = (
    cc.alias("cc")
    .join(
        cu.alias("cu"),
        (F.col("cc.block_zip") == F.col("cu.block_zip"))
        & (F.col("cc.block_ln") == F.col("cu.block_ln"))
        & (F.col("cc.block_zip") != "")
        & (F.length(F.col("cc.block_zip")) == 5),
        "inner",
    )
    .select(
        F.col("cc.record_id"),
        F.col("cu.customer_record_id"),
        dot,
        F.col("cc.cc_first_name"),
        F.col("cc.cc_last_name"),
        F.col("cc.cc_address"),
        F.col("cu.cu_first_name"),
        F.col("cu.cu_last_name"),
        F.col("cu.cu_address"),
    )
    .filter(F.col("cosine_similarity") >= SIM_THRESHOLD)
)

w = Window.partitionBy("record_id").orderBy(F.col("cosine_similarity").desc())
df_out = (
    pairs.withColumn("rn", F.row_number().over(w))
    .filter(F.col("rn") <= TOP_K_PER_CC)
    .drop("rn")
    .orderBy(F.col("cosine_similarity").desc())
)

df_out.createOrReplaceTempView("tmp_pii_candidate_pairs")
if OUT_CANDIDATES:
    df_out.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(OUT_CANDIDATES)

display(df_out.limit(100))
