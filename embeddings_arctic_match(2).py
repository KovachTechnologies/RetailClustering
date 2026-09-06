# Databricks notebook cell — PII record matching with Snowflake Arctic embeddings
# Paste into a single cell (or split on the SECTION comments).
#
# What was wrong in the Word2Vec notebook
# ---------------------------------------
# 1. Embedding a JSON blob whose KEYS are identical on every row
#    ("address","full_name","first_name",...). Word2Vec / token averages
#    then collapse onto those shared tokens. Rare names are dropped by
#    minCount=20. Unrelated people get ~0.9 "similarity".
# 2. cosine_distance = 1 - cosine_similarity, then ORDER BY distance DESC.
#    That surfaces the WORST matches. Cosine similarity is high when
#    records look alike; cosine distance is low.
# 3. `EXACT NEAREST 1 BY SIMILARITY` is Vector Search syntax, not valid
#    against temp views of ARRAY<FLOAT>.
# 4. Tables were crossed (cc_pre loaded as "customer", re_pre as "cc"),
#    create_json was incomplete, and JSON used `=` instead of `:`.
#
# What this cell does
# -------------------
# * Builds a short labeled text string (not JSON) from PII fields.
# * Encodes with Snowflake/snowflake-arctic-embed-m-v1.5 (768-d, L2-normalized).
# * Blocks on zip5 + first 3 letters of last name so we never cartesian-join.
# * Scores cosine SIMILARITY (dot product of L2-normalized vectors).
# * Prints a sanity check: known-same vs known-different pairs.

# %pip install -q sentence-transformers==3.3.1
# dbutils.library.restartPython()

# =============================================================================
# SECTION 0 — imports and parameters
# =============================================================================
import os
import re
import numpy as np
import pandas as pd

from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    StringType,
    DoubleType,
    IntegerType,
    StructType,
    StructField,
)
from pyspark.sql.window import Window

# Notebook widgets — no catalog.schema prefix required.
# Fill these in the notebook UI or leave the defaults if the hive-metastore
# / current database already has cc_pre and re_pre.
dbutils.widgets.text("cc_source", "cc_pre", "Credit-card source table")
dbutils.widgets.text("cu_source", "re_pre", "Customer source table")
dbutils.widgets.text("idr_run_date", "2026-08-31", "idr_run_date filter (if column exists)")
dbutils.widgets.text("out_candidates", "", "Optional output table for pairs (blank = session only)")

IDR_RUN_DATE = dbutils.widgets.get("idr_run_date")
CC_PRE_TABLE = dbutils.widgets.get("cc_source").strip()
RE_PRE_TABLE = dbutils.widgets.get("cu_source").strip()
OUT_CANDIDATES = dbutils.widgets.get("out_candidates").strip()

# Session-scoped names only. Nothing is written to Unity Catalog unless
# out_candidates is filled in.
TMP_CC = "tmp_cc_embeddings"
TMP_CU = "tmp_cu_embeddings"
TMP_PAIRS = "tmp_pii_candidate_pairs"

CC_ID_COL = "record_id"
CU_ID_COL = "customer_record_id"

MODEL_NAME = "Snowflake/snowflake-arctic-embed-m-v1.5"
MODEL_LOCAL_DIR = os.path.expanduser("~/shared/snowflake-arctic-embed-m-v1.5")

EMBED_DIM = 768
MIN_NONEMPTY_FIELDS = 4
SIM_THRESHOLD = 0.78          # tune after looking at the sanity-check + histogram
TOP_K_PER_CC = 5
BLOCK_LAST3 = True            # False = block on zip only (recall up, cost up)


# =============================================================================
# SECTION 1 — text building (this is the embedding input, not JSON)
# =============================================================================
def _clean_sql(col):
    """Uppercase, collapse whitespace, drop nulls. Keep alphanumerics + #."""
    return F.upper(
        F.trim(
            F.regexp_replace(F.coalesce(F.col(col).cast("string"), F.lit("")), r"\s+", " ")
        )
    )


def zip5(colname):
    return F.regexp_extract(_clean_sql(colname), r"(\d{5})", 1)


def last3(colname):
    return F.substring(
        F.regexp_replace(_clean_sql(colname), r"[^A-Z0-9]", ""), 1, 3
    )


def build_embed_text(df, field_map):
    """
    field_map: dict label -> column name, e.g. {"FIRST": "first_name", ...}
    Produces a compact labeled document. Labels are short and the VALUES
    dominate the tokens, unlike a JSON blob where keys repeat on every row.
    """
    parts = []
    nonempty = []
    for label, colname in field_map.items():
        cleaned = _clean_sql(colname)
        parts.append(
            F.when(cleaned == "", F.lit("")).otherwise(F.concat(F.lit(f"{label}: "), cleaned))
        )
        nonempty.append(F.when(cleaned == "", F.lit(0)).otherwise(F.lit(1)))
    text = F.regexp_replace(F.concat_ws(" | ", *parts), r"( \| )+", " | ")
    text = F.regexp_replace(text, r"^( \| )+|( \| )+$", "")
    n_filled = sum(nonempty)
    return df.withColumn("embed_text", text).withColumn("n_filled", n_filled)


CC_FIELDS = {
    "NAME": "full_name",
    "FIRST": "first_name",
    "MIDDLE": "middle_name",
    "LAST": "last_name",
    "TITLE": "title",
    "SUFFIX": "suffix",
    "ADDR": "physical_address",
    "CITY": "city",
    "ZIP": "zip_code",
    "DOB": "date_of_birth",
}
CU_FIELDS = {
    "NAME": "full_name",
    "FIRST": "first_name",
    "MIDDLE": "middle_name",
    "LAST": "last_name",
    "TITLE": "title",
    "SUFFIX": "suffix",
    "ADDR": "address_all",
    "CITY": "city",
    "ZIP": "zip_code",
    "DOB": "date_of_birth",
}


# =============================================================================
# SECTION 2 — load + prepare both sides
# =============================================================================
# Prefer the prepped tables if they already have the PII columns.
# Fall back comments show the raw prod tables.

df_cc_raw = spark.table(CC_PRE_TABLE)
df_cu_raw = spark.table(RE_PRE_TABLE)

if "idr_run_date" in df_cu_raw.columns:
    df_cu_raw = df_cu_raw.filter(F.col("idr_run_date") == F.lit(IDR_RUN_DATE))
if "idr_run_date" in df_cc_raw.columns:
    df_cc_raw = df_cc_raw.filter(F.col("idr_run_date") == F.lit(IDR_RUN_DATE))

# Harmonize types; DOB as string so the encoder sees the same token shape.
for c in ("date_of_birth",):
    if c in df_cc_raw.columns:
        df_cc_raw = df_cc_raw.withColumn(c, F.col(c).cast("string"))
    if c in df_cu_raw.columns:
        df_cu_raw = df_cu_raw.withColumn(c, F.col(c).cast("string"))

df_cc = (
    build_embed_text(df_cc_raw, CC_FIELDS)
    .filter(F.col("n_filled") >= MIN_NONEMPTY_FIELDS)
    .filter(F.col(CC_ID_COL).isNotNull())
    .withColumn("block_zip", zip5("zip_code"))
    .withColumn("block_ln", last3("last_name"))
    .dropDuplicates([CC_ID_COL])
)

df_cu = (
    build_embed_text(df_cu_raw, CU_FIELDS)
    .filter(F.col("n_filled") >= MIN_NONEMPTY_FIELDS)
    .filter(F.col(CU_ID_COL).isNotNull())
    .withColumn("block_zip", zip5("zip_code"))
    .withColumn("block_ln", last3("last_name"))
    .dropDuplicates([CU_ID_COL])
)

print("CC rows after filter:", df_cc.count())
print("CU rows after filter:", df_cu.count())
display(df_cc.select(CC_ID_COL, "embed_text", "block_zip", "block_ln", "n_filled").limit(5))
display(df_cu.select(CU_ID_COL, "embed_text", "block_zip", "block_ln", "n_filled").limit(5))


# =============================================================================
# SECTION 3 — load Arctic on the DRIVER only
# Serverless is Spark Connect: there is no `sc`, and worker processes cannot
# see the driver's local disk or a broadcast variable. Do not use a pandas
# UDF / mapInPandas to load SentenceTransformer. Encode on the driver and
# hand vectors back to Spark as ordinary arrays.
# =============================================================================
from sentence_transformers import SentenceTransformer

ENCODE_BATCH = 64          # model.encode batch
CHUNK_ROWS = 400           # rows pulled per Spark query (keep small on serverless)
SAMPLE_CC = 0.05           # 1.0 = full table; start small so the run finishes
SAMPLE_CU = 0.05
SAMPLE_SEED = 42

_MODEL = None


def get_model():
    """Single driver-side singleton. Never call this on a worker."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if os.path.isdir(MODEL_LOCAL_DIR) and os.listdir(MODEL_LOCAL_DIR):
        path = MODEL_LOCAL_DIR
        print("Loading model from", path)
        _MODEL = SentenceTransformer(path)
    else:
        print("Downloading", MODEL_NAME, "onto the driver …")
        _MODEL = SentenceTransformer(MODEL_NAME)
        try:
            os.makedirs(MODEL_LOCAL_DIR, exist_ok=True)
            _MODEL.save(MODEL_LOCAL_DIR)
            print("Saved model to", MODEL_LOCAL_DIR)
        except Exception as exc:
            print("Could not cache model locally:", exc)
    return _MODEL


def encode_texts(texts):
    model = get_model()
    vecs = model.encode(
        [t if t is not None else "" for t in texts],
        batch_size=ENCODE_BATCH,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [v.astype(np.float32).tolist() for v in vecs]


def _as_str(v):
    if v is None:
        return ""
    return str(v)


def _pdf_to_records(pdf, field_names):
    recs = []
    for _, row in pdf.iterrows():
        rec = {}
        for name in field_names:
            if name == "n_filled":
                rec[name] = int(row[name] or 0)
            else:
                rec[name] = _as_str(row[name])
        recs.append(rec)
    return recs


def embed_on_driver(df, schema, label="df"):
    """
    Pull short-lived chunks with toPandas() (query ends immediately),
    then encode on the driver. Do NOT use toLocalIterator() on serverless:
    the Connect handle is killed for inactivity while encode() runs
    (INVALID_HANDLE.OPERATION_ABANDONED / HY000).
    """
    field_names = [f.name for f in schema.fields if f.name != "vec"]
    n = df.count()
    print(f"{label}: {n} rows to embed")
    if n == 0:
        raise RuntimeError(f"{label} is empty")

    frames = []
    done = 0
    # row_number is computed once; each filter+toPandas is its own query
    ranked = df.withColumn("_rn", F.row_number().over(Window.orderBy(F.lit(1))))
    start = 1
    while start <= n:
        end = min(start + CHUNK_ROWS - 1, n)
        pdf = (
            ranked.filter((F.col("_rn") >= start) & (F.col("_rn") <= end))
            .drop("_rn")
            .toPandas()
        )
        recs = _pdf_to_records(pdf, field_names)
        texts = [r.get("embed_text") or "" for r in recs]
        for i in range(0, len(texts), ENCODE_BATCH):
            vecs = encode_texts(texts[i : i + ENCODE_BATCH])
            for rec, vec in zip(recs[i : i + ENCODE_BATCH], vecs):
                rec["vec"] = [float(x) for x in vec]
        frames.append(spark.createDataFrame(pd.DataFrame(recs), schema=schema))
        done += len(recs)
        print(f"{label}: embedded {done}/{n}")
        start = end + 1

    out = frames[0]
    for frm in frames[1:]:
        out = out.unionByName(frm)
    return out


CC_EMB_SCHEMA = StructType(
    [
        StructField("record_id", StringType(), True),
        StructField("embed_text", StringType(), True),
        StructField("block_zip", StringType(), True),
        StructField("block_ln", StringType(), True),
        StructField("n_filled", IntegerType(), True),
        StructField("cc_first_name", StringType(), True),
        StructField("cc_last_name", StringType(), True),
        StructField("cc_address", StringType(), True),
        StructField("cc_zip", StringType(), True),
        StructField("vec", ArrayType(FloatType()), False),
    ]
)
CU_EMB_SCHEMA = StructType(
    [
        StructField("customer_record_id", StringType(), True),
        StructField("embed_text", StringType(), True),
        StructField("block_zip", StringType(), True),
        StructField("block_ln", StringType(), True),
        StructField("n_filled", IntegerType(), True),
        StructField("cu_first_name", StringType(), True),
        StructField("cu_last_name", StringType(), True),
        StructField("cu_address", StringType(), True),
        StructField("cu_zip", StringType(), True),
        StructField("vec", ArrayType(FloatType()), False),
    ]
)


# =============================================================================
# SECTION 4 — sanity check BEFORE touching the lake
# Runs entirely on the driver. If this fails, do not embed the full tables.
# =============================================================================
probe = {
    "same_a": "NAME: ADAM YANELLI | FIRST: ADAM | LAST: YANELLI | ADDR: 5353 SPACE CENTER BLVD #2301 | CITY: PASADENA | ZIP: 77505",
    "same_b": "NAME: ADAM YANELLI | FIRST: ADAM | LAST: YANELLI | ADDR: 5353 SPACE CENTER BOULEVARD UNIT 2301 | CITY: PASADENA | ZIP: 77505",
    "diff_a": "NAME: ADAM YANELLI | FIRST: ADAM | LAST: YANELLI | ADDR: 5353 SPACE CENTER BLVD #2301 | CITY: PASADENA | ZIP: 77505",
    "diff_b": "NAME: GATEWAY STATION | FIRST: GATEWAY | LAST: STATION | ADDR: 117 HOLLEMAN DRIVE WEST | CITY: COLLEGE STATION | ZIP: 77840",
    "diff_c": "NAME: KIMBERLY JOHNS | FIRST: KIMBERLY | LAST: JOHNS | ADDR: 634 VILLAGE SHORE DRIVE | CITY: CANYON LAKE | ZIP: 78133",
    "diff_d": "NAME: ANTONIO GARCIA | FIRST: ANTONIO | LAST: GARCIA | ADDR: 7600 CALLAGHAN ROAD 118 | CITY: SAN ANTONIO | ZIP: 78229",
}
probe_vecs = dict(zip(probe.keys(), encode_texts(list(probe.values()))))


def cos(a, b):
    return float(np.dot(np.asarray(a, np.float32), np.asarray(b, np.float32)))


print("SANITY cosine_similarity (expect same_* ~0.90+, diff_* ~0.3-0.6)")
print("  same person, lightly rewritten addr :", round(cos(probe_vecs["same_a"], probe_vecs["same_b"]), 4))
print("  ADAM YANELLI vs GATEWAY STATION     :", round(cos(probe_vecs["diff_a"], probe_vecs["diff_b"]), 4))
print("  KIMBERLY JOHNS vs ANTONIO GARCIA    :", round(cos(probe_vecs["diff_c"], probe_vecs["diff_d"]), 4))


# =============================================================================
# SECTION 5 — project to known columns, then embed on the driver
# =============================================================================
cc_slim = df_cc.select(
    F.col(CC_ID_COL).cast("string").alias("record_id"),
    F.col("embed_text").cast("string").alias("embed_text"),
    F.col("block_zip").cast("string").alias("block_zip"),
    F.col("block_ln").cast("string").alias("block_ln"),
    F.col("n_filled").cast("int").alias("n_filled"),
    F.col("first_name").cast("string").alias("cc_first_name"),
    F.col("last_name").cast("string").alias("cc_last_name"),
    F.col("physical_address").cast("string").alias("cc_address"),
    F.col("zip_code").cast("string").alias("cc_zip"),
)
cu_slim = df_cu.select(
    F.col(CU_ID_COL).cast("string").alias("customer_record_id"),
    F.col("embed_text").cast("string").alias("embed_text"),
    F.col("block_zip").cast("string").alias("block_zip"),
    F.col("block_ln").cast("string").alias("block_ln"),
    F.col("n_filled").cast("int").alias("n_filled"),
    F.col("first_name").cast("string").alias("cu_first_name"),
    F.col("last_name").cast("string").alias("cu_last_name"),
    F.col("address_all").cast("string").alias("cu_address"),
    F.col("zip_code").cast("string").alias("cu_zip"),
)

# Arctic is already trained — there is no fit() step. Sampling is only
# to keep driver-side encode() inside serverless idle limits.
if SAMPLE_CC < 1.0:
    cc_slim = cc_slim.sample(False, SAMPLE_CC, SAMPLE_SEED)
if SAMPLE_CU < 1.0:
    cu_slim = cu_slim.sample(False, SAMPLE_CU, SAMPLE_SEED)

cc_out = embed_on_driver(cc_slim, CC_EMB_SCHEMA, label="cc")
cu_out = embed_on_driver(cu_slim, CU_EMB_SCHEMA, label="cu")

cc_out.createOrReplaceTempView(TMP_CC)
cu_out.createOrReplaceTempView(TMP_CU)
cc_out.cache()
cu_out.cache()
print(f"Cached session views {TMP_CC} ({cc_out.count()} rows) and {TMP_CU} ({cu_out.count()} rows)")


# =============================================================================
# SECTION 6 — blocked similarity join
# Cosine similarity of L2-normalized vectors = dot product.
# Implemented in Spark so we do not depend on Vector Search syntax.
# =============================================================================
cc = spark.table(TMP_CC).alias("cc")
cu = spark.table(TMP_CU).alias("cu")

dot = F.aggregate(
    F.arrays_zip(F.col("cc.vec"), F.col("cu.vec")),
    F.lit(0.0),
    lambda acc, x: acc + x["0"].cast("double") * x["1"].cast("double"),
).alias("cosine_similarity")

join_keys = [F.col("cc.block_zip") == F.col("cu.block_zip")]
if BLOCK_LAST3:
    join_keys.append(F.col("cc.block_ln") == F.col("cu.block_ln"))

# Drop empty blocks — they would otherwise become a giant bucket.
pairs = (
    cc.join(cu, join_keys, "inner")
    .filter(F.col("cc.block_zip") != "")
    .filter(F.length(F.col("cc.block_zip")) == 5)
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
        F.col("cc.embed_text").alias("cc_embed_text"),
        F.col("cu.embed_text").alias("cu_embed_text"),
        F.col("cc.block_zip"),
    )
    .filter(F.col("cosine_similarity") >= SIM_THRESHOLD)
)

w = Window.partitionBy("record_id").orderBy(F.col("cosine_similarity").desc())
top = (
    pairs.withColumn("rn", F.row_number().over(w))
    .filter(F.col("rn") <= TOP_K_PER_CC)
    .drop("rn")
)

top.createOrReplaceTempView(TMP_PAIRS)
top.cache()
n_pairs = top.count()
print(f"Cached session view {TMP_PAIRS} ({n_pairs} rows)")

if OUT_CANDIDATES:
    (
        top.write.mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(OUT_CANDIDATES)
    )
    print("Also wrote table", OUT_CANDIDATES)

display(
    spark.table(TMP_PAIRS)
    .orderBy(F.col("cosine_similarity").desc())
    .select(
        "cosine_similarity",
        "cc_first_name",
        "cc_last_name",
        "cc_address",
        "cu_first_name",
        "cu_last_name",
        "cu_address",
    )
    .limit(50)
)

# Histogram so you can set SIM_THRESHOLD from data rather than a guess.
display(
    spark.table(TMP_PAIRS)
    .select(F.round(F.col("cosine_similarity"), 2).alias("bin"))
    .groupBy("bin")
    .count()
    .orderBy("bin")
)
)
