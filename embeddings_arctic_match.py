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

IDR_RUN_DATE = "2026-08-31"
# Fully-qualified catalog.schema in THIS tenant (outputs + prepped sources).
CATALOG_SCHEMA = "catalog.schema"
VERSION = "arctic_v2"

# Source tables — confirm these names in your workspace.
# The original notebook swapped cc_pre / re_pre; keep them explicit here.
CC_PRE_TABLE = f"{CATALOG_SCHEMA}.cc_pre"          # credit-card side, prepped
RE_PRE_TABLE = f"{CATALOG_SCHEMA}.re_pre"          # customer / RE side, prepped
CC_RAW_TABLE = "prod.customer.credit_card_attribute_history"
CU_RAW_TABLE = "prod.customer_splink.linker_base"

CC_ID_COL = "record_id"
CU_ID_COL = "customer_record_id"

# If serverless local I/O was the original blocker, put the model on a
# UC Volume the cluster can read. Fallback: driver-local cache.
MODEL_NAME = "Snowflake/snowflake-arctic-embed-m-v1.5"
MODEL_VOLUME_DIR = "/Volumes/catalog/schema/models/snowflake-arctic-embed-m-v1.5"
MODEL_LOCAL_DIR = os.path.expanduser("~/shared/snowflake-arctic-embed-m-v1.5")

EMBED_DIM = 768
MIN_NONEMPTY_FIELDS = 4
SIM_THRESHOLD = 0.78          # tune after looking at the sanity-check + histogram
TOP_K_PER_CC = 5
BLOCK_LAST3 = True            # False = block on zip only (recall up, cost up)

OUT_EMBED_CC = f"{CATALOG_SCHEMA}.cc_embeddings_{VERSION}"
OUT_EMBED_CU = f"{CATALOG_SCHEMA}.cu_embeddings_{VERSION}"
OUT_CANDIDATES = f"{CATALOG_SCHEMA}.pii_candidate_pairs_{VERSION}"


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

# If you instead need the prod tables, swap to:
# df_cc_raw = (
#     spark.table(CC_RAW_TABLE)
#     .withColumnRenamed("physical_address", "physical_address")
# )
# df_cu_raw = (
#     spark.table(CU_RAW_TABLE)
#     .filter(F.col("idr_run_date") == IDR_RUN_DATE)
#     .filter(F.col(CU_ID_COL).isNotNull())
# )

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
# SECTION 3 — load Arctic once, share path with workers
# =============================================================================
from sentence_transformers import SentenceTransformer


def resolve_model_path():
    """
    Prefer a UC Volume (survives serverless). Otherwise a local folder
    populated on the driver. Workers must be able to read the same path.
    """
    for path in (MODEL_VOLUME_DIR, MODEL_LOCAL_DIR):
        if path and os.path.isdir(path) and os.listdir(path):
            return path
    # Download once on the driver, then save to both locations if possible.
    print(f"Downloading {MODEL_NAME} …")
    mdl = SentenceTransformer(MODEL_NAME)
    for path in (MODEL_VOLUME_DIR, MODEL_LOCAL_DIR):
        try:
            os.makedirs(path, exist_ok=True)
            mdl.save(path)
            print(f"Saved model to {path}")
            return path
        except Exception as exc:
            print(f"Could not save to {path}: {exc}")
    # Last resort: HuggingFace hub id (workers will each download).
    return MODEL_NAME


MODEL_PATH = resolve_model_path()
print("MODEL_PATH =", MODEL_PATH)

# Broadcast the path so executors agree.
model_path_bc = sc.broadcast(MODEL_PATH)

_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(model_path_bc.value)
    return _MODEL


embed_schema = ArrayType(FloatType())


@F.pandas_udf(embed_schema)
def embed_texts(texts: pd.Series) -> pd.Series:
    model = get_model()
    clean = texts.fillna("").astype(str).tolist()
    vecs = model.encode(
        clean,
        batch_size=64,
        normalize_embeddings=True,   # cosine(sim) == dot product
        show_progress_bar=False,
    )
    return pd.Series([v.astype(np.float32).tolist() for v in vecs])


# =============================================================================
# SECTION 4 — sanity check BEFORE touching the lake
# If this section fails, do not bother embedding millions of rows.
# =============================================================================
probe_rows = [
    # near-duplicate of a real CC record from your output sample
    (
        "same_a",
        "NAME: ADAM YANELLI | FIRST: ADAM | LAST: YANELLI | ADDR: 5353 SPACE CENTER BLVD #2301 | CITY: PASADENA | ZIP: 77505",
    ),
    (
        "same_b",
        "NAME: ADAM YANELLI | FIRST: ADAM | LAST: YANELLI | ADDR: 5353 SPACE CENTER BOULEVARD UNIT 2301 | CITY: PASADENA | ZIP: 77505",
    ),
    # the false pair from output.xlsx
    (
        "diff_a",
        "NAME: ADAM YANELLI | FIRST: ADAM | LAST: YANELLI | ADDR: 5353 SPACE CENTER BLVD #2301 | CITY: PASADENA | ZIP: 77505",
    ),
    (
        "diff_b",
        "NAME: GATEWAY STATION | FIRST: GATEWAY | LAST: STATION | ADDR: 117 HOLLEMAN DRIVE WEST | CITY: COLLEGE STATION | ZIP: 77840",
    ),
    (
        "diff_c",
        "NAME: KIMBERLY JOHNS | FIRST: KIMBERLY | LAST: JOHNS | ADDR: 634 VILLAGE SHORE DRIVE | CITY: CANYON LAKE | ZIP: 78133",
    ),
    (
        "diff_d",
        "NAME: ANTONIO GARCIA | FIRST: ANTONIO | LAST: GARCIA | ADDR: 7600 CALLAGHAN ROAD 118 | CITY: SAN ANTONIO | ZIP: 78229",
    ),
]
probe = spark.createDataFrame(probe_rows, ["tag", "embed_text"]).withColumn(
    "vec", embed_texts(F.col("embed_text"))
)
pdf = probe.toPandas()


def cos(a, b):
    va, vb = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.dot(va, vb))  # already L2-normalized


by_tag = dict(zip(pdf["tag"], pdf["vec"]))
print("SANITY cosine_similarity (expect same_* ~0.90+, diff_* ~0.3-0.6)")
print("  same person, lightly rewritten addr :", round(cos(by_tag["same_a"], by_tag["same_b"]), 4))
print("  ADAM YANELLI vs GATEWAY STATION     :", round(cos(by_tag["diff_a"], by_tag["diff_b"]), 4))
print("  KIMBERLY JOHNS vs ANTONIO GARCIA    :", round(cos(by_tag["diff_c"], by_tag["diff_d"]), 4))

# If same-person is not clearly above the false pairs, stop and inspect embed_text.


# =============================================================================
# SECTION 5 — embed both tables and persist
# =============================================================================
cc_emb = df_cc.withColumn("vec", embed_texts(F.col("embed_text")))
cu_emb = df_cu.withColumn("vec", embed_texts(F.col("embed_text")))

(
    cc_emb.select(
        F.col(CC_ID_COL).alias("record_id"),
        "embed_text",
        "block_zip",
        "block_ln",
        "n_filled",
        "vec",
        F.col("first_name").alias("cc_first_name"),
        F.col("last_name").alias("cc_last_name"),
        F.col("physical_address").alias("cc_address"),
        F.col("zip_code").alias("cc_zip"),
    )
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(OUT_EMBED_CC)
)

(
    cu_emb.select(
        F.col(CU_ID_COL).alias("customer_record_id"),
        "embed_text",
        "block_zip",
        "block_ln",
        "n_filled",
        "vec",
        F.col("first_name").alias("cu_first_name"),
        F.col("last_name").alias("cu_last_name"),
        F.col("address_all").alias("cu_address"),
        F.col("zip_code").alias("cu_zip"),
    )
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(OUT_EMBED_CU)
)

print("Wrote", OUT_EMBED_CC, "and", OUT_EMBED_CU)


# =============================================================================
# SECTION 6 — blocked similarity join
# Cosine similarity of L2-normalized vectors = dot product.
# Implemented in Spark so we do not depend on Vector Search syntax.
# =============================================================================
cc = spark.table(OUT_EMBED_CC).alias("cc")
cu = spark.table(OUT_EMBED_CU).alias("cu")

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

(
    top.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(OUT_CANDIDATES)
)

print("Wrote", OUT_CANDIDATES, "rows:", spark.table(OUT_CANDIDATES).count())
display(
    spark.table(OUT_CANDIDATES)
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
    spark.table(OUT_CANDIDATES)
    .select(F.round(F.col("cosine_similarity"), 2).alias("bin"))
    .groupBy("bin")
    .count()
    .orderBy("bin")
)
