from pyspark.sql import functions as F
import pyspark.pandas as ps
import plotly.express as px

TABLE = "prod.customer.embeddings_pairs"
COL = "cosine_distance"   # or "cosine_similarity"
OUTER_N = 200_000
BINS = 60
SEED = 42

pairs = spark.table(TABLE)

df_inner = pairs.where(F.col("basket_record_id") == F.col("customer_record_id")).select(COL)

n_outer = pairs.where(F.col("basket_record_id") != F.col("customer_record_id")).count()
frac = min(1.0, OUTER_N / float(n_outer))

df_outer = (
    pairs
    .where(F.col("basket_record_id") != F.col("customer_record_id"))
    .select(COL)
    .sample(False, frac, seed=SEED)
)

# same style as psdf = dfj.pandas_api(); psdf[[col]].plot.hist(...)
ps.options.plotting.backend = "plotly"

pdf = ps.concat(
    [
        df_inner.pandas_api().assign(pair="match (same id)"),
        df_outer.pandas_api().assign(pair="non-match"),
    ]
).to_pandas()

fig = px.histogram(
    pdf,
    x=COL,
    color="pair",
    barmode="overlay",
    nbins=BINS,
    histnorm="probability density",
    opacity=0.65,
    title="Basket vs customer embedding cosine distance",
)
fig.update_layout(xaxis_title=COL, yaxis_title="density", bargap=0.05)
fig.show()

print(pdf.groupby("pair")[COL].agg(["count", "mean", "median", "std", "min", "max"]).round(4))
