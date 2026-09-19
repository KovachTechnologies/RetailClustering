import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# inner is ~1 row per id; outer is almost the full cartesian product
INNER_MAX = 50_000
OUTER_MAX = 200_000
N_BINS = 60
SEED = 42

n_inner = df_inner.count()
n_outer = df_outer.count()

pdf_inner = (
    df_inner.limit(INNER_MAX).toPandas()
    if n_inner <= INNER_MAX
    else df_inner.sample(False, min(1.0, INNER_MAX / max(n_inner, 1)), seed=SEED).toPandas()
)
pdf_outer = (
    df_outer.limit(OUTER_MAX).toPandas()
    if n_outer <= OUTER_MAX
    else df_outer.sample(False, min(1.0, OUTER_MAX / max(n_outer, 1)), seed=SEED).toPandas()
)

pdf = pd.concat(
    [
        pdf_inner.assign(pair="match (same id)"),
        pdf_outer.assign(pair="non-match"),
    ],
    ignore_index=True,
)

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(
    pdf_outer["cosine_similarity"],
    bins=np.linspace(0, 1, N_BINS + 1),
    density=True,
    alpha=0.55,
    color="#4C78A8",
    label=f"non-match (n≈{len(pdf_outer):,})",
)
ax.hist(
    pdf_inner["cosine_similarity"],
    bins=np.linspace(0, 1, N_BINS + 1),
    density=True,
    alpha=0.70,
    color="#F58518",
    label=f"match, same id (n≈{len(pdf_inner):,})",
)
ax.axvline(pdf_inner["cosine_similarity"].median(), color="#F58518", ls="--", lw=1.5)
ax.axvline(pdf_outer["cosine_similarity"].median(), color="#4C78A8", ls="--", lw=1.5)
ax.set_xlim(0, 1)
ax.set_xlabel("cosine similarity")
ax.set_ylabel("density")
ax.set_title("Basket vs customer embedding similarity")
ax.legend(frameon=False)
fig.tight_layout()
plt.show()

print(
    pdf.groupby("pair")["cosine_similarity"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .round(4)
)
