from pyspark.sql import functions as F
import matplotlib.pyplot as plt
import pandas as pd

# Assume `popular_items` is the aggregated Spark DataFrame:
# consumer_unit_code_id | purchase_count

# Take top N for the plot (full ranking can stay in Spark)
N = 30
top_n = popular_items.orderBy(F.desc("purchase_count")).limit(N)

# Bring only the top N to the driver
pdf = top_n.toPandas()

# Optional: sort for the bar chart (highest at top)
pdf = pdf.sort_values("purchase_count", ascending=True)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))

bars = ax.barh(
    pdf["consumer_unit_code_id"].astype(str),
    pdf["purchase_count"],
    color="#1f4e79",
    edgecolor="white",
    linewidth=0.4,
)

ax.set_xlabel("Purchase count")
ax.set_ylabel("Item (consumer_unit_code_id)")
ax.set_title(f"Top {N} items by purchase volume")
ax.grid(axis="x", linestyle="--", alpha=0.4)

# Format x-axis with commas
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

# Add value labels on the bars (optional, useful for the top few)
for bar in bars:
    width = bar.get_width()
    ax.text(
        width,
        bar.get_y() + bar.get_height() / 2,
        f"{int(width):,}",
        va="center",
        ha="left",
        fontsize=8,
        color="#333333",
    )

fig.tight_layout()
fig.savefig("top_items_by_purchase_count.png", dpi=140)
plt.show()  # or plt.close() if running as a script
