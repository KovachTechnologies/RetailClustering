"""Histogram + KS + ROC-AUC for basket-vs-person cosine distance.

Reads (same directory as this script):
  cosine_distance_positive.csv   same-customer pairs
  cosine_distance_negative.csv   different-customer pairs

Cosine distance = 1 - cosine similarity. Lower is a closer match.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu

HERE = Path(__file__).resolve().parent
POS_CSV = HERE / "cosine_distance_positive.csv"
NEG_CSV = HERE / "cosine_distance_negative.csv"
OUT_PNG = HERE / "cosine_distance_histograms.png"


def load_distances(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if "cosine_distance" not in df.columns:
        raise SystemExit(f"{path.name} needs a cosine_distance column, found {list(df.columns)}")
    return df["cosine_distance"].to_numpy(dtype=float)


def roc_auc_lower_is_better(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(neg_distance > pos_distance) + 0.5 P(tie).

    Mann-Whitney U on (neg, pos) counts pairs where a negative distance
    is larger than a positive distance. That is ROC-AUC when the
    same-customer class is the low-distance class.
    """
    u = mannwhitneyu(neg, pos, alternative="two-sided").statistic
    return float(u / (len(pos) * len(neg)))


def main() -> None:
    pos = load_distances(POS_CSV)
    neg = load_distances(NEG_CSV)

    ks = ks_2samp(pos, neg, alternative="two-sided")
    auc = roc_auc_lower_is_better(pos, neg)

    print(f"n positive (same customer):      {len(pos):,}")
    print(f"n negative (different customer): {len(neg):,}")
    print(f"positive distance  mean={pos.mean():.3f}  median={np.median(pos):.3f}  std={pos.std():.3f}")
    print(f"negative distance  mean={neg.mean():.3f}  median={np.median(neg):.3f}  std={neg.std():.3f}")
    print(f"KS statistic:  {ks.statistic:.4f}")
    print(f"KS p-value:    {ks.pvalue:.2e}   (use the statistic, not p, at this N)")
    print(f"ROC-AUC:       {auc:.4f}         (lower distance = better match; 0.5 = chance)")
    print(f"P(dist < 0.20 | same):      {np.mean(pos < 0.20):.1%}")
    print(f"P(dist < 0.20 | different): {np.mean(neg < 0.20):.1%}")
    print(f"P(dist > 0.55 | same):      {np.mean(pos > 0.55):.1%}")
    print(f"P(dist > 0.55 | different): {np.mean(neg > 0.55):.1%}")

    bins = np.linspace(0.0, 1.0, 41)
    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [1, 1.05]},
    )

    axes[0, 0].hist(pos, bins=bins, density=True, color="#1f4e79", alpha=0.88, edgecolor="white", linewidth=0.4)
    axes[0, 0].axvline(np.median(pos), color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_title("Positive (same customer) — mass toward 0")
    axes[0, 0].set_ylabel("Density")

    axes[0, 1].hist(neg, bins=bins, density=True, color="#c45911", alpha=0.88, edgecolor="white", linewidth=0.4)
    axes[0, 1].axvline(np.median(neg), color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_title("Negative (different customer) — longer tail")

    axes[1, 0].hist(pos, bins=bins, density=True, color="#1f4e79", alpha=0.65, label="Same customer")
    axes[1, 0].hist(neg, bins=bins, density=True, color="#c45911", alpha=0.45, label="Different customer")
    axes[1, 0].set_xlabel("Cosine distance")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_title("Overlay (middle overlap is expected)")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.0,
        0.75,
        "\n".join(
            [
                f"n same-customer pairs:        {len(pos):>8,}",
                f"n different-customer pairs:   {len(neg):>8,}",
                "",
                f"KS statistic:                 {ks.statistic:>8.4f}",
                f"ROC-AUC (lower dist. better): {auc:>8.4f}",
                "",
                f"P(dist < 0.20 | same):        {np.mean(pos < 0.20):>8.1%}",
                f"P(dist < 0.20 | different):   {np.mean(neg < 0.20):>8.1%}",
                f"P(dist > 0.55 | same):        {np.mean(pos > 0.55):>8.1%}",
                f"P(dist > 0.55 | different):   {np.mean(neg > 0.55):>8.1%}",
            ]
        ),
        family="monospace",
        fontsize=10,
        va="top",
    )

    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xlim(0, 1)

    fig.suptitle("Pilot cosine-distance histograms", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    print(f"wrote {OUT_PNG}")
    plt.close(fig)


if __name__ == "__main__":
    main()
