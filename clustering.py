#!/usr/bin/env python3
"""
clustering.py – Clustering & Evaluation on Pre-computed Embeddings (2026 updated version)

Compares 5 embedding strategies:
  1. name (title only)
  2. description only
  3. image only
  4. combined (simple concatenation – late fusion, no mixing)
  5. cross_embedding (true CLIP cross-modal – title+desc mixed with image in shared space)

Goal: clearly demonstrate that the cross_embedding is superior due to neural mixing.

Metrics (higher is better unless noted):
  • ARI                  – how well clusters recover true product_type (higher better)
  • Silhouette (true)    – separation & cohesion w.r.t. product_type (higher better)
  • Homogeneity / V-measure – clustering quality (higher better)
  • Intra Cohesion       – average tightness of each product_type group (LOWER better)

Visuals:
  • One PCA scatter plot per embedding type, colored by product_type
  • Saved in clustering_results/ folder

Uses number of unique product_types as n_clusters for fair comparison.
"""

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
)
import matplotlib.pyplot as plt


def load_embeddings(json_path: str = "embeddings.json") -> tuple[list[dict], dict[str, np.ndarray]]:
    """Load embeddings.json produced by the updated embeddings.py"""
    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    print(f"Loaded {len(items):,} items from {json_path}")

    # Extract the five embedding types
    name_emb = np.array([item["embeddings"]["name"] for item in items], dtype=np.float32)
    desc_emb = np.array([item["embeddings"]["description"] for item in items], dtype=np.float32)
    image_emb = np.array([item["embeddings"]["image"] for item in items], dtype=np.float32)
    combined_emb = np.array([item["embeddings"]["combined"] for item in items], dtype=np.float32)
    cross_emb = np.array([item["embeddings"]["cross_embedding"] for item in items], dtype=np.float32)

    modalities = {
        "name": name_emb,
        "description": desc_emb,
        "image": image_emb,
        "combined": combined_emb,
        "cross_embedding": cross_emb,
    }

    print("Embedding shapes loaded:")
    for name, emb in modalities.items():
        print(f"  • {name:15} : {emb.shape}")

    return items, modalities


def get_product_type_labels(items: list[dict]) -> tuple[np.ndarray, dict, int]:
    """Convert product_type to integer labels and return number of unique types."""
    product_types = [item.get("product_type", "Unknown") for item in items]
    le = LabelEncoder()
    true_labels = le.fit_transform(product_types)
    label_map = {i: name for i, name in enumerate(le.classes_)}
    n_types = len(le.classes_)
    print(f"Found {n_types} unique product_types → using as n_clusters for fair comparison")
    return true_labels, label_map, n_types


def compute_intra_cohesion(emb: np.ndarray, labels: np.ndarray) -> float:
    """Lower = better (tighter groups within each product_type)."""
    unique_labels = np.unique(labels)
    intra_scores = []
    for lbl in unique_labels:
        group = emb[labels == lbl]
        if len(group) < 2:
            continue
        centroid = np.mean(group, axis=0)
        dists = np.linalg.norm(group - centroid, axis=1)
        intra_scores.append(np.mean(dists))
    return float(np.mean(intra_scores)) if intra_scores else 0.0


def cluster_and_evaluate(
    emb: np.ndarray,
    modality_name: str,
    true_labels: np.ndarray,
    n_clusters: int,
) -> tuple[dict, np.ndarray]:
    """Run KMeans and compute all metrics for one modality."""
    print(f"\n→ Processing {modality_name} ({emb.shape[1]} dimensions, {n_clusters} clusters)")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(emb)

    # Metrics
    ari = adjusted_rand_score(true_labels, pred_labels)
    sil_true = silhouette_score(emb, true_labels, metric="cosine")
    hom = homogeneity_score(true_labels, pred_labels)
    comp = completeness_score(true_labels, pred_labels)
    vmeas = v_measure_score(true_labels, pred_labels)
    intra = compute_intra_cohesion(emb, true_labels)

    metrics = {
        "modality": modality_name,
        "dimensions": int(emb.shape[1]),
        "n_clusters": int(n_clusters),
        "ARI": float(ari),
        "Silhouette_true": float(sil_true),
        "Homogeneity": float(hom),
        "Completeness": float(comp),
        "V_measure": float(vmeas),
        "Intra_cohesion": float(intra),
    }

    print(f"   ARI               : {ari:.4f}")
    print(f"   Silhouette (true) : {sil_true:.4f}")
    print(f"   Intra cohesion    : {intra:.4f}  (lower = tighter groups)")
    print(f"   V-measure         : {vmeas:.4f}")

    return metrics, pred_labels


def visualize_pca(
    emb: np.ndarray,
    true_labels: np.ndarray,
    modality_name: str,
    output_dir: Path,
):
    """Create PCA 2D plot colored by product_type."""
    print(f"   Generating PCA visualization for {modality_name}...")

    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(emb)
    pca = PCA(n_components=2, random_state=42)
    emb_2d = pca.fit_transform(emb_scaled)

    explained = pca.explained_variance_ratio_.sum()
    print(f"   PCA explained variance: {explained:.1%}")

    plt.figure(figsize=(12, 9))
    scatter = plt.scatter(
        emb_2d[:, 0], emb_2d[:, 1],
        c=true_labels,
        cmap="tab20",
        alpha=0.75,
        s=45,
        edgecolor="k",
        linewidth=0.2,
    )

    plt.colorbar(scatter, label="Product Type ID", shrink=0.8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"{modality_name.replace('_', ' ').title()} Embedding\nColored by product_type")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_file = output_dir / f"{modality_name}_by_product_type.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"   Saved → {out_file.name}")


def save_results(items: list[dict], cross_labels: np.ndarray, output_path: str):
    """Save items with cross_embedding cluster label."""
    results = []
    for item, label in zip(items, cross_labels):
        result = item.copy()
        result["cross_cluster"] = int(label)
        # Optionally remove large embeddings to keep file smaller
        # result.pop("embeddings", None)
        results.append(result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nClustered results (using cross_embedding) saved to {output_path}")


def main():
    output_dir = Path("clustering_results")
    output_dir.mkdir(exist_ok=True)

    # Load data
    items, modalities = load_embeddings("embeddings.json")
    true_labels, label_map, n_clusters = get_product_type_labels(items)

    # Evaluate all 5 modalities
    all_metrics = []
    cross_labels = None

    for modality_name, emb in modalities.items():
        metrics, pred_labels = cluster_and_evaluate(
            emb, modality_name, true_labels, n_clusters
        )
        all_metrics.append(metrics)

        visualize_pca(emb, true_labels, modality_name, output_dir)

        if modality_name == "cross_embedding":
            cross_labels = pred_labels

    # Summary table
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics = df_metrics.set_index("modality")

    print("\n" + "="*90)
    print("FINAL METRICS COMPARISON (higher is better except Intra_cohesion)")
    print("="*90)
    print(df_metrics.round(4))

    # Save metrics
    metrics_path = output_dir / "embeddings_clustering_metrics.json"
    df_metrics.to_json(metrics_path, indent=2)
    print(f"\nFull metrics saved to {metrics_path}")

    # Save main results file
    save_results(items, cross_labels, str(output_dir / "embeddings_clustered_results.json"))

    print("\n" + "="*90)
    print("Clustering & evaluation complete!")
    print(f"→ All outputs are in the '{output_dir}' folder")
    print("\nExpected outcome:")
    print("   • cross_embedding should show highest ARI / Silhouette / V-measure")
    print("   • and lowest Intra_cohesion compared to name, description, image, and combined.")
    print("This demonstrates the advantage of true neural cross-modal mixing.")


if __name__ == "__main__":
    main()
