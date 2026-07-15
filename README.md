# Multimodal Retail Embeddings — Experiment

Research code exploring whether **cross-modal fusion** (text + image embeddings mapped into a shared,
jointly-aligned space) produces better product representations than single-modality embeddings or naive
concatenation. This was built as a proof-of-concept with the extent on expanding for retail applications.

## Pipeline overview

```
preprocess.py   →   embeddings.py   →   clustering.py
(build dataset)     (generate 5           (evaluate + visualize
                     embedding variants     embedding quality)
                     per item)
```

1. **`preprocess.py`** — parses the raw [Amazon Berkeley Objects (ABO)](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) listings metadata, filters to English-language entries, joins each item to its main product image path, and writes a flat `abo_preprocessed.json`.
2. **`embeddings.py`** — loads `abo_preprocessed.json` and generates five embedding variants per item, saved to `embeddings.json`.
3. **`clustering.py`** — loads `embeddings.json`, clusters each variant with K-Means, scores each against ground-truth `product_type`, and saves metrics + PCA visualizations to `clustering_results/`.

## Repository structure

```
.
├── preprocess.py     # Step 1: build abo_preprocessed.json from raw ABO metadata
├── embeddings.py      # Step 2: generate 5 embedding variants -> embeddings.json
├── clustering.py      # Step 3: cluster + evaluate -> clustering_results/
├── abo_preprocessed.json   # (generated) flat item records
├── embeddings.json         # (generated) items + embeddings
└── clustering_results/     # (generated) metrics, PCA plots, clustered results
```

## Requirements

- Python 3.10+
- A CUDA-capable GPU is recommended for `embeddings.py` (falls back to CPU automatically)

```bash
pip install torch pillow numpy pandas scikit-learn matplotlib
pip install sentence-transformers
pip install git+https://github.com/openai/CLIP.git
```

## Data

This project uses the [ABO dataset](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) (listings metadata + small product images). Before running `preprocess.py`, download the dataset and update the hardcoded paths at the top of the file to match your environment:

```python
data_dir = "/home/daniel/mnt2/Data/abo-listings/listings/metadata"
img_base_dir = "/home/daniel/mnt2/Data/abo-images-small/images"
```

`preprocess.py` expects:
- `listings/metadata/*.json` — line-delimited JSON listing files
- `images/metadata/images.csv` — maps `image_id` → image file path
- `images/small/` — the actual image files

## Usage

### 1. Preprocess the raw dataset

```bash
python preprocess.py
```

Produces `abo_preprocessed.json`, a list of records with:

| Field | Description |
|---|---|
| `item_id` | ABO item identifier |
| `item_name` | Product title (English) |
| `product_type` | Category label used as ground truth for evaluation |
| `description` | Bullet points joined into a single string |
| `keywords` | List of item keywords |
| `image_location` | Local path to the main product image |

### 2. Generate embeddings

```bash
python embeddings.py
```

Produces `embeddings.json`. For each item, five embedding variants are generated:

| Variant | Model(s) | Description |
|---|---|---|
| `name` | Sentence-Transformer (`all-MiniLM-L6-v2`) | Title text only |
| `description` | Sentence-Transformer (`all-MiniLM-L6-v2`) | Description text only |
| `image` | CLIP ViT-B/32 (image encoder) | Product image only |
| `combined` | Sentence-Transformer + CLIP (concatenated) | Late fusion — title + description + image vectors concatenated, no cross-modal mixing |
| `cross_embedding` | CLIP ViT-B/32 (text + image encoders) | True cross-modal fusion — title and description combined into one prompt, encoded with CLIP's text encoder, and concatenated with the CLIP image embedding. Because CLIP was contrastively trained, text and image embeddings share a jointly-aligned space |

Each record in `embeddings.json` looks like:

```json
{
  "item_id": "...",
  "item_name": "...",
  "product_type": "...",
  "embeddings": {
    "name": [...],
    "description": [...],
    "image": [...],
    "combined": [...],
    "cross_embedding": [...]
  }
}
```

> **Note:** `main()` expects the input/output filenames `abo_preprocessed.json` / `embeddings.json` in the working directory — edit `items_json_path` / `output_path` in `embeddings.py` if you need different locations.

### 3. Cluster and evaluate

```bash
python clustering.py
```

For each of the five embedding variants, runs K-Means (with `k` = number of unique `product_type` values) and computes:

| Metric | Better when | What it measures |
|---|---|---|
| ARI (Adjusted Rand Index) | Higher | Agreement between predicted clusters and true `product_type` |
| Silhouette (true labels, cosine) | Higher | Separation/cohesion of true `product_type` groups in embedding space |
| Homogeneity | Higher | Each cluster contains only members of a single true class |
| Completeness | Higher | All members of a true class are assigned to the same cluster |
| V-measure | Higher | Harmonic mean of homogeneity and completeness |
| Intra-cohesion | Lower | Average distance of items to their `product_type` group centroid |

Outputs, all written to `clustering_results/`:

- `{modality}_by_product_type.png` — 2D PCA scatter plot per embedding variant, colored by `product_type`
- `embeddings_clustering_metrics.json` — full metrics table across all five variants
- `embeddings_clustered_results.json` — original items with an added `cross_cluster` label from the `cross_embedding` K-Means run

## Interpreting results

The hypothesis under test: `cross_embedding` should outperform `name`, `description`, `image`, and `combined` on ARI / silhouette / V-measure, and show the lowest intra-cohesion — evidence that jointly-aligned cross-modal fusion captures product identity better than single modalities or unaligned concatenation.

## Known limitations / next steps

- Ground truth (`product_type`) is a proxy for clustering quality — it validates the *fusion mechanism*, not the downstream identity-resolution task directly.
- `all-MiniLM-L6-v2` and CLIP ViT-B/32 were chosen for fast iteration; see the companion embedding-model proposal for stronger production candidates (e.g. Databricks-hosted text embedding models, SigLIP-2).
- `preprocess.py` paths are hardcoded to a local machine and should be parameterized (CLI args or a config file) before this becomes a shared/production pipeline.
- No train/test split — current evaluation clusters the full dataset; consider a held-out set for more robust metrics.
- Image loading failures fall back to a zero vector (`np.zeros(512)`), which will pull the `image` and `cross_embedding` centroids toward the origin for items with broken/missing images — worth flagging/filtering these out rather than silently zero-filling in a production version.

## License - MIT License

Copyright (c) 2026 Kovach Technologies

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
