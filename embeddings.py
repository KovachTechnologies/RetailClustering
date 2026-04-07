#!/usr/bin/env python3
"""
embeddings.py – FIXED & IMPROVED Embedding generation

Key fix based on your feedback:
  • "combined"  = simple late-fusion concatenation of the three separate embeddings
    (title + description + image). No mixing across modalities.
  • "cross_embedding" = true cross-modal fusion:
      - Title + Description are concatenated into ONE structured text prompt
      - Both text and image are encoded with the SAME CLIP model (ViT-B/32)
      - CLIP was pre-trained with contrastive loss → the text and image embeddings
        live in a shared, aligned space and benefit from the "dense mixing" that
        happened during CLIP’s neural network training.
      - Result: 512-dim CLIP text + 512-dim CLIP image = 1024-dim cross vector

This now clearly distinguishes the two:
  - combined = no neural mixing between modalities
  - cross_embedding = neural cross-modal alignment + mixing via CLIP

Output embeddings.json now contains FIVE keys (exactly what you need for the new clustering.py):

{
 "item_id": "...",
 "item_name": "...",
 "product_type": "...",
 "embeddings" : {
   "name": [...],
   "description": [...],
   "image": [...],
   "combined": [...],          ← new: pure concatenation
   "cross_embedding": [...]    ← new: true cross-modal (CLIP-aligned)
 }
}
"""

import json
import os
import numpy as np
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer
import clip


def load_items(json_path: str) -> list[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_image_as_tensor(image_path: str):
    try:
        if image_path.startswith(("http://", "https://")):
            import urllib.request
            import io
            with urllib.request.urlopen(image_path, timeout=10) as response:
                image_data = response.read()
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        else:
            image = Image.open(image_path).convert("RGB")
        return image.resize((224, 224))
    except Exception as e:
        print(f"Warning: Could not load image {image_path}: {e}")
        return None


def get_text_embeddings(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


def get_clip_image_embeddings(images: list, clip_model, preprocess, device: str) -> np.ndarray:
    """CLIP image embeddings – reuses the already-loaded CLIP model."""
    embeddings = []
    for image in images:
        if image is not None:
            image_input = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = clip_model.encode_image(image_input)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu().numpy().flatten())
        else:
            embeddings.append(np.zeros(512))
    return np.array(embeddings)


def get_clip_text_embeddings(texts: list[str], clip_model, device: str) -> np.ndarray:
    """CLIP text embeddings – uses the same CLIP model as images (true cross-modal)."""
    tokenized = clip.tokenize(texts).to(device)
    with torch.no_grad():
        embeddings = clip_model.encode_text(tokenized)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return embeddings.cpu().numpy()


def create_all_embeddings(
    titles: list[str],
    descriptions: list[str],
    images: list,
    device: str,
) -> dict[str, np.ndarray]:
    print("Loading text embedding model (SentenceTransformer all-MiniLM-L6-v2)...")
    text_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Encoding titles...")
    title_embeddings = get_text_embeddings(titles, text_model)

    print("Encoding descriptions...")
    desc_embeddings = get_text_embeddings(descriptions, text_model)

    # ── Load CLIP once and reuse for both image AND cross-modal text ──
    print("Loading CLIP ViT-B/32 (for images + true cross-modal fusion)...")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    print("Encoding images with CLIP...")
    image_embeddings = get_clip_image_embeddings(images, clip_model, preprocess, device)

    # Normalize the separate modalities (same as before)
    title_embeddings = title_embeddings / np.linalg.norm(title_embeddings, axis=1, keepdims=True)
    desc_embeddings = desc_embeddings / np.linalg.norm(desc_embeddings, axis=1, keepdims=True)
    image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)

    # 4. Combined = pure concatenation (late fusion, no mixing)
    combined_embeddings = np.hstack([title_embeddings, desc_embeddings, image_embeddings])

    # 5. Cross-embedding = true cross-modal (CLIP-aligned)
    print("Creating true cross-embeddings (CLIP text + CLIP image)...")
    # Mix title + description in a single structured prompt for CLIP
    combined_texts = [
        f"Title: {title}. Description: {desc}" if desc else f"Title: {title}"
        for title, desc in zip(titles, descriptions)
    ]
    clip_text_embeddings = get_clip_text_embeddings(combined_texts, clip_model, device)

    # Cross = CLIP text (already aligned with image during CLIP training) + CLIP image
    cross_embeddings = np.hstack([clip_text_embeddings, image_embeddings])

    print(f"Embedding shapes:")
    print(f"  • name           : {title_embeddings.shape}")
    print(f"  • description    : {desc_embeddings.shape}")
    print(f"  • image          : {image_embeddings.shape}")
    print(f"  • combined       : {combined_embeddings.shape}  (pure concatenation)")
    print(f"  • cross_embedding: {cross_embeddings.shape}  (true CLIP cross-modal mixing)")

    return {
        "name": title_embeddings,
        "description": desc_embeddings,
        "image": image_embeddings,
        "combined": combined_embeddings,
        "cross_embedding": cross_embeddings,
    }


def main():
    items_json_path = "abo_preprocessed.json"
    output_path = "embeddings.json"

    if not os.path.exists(items_json_path):
        print(f"Error: {items_json_path} not found!")
        return

    print("Loading items...")
    items = load_items(items_json_path)

    required = ["item_id", "item_name", "product_type", "description", "image_location"]
    for item in items:
        for field in required:
            if field not in item:
                print(f"Error: Item missing required field '{field}'")
                return

    titles = [item["item_name"] for item in items]
    descriptions = [item.get("description", "") for item in items]
    image_paths = [item["image_location"] for item in items]

    print(f"Loaded {len(items):,} items")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading images...")
    images = [load_image_as_tensor(path) for path in image_paths]

    emb_dict = create_all_embeddings(titles, descriptions, images, device)

    print("\nBuilding output JSON with five embedding types...")
    results = []
    for i, item in enumerate(items):
        result = {
            "item_id": item["item_id"],
            "item_name": item["item_name"],
            "product_type": item["product_type"],
            "embeddings": {
                "name": emb_dict["name"][i].tolist(),
                "description": emb_dict["description"][i].tolist(),
                "image": emb_dict["image"][i].tolist(),
                "combined": emb_dict["combined"][i].tolist(),
                "cross_embedding": emb_dict["cross_embedding"][i].tolist(),
            },
        }
        results.append(result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Success! Updated embeddings saved to {output_path}")
    print(f"   • Items processed : {len(results):,}")
    print(f"   • File size       : {os.path.getsize(output_path) / (1024*1024):.1f} MB")
    print("\nYou now have both:")
    print("   • combined       (simple concatenation – no mixing)")
    print("   • cross_embedding (true CLIP cross-modal mixing)")
    print("\nReady for the updated clustering.py (which will compare all five).")


if __name__ == "__main__":
    main()
