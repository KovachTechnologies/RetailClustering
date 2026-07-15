# Multimodal Embeddings for Retail Item Clustering:  
# The Theoretical Advantage of Cross-Modal Alignment over Unimodal and Late-Fusion Representations

## Abstract

In e-commerce and retail analytics, effective product representation is essential for tasks such as clustering, categorization, retrieval, and recommendation. Products are inherently multimodal, described through textual attributes (titles and descriptions) and visual content (images). While unimodal embeddings capture information from individual modalities and late-fusion approaches (simple concatenation) combine them post hoc, true cross-modal embeddings learned in a shared, aligned vector space enable richer semantic interactions.

This work theoretically examines why cross-modal representations—particularly those derived from contrastively pre-trained models such as CLIP—offer superior clustering performance compared to unimodal baselines and naive concatenation. We ground the analysis in the geometry of embedding spaces, the benefits of joint training objectives, and the mitigation of the modality gap. Empirical validation on the Amazon Berkeley Objects (ABO) dataset demonstrates that cross-modal embeddings produce tighter intra-category clusters and better recovery of ground-truth product types, highlighting the value of neural mixing across modalities.

**Keywords**: multimodal embeddings, cross-modal alignment, contrastive learning, product representation, retail clustering, shared embedding space

## 1. Introduction

Modern retail catalogs contain heterogeneous data: natural language titles and descriptions convey functional and categorical information, while images capture appearance, style, and contextual cues. Representing these modalities independently limits the model's ability to exploit their complementary nature. 

Two dominant strategies exist for multimodal integration:

- **Unimodal representations**: Separate embeddings for text (e.g., via SentenceTransformers) and images (e.g., via CLIP vision encoder).
- **Late fusion (concatenation)**: Independent unimodal embeddings are computed and then concatenated into a single vector. This approach performs "mixing" only at the downstream task level, with no interaction during feature extraction.

In contrast, **cross-modal embeddings** project different modalities into a **shared semantic space** during or after joint training. Models like CLIP achieve this through large-scale contrastive learning on image-text pairs, encouraging semantically related content (regardless of modality) to lie close together in the embedding space.

This paper focuses on the **theoretical foundations** underlying the superiority of cross-modal alignment for downstream clustering tasks in retail. We argue that the dense neural mixing enabled by shared-space training yields more semantically coherent representations than either isolated unimodal features or post-hoc concatenation.

## 2. Theoretical Background

### 2.1 Embedding Spaces and Semantic Geometry

Embeddings map discrete data (text tokens, image pixels) into continuous vector spaces where geometric proximity reflects semantic similarity. In a well-structured embedding space, cosine similarity or Euclidean distance becomes a proxy for conceptual relatedness.

For multimodal data, the challenge is bridging the **modality gap**—the inherent distributional differences between text and visual feature spaces. Unimodal models operate in separate spaces, making direct comparison impossible without additional projection layers. Late fusion via concatenation preserves these separate subspaces within a higher-dimensional vector but does not enforce alignment or interaction between them during representation learning.

### 2.2 Contrastive Learning and Shared Embedding Spaces

Contrastive pre-training, as popularized by CLIP (Radford et al., 2021), optimizes a symmetric cross-entropy loss over large batches of image-text pairs. Matched pairs are pulled closer in the joint space, while mismatched pairs are pushed apart. This process induces several desirable properties:

- **Semantic alignment**: Text and image embeddings of the same product become neighboring points, even though they originate from different sensory domains.
- **Emergent compositionality**: Because the model learns a single shared space, information from one modality can "fill in" or disambiguate the other. For instance, a vague title benefits from visual context, and an ambiguous image gains categorical precision from textual description.
- **Zero-shot transferability**: Concepts learned jointly generalize across modalities without task-specific fine-tuning.

In theoretical terms, contrastive objectives maximize mutual information between modalities while minimizing intra-modal redundancy, leading to a more compact and disentangled latent manifold.

### 2.3 Limitations of Late Fusion (Concatenation)

Simple concatenation treats modalities as independent feature sets. While computationally straightforward, it suffers from several theoretical drawbacks:

- **Lack of cross-modal interaction**: No mechanism exists for features from one modality to influence the representation of another during encoding. Textual semantics and visual attributes remain isolated until the final vector is formed.
- **Curse of dimensionality**: Concatenated vectors grow linearly with the sum of individual dimensions, potentially introducing noise and requiring stronger downstream regularization.
- **Suboptimal geometry**: The resulting space does not guarantee that semantically equivalent products cluster tightly when their dominant cues appear in different modalities. Intra-category variance may remain high if one modality is noisy or incomplete.

In contrast, true cross-modal fusion (e.g., feeding a combined text prompt into CLIP's text encoder alongside the image encoder) allows gradients to flow across modalities during pre-training, enabling **dense neural mixing**. This produces embeddings where title, description, and image information are not merely juxtaposed but semantically entangled in a unified representation.

### 2.4 Expected Benefits for Clustering

Clustering quality depends on both **intra-cluster compactness** (low variance within true categories) and **inter-cluster separation** (high distance between different categories).

Cross-modal embeddings are theoretically expected to excel because:

1. **Tighter intra-category cohesion**: Complementary signals reduce representational ambiguity. A product's visual style reinforces its textual category, pulling all instances of the same `product_type` closer to a shared centroid.
2. **Better separation**: The shared space encodes higher-level semantics (function, usage context, material properties) that distinguish fine-grained categories more effectively than any single modality.
3. **Robustness to missing or noisy data**: When one modality is weak, the aligned space allows the other to compensate implicitly through the learned joint distribution.

These advantages manifest in standard clustering metrics such as Adjusted Rand Index (ARI) against ground-truth labels, silhouette scores, and mean intra-group distance to centroids.

## 3. Methodological Framework

We evaluate five representation strategies on the Amazon Berkeley Objects (ABO) dataset:

- **Name-only** and **Description-only**: Unimodal text embeddings using a SentenceTransformer model.
- **Image-only**: Unimodal visual embeddings from CLIP's vision encoder.
- **Combined**: Late-fusion concatenation of the three unimodal embeddings (no cross-modal interaction during encoding).
- **Cross-embedding**: True cross-modal representation obtained by encoding a structured text prompt (title + description) with CLIP's text encoder and combining it with the corresponding image embedding. This leverages CLIP's pre-trained alignment for dense multimodal mixing.

All embeddings are L2-normalized. Clustering is performed with KMeans using the number of unique `product_type` categories as the target number of clusters (providing a fair, supervised-like evaluation). Evaluation uses ARI, silhouette score (with respect to true product types), V-measure, and average intra-product-type cohesion (mean distance to group centroid).

## 4. Theoretical Implications and Discussion

The empirical separation observed between late-fusion ("combined") and true cross-modal ("cross_embedding") representations underscores a fundamental principle in multimodal learning: **alignment during representation learning outperforms post-hoc fusion**.

By projecting text and images into a shared space trained contrastively, the model learns invariant semantic features that transcend surface-level modality differences. This results in a lower-dimensional effective manifold where retail concepts (e.g., "durable filament for 3D printing") are more compactly encoded.

In retail applications, such representations enable more robust downstream tasks: improved product deduplication, category discovery, visual search, and cold-start recommendations. The approach also scales naturally to additional modalities (video, tabular attributes) provided they can be aligned in the same contrastive framework.

Limitations of the current analysis include reliance on off-the-shelf pre-trained models without domain-specific fine-tuning and the use of coarse `product_type` as ground truth. Future work could explore learned fusion layers, hierarchical product taxonomies, or contrastive objectives tailored to fine-grained retail semantics.

## 5. Conclusion

This study provides a theoretical and empirical case for preferring cross-modal aligned embeddings over unimodal or simple late-fusion strategies in retail product representation. The dense neural mixing afforded by shared embedding spaces—induced through contrastive pre-training—yields semantically richer representations that better capture the multifaceted nature of commercial items.

By demonstrating clearer cluster structure and superior recovery of product categories, we highlight the practical value of investing in true multimodal alignment for e-commerce intelligence. These insights contribute to the broader understanding of when and why joint multimodal representations outperform modular alternatives.

## References

- Radford, A., et al. (2021). Learning transferable visual models from natural language supervision. *ICML*.
- Relevant literature on multimodal fusion in e-commerce (e.g., studies using the ABO dataset for product similarity and categorization).
- Recent work on unified multimodal embedding models emphasizing shared vector spaces for cross-modal retrieval.

---

*This document focuses exclusively on the theoretical motivations and conceptual framework. Implementation details, code structure, dataset preparation, and experimental results are provided in `README.md` and supporting scripts.*
