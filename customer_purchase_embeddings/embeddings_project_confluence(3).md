# Customer and Item Embeddings — Pilot Evaluation

## 1. Purpose

This page documents a pilot that represents **customers** and **purchased items** as embedding vectors, then asks a simple question:

> Does a person’s historical item mix sit closer, in embedding space, to a basket that person actually bought than it does to a basket bought by someone else?

If that gap is real — even if it is still modest — the same geometry can later absorb **card attributes** and **store context**. Those two additions are the immediate next steps of this project.

This page covers:

- Background on embeddings and why they fit this problem
- The representation and comparison approach
- What has been built vs what is planned
- A pilot evaluation on labeled same-customer vs different-customer pairs
- Whether the current numbers clear the go / no-go bar

---

## 2. Background on Embeddings

An **embedding** is a fixed-length numeric vector that stands in for a piece of text (or a structured record rendered as text). Items that mean similar things land near each other; unrelated items land farther apart.

Three properties matter for this pilot:

| Property | Why it matters here |
|---|---|
| Geometry | Cosine similarity (or its complement, cosine distance) is a single score we can compute between any two vectors of the same dimension. |
| Composition | Vectors can be added. A basket becomes the sum of its item vectors. The sum of item vectors represents a person. |
| Incomplete records | A missing field just drops out of the text. We do not need every customer to have the same columns populated. |

**Cosine similarity** is the dot product of two L2-normalized vectors. It runs from −1 to 1. Values near 1 mean the two texts point the same way. **Cosine distance** is `1 − cosine similarity`. High similarity and low distance are the same statement. The hypothesis and the evaluation on this page use **cosine distance** so that “closer to zero = better match” matches the histograms that were run.

**Lessons from previous runs:** if the input text is a JSON blob whose *keys* are identical on every row (`"zip_code"`, `"card_type"`, `"item_class"`, …), token-based models overweight those repeated keys and underweight the rare values we actually care about. Prefer a short **labeled text string** (values, with light field labels) over raw JSON. The pilot below still started from a JSON-shaped payload because that is what was assembled from the joins; tightening the text template is part of the next pass, not a rewrite of the pipeline.

---

## 3. Business Framing

We already know who bought what when the customer id is present. The useful claim is stronger than that:

1. A customer has a stable *taste vector* that can be estimated from the items they buy.
2. A new basket has a *content vector* that can be estimated from the items in that ticket.
3. Those two vectors should be closer when the basket belongs to that customer than when it does not.

If (3) holds even weakly, the same vectors become a feature for:

- Linking a ticket or card-present event to a customer when the join is incomplete
- Ranking “which of these customers is this basket more like”
- Downstream graph / identity work that needs a non-PII similarity signal
- Store- and card-conditioned versions of the same score (the next two steps)

It must be emphasized that we are **not** trying to replace the customer id. We are trying to aid in identifying customers based on their spending patterns, store affinity, and any other information we can collect.

---

## 4. Approach

### 4.1 Sources

Joined from the current-tenant credit card, transaction, and store tables, plus item master attributes. All person-level aggregates on this page are limited to **`idr_run_date = 2026-06-01`**.

### 4.2 Customer text

One document per customer-side record, assembled from:

- Zip code
- Expiration date
- Card type
- Online transaction (true / false)
- Name on card
- Address, city, state, postal code

Card fields are in the document so the same pipeline can accept card embeddings in the next step without a new join pattern. In the current cut they are present in the payload but **not yet a separate vector** added into the person or basket score.

### 4.3 Item text

One document per item, assembled from:

- Item description
- Item class, department, category
- Brand, vendor, manufacturing company, parent company
- Seasonality
- Full product description
- Category hierarchy description

### 4.4 Vectors

| Object | How the vector is built |
|---|---|
| Item | Embedding of the item text |
| Customer (person vector) | Sum of item embeddings for that customer, window ending at `idr_run_date = 2026-06-01` |
| Basket (ticket vector) | Sum of item embeddings for the items on that purchase |

Sums are **L2-normalized** before comparison so a 20-line ticket is not automatically “closer” to a heavy shopper than a 4-line ticket is. That normalization is necessary; without it, cosine largely tracks basket size.

The **purchase centroid** (basket sum) and the **person centroid** (customer sum) are computed from the items that belong to that basket or that person. They are **not** restricted by the basket-size filter in §6. That filter only decides which baskets are scored in the labeled comparison.

### 4.5 Comparison (Current Step)

**Step 1** — Calculate embeddings by person  
**Step 2** — Calculate embeddings by basket  
**Step 3** — Compare embeddings by basket purchase to person via cosine distance

**Hypothesis:** Embeddings by basket should have a **lower cosine distance** when scored against that person’s own embedding than when scored against someone else’s. Positive pairs should pile up toward zero. Negative pairs should share a middle band (common needs) and show a **longer tail** toward high distance.

Person-vs-basket is the cheapest test of whether item text alone carries identity-relevant signal. Card attributes and store weights can only help if that base geometry is not noise. They also cannot be interpreted if we skip the labeled same-id vs different-id check.

---

## 5. What has been done

- Joined credit card, transaction, and store data
- Built the customer-side JSON-shaped document listed above
- Built the item-side document from merchandising attributes
- Embedded customers and items
- Aggregated item embeddings to a person vector (`idr_run_date = 2026-06-01`)
- Aggregated item embeddings to a per-purchase basket vector
- Scored cosine **distance** of each basket against its owner’s person vector and against non-owner person vectors
- Reviewed the two score distributions (histogram / summary stats) against the go / no-go bars below

Not done yet, and explicitly in scope for the next iteration:

- A standalone **card embedding** added into the person and/or basket vector
- **Store weighting** (tickets from the customer’s usual stores should pull harder than an incidental store)
- Replacing leftover JSON-key text with labeled value strings
- Threshold selection for an operating point (precision/recall), not just distribution shift

---

## 6. Evaluation

The only labels we trust in this pilot are **known customer ids**. That is the right label for a “does this geometry work at all” test. It is not the label for production matching.

| Split | Definition |
|---|---|
| Positives | Basket *B* scored against the person vector of the customer who bought *B* |
| Negatives | Same basket scored against a person vector from a different customer |
| Evaluation filter | Person vector restricted to `idr_run_date = 2026-06-01`; comparison limited to baskets with **size ≥ 20**. This filter is **not** used to calculate the purchase centroid or the person centroid. |
| Normalization | Person and basket sums L2-normalized before cosine |
| Negatives per positive | 5, sampled from other customers in the slice |

**Overlap is expected.** Customers have common needs — milk, paper goods, seasonal staples, the same store assortment. Positive and negative histograms will share a middle band for that reason. A clean split of the two distributions is not the success criterion.

**What should differ is the tail.** On cosine **distance**, positives should be more isolated toward zero. The negative histogram should have the longer right tail (more mass at high distance). Many strangers will still look somewhat like each other because of shared needs; the *worst* stranger matches should be worse than the *worst* true-customer matches.

**Go / no-go used for this page**

| Criterion | Bar to keep going | Bar that would stop this path |
|---|---|---|
| Mean distance gap (negative − positive) | ≥ 0.08 | ≈ 0 after the same filters |
| ROC-AUC on pos vs neg (lower distance = better match) | ≥ 0.65 | ≤ 0.55 |
| KS statistic between the two histograms | ≥ 0.20 | ≤ 0.08 |
| Negative tail vs positive tail | Negatives have more mass at high distance than positives | Tails indistinguishable |

These bars are low on purpose. The current representation is incomplete (no card vector, no store weight). We only need evidence that item-sum geometry is not a dead end.

---

## 7. Results

Labeled same-customer vs different-customer cosine **distance**, after the §6 filter (basket size ≥ 20 for scoring only).

| | Same customer (positive) | Different customer (negative) |
|---|---|---|
| n pairs | 25,000 | 125,000 |
| Mean distance | 0.337 | 0.495 |
| Median distance | 0.320 | 0.483 |
| Std | 0.151 | 0.181 |
| P(distance < 0.20) | 19.5% | 3.9% |
| P(distance > 0.55) | 9.6% | 36.9% |

| Metric | Value | Go bar | Result |
|---|---|---|---|
| Mean distance gap (neg − pos) | 0.158 | ≥ 0.08 | **Go** |
| KS statistic | 0.3548 | ≥ 0.20 | **Go** |
| KS p-value | 0.00e+00 | — | Not used (N is large; use the statistic) |
| ROC-AUC (lower distance = better) | 0.7453 | ≥ 0.65 | **Go** |

The shape matches the thesis:

- Positives are more isolated toward zero (mean 0.337 / median 0.320; 19.5% below 0.20 vs 3.9% of negatives).
- The middle band still overlaps — expected, given common needs.
- The negative tail is longer (36.9% of different-customer pairs above 0.55 vs 9.6% of same-customer pairs).

---

## 8. Conclusion

The pilot data met the go expectations on every bar we set: mean distance gap 0.158, KS 0.35, ROC-AUC 0.75, and a longer high-distance tail on the negatives. Item-sum geometry is not a dead end. Proceed.

---

*Pilot comparison window: person vectors at `idr_run_date = 2026-06-01`. Evaluation filter: basket size ≥ 20 for scoring only; centroids are not computed under that filter. Sources: current-tenant credit card, transaction, store, and item attributes.*
