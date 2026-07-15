# Proposal: Embedding-Based Customer Identity Resolution (IDR) — Vector Storage Strategy

**Status:** Draft for review
**Owner:** [Your name]
**Audience:** Data Engineering / ML Platform
**Date:** 2026-07-15

## 1. Objective

Evaluate the use of embeddings (name-on-card, item title/description/photo, store address, and related identity signals) to improve customer identity resolution (IDR) — i.e., linking transactions, accounts, and interactions that refer to the same real-world customer even when identifiers are inconsistent, misspelled, or missing.

This document proposes a **two-tier, cost-conscious rollout** for where and how we store and query the resulting vectors, built entirely on our existing Databricks stack.

## 2. Guiding principles

- **No new infra spend until the method is proven.** Tier 1 uses only Delta Lake + Spark/Photon, which we already pay for.
- **No dead-end architecture.** Tier 1 must be structured so migration to Tier 2 is a data-copy operation, not a re-architecture.
- **Governance from day one.** Name-on-card and address are sensitive/PII fields — embeddings derived from them are quasi-PII too (they can leak the underlying string) and must live in Unity Catalog with the same access controls as the source data.

## 3. Data sources for embedding generation

| Signal | Modality | Candidate embedding model class | Notes |
|---|---|---|---|
| Name on credit card | Text (short string) | Text embedding (sentence-transformer style, or fine-tuned name-matching model) | Needs normalization (case, punctuation, nicknames) before embedding |
| Store item title/description | Text | Text embedding (general-purpose LLM embedding) | Also useful for purchase-affinity clustering, not just IDR |
| Store item photo | Image | CLIP-style image embedding | Only if visual similarity is expected to add IDR signal (e.g., matching receipts/photos) |
| Store address | Text (structured) | Text embedding on normalized/geocoded address string | Consider concatenating with geocode (lat/long) as auxiliary features, not just raw embedding |

Recommendation: keep **one embedding table per source entity type**, not one giant table — simplifies governance, backfills, and lets each tier scale independently.

## 4. Tiered plan

### Tier 1 — Delta Lake–native storage (no vector DB spend)

**Goal:** prove that embedding similarity meaningfully improves match rate / precision-recall over the existing IDR rules engine, at low cost and low commitment.

**Storage:**
- Store embeddings as a native array column in a Delta table:
```sql
  CREATE TABLE identity.embeddings.card_name_embeddings (
    entity_id       STRING,        -- FK to source record
    source_table    STRING,
    embedding       ARRAY<FLOAT>,  -- e.g. 384 or 768 dims
    model_name      STRING,
    model_version   STRING,
    created_at      TIMESTAMP
  )
  USING DELTA
  TBLPROPERTIES (delta.enableChangeDataFeed = true);
```
- Partition/Z-order by a coarse blocking key (e.g., first-letter-of-name, store region, zip prefix) so downstream similarity search doesn't require a full table scan — this is the cheap substitute for what a vector DB's index would do.
- Use Change Data Feed so Tier 2 migration later can pick up only new/changed rows.

**Similarity search options at this tier (roughly in order of effort):**
1. **Blocked brute-force dot product / cosine similarity in Spark SQL or Pandas UDF** — compute similarity only within a blocking key/partition (e.g., candidates sharing zip code or store region), not across the whole table. This is the standard "blocking + scoring" pattern used in classical entity resolution and keeps cost bounded.
2. **Spark MLlib `BucketedRandomProjectionLSH`** (or MinHash for the categorical parts) — gives you approximate nearest-neighbor search natively in Spark, no external service, and scales with the cluster you already have.
3. **FAISS index built in a notebook/job, persisted to DBFS/Volumes as an artifact** — for ad hoc, notebook-driven candidate generation during the research phase. Rebuild on a schedule (e.g., nightly job) rather than serving it live.

**What "proven" should mean before moving to Tier 2** (define explicit exit criteria up front):
- Match precision/recall vs. current rules-based IDR, measured on a labeled holdout set.
- Latency/throughput requirements for the *target* use case (batch nightly resolution vs. near-real-time lookup at point-of-sale/checkout) — this is the real driver of whether Tier 2 is needed.
- Estimated candidate volume growth (rows × query rate) that would make blocked brute-force too slow or too expensive in Spark compute hours.

**Cost profile:** marginal — Delta storage cost only, plus whatever Spark job time we already budget for pipelines. No new service, no new SKU.

**Limitations to flag to stakeholders now** (so Tier 1 isn't oversold):
- Not real-time; this is batch/near-batch similarity, fine for research and even for daily-batch IDR, not for sub-second lookups.
- ANN quality from LSH/FAISS-in-a-job is good but not as tunable/production-grade as a managed ANN index (HNSW).
- No built-in incremental index maintenance — recompute cadence is a job we own and schedule ourselves.

### Tier 2 — Managed vector search (Databricks-native)

**Goal:** once Tier 1 shows a positive signal and we have a concrete latency/scale requirement, move to a managed ANN index without leaving the Databricks/Unity Catalog boundary.

**Recommendation:** **Databricks Vector Search** (Unity Catalog–native; also marketed as part of Databricks AI Search). It's the natural next tier because:

- **Delta Sync Index**: the index is created from a Delta table, includes embedded data with metadata, and can be structured to automatically sync when the underlying Delta table is updated, with a pipeline execution mode of either triggered (refresh once per run) or continuous (index stays fresh as new rows land) — i.e., our Tier 1 embedding tables become the source table for the index with no data-movement pipeline to build ourselves. 
- Governance carries over: access control lists (ACLs) are used to manage the search endpoints, consistent with Unity Catalog permissions we'd already have on the source tables. 
- Flexible embedding management: you can use either Databricks-managed embeddings or self-managed embeddings, and for self-managed embeddings you specify the source table's text/vector column directly — so we're not locked into Databricks' own embedding models if we've already invested in a custom name-matching or CLIP model in Tier 1. 
- There's also a **Direct Vector Access Index** option that supports direct read and write of vectors and metadata via REST/SDK, for cases where we want to manage index updates ourselves rather than syncing from Delta — useful only if we need sub-second write-then-read (e.g., checkout-time resolution), otherwise Delta Sync is simpler and less code to maintain. 
- Query-time reranking is built in: a reranker config can rerank the top ~50 results using a Databricks reranking model, based on selected columns concatenated per result — potentially useful for combining name + address signals at query time rather than only at embedding time. 

**Migration path Tier 1 → Tier 2 (low-risk, incremental):**
1. Keep Tier 1 embedding Delta tables as the system of record — Tier 2 is additive, not a replacement.
2. Point a Delta Sync Index at the existing embedding table(s); no reprocessing of source data required.
3. Run both systems in parallel for a validation window; compare match quality and latency.
4. Cut over query traffic to the Vector Search endpoint once validated; keep Tier 1 blocked-brute-force path as a documented fallback / audit mechanism.

**Cost profile:** compute for a serving endpoint (pay for uptime/throughput of the endpoint) plus storage for the index — a real new line item, which is exactly why we gate it behind Tier 1 proof.

## 5. Summary comparison

| | Tier 1 (now) | Tier 2 (after proof) |
|---|---|---|
| Storage | Delta table, `ARRAY<FLOAT>` column | Same Delta table + synced Vector Search index |
| Similarity search | Blocked brute-force / Spark LSH / FAISS batch job | Managed ANN (HNSW-backed) via Delta Sync Index |
| Latency profile | Batch / near-batch | Near-real-time, query-endpoint driven |
| New cost | ~$0 incremental | Vector Search endpoint compute + index storage |
| Governance | Native Unity Catalog table ACLs | Unity Catalog + Vector Search endpoint ACLs |
| Effort to stand up | Low (SQL + a scheduled job) | Low (point Delta Sync Index at existing table) |
| Migration risk | N/A | Low — additive, table stays source of truth |

## 6. Open questions for engineering review

1. What's the target latency for IDR resolution in production — nightly batch, intra-day, or point-of-sale real-time? This is the single biggest driver of whether/when we need Tier 2.
2. Do we have (or need) a labeled ground-truth set of "known same customer" pairs to measure precision/recall during Tier 1?
3. Blocking key strategy for Tier 1 — zip/region, store ID, or something derived from existing rules-engine logic? This determines how well blocked brute-force scales.
4. Data retention/PII handling for embeddings derived from name-on-card and address — do these need the same masking/retention policy as the source PII columns?
5. Does image embedding (item photos) add enough IDR signal to justify the extra pipeline complexity in Tier 1, or should it be deferred to a later phase?

## 7. Recommendation

Proceed with **Tier 1 now** using Delta-native storage and blocked similarity search in Spark. Define explicit quantitative exit criteria (precision/recall lift, latency requirement, data volume) before greenlighting **Tier 2 (Databricks Vector Search)**, which can be layered on top of the same tables with minimal rework.
