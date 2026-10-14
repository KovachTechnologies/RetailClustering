# Embeddings Pilot: Customer ID and PAR Comparisons

## Embeddings

- **Method**: Word2Vec embeddings with regex tokenization.
- **Reasoning**: Faster training and simpler implementation. Can run on a serverless Databricks instance, whereas alternatives (e.g. Snowflake Arctic Embed) require a provisioned server instance.

## Testing Scope

Aggregated customer purchases were compared in two ways:

1. Against **customer ID**
2. Against **PAR** (Payment Account Reference) IDs

### Customer IDs

- A longer-term window of purchases (from 2026-08-31) was aggregated into a single embedding vector per customer.
- The comparatively short “long-term” window was chosen because datasets become untenable quickly at larger scales.
- Individual basket purchases (of 20 items or more) were compared against this reference vector via cosine distance.

**Expectations**

| | Low Cosine Distance | High Cosine Distance |
|---|---|---|
| Matching customer ID | Expected | Less expected |
| Non-matching customer ID | Less expected | Expected |

### PAR IDs

- All purchases over a given window (from 2026-10-06) were aggregated by `external_id`.
- Because PAR IDs contain multiple external IDs, we first selected PAR IDs with at least two external IDs, then compared the aggregated purchases for matching vs. non-matching PAR.

**Expectations**

| | Low Cosine Distance | High Cosine Distance |
|---|---|---|
| Matching PAR | Expected | Less expected |
| Non-matching PAR | Less expected | Expected |

## Comments on Expectations

- In general, matching identities (same customer ID or same PAR) should produce lower cosine distances.
- However, a common PAR can still show higher cosine distance when the same person uses different cards under that PAR for different purposes (e.g. business, child/spouse, gas).
- Differing purchases under the same customer ID can arise from extraneous or seasonal events (birthday supplies, Halloween decorations, etc.).
- Common items are purchased by many customers regardless of identity, so some overlap is expected. Filtering of common items is planned for later work.

## Results

### Customer IDs

- **Common customer ID – low cosine distance (expected)**
  - High overlap observed; often the exact same items were purchased.
  - `<screenshot>`

- **Common customer ID – high cosine distance (less expected)**
  - Higher cosine distance driven by seasonal / special-case items (e.g. Halloween, birthday).
  - `<screenshot>`

- **Differing customer ID – low cosine distance (less expected)**
  - Lower cosine distance observed when common items matched.
  - `<screenshot>`

- **Differing customer ID – high cosine distance (expected)**
  - Generally higher cosine distance among disparate customer IDs, with only minor overlap on common items.
  - `<screenshot>`

### PAR

- **Common PAR – low cosine distance (expected)**
  - High degree of overlap in items (often 100%).
  - `<screenshot>`

- **Common PAR – high cosine distance (less expected)**
  - Gas and gift-card type purchases stood out against retail purchases.
  - `<screenshot>`

- **Differing PAR – low cosine distance (less expected)**
  - Common items contributed to overlap (e.g. produce).
  - `<screenshot>`

- **Differing PAR – high cosine distance (expected)**
  - Often completely disparate baskets observed (zero common items).
  - `<screenshot>`

## Key Observation

“True positives” (low cosine distance with a definite match) showed not only a low embedding cosine distance but also a high exact overlap (Jaccard index).

- `<screenshot of Jaccard index (true positive)>`
- `<screenshot of Jaccard index (false positive)>`

## Overall Recommendation

For determining a match when customer ID or PAR is **not** given:

- Select for a “low” cosine distance: **< 0.05**
- Select for a “high” Jaccard index: **> 0.1**

## Immediate Next Steps

- Determine parameters for a probability metric combining low cosine distance + Jaccard index.
- Use longer time frames.
- Implement blacklisting of common items (gas, produce, etc.).
- Ensure multiple *transactions* per identity, not just items in a single basket.
