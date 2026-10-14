# Customer and PAR Identity via Basket Embeddings — Results and Recommendations

## 1. Purpose

This page documents the results of comparing aggregated purchase embeddings against known identity keys (customer ID and Payment Account Reference / PAR). The goal is to determine whether basket-level cosine distance, alone or combined with exact item overlap (Jaccard index), can serve as a useful signal for linking identities when a direct key is missing or incomplete.

The evaluation covers two parallel experiments:

- Matching and non-matching **customer IDs**
- Matching and non-matching **PAR** IDs

Both experiments use the same underlying representation (Word2Vec item embeddings aggregated into person- or external-id-level vectors) so that differences in outcome can be attributed to the identity grain rather than to model choice.

---

## 2. Embeddings Approach

| Aspect | Detail |
|---|---|
| Model | Word2Vec with regex tokenization |
| Why this model | Faster training and simpler implementation; runs on a serverless Databricks instance. Alternatives such as Snowflake Arctic Embed require a provisioned server. |
| Input | Item descriptions (`DSC_ITEM`) tokenized via regex |
| Aggregation | Sum (or mean) of item vectors over a time window to produce a single vector per identity |

**Note on prior lessons:** Earlier experiments that embedded raw JSON payloads suffered from shared key tokens dominating the vectors. The current pipeline embeds the item description text itself, avoiding that collapse.

---

## 3. Experimental Design

### 3.1 Customer ID Experiment

- **Reference vector:** All purchases for a customer aggregated over a window starting 2026-08-31.
- **Query vectors:** Individual baskets containing 20 or more items.
- **Comparison:** Cosine distance between the reference vector and each basket vector.
- **Rationale for window length:** Longer windows improve stability of the reference vector but grow the dataset size rapidly; the chosen window was the practical maximum for the serverless environment.

**Expectations**

| Condition | Low Cosine Distance | High Cosine Distance |
|---|---|---|
| Matching customer ID | Expected | Less expected |
| Non-matching customer ID | Less expected | Expected |

### 3.2 PAR Experiment

- **Aggregation:** All purchases in a window starting 2026-10-06 grouped by `external_id`.
- **Selection:** Only PAR values that contain at least two distinct external IDs were retained.
- **Comparison:** Cosine distance between the aggregated vectors of two external IDs that share the same PAR versus two external IDs that do not.

**Expectations**

| Condition | Low Cosine Distance | High Cosine Distance |
|---|---|---|
| Matching PAR | Expected | Less expected |
| Non-matching PAR | Less expected | Expected |

---

## 4. Interpretation of Expectations

Matching identities (same customer ID or same PAR) are expected to produce lower cosine distances because the underlying purchasing behavior is more similar. Several real-world factors, however, can push same-identity pairs toward higher distance:

- A single PAR may cover cards used for different purposes (personal retail, business, gas, child/spouse).
- Seasonal or one-off events (birthday supplies, Halloween decorations, travel) can temporarily shift a customer’s basket away from their long-term vector.
- Many customers purchase the same everyday items (produce, staples, gas). This shared background creates incidental overlap even between unrelated identities and will be addressed by common-item blacklisting in later work.

These factors explain why the “less expected” cells in the tables above are not empty; they are simply less frequent than the expected cells.

---

## 5. Results

### 5.1 Customer ID Results

**Matching customer ID — low cosine distance (expected)**  
High item overlap was observed, frequently reaching 100% identical items between the long-term reference and the individual basket.  
`<screenshot>`

**Matching customer ID — high cosine distance (less expected)**  
Elevated cosine distance was driven by seasonal or special-occasion items (Halloween, birthday supplies). Removing those items would bring the pair closer to the expected low-distance regime.  
`<screenshot>`

**Non-matching customer ID — low cosine distance (less expected)**  
When low distance occurred, it was almost always explained by overlap on common everyday items rather than by a true identity match.  
`<screenshot>`

**Non-matching customer ID — high cosine distance (expected)**  
Disparate customer IDs generally produced higher cosine distances, with only minor residual overlap on common items.  
`<screenshot>`

### 5.2 PAR Results

**Matching PAR — low cosine distance (expected)**  
High degree of item overlap, often 100%. This is the strongest positive signal observed in the study.  
`<screenshot>`

**Matching PAR — high cosine distance (less expected)**  
Gas purchases and gift-card transactions stood out against ordinary retail baskets under the same PAR, producing higher distance.  
`<screenshot>`

**Non-matching PAR — low cosine distance (less expected)**  
Overlap was driven by common items (especially produce). Once those items are blacklisted, the residual distance is expected to rise.  
`<screenshot>`

**Non-matching PAR — high cosine distance (expected)**  
Frequently zero common items; the baskets were completely disparate.  
`<screenshot>`

---

## 6. Key Observation: Cosine Distance + Exact Overlap

Pairs that are true positives (low cosine distance and a confirmed identity match) exhibit **both**:

1. Low embedding cosine distance, and  
2. High exact item overlap measured by the Jaccard index (`|A ∩ B| / |A ∪ B|`).

False positives (low cosine distance but different identities) show low Jaccard, indicating that the embedding similarity is driven by a few shared common items rather than by broad basket alignment.

- `<screenshot of Jaccard index distribution for true positives>`
- `<screenshot of Jaccard index distribution for false positives>`

This dual-signal pattern is the strongest practical finding of the pilot.

---

## 7. Recommendation for Identity Linking (No Customer ID or PAR Given)

When a direct key is unavailable, apply the following thresholds:

| Signal | Threshold | Direction |
|---|---|---|
| Cosine distance | < 0.05 | Lower is better |
| Jaccard index | > 0.1 | Higher is better |

A candidate pair must satisfy **both** conditions. The combination substantially reduces false positives caused by common-item overlap while retaining the true matches observed in the customer-ID and PAR experiments.

---

## 8. Immediate Next Steps

1. **Probability metric** — Calibrate a combined score (cosine distance + Jaccard) and determine operating thresholds on a held-out set.
2. **Longer time windows** — Re-run both experiments with extended aggregation periods once larger compute is available.
3. **Common-item blacklist** — Remove high-frequency items (gas, produce, staples) before aggregation and re-evaluate both cosine distance and Jaccard.
4. **Transaction-level identity** — Ensure that each identity is represented by multiple distinct transactions rather than by items within a single basket, improving robustness of the reference vector.

---

## 9. Appendix: Data Windows and Selection Criteria

| Experiment | Aggregation Start | Selection Rule | Query Grain |
|---|---|---|---|
| Customer ID | 2026-08-31 | All customers with sufficient history | Baskets ≥ 20 items |
| PAR | 2026-10-06 | PAR values with ≥ 2 external IDs | Aggregated external-id vectors |

All cosine-distance extremes examined in the qualitative review were drawn from the tails of the respective distributions (highest and lowest 20 pairs).
