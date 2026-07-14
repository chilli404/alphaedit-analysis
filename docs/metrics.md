# Metrics & Statistical Framework

## Standard Editing Metrics

| Metric | What it measures |
|--------|-----------------|
| Efficacy | Edit succeeds on the exact prompt |
| Generalization | Edit succeeds on paraphrases |
| Specificity | Unrelated facts remain correct |
| Fluency | Output remains coherent |
| Consistency | Open-ended generation aligns with the edit |

## Mechanistic Metrics (Novel)

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Projection loss** | `1 - ‖P @ K @ resid^T‖ / ‖K @ resid^T‖` | Fraction of desired edit stripped by the null-space constraint. **The central measurement.** |
| Consumption ratio | `rank(cache_c) / rank(P)` per layer | How much null-space remains — predicts capacity failure |
| Perplexity | `exp(total_NLL / total_tokens)` on WikiText-103 | Global capability preservation |

## Statistical Framework

- Core reproduction (MVE1-4): 5 seeds (42, 137, 2024, 7, 99)
- Extensions: 3 seeds (42, 137, 2024)
- BCa bootstrap CIs (10,000 resamples)
- Holm-Bonferroni correction for multiple comparisons
- Effect sizes: Cohen's d, Cliff's delta
- Paired bootstrap test for AlphaEdit vs MEMIT
