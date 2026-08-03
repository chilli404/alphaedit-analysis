# Geometric Validation Report — Seed 2024

Generated: 2026-08-03 17:32 UTC

## Gate Criterion

**PASSED**: GATE PASSED: greedy_minmax frac>0.3 = 0.0000 < key_clustered = 0.4440 (improvement: 0.4440)

## First-1K Cohort Metrics

| Ordering | max_cos_mean | frac > 0.3 | frac > 0.4 | n_high_mean | within_batch_cos |
|----------|:------------:|:----------:|:----------:|:-----------:|:----------------:|
| cluster_topo     | 0.3283 | 0.5970 | 0.1450 | 3.9 | 0.1907 |
| greedy_minmax    | 0.2417 | 0.0000 | 0.0000 | 0.0 | 0.1142 |
| key_clustered    | 0.2987 | 0.4440 | 0.0580 | 2.3 | 0.1803 |
| key_dispersed    | 0.4230 | 0.8080 | 0.4350 | 20.7 | 0.1024 |
| random           | 0.4146 | 0.8080 | 0.4060 | 22.4 | 0.1021 |

## First-5K Cohort Metrics

| Ordering | max_cos_mean | frac > 0.3 | frac > 0.4 | n_high_mean |
|----------|:------------:|:----------:|:----------:|:-----------:|
| cluster_topo     | 0.2573 | 0.2644 | 0.0366 | 1.6 |
| greedy_minmax    | 0.2936 | 0.4880 | 0.0000 | 2.7 |
| key_clustered    | 0.3029 | 0.4594 | 0.0778 | 2.8 |
| key_dispersed    | 0.3683 | 0.6856 | 0.2862 | 10.5 |
| random           | 0.3942 | 0.7532 | 0.3562 | 12.2 |

## Age-Binned Max-Cos (first 5 cohorts)

| Ordering | 0-1K | 1K-2K | 2K-3K | 3K-4K | 4K-5K |
|----------|:----:|:-----:|:-----:|:-----:|:-----:|
| cluster_topo     | 0.328 | 0.303 | 0.315 | 0.266 | 0.280 |
| greedy_minmax    | 0.242 | 0.286 | 0.309 | 0.330 | 0.348 |
| key_clustered    | 0.299 | 0.306 | 0.358 | 0.321 | 0.327 |
| key_dispersed    | 0.423 | 0.424 | 0.416 | 0.389 | 0.376 |
| random           | 0.415 | 0.403 | 0.405 | 0.409 | 0.388 |

## Interpretation

The greedy_minmax scheduler successfully reduces first-1K exposure below
the key_clustered baseline. GPU runs are recommended.

Key observation: within-batch cosine should be SIMILAR across orderings
(scheduling manipulates cross-batch exposure, not within-batch similarity).