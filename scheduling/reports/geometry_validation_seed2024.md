# Geometric Validation Report — Seed 2024

Generated: 2026-08-04 17:49 UTC

## Gate Criterion

**PASSED**: GATE PASSED: greedy_minmax frac>0.3 = 0.0000 < key_clustered = 0.8060 (improvement: 0.8060)

## First-1K Cohort Metrics

| Ordering | max_cos_mean | frac > 0.3 | frac > 0.4 | n_high_mean | within_batch_cos |
|----------|:------------:|:----------:|:----------:|:-----------:|:----------------:|
| cluster_topo     | 0.3173 | 0.5410 | 0.1340 | 2.9 | 0.1813 |
| greedy_minmax    | 0.2421 | 0.0000 | 0.0000 | 0.0 | 0.1139 |
| key_clustered    | 0.4345 | 0.8060 | 0.4450 | 22.3 | 0.1023 |
| key_dispersed    | 0.4278 | 0.8110 | 0.4430 | 21.6 | 0.1024 |
| random           | 0.4217 | 0.8050 | 0.4170 | 21.2 | 0.1023 |

## First-5K Cohort Metrics

| Ordering | max_cos_mean | frac > 0.3 | frac > 0.4 | n_high_mean |
|----------|:------------:|:----------:|:----------:|:-----------:|
| cluster_topo     | 0.2437 | 0.1910 | 0.0196 | 0.5 |
| greedy_minmax    | 0.2944 | 0.4888 | 0.0000 | 2.9 |
| key_clustered    | 0.4018 | 0.7518 | 0.3674 | 12.1 |
| key_dispersed    | 0.4019 | 0.7486 | 0.3626 | 12.1 |
| random           | 0.4026 | 0.7556 | 0.3634 | 12.0 |

## Age-Binned Max-Cos (first 5 cohorts)

| Ordering | 0-1K | 1K-2K | 2K-3K | 3K-4K | 4K-5K |
|----------|:----:|:-----:|:-----:|:-----:|:-----:|
| cluster_topo     | 0.317 | 0.316 | 0.295 | 0.264 | 0.245 |
| greedy_minmax    | 0.242 | 0.287 | 0.310 | 0.330 | 0.351 |
| key_clustered    | 0.435 | 0.427 | 0.410 | 0.405 | 0.395 |
| key_dispersed    | 0.428 | 0.418 | 0.421 | 0.411 | 0.401 |
| random           | 0.422 | 0.427 | 0.419 | 0.415 | 0.400 |

## Interpretation

The greedy_minmax scheduler successfully reduces first-1K exposure below
the key_clustered baseline. GPU runs are recommended.

Key observation: within-batch cosine should be SIMILAR across orderings
(scheduling manipulates cross-batch exposure, not within-batch similarity).