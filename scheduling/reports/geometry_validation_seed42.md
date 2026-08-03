# Geometric Validation Report — Seed 42

Generated: 2026-08-03 17:28 UTC

## Gate Criterion

**PASSED**: GATE PASSED: greedy_minmax frac>0.3 = 0.0000 < key_clustered = 0.5157 (improvement: 0.5157)

## First-1K Cohort Metrics

| Ordering | max_cos_mean | frac > 0.3 | frac > 0.4 | n_high_mean | within_batch_cos |
|----------|:------------:|:----------:|:----------:|:-----------:|:----------------:|
| cluster_topo     | 0.2957 | 0.3714 | 0.1163 | 1.7 | 0.1868 |
| greedy_minmax    | 0.2328 | 0.0000 | 0.0000 | 0.0 | 0.1139 |
| key_clustered    | 0.3345 | 0.5157 | 0.2109 | 1.8 | 0.1786 |
| key_dispersed    | 0.4056 | 0.7835 | 0.3691 | 9.8 | 0.1021 |
| random           | 0.3934 | 0.7474 | 0.3437 | 10.5 | 0.1019 |

## First-5K Cohort Metrics

| Ordering | max_cos_mean | frac > 0.3 | frac > 0.4 | n_high_mean |
|----------|:------------:|:----------:|:----------:|:-----------:|
| cluster_topo     | 0.1258 | 0.1174 | 0.0132 | 0.4 |
| greedy_minmax    | 0.1415 | 0.1960 | 0.0000 | 0.6 |
| key_clustered    | 0.1356 | 0.1486 | 0.0184 | 0.6 |
| key_dispersed    | 0.1763 | 0.3060 | 0.1132 | 2.5 |
| random           | 0.1822 | 0.3302 | 0.1332 | 2.8 |

## Age-Binned Max-Cos (first 5 cohorts)

| Ordering | 0-1K | 1K-2K | 2K-3K | 3K-4K | 4K-5K |
|----------|:----:|:-----:|:-----:|:-----:|:-----:|
| cluster_topo     | 0.145 | 0.154 | 0.136 | 0.131 | 0.137 |
| greedy_minmax    | 0.118 | 0.139 | 0.144 | 0.158 | 0.172 |
| key_clustered    | 0.160 | 0.162 | 0.136 | 0.129 | 0.144 |
| key_dispersed    | 0.197 | 0.179 | 0.194 | 0.183 | 0.188 |
| random           | 0.190 | 0.190 | 0.189 | 0.179 | 0.191 |

## Interpretation

The greedy_minmax scheduler successfully reduces first-1K exposure below
the key_clustered baseline. GPU runs are recommended.

Key observation: within-batch cosine should be SIMILAR across orderings
(scheduling manipulates cross-batch exposure, not within-batch similarity).