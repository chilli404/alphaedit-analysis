# Scheduling Forensics Report

Seeds: [42, 2024]

## 1. Collapse Timeline (first_1k retention)

| Algorithm | Ordering | Seed | 1K | 2K | 3K | 4K | 5K | 6K | 7K | 8K | 9K | 10K | Collapse Onset |
|-----------|----------|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| AE | cluster_topo | 42 | 0.882 | 0.853 | 0.788 | 0.764 | 0.724 | 0.679 | 0.631 | 0.486 | **0.206** | **0.109** | **6K** |
| AE | greedy_minmax | 42 | 0.988 | 0.961 | 0.916 | 0.806 | 0.536 | **0.112** | **0.050** | **0.019** | **0.015** | **0.020** | **5K** |
| AE | greedy_minmax | 2024 | 0.985 | 0.958 | 0.895 | 0.647 | **0.233** | **0.092** | **0.056** | **0.018** | **0.011** | **0.013** | **4K** |
| AE | key_clustered | 42 | 0.948 | 0.933 | 0.907 | 0.827 | 0.801 | 0.762 | 0.733 | 0.640 | 0.584 | 0.551 | **8K** |
| AE | key_clustered | 2024 | 0.979 | 0.964 | 0.946 | 0.923 | 0.875 | 0.835 | 0.792 | 0.742 | 0.690 | 0.647 | **9K** |
| AE | key_dispersed | 42 | 0.957 | 0.906 | 0.852 | 0.778 | 0.710 | 0.612 | 0.479 | 0.333 | **0.221** | **0.148** | **6K** |
| AE | key_dispersed | 2024 | 0.963 | 0.910 | 0.835 | 0.729 | 0.599 | 0.486 | **0.297** | **0.100** | **0.059** | **0.031** | **5K** |
| AE | random | 42 | 0.972 | 0.934 | 0.877 | 0.790 | 0.711 | 0.623 | 0.522 | 0.385 | **0.262** | **0.137** | **6K** |
| M-Seq | greedy_minmax | 42 | 0.997 | 0.985 | 0.975 | 0.965 | 0.939 | 0.902 | 0.780 | **0.154** | **0.043** | **0.040** | **8K** |
| M-Seq | key_clustered | 42 | 0.977 | 0.980 | 0.971 | 0.955 | 0.952 | 0.941 | 0.918 | 0.917 | 0.900 | 0.878 | None |
| M-Seq | key_clustered | 2024 | 0.991 | 0.988 | 0.986 | 0.979 | 0.973 | 0.960 | 0.945 | 0.924 | 0.904 | 0.889 | None |
| M-Seq | key_dispersed | 42 | 0.981 | 0.967 | 0.949 | 0.924 | 0.905 | 0.869 | 0.835 | 0.800 | 0.760 | 0.723 | None |
| M-Seq | key_dispersed | 2024 | 0.976 | 0.953 | 0.940 | 0.922 | 0.898 | 0.872 | 0.825 | 0.785 | 0.739 | 0.673 | **10K** |

## 2. Installation Quality (latest_100 efficacy)

| Algorithm | Ordering | Seed | 1K | 2K | 3K | 4K | 5K | 6K | 7K | 8K | 9K | 10K |
|-----------|----------|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| AE | cluster_topo | 42 | 1.000 | 0.960 | 1.000 | 1.000 | 0.990 | 1.000 | 1.000 | 1.000 | 0.970 | 0.950 |
| AE | greedy_minmax | 42 | 0.990 | 1.000 | 1.000 | 1.000 | 1.000 | **0.780** | **0.740** | **0.600** | **0.710** | **0.160** |
| AE | greedy_minmax | 2024 | 1.000 | 1.000 | 1.000 | 0.990 | 0.950 | 0.880 | **0.760** | **0.490** | **0.670** | **0.220** |
| AE | key_clustered | 42 | 0.980 | 0.990 | 0.970 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.800 |
| AE | key_clustered | 2024 | 0.990 | 1.000 | 0.990 | 0.950 | 0.990 | 1.000 | 1.000 | 0.980 | 0.910 | 0.960 |
| AE | key_dispersed | 42 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.990 | 1.000 | 0.980 |
| AE | key_dispersed | 2024 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.980 | 0.870 | 0.880 | **0.730** |
| AE | random | 42 | 0.980 | 1.000 | 1.000 | 1.000 | 1.000 | 0.990 | 0.990 | 0.980 | 0.980 | 0.980 |
| M-Seq | greedy_minmax | 42 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.970 | 0.870 | 0.950 | **0.440** |
| M-Seq | key_clustered | 42 | 0.990 | 0.970 | 0.990 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.910 |
| M-Seq | key_clustered | 2024 | 0.990 | 1.000 | 1.000 | 0.950 | 1.000 | 1.000 | 1.000 | 0.990 | 0.930 | 0.950 |
| M-Seq | key_dispersed | 42 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| M-Seq | key_dispersed | 2024 | 1.000 | 0.990 | 1.000 | 1.000 | 1.000 | 1.000 | 0.990 | 1.000 | 0.990 | 1.000 |

## 3. Geometric Exposure (Complete)

| Ordering | Seed | frac>0.3 | frac>0.4 | mean_max_cos | within_batch_cos |
|----------|------|:--------:|:--------:|:------------:|:----------------:|
| key_clustered | 42 | 0.4150 | 0.0550 | 0.3030 | 0.2197 |
| key_clustered | 2024 | 0.1980 | 0.0200 | 0.2616 | 0.1578 |
| key_dispersed | 42 | 0.7550 | 0.3380 | 0.3887 | 0.0969 |
| key_dispersed | 2024 | 0.5060 | 0.1810 | 0.3265 | 0.0956 |
| greedy_minmax | 42 | 0.0250 | 0.0000 | 0.2521 | 0.0936 |
| greedy_minmax | 2024 | 0.2270 | 0.0000 | 0.2617 | 0.0924 |
| cluster_topo | 42 | 0.3810 | 0.0570 | 0.2899 | 0.1958 |
| cluster_topo | 2024 | 0.1110 | 0.0090 | 0.2257 | 0.2259 |
| random | 42 | 0.7430 | 0.3300 | 0.3855 | 0.1039 |
| random | 2024 | 0.5810 | 0.2190 | 0.3446 | 0.1022 |

## 4. Per-Batch Conditioning Profile

### 4a. Within-Batch Cosine
| Ordering | Seed | Early (0-10) | Mid (40-50) | Late (90-99) |
|----------|------|:------------:|:-----------:|:------------:|
| key_clustered | 42 | 0.2197 | 0.1749 | 0.1821 |
| key_clustered | 2024 | 0.1578 | 0.1952 | 0.2004 |
| key_dispersed | 42 | 0.0969 | 0.1014 | 0.1127 |
| key_dispersed | 2024 | 0.0956 | 0.1025 | 0.1089 |
| greedy_minmax | 42 | 0.0936 | 0.1132 | 0.1319 |
| greedy_minmax | 2024 | 0.0924 | 0.1169 | 0.1263 |
| cluster_topo | 42 | 0.1958 | 0.2081 | 0.1700 |
| cluster_topo | 2024 | 0.2259 | 0.1595 | 0.1820 |
| random | 42 | 0.1039 | 0.1020 | 0.1011 |
| random | 2024 | 0.1022 | 0.1051 | 0.1021 |

### 4b. Batch K@K^T Condition Number (cosine-normalized)
| Ordering | Seed | Early (0-10) | Mid (40-50) | Late (90-99) |
|----------|------|:------------:|:-----------:|:------------:|
| key_clustered | 42 | 48.8 | 48.2 | 39.9 |
| key_clustered | 2024 | 6.8 | 11.3 | 34.1 |
| key_dispersed | 42 | 11.1 | 11.2 | 17.6 |
| key_dispersed | 2024 | 4.8 | 5.0 | 5.8 |
| greedy_minmax | 42 | 8.4 | 12.2 | 71.2 |
| greedy_minmax | 2024 | 4.4 | 5.9 | 24.5 |
| cluster_topo | 42 | 126.2 | 35.3 | 18.4 |
| cluster_topo | 2024 | 47.0 | 9.0 | 10.0 |
| random | 42 | 13.3 | 14.5 | 13.0 |
| random | 2024 | 5.9 | 5.3 | 5.2 |

### 4b'. Batch K@K^T Condition Number (unnormalized — reflects actual solve)
| Ordering | Seed | Early (0-10) | Mid (40-50) | Late (90-99) |
|----------|------|:------------:|:-----------:|:------------:|
| key_clustered | 42 | 49.6 | 50.7 | 41.2 |
| key_clustered | 2024 | 7.8 | 12.4 | 35.1 |
| key_dispersed | 42 | 12.0 | 12.6 | 17.6 |
| key_dispersed | 2024 | 5.5 | 6.0 | 7.2 |
| greedy_minmax | 42 | 9.3 | 14.5 | 74.4 |
| greedy_minmax | 2024 | 4.9 | 6.6 | 27.0 |
| cluster_topo | 42 | 120.8 | 37.8 | 20.3 |
| cluster_topo | 2024 | 47.4 | 9.8 | 10.8 |
| random | 42 | 15.0 | 17.7 | 14.1 |
| random | 2024 | 6.4 | 6.3 | 6.4 |

### 4c. Batch Effective Rank
| Ordering | Seed | Early (0-10) | Mid (40-50) | Late (90-99) |
|----------|------|:------------:|:-----------:|:------------:|
| key_clustered | 42 | 44.4 | 47.7 | 47.5 |
| key_clustered | 2024 | 22.0 | 22.8 | 22.2 |
| key_dispersed | 42 | 47.3 | 51.9 | 46.8 |
| key_dispersed | 2024 | 23.8 | 21.8 | 23.8 |
| greedy_minmax | 42 | 49.5 | 50.2 | 43.0 |
| greedy_minmax | 2024 | 24.4 | 23.1 | 21.2 |
| cluster_topo | 42 | 44.3 | 45.7 | 48.0 |
| cluster_topo | 2024 | 21.4 | 23.1 | 22.6 |
| random | 42 | 47.0 | 50.9 | 47.8 |
| random | 2024 | 23.6 | 22.4 | 24.1 |

### 4d. Prefix Cache Spectrum (Cumulative Condition Number)
| Ordering | Seed | @1K | @2K | @3K | @4K | @5K | @7K | @10K |
|----------|------|:---:|:---:|:---:|:---:|:---:|:---:|:----:|
| key_clustered | 42 | 33 | 47 | 86 | 100 | 115 | 145 | 204 |
| key_clustered | 2024 | 10 | 19 | 35 | 42 | 49 | 61 | 79 |
| key_dispersed | 42 | 20 | 51 | 72 | 91 | 107 | 152 | 204 |
| key_dispersed | 2024 | 10 | 21 | 36 | 43 | 49 | 62 | 79 |
| greedy_minmax | 42 | 11 | 18 | 26 | 34 | 42 | 61 | 204 |
| greedy_minmax | 2024 | 7 | 11 | 15 | 19 | 23 | 31 | 79 |
| cluster_topo | 42 | 45 | 65 | 87 | 103 | 117 | 149 | 204 |
| cluster_topo | 2024 | 22 | 30 | 38 | 44 | 49 | 61 | 79 |
| random | 42 | 19 | 30 | 54 | 65 | 77 | 118 | 204 |
| random | 2024 | 11 | 18 | 23 | 45 | 50 | 62 | 79 |

## 5. Pre-Collapse Survival Model

Correlation between per-cohort geometric exposure (mean max-cos to subsequent keys)
and cohort retention, measured at the last healthy checkpoint.

| Ordering | Seed | Test Edits | N cohorts | Exposure→Retention r | p-value | Position→Retention r | p-value |
|----------|------|:----------:|:---------:|:--------------------:|:-------:|:--------------------:|:-------:|
| greedy_minmax | 42 | 4000 | 39 | **0.687** | 0.0000 | 0.919 | 0.0000 |
| greedy_minmax | 2024 | 4000 | 39 | **0.360** | 0.0244 | 0.944 | 0.0000 |
| key_clustered | 42 | 5000 | 49 | **-0.586** | 0.0000 | 0.622 | 0.0000 |
| key_clustered | 2024 | 5000 | 49 | -0.116 | 0.4269 | 0.621 | 0.0000 |

Interpretation:
- Negative exposure→retention r: higher geometric exposure predicts more forgetting (hypothesis supported)
- Negative position→retention r: older cohorts degrade more (expected, age effect)
- If exposure→retention is significant AFTER controlling for position: geometry matters beyond age

## 6. Mechanism Trajectories

Requires JSONL logs from S3. Run `scheduling/forensics_mechanism.py` after pulling logs.

## 7. Capability Probes

Requires GPU. Run `scripts/run_capability_probe_ordering.sh` on cluster.