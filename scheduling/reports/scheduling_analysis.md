# Scheduling Experiment — Analysis Report

Seeds: [42, 2024]
Arms: 6 (4 with results)

## Headline Table: Scheduling Experiment Results

| Arm | Eff@2K | Eff@5K | Eff@7K | Eff@10K | 1st-1K Ret | Latest-1K Eff |
|-----|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| AlphaEdit/key_clustered/s42 | 0.964 | 0.922 | 0.912 | 0.822 | 0.551 | 0.928 |
| AlphaEdit/key_clustered/s2024 | 0.966 | 0.947 | 0.924 | 0.837 | 0.647 | 0.941 |
| AlphaEdit/key_dispersed/s42 | 0.949 | 0.901 | 0.791 | 0.397 | 0.148 | 0.901 |
| AlphaEdit/key_dispersed/s2024 | 0.950 | 0.846 | 0.566 | 0.132 | 0.031 | 0.592 |

### Seed 42

## Prospective Validation

Correlating geometry-predicted interference with actual retention:

| Ordering | Predicted Exposure (frac>0.3) | Actual 1st-1K Retention@10K |
|----------|:-----------------------------:|:---------------------------:|
| key_clustered | 0.5157 | 0.5510 |
| key_dispersed | 0.7835 | 0.1480 |
| greedy_minmax | 0.0000 | — |
| cluster_topo | 0.3714 | — |
| random | 0.7474 | — |

### Seed 2024

## Prospective Validation

Correlating geometry-predicted interference with actual retention:

| Ordering | Predicted Exposure (frac>0.3) | Actual 1st-1K Retention@10K |
|----------|:-----------------------------:|:---------------------------:|
| key_clustered | 0.4440 | 0.6470 |
| key_dispersed | 0.8080 | 0.0310 |
| greedy_minmax | 0.0000 | — |
| cluster_topo | 0.5970 | — |
| random | 0.8080 | — |

## Installation-Quality Equivalence

Verifying that scheduling does not degrade edit installation:

| Arm | Latest-1K Efficacy@10K | Deviation from canonical |
|-----|:----------------------:|:------------------------:|
| AlphaEdit/key_clustered/s42 | 0.9280 | -1.4% |
| AlphaEdit/key_clustered/s2024 | 0.9410 | +0.0% |
| AlphaEdit/key_dispersed/s42 | 0.9010 | -4.3% **FLAG** |
| AlphaEdit/key_dispersed/s2024 | 0.5920 | -37.1% **FLAG** |

Flag threshold: >3% deviation from canonical (key_clustered) baseline.
