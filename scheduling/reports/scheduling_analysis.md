# Scheduling Experiment — Analysis Report

Seeds: [42, 2024]
Arms: 6 (8 with results)

## Headline Table: Scheduling Experiment Results

| Arm | Eff@2K | Eff@5K | Eff@7K | Eff@10K | 1st-1K Ret | Latest-1K Eff |
|-----|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| AlphaEdit/cluster_topo/s42 | 0.910 | 0.904 | 0.898 | 0.333 | 0.109 | 0.862 |
| AlphaEdit/greedy_minmax/s42 | 0.980 | 0.722 | 0.161 | 0.124 | 0.020 | 0.423 |
| AlphaEdit/greedy_minmax/s2024 | 0.978 | 0.407 | 0.207 | 0.147 | 0.013 | 0.435 |
| AlphaEdit/key_clustered/s42 | 0.964 | 0.922 | 0.912 | 0.822 | 0.551 | 0.928 |
| AlphaEdit/key_clustered/s2024 | 0.966 | 0.947 | 0.924 | 0.837 | 0.647 | 0.941 |
| AlphaEdit/key_dispersed/s42 | 0.949 | 0.901 | 0.791 | 0.397 | 0.148 | 0.901 |
| AlphaEdit/key_dispersed/s2024 | 0.950 | 0.846 | 0.566 | 0.132 | 0.031 | 0.592 |
| AlphaEdit/random/s42 | 0.965 | 0.901 | 0.825 | 0.336 | 0.137 | 0.841 |

### Seed 42

## Prospective Validation

Correlating geometry-predicted interference with actual retention:

| Ordering | Predicted Exposure (frac>0.3) | Actual 1st-1K Retention@10K |
|----------|:-----------------------------:|:---------------------------:|
| key_clustered | 0.5157 | 0.5510 |
| key_dispersed | 0.7835 | 0.1480 |
| greedy_minmax | 0.0000 | 0.0200 |
| cluster_topo | 0.3714 | 0.1090 |
| random | 0.7474 | 0.1370 |

Pearson correlation (exposure vs retention): r = 0.290
Expected: negative (more exposure → less retention).

### Seed 2024

## Prospective Validation

Correlating geometry-predicted interference with actual retention:

| Ordering | Predicted Exposure (frac>0.3) | Actual 1st-1K Retention@10K |
|----------|:-----------------------------:|:---------------------------:|
| key_clustered | 0.4440 | 0.6470 |
| key_dispersed | 0.8080 | 0.0310 |
| greedy_minmax | 0.0000 | 0.0130 |
| cluster_topo | 0.5970 | — |
| random | 0.8080 | — |

Pearson correlation (exposure vs retention): r = 0.082
Expected: negative (more exposure → less retention).

## Installation-Quality Equivalence

Verifying that scheduling does not degrade edit installation:

| Arm | Latest-1K Efficacy@10K | Deviation from canonical |
|-----|:----------------------:|:------------------------:|
| AlphaEdit/cluster_topo/s42 | 0.8620 | -8.4% **FLAG** |
| AlphaEdit/greedy_minmax/s42 | 0.4230 | -55.0% **FLAG** |
| AlphaEdit/greedy_minmax/s2024 | 0.4350 | -53.8% **FLAG** |
| AlphaEdit/key_clustered/s42 | 0.9280 | -1.4% |
| AlphaEdit/key_clustered/s2024 | 0.9410 | +0.0% |
| AlphaEdit/key_dispersed/s42 | 0.9010 | -4.3% **FLAG** |
| AlphaEdit/key_dispersed/s2024 | 0.5920 | -37.1% **FLAG** |
| AlphaEdit/random/s42 | 0.8410 | -10.6% **FLAG** |

Flag threshold: >3% deviation from canonical (key_clustered) baseline.
