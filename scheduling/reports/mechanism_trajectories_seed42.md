## 6. Mechanism Trajectories (from JSONL logs)

Seed: 42

### ||dW|| Update Norm Trajectory
| Ordering | Batches | Early (0-10) | Mid (30-50) | Late (70-99) | Spike Onset |
|----------|:-------:|:------------:|:-----------:|:------------:|:-----------:|
| greedy_minmax | 80 | 2.8180 | 5.4145 | 12.5006 | batch 59 |
| key_clustered | 75 | 2.6557 | 3.8254 | 4.6380 | None |
| random | 87 | 2.8969 | 4.3542 | 5.3375 | None |

### Removed Fraction (signal lost to projection)
| Ordering | Early (0-10) | Mid (30-50) | Late (70-99) | Spike Onset |
|----------|:------------:|:-----------:|:------------:|:-----------:|
| greedy_minmax | 0.0631 | 0.0531 | 0.0556 | None |
| key_clustered | 0.0777 | 0.0481 | 0.0479 | None |
| random | 0.0651 | 0.0414 | 0.0515 | batch 0 |

### q_t (Functional Signal Preservation Ratio)
| Ordering | Early (0-10) | Mid (30-50) | Late (70-99) |
|----------|:------------:|:-----------:|:------------:|
| greedy_minmax | 0.9948 | 0.9983 | 0.9983 |
| key_clustered | 0.9956 | 0.9981 | 0.9985 |
| random | 0.9950 | 0.9983 | 0.9983 |