# AlphaEdit Reproducibility Study

A mechanistic reproducibility study of [AlphaEdit](https://github.com/jianghoucheng/AlphaEdit) (ICLR 2025 Outstanding Paper), targeting TMLR Reproducibility Certification and NeurIPS 2026 MLRC track.

---

## Central Thesis

> AlphaEdit's reliability is limited by two mechanisms: **capacity saturation** (the null-space fills up) and **semantic interference** (the desired edit overlaps with preserved knowledge). The second failure mode is more fundamental — it challenges the assumption that editability and preservation are always separable.

---

## Experiment Status

### Core Reproduction

- [x] Obtain HuggingFace access to Llama-3-8B-Instruct
- [x] GPU smoke test (validate pipeline end-to-end)
- [x] MVE1 — AlphaEdit on MultiCounterFact (5 seeds)
- [x] MVE2 — MEMIT on MultiCounterFact (5 seeds)
- [x] MVE3 — AlphaEdit on zsRE (5 seeds)

### Extended Experiments (3 seeds: 42, 137, 2024)

- [x] Failure curve (500–10K edits)
- [ ] Null-space rank tracking
- [ ] MEMIT+SeqReg calibration
- [ ] Cache mitigation sweep
- [ ] Capability probe (WikiText perplexity + MMLU)
- [ ] Second model (Mistral-7B)

### Write-Up

- [ ] Write up for TMLR (target: 2026-07-24)

---

## Experiments

### Core Reproduction (MVE)

| ID | Experiment | Tests |
|----|-----------|-------|
| MVE1 | AlphaEdit on MultiCounterFact | Primary benchmark: 2000 facts, 5 seeds |
| MVE2 | MEMIT on MultiCounterFact | Fair comparison under identical conditions |
| MVE3 | AlphaEdit on zsRE | Cross-dataset generalization |

All MVEs: batches of 100, evaluation every 5 batches, 5 seeds.

### Novel Extensions

| Priority | Experiment | Core question |
|----------|-----------|---------------|
| **P0** | Matched ordering | Does key-space geometry of edit sequences affect performance? |
| P1 | Failure curve (500–10K edits) | Where does AlphaEdit's advantage disappear? |
| P1 | Null-space rank tracking | Which layers saturate first? |
| P1 | MEMIT+PrevKeyReg+Ridge | Is null-space projection necessary, or does key-direction regularization suffice? |
| P2 | Capability probe (WikiText-103) | Does editing destroy general language ability? |
| P2 | Second model (Mistral-7B) | Do findings generalize beyond Llama-3-8B? |

See [docs/extensions.md](docs/extensions.md) for detailed experimental designs and [docs/metrics.md](docs/metrics.md) for the full metrics/statistical framework.

---

## Setup

```bash
bash scripts/setup_env.sh
bash scripts/link_stats.sh /path/to/wikipedia_stats/
export HF_TOKEN=hf_...
uv run huggingface-cli login --token $HF_TOKEN
bash scripts/smoke_test.sh  # requires GPU
```

**Requirements**: Python 3.10, NVIDIA GPU ≥ 48GB VRAM, HuggingFace access to `meta-llama/Meta-Llama-3-8B-Instruct`.

---

## Running Experiments

### Core Experiments

```bash
bash scripts/run_mve1_alphaedit_mcf.sh 42
bash scripts/run_mve2_memit_mcf.sh 42
bash scripts/run_mve3_alphaedit_zsre.sh 42
```

### Failure Curve (Checkpointed)

```bash
# Milestone evaluation (RECOMMENDED)
EVAL_AT_CHECKPOINTS_ONLY=true bash scripts/run_failure_curve_checkpointed.sh 42 both 5000

# Fast mode (testing/iteration)
FAST_CHECKPOINT=true bash scripts/run_failure_curve_checkpointed.sh 42 AlphaEdit 2000
```

### Extensions

```bash
bash scripts/run_nullspace_analysis.sh 42
bash scripts/run_capability_probe.sh 42
bash scripts/run_matched_ordering.sh 42
FAST_CHECKPOINT=true bash scripts/run_memit_sequential.sh 42 1 1
```

### Multi-Seed / Full Sweep

```bash
bash scripts/run_all_seeds.sh mve     # All MVEs × 5 seeds (42, 137, 2024, 7, 99)
bash scripts/run_all_seeds.sh failure_curve  # Extensions × 3 seeds (42, 137, 2024)
bash scripts/run_all_seeds.sh all     # Everything with appropriate seed counts
```

### SkyPilot (Cloud GPU)

```bash
bash sky/sky_launch.sh mve1           # MVE1 × 5 seeds
bash sky/sky_launch.sh mve1 42        # MVE1, single seed override
bash sky/sky_launch.sh failure_curve_ckpt  # Extension × 3 seeds
bash sky/sky_launch.sh all            # All experiments with appropriate seeds
sky status                            # Monitor clusters
sky logs ae-mve1_alphaedit_mcf-s42    # Stream logs
sky down -a                           # Tear down all clusters
```

---

## Analysis

```bash
uv run python analysis/aggregate.py --results_dir results
uv run python analysis/paired_bootstrap.py --results_dir results
uv run python analysis/plots.py --results_dir results --output_dir results/figures
uv run python analysis/nullspace_analysis.py --results_dir results/nullspace_tracking
```

---

## Project Structure

```
├── sky/                     # SkyPilot cloud GPU orchestration
├── scripts/                 # Shell scripts (setup, experiments)
├── src/                     # Python runners (source injection pattern)
├── configs/                 # Frozen experiment manifest
├── vendor/AlphaEdit/        # Git submodule pinned at b84624f
├── analysis/                # Post-hoc statistical analysis
├── docs/                    # Detailed documentation
├── tests/                   # Unit tests
└── results/                 # Output directory
```

---

## Citation

```bibtex
@inproceedings{fang2024alphaedit,
  title={AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models},
  author={Fang, Junfeng and Jiang, Houcheng and ...},
  booktitle={ICLR},
  year={2025}
}
```

Upstream code: https://github.com/jianghoucheng/AlphaEdit (pinned at commit `b84624f`)
