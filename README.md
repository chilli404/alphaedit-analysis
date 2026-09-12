# AlphaEdit Reproducibility Study

Rigorous reproducibility study of [AlphaEdit](https://github.com/jianghoucheng/AlphaEdit) (ICLR 2025 Outstanding Paper), investigating temporal path dependence in sequential model editing.

**Key finding:** The order in which edits are applied — not just what is edited — determines which memories survive. Identical edit batches produce radically different outcomes under different temporal sequences, and this path sensitivity varies across editing methods.

## Key Results

- **Age-selective forgetting**: Older edits are disproportionately lost (42pp gap at 10K edits, 3 seeds)
- **Path dependence**: Reordering identical fixed batches creates 25pp efficacy differences
- **Method dependence**: AlphaEdit and EvoEdit are order-sensitive (-22pp, -12pp); MEMIT-Seq and PathGuard are robust (~0pp)
- **Cross-architecture**: Pattern replicates on Llama-3-8B, GPT-J-6B, and Qwen-2.5-7B
- **Causal evidence**: Same-state A/B interventions show high-overlap batches cause more immediate logit damage (47/60 trials, p < 0.0001)

## Setup

```bash
git clone --recurse-submodules <repo-url>
cd alphaedit-analysis
uv sync
bash scripts/setup_env.sh
export HF_TOKEN=hf_...
```

**Requirements:** Python 3.10, NVIDIA GPU with ≥ 48GB VRAM, HuggingFace access to `meta-llama/Meta-Llama-3-8B-Instruct`.

### Quick Validation

```bash
bash scripts/smoke_test.sh   # 5-minute GPU smoke test (2 edits, 10 samples)
uv run pytest tests/ -q      # 1175 CPU tests (~10 seconds)
```

## Running Experiments

### Core Reproduction (5 seeds)

```bash
bash scripts/run_mve1_alphaedit_mcf.sh 42    # AlphaEdit on MultiCounterFact
bash scripts/run_mve2_memit_mcf.sh 42        # MEMIT baseline
bash scripts/run_mve3_alphaedit_zsre.sh 42   # AlphaEdit on zsRE
```

### Failure Curve (10K edits with checkpoints)

```bash
EVAL_AT_CHECKPOINTS_ONLY=true \
  bash scripts/run_failure_curve_checkpointed.sh 42 both 10000
```

### Fixed-Batch Ordering Experiment

```bash
bash scripts/run_matched_ordering.sh 42 AlphaEdit fb_high_exposure
bash scripts/run_matched_ordering.sh 42 AlphaEdit fb_low_exposure
```

### Cross-Method Comparison

```bash
bash scripts/run_evoedit_baseline.sh 42 fb_high_exposure
bash scripts/run_nse_baseline.sh 42 fb_high_exposure
bash scripts/run_revive_baseline.sh 42 fb_high_exposure
```

### SkyPilot (Cloud GPU)

```bash
bash sky/sky_launch.sh mve1 42          # Single experiment + seed
bash sky/sky_launch.sh all              # All experiments
sky status                              # Monitor
sky down -a                             # Tear down
```

See [docs/experiments.md](docs/experiments.md) for the full experiment reference including environment variables, path conventions, and all available scripts.

## Project Structure

```
alphaedit-analysis/
├── src/
│   ├── evaluate_harness.py      # Main experiment loop (replaces vendor evaluate.py)
│   ├── algorithms/              # Composable per-layer hooks (SeqReg, REVIVE, PathGuard)
│   │   ├── hooks.py             # AlgorithmHooks dataclass + compose_hooks()
│   │   ├── hook_presets.py      # seqreg_hooks, revive_hooks, pathguard_hooks, c0_hooks
│   │   ├── memit_with_hooks.py  # MEMIT with hook call points
│   │   └── alphaedit_with_hooks.py
│   ├── runners/                 # Experiment runners (thin wrappers around harness)
│   ├── util/
│   │   ├── experiment_config.py # Single source of truth for variant names + paths
│   │   ├── checkpoint_io.py     # Shared save/load with path validation
│   │   └── paths.py             # Centralized path resolution + S3 guard
│   ├── mechanism/               # Mechanistic analysis tools
│   ├── polykernel/              # Polynomial kernel experiments
│   └── revive/                  # REVIVE spectral filter
├── analysis/                    # Paper figure + table generation (Makefile-driven)
│   └── _standalone/             # One-off analysis scripts
├── scripts/                     # Shell scripts for experiments + utilities
│   └── patches/                 # Runtime patches for vendor code
├── sky/                         # SkyPilot cloud GPU orchestration
├── tests/                       # 1175 CPU tests + GPU smoke tests
├── configs/                     # Frozen experiment manifest
├── vendor/AlphaEdit/            # Git submodule (pinned at b84624f)
├── docs/                        # Experiment design + technical analysis
└── paper/                       # Frozen result numbers for the paper
```

## Architecture

The codebase uses two patterns:

**Harness + hooks** (primary): `evaluate_harness.run_experiment()` runs the edit-eval loop. Algorithm-specific behavior is injected via composable `AlgorithmHooks`:

```python
from algorithms.hook_presets import seqreg_hooks, revive_hooks, compose_hooks

hooks = compose_hooks(
    seqreg_hooks(lambda_prev=1.0),
    revive_hooks(revive_tau=0.1),
)
run_experiment(model, tok, hparams, dataset, apply_fn, hooks=hooks, ...)
```

**Legacy source injection** (secondary): Some runners still read vendor source files as text, patch them, and `exec(compile())`. These are labeled with `STATUS: LEGACY` headers.

## Reproducing Paper Results

Every number in the paper traces to a source file:
- **`paper/RESULTS_MANIFEST_v2.md`** — authoritative prob-pref numbers for every claim
- **`paper/result_registry.json`** — machine-readable mapping of every number to its source file, metric type, and checkpoint

## Evaluation Metrics

Results use probability-preference metrics (pairwise NLL comparison, matching published AlphaEdit/EvoEdit papers). Argmax metrics are reported as secondary. See `scripts/eval_prob_preference.py` for the rescoring tool and `analysis/loaders.py` for metric extraction.

## Citation

```bibtex
@article{alphaedit-reproducibility-2026,
  title={Temporal Paths Determine Which Model Edits Survive},
  author={},
  year={2026}
}
```

**Upstream:** [AlphaEdit](https://github.com/jianghoucheng/AlphaEdit) (Fang et al., ICLR 2025), pinned at commit `b84624f`.

## License

MIT — see [LICENSE](LICENSE).
