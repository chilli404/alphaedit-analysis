# AlphaEdit Reproducibility Study — Project Reference

## Purpose

This repository is a rigorous reproducibility study targeting TMLR (Transactions on Machine Learning Research) certification and the NeurIPS 2026 MLRC track. It reproduces and extends AlphaEdit (ICLR 2025 Outstanding Paper), a knowledge editing method for LLMs that uses null-space projection to edit model facts while preserving existing knowledge.

The central thesis **challenges AlphaEdit's core assumption**: that editability and preservation can always be separated via null-space projection. The project provides empirical evidence of failure modes where the null-space becomes exhausted or where semantically coupled facts degrade under editing.

---

## Architecture

```
alphaedit-reproducibility/
├── sky/                     # SkyPilot cloud GPU orchestration
│   ├── alphaedit_gpu.yaml   # Cluster resource spec + setup/run task definition
│   └── sky_launch.sh        # Batch job launcher across experiments × seeds
├── scripts/                 # Shell scripts (setup, experiments, analysis)
├── src/                     # Python runners (seeded_runner, trackers, probes)
├── configs/                 # Frozen experiment manifest (experiment_manifest.yaml)
├── vendor/AlphaEdit/        # Git submodule pinned at b84624f
├── analysis/                # Post-hoc statistical analysis
├── data/                    # Datasets & covariance stats (symlinked from S3)
└── results/                 # Output directory
```

---

## SkyPilot Integration

### What is SkyPilot

SkyPilot is a framework for running ML workloads on any cloud (AWS, GCP, Azure, Lambda, etc.). It handles provisioning, file syncing, setup, execution, and teardown. In this project it orchestrates GPU experiments across multiple seeds and experiment types.

### YAML Task Definition (`sky/alphaedit_gpu.yaml`)

The YAML declares:
- **`resources:`** — GPU type (`L40s:1`), CPU (8+), memory (64+GB)
- **`workdir: .`** — Uploads the entire project root to `~/sky_workdir` on the cluster
- **`file_mounts:`** — Mounts `.env` (contains HF_TOKEN and other secrets) into the cluster
- **`setup:`** — Runs **once** on cluster creation: calls `scripts/remote_setup.sh` to install Python 3.10, uv, dependencies, initialize the submodule
- **`run:`** — Runs **each time** a job executes: links S3 data, patches vendor code, runs the experiment, copies results back to S3

### Environment Variables

The YAML `run:` block expects these env vars (passed via `--env`):
- `EXPERIMENT_NAME` — Which experiment script to run (e.g., `mve1_alphaedit_mcf`)
- `SEED` — Random seed (e.g., 42, 137, 2024, 7, 99)
- `CUDA_DEVICE` — GPU device index (default 0)
- `TOKENIZERS_PARALLELISM` — HuggingFace tokenizer setting
- `HF_TOKEN` — HuggingFace access token (from `.env` file)

### S3 Mounting

Within SkyPilot clusters, **S3 is already FUSE-mounted at `/s3-data`**. This means:
- **Input data** is read from:
  - `/s3-data/continual-learning/alphaedit/stats/` (covariance statistics)
  - `/s3-data/continual-learning/alphaedit/dsets/` (datasets)
- **Output results** are copied to:
  - `/s3-data/continual-learning/alphaedit/results/${EXPERIMENT_NAME}/seed${SEED}/`
- No explicit `aws s3 cp` needed — standard filesystem operations (`cp -r`, `ln -s`) work directly.

### The Launcher (`sky/sky_launch.sh`)

```bash
bash sky/sky_launch.sh              # All experiments × 5 seeds (55+ jobs)
bash sky/sky_launch.sh mve1         # MVE1 only × 5 seeds
bash sky/sky_launch.sh mve1 42      # MVE1, seed 42 only
```

Creates clusters named `ae-{experiment}-s{seed}` (e.g., `ae-mve1_alphaedit_mcf-s42`). If the cluster already exists, it uses `sky exec` (reuses existing cluster) instead of `sky launch` (creates new cluster). All jobs use `--detach-run` for asynchronous execution.

### Key SkyPilot CLI Commands

| Command | Purpose |
|---------|---------|
| `sky launch yaml --cluster name -y` | Create new cluster + run the task |
| `sky exec cluster yaml` | Run a new job on existing cluster (skips setup) |
| `sky ssh cluster` | SSH into a running cluster interactively |
| `sky logs cluster` | Stream logs from the most recent job |
| `sky queue` | View all jobs across all clusters |
| `sky status` | List all clusters and their states |
| `sky stop cluster` | Pause cluster (keeps disk, stops billing) |
| `sky down cluster` | Terminate and delete cluster entirely |
| `sky down -a` | Terminate ALL clusters |

### sky launch vs sky exec

- **`sky launch`**: Provisions new infrastructure, runs `setup:` block, then runs `run:` block. Use for first-time cluster creation.
- **`sky exec`**: Runs only the `run:` block on an already-provisioned cluster. Faster because setup is skipped. Use to submit additional jobs to existing clusters.

### Interactive Mode

If `EXPERIMENT_NAME` is not set, the YAML enters interactive mode — it prints SSH instructions and exits. You can then `sky ssh <cluster>` and run experiments manually:
```bash
sky ssh ae-mve1_alphaedit_mcf-s42
cd ~/sky_workdir
bash scripts/run_mve1_alphaedit_mcf.sh 42
```

---

## Scripts Reference

### Setup Scripts

| Script | Purpose |
|--------|---------|
| `scripts/setup_env.sh` | Local one-time setup: installs Python 3.10, syncs deps with uv, inits submodule, patches vendor code |
| `scripts/remote_setup.sh` | Cluster setup: same as above but includes nvidia-smi verification, Artifactory auth fallback |
| `scripts/link_stats.sh` | Symlinks covariance statistics from `/s3-data` or local `data/stats/` into `vendor/AlphaEdit/data/stats/` |
| `scripts/link_dsets.sh` | Symlinks datasets from `/s3-data` or local `data/dsets/` into `vendor/AlphaEdit/data/` |
| `scripts/download_datasets.sh` | Downloads CounterFact/zsRE datasets from `memit.baulab.info` (for initial S3 upload or local use) |

### Core Experiment Scripts (MVEs)

| Script | Algorithm | Dataset | What It Tests |
|--------|-----------|---------|---------------|
| `scripts/run_mve1_alphaedit_mcf.sh` | AlphaEdit | MultiCounterFact | Primary claim: 2000 facts, 100-edit batches, measures efficacy/paraphrase/neighborhood/GLUE |
| `scripts/run_mve2_memit_mcf.sh` | MEMIT | MultiCounterFact | Fair comparison baseline (same data, same batches, unconstrained editing) |
| `scripts/run_mve3_alphaedit_zsre.sh` | AlphaEdit | zsRE | Cross-dataset generalization |
| `scripts/run_mve4_conflict_seq.sh` | AlphaEdit | Conflict (generated) | Sequential single edits with conflicting targets for same subject |
| `scripts/run_rome_baseline.sh` | ROME | MultiCounterFact | Lower-bound calibration (1 edit at a time) |

### Extended Analysis Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_failure_curve.sh` | Tests at [500, 1000, 1500, 2000, 3000, 5000, 7500, 10000] edits to find where AlphaEdit's null-space advantage disappears |
| `scripts/run_nullspace_analysis.sh` | Tracks null-space rank consumption per layer per batch via SVD |
| `scripts/run_order_sensitivity.sh` | 10 random orderings × 2 algorithms — tests if edit order affects final metrics |
| `scripts/run_coupling_stress.sh` | Measures "projection loss" under 4 semantic coupling types (synonym, hypernym, co-occurrence, causal) |
| `scripts/run_capability_probe.sh` | Measures WikiText perplexity + few-shot MMLU at intervals to detect general capability damage |

### Meta-Orchestration

| Script | Purpose |
|--------|---------|
| `scripts/run_all_seeds.sh` | Runs any experiment across all 5 seeds locally |
| `scripts/smoke_test.sh` | 5-minute validation run (2 edits, 10 samples) to verify environment |

---

## Python Source (`src/`)

### Source Injection Pattern

All runners use a **source injection** approach: they read `vendor/AlphaEdit/experiments/evaluate.py` as text, inject measurement/tracking code at specific anchor points, then `compile()` + `exec()` the modified source. This avoids import path issues with the vendor code and lets measurement code access internal variables (like the projection matrix P).

### Key Files

| File | Purpose |
|------|---------|
| `src/seeded_runner.py` | Main wrapper: sets all RNG seeds (Python, NumPy, PyTorch, CUDA), patches CUDA device lines, launches experiment via source injection |
| `src/nullspace_tracker.py` | Injects SVD tracking of the projection matrix P and covariance cache at each edit batch |
| `src/coupling_stress_runner.py` | Injects projection-loss measurement inside the edit loop |
| `src/order_sensitivity_runner.py` | Injects dataset shuffling with an independent order seed |
| `src/capability_probe_runner.py` | Hooks into GLUEEval to run perplexity/MMLU probes at configurable intervals |
| `src/coupling_dataset.py` | Generates anchor-probe pairs stratified by 4 semantic coupling types |
| `src/conflict_dataset.py` | Generates conflicting edit pairs (same subject, contradictory targets) |
| `src/capability_probe.py` | Computes WikiText-2 perplexity and few-shot MMLU accuracy |
| `src/model_download.py` | Resolves model paths — tries local cache, then HuggingFace Hub, with optional Artifactory fallback |

---

## Vendor Submodule (`vendor/AlphaEdit/`)

**Pinned at commit `b84624f`** from `github.com/jianghoucheng/AlphaEdit`.

### Key Components

- **`experiments/evaluate.py`** — Main evaluation harness: loads model, applies edits in configurable batches, measures efficacy/paraphrase/neighborhood/GLUE metrics
- **`AlphaEdit/AlphaEdit_main.py`** — Core algorithm: computes null-space projection P from precomputed covariance statistics, solves for weight updates constrained to P's null-space
- **`memit/memit_main.py`** — MEMIT baseline: unconstrained mass editing (no null-space projection)
- **`rome/rome_main.py`** — ROME baseline: rank-one model editing (single edit at a time)
- **`hparams/`** — Per-algorithm, per-model hyperparameters (which layers to edit, learning rates, batch sizes, etc.)
- **`data/stats/`** — Precomputed covariance statistics from Wikipedia (C matrices needed for projection)
- **`glue_eval/`** — GLUE benchmark evaluation for measuring general language understanding preservation

### Runtime Patches

These patches are applied at runtime (in the YAML `run:` block) rather than committed to the submodule:
1. **Model name variants**: Adds `meta-llama-3-8b-instruct` and `nousresearch--meta-llama-3-8b-instruct` to the context length map in `glue_eval/useful_functions.py`
2. **Extra kwargs**: Adds `**_kwargs` to MEMIT and AlphaEdit main functions to handle `return_orig_weights_device` kwarg passed by `evaluate.py`

---

## Data Flow

```
Local machine                    SkyPilot Cluster
─────────────                    ────────────────
sky/sky_launch.sh ──────────→  sky launch/exec
  passes: EXPERIMENT_NAME,        │
          SEED, .env              ▼
                              ~/sky_workdir (uploaded via workdir: .)
                                  │
                                  ▼
                              setup: remote_setup.sh (installs everything, once)
                                  │
                                  ▼
                              run:
                                link_stats.sh  ← /s3-data/.../stats/
                                link_dsets.sh  ← /s3-data/.../dsets/
                                patch vendor code (sed)
                                download NLTK data
                                  │
                                  ▼
                              bash scripts/run_${EXPERIMENT_NAME}.sh $SEED
                                  │
                                  ▼
                              uv run python src/seeded_runner.py (or specialized runner)
                                  │
                                  ▼
                              vendor/AlphaEdit/experiments/evaluate.py (injected)
                                  │
                                  ▼
                              Results written to results/ and vendor/AlphaEdit/results/
                                  │
                                  ▼
                              cp -r results/ → /s3-data/.../results/${EXPERIMENT_NAME}/seed${SEED}/
```

---

## Reproducibility Design

- **5 seeds per experiment**: 42, 137, 2024, 7, 99
- **Deterministic seeding**: All RNG sources (Python `random`, NumPy, PyTorch CPU/CUDA) seeded identically
- **Frozen dependencies**: `uv.lock` pins all Python packages
- **Pinned submodule**: Vendor code at exact commit hash
- **Frozen config**: `configs/experiment_manifest.yaml` declares all experiment parameters

---

## Common Workflows

### Run a smoke test locally
```bash
bash scripts/setup_env.sh
bash scripts/smoke_test.sh
```

### Launch one experiment on SkyPilot
```bash
bash sky/sky_launch.sh mve1 42
```

### Launch all experiments
```bash
bash sky/sky_launch.sh all
```

### Monitor running jobs
```bash
sky status          # Cluster list
sky queue           # Job queue across all clusters
sky logs ae-mve1_alphaedit_mcf-s42   # Stream logs
```

### Debug on cluster
```bash
sky ssh ae-mve1_alphaedit_mcf-s42
cd ~/sky_workdir
nvidia-smi
ls results/
```

### Retrieve results
Results auto-sync to S3 at job completion. From any machine with S3 access:
```bash
ls /s3-data/continual-learning/alphaedit/results/
# or
aws s3 ls s3://<bucket>/continual-learning/alphaedit/results/
```

### Tear down all clusters
```bash
sky down -a
```
