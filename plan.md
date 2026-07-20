# Plan: Streamlined Analysis Pipeline

## Goal

Replace the current monolithic `analysis/paper_figures.py` with modular per-figure scripts that are:
- Easy to understand (one file = one figure)
- Easy to run (CLI args, clear defaults, Makefile orchestration)
- Properly use shared utilities (no duplicated aggregation code)
- Cover both main figures AND key appendix figures
- Support local-first loading with S3 fallback

## Current Problems

1. `paper_figures.py` duplicates `extract_metrics_from_case()` logic instead of importing from `analysis/stats/aggregate.py`
2. `paper_figures.py` only extracts `_correct` fields — never `_probs` (probability locality)
3. No cohort-retention computation in `paper_figures.py`
4. No SeqReg or order-at-scale analysis
5. Inconsistent invocation across scripts (some CLI, some hardcoded)
6. No single command to regenerate all paper outputs
7. Figures don't match the new paper spec (different panels, different content)

## New Structure

```
analysis/
├── Makefile                          # Single entry point: `make all`, `make fig1`, etc.
├── loaders.py                        # Shared data loading (local + S3 fallback)
├── style.py                          # Shared matplotlib config + color schemes
├── fig1_reproduction.py              # Figure 1: Reproduction + long-horizon boundary
├── fig2_forgetting.py                # Figure 2: Anatomy of forgetting
├── fig3_coupling.py                  # Figure 3: Controlled semantic concentration
├── fig4_attribution.py              # Figure 4: Editability, interference, attribution
├── tables.py                         # Tables 1-4 + paper_numbers.json
├── appendix_figures.py              # A1 (per-seed curves), A3 (heatmaps), A8 (SeqReg)
├── paper_figures.py                  # [ARCHIVED] Kept for reference, not used
├── stats/                            # [UNCHANGED] Existing statistical infrastructure
│   ├── aggregate.py
│   ├── confidence_intervals.py
│   ├── paired_bootstrap.py
│   └── tables.py
└── plots/                            # [UNCHANGED] Existing specialized plotters
    ├── failure_curve.py
    ├── failure_curve_4panel.py
    ├── controlled_coupling_plots.py
    ├── cache_ablation_plots.py
    └── mechanism_analysis_plots.py
```

## Files to Create

### 1. `analysis/style.py` — Shared visual config

Consolidates matplotlib rcParams, color palettes, and helper functions used across all figures:
- `COLORS` dict (algorithm → hex)
- `SEED_COLORS` dict
- `STREAM_COLORS` dict
- `setup_style()` function for rcParams
- `save_figure(fig, name)` helper (saves both PNG + PDF to output dir)

### 2. `analysis/loaders.py` — Shared data loading module

Unified data access layer. All figure scripts import from here instead of writing their own loaders.

Functions:
- `load_checkpoint_metrics(seed, edits, alg)` → dict with efficacy, paraphrase, neighborhood, prob-locality, n_facts
- `load_checkpoint_cohorts(seed, edits, alg, batch_size=100)` → dict mapping cohort_idx → {efficacy, paraphrase, neighborhood, n_facts}
- `load_checkpoint_glue(seed, edits, alg)` → dict with mmmlu, sst, cola, etc.
- `load_controlled_coupling_jsonl(stream, seed)` → list of per-batch mechanism records
- `load_controlled_coupling_behavioral(seed)` → dict with low/high coupling cohort data
- `load_comparison_ordered(seed, edits)` → list of per-order metric dicts
- `load_mechanism_metrics(seed)` → list of mechanism JSONL records
- `load_seqreg_eval(seed, lambda_prev, lambda_delta, edits)` → dict with metrics + cohorts
- `load_seqreg_logs(seed, lambda_prev, lambda_delta)` → list of per-batch JSONL records
- `load_weight_drift(seed)` → dict with per-layer drift metrics
- `discover_available_data()` → summary of what data exists locally

S3 fallback logic:
- Each loader tries local path first
- If not found, constructs S3 path using known bucket structure
- Uses `subprocess.run(["aws", "s3", "cp", ...])` — no boto3 dependency
- Caches S3 downloads to `.cache/s3_downloads/`
- Prints warning when falling back to S3

Key difference from current code: extracts BOTH `_correct` AND `_probs` fields from per-case JSONs using `extract_metrics_from_case()` from `stats/aggregate.py`.

### 3. `analysis/fig1_reproduction.py` — Figure 1

**Panels per spec:**
- A: Standard-scale reproduction (AlphaEdit vs MEMIT through 3K, 5 seeds with uncertainty)
- B: Long-horizon efficacy (1K–10K, individual seed traces + mean)
- C: Probability locality trajectory (both algos, both seeds)
- D: Capability trajectory (MMLU + SST, or probability-locality as substitute)

CLI:
```
python -m analysis.fig1_reproduction [--output-dir results/figures/paper]
```

### 4. `analysis/fig2_forgetting.py` — Figure 2

**Panels per spec:**
- A: Cohort-retention heatmap (x=checkpoint, y=cohort, color=efficacy)
- B: First-1K retention trajectory (seed 42 + 2024 separately)
- C: First-1K, middle, latest-1K curves
- D: Order sensitivity CV at 3K and 7K (and 5K if available)

CLI:
```
python -m analysis.fig2_forgetting [--output-dir results/figures/paper]
```

### 5. `analysis/fig3_coupling.py` — Figure 3

**Panels per spec:**
- A: Stream construction summary (bar plot: reuse rate, unique subjects, overlap)
- B: Old vs recent retention (first-1K/latest-1K for seeds 42, 137, paired)
- C: Retention trajectories + AUC (both seeds, both streams)
- D: Effective rank over edit count (low vs high coupling)

CLI:
```
python -m analysis.fig3_coupling [--output-dir results/figures/paper]
```

### 6. `analysis/fig4_attribution.py` — Figure 4

**Panels per spec:**
- A: Functional projection signal (q_t ≈ latest-cohort efficacy over time, or 1-removed_fraction)
- B: Weight drift negative control (Frobenius drift low/high + retention overlay)
- C: Matched method comparison at 3K (MEMIT vs AlphaEdit vs SeqReg)
- D: Matched method comparison at 5K (AlphaEdit vs SeqReg)

CLI:
```
python -m analysis.fig4_attribution [--output-dir results/figures/paper]
```

### 7. `analysis/tables.py` (new, replaces stats/tables.py for paper)

Produces:
- `table1_reproduction.csv` + LaTeX
- `table2_controlled_coupling.csv` + LaTeX
- `table3_matched_comparison.csv` + LaTeX
- `table4_stream_audit.csv` + LaTeX
- `paper_numbers.json` (all numbers cited in prose)

CLI:
```
python -m analysis.tables [--output-dir results/figures/paper]
```

### 8. `analysis/appendix_figures.py` — Key appendix

Produces:
- A1: Full per-seed failure curves (all seeds, all orderings)
- A3: Full cohort heatmaps (one per trajectory)
- A8: SeqReg mechanism trajectory (cache size, disruption ratio, update norm vs edits)

CLI:
```
python -m analysis.appendix_figures [--output-dir results/figures/appendix]
```

### 9. `analysis/Makefile` — Orchestration

```makefile
OUTPUT := ../results/figures/paper
APPENDIX := ../results/figures/appendix

.PHONY: all fig1 fig2 fig3 fig4 tables appendix clean

all: fig1 fig2 fig3 fig4 tables appendix

fig1:
	uv run python -m analysis.fig1_reproduction --output-dir $(OUTPUT)

fig2:
	uv run python -m analysis.fig2_forgetting --output-dir $(OUTPUT)

fig3:
	uv run python -m analysis.fig3_coupling --output-dir $(OUTPUT)

fig4:
	uv run python -m analysis.fig4_attribution --output-dir $(OUTPUT)

tables:
	uv run python -m analysis.tables --output-dir $(OUTPUT)

appendix:
	uv run python -m analysis.appendix_figures --output-dir $(APPENDIX)

clean:
	rm -f $(OUTPUT)/*.png $(OUTPUT)/*.pdf $(OUTPUT)/*.csv $(OUTPUT)/*.json
```

## Implementation Order

1. `style.py` (10 min) — trivial shared config
2. `loaders.py` (main effort) — shared loading with S3 fallback, prob-locality, cohort computation
3. `fig1_reproduction.py` — uses loaders, produces 4-panel figure
4. `fig2_forgetting.py` — cohort heatmap + order sensitivity
5. `fig3_coupling.py` — controlled coupling panels
6. `fig4_attribution.py` — projection signal + SeqReg comparison
7. `tables.py` — all 4 tables + paper_numbers.json
8. `appendix_figures.py` — A1, A3, A8
9. `Makefile` — orchestration

## Key Design Decisions

- **Import from stats/aggregate.py**: `extract_metrics_from_case()` already handles both `_correct` and `_probs`. Use it everywhere.
- **Cohort computation**: Determined by `case_id // batch_size` (case_id encodes insertion order). This is how `failure_curve_4panel.py` already does it.
- **S3 fallback**: Uses `aws s3 cp` subprocess, not boto3. Caches to avoid repeat downloads. The S3 structure differs from local (has `alphaedit_results/` prefix and tar.gz archives for some experiments) — the loader handles both.
- **No breaking changes**: Existing scripts in `plots/` and `stats/` remain untouched. The old `paper_figures.py` stays but is no longer the primary entry point.
- **Graceful degradation**: Each figure script prints clear messages about missing data and still generates what it can.
