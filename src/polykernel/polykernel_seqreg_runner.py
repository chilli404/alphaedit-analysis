#!/usr/bin/env python3
"""
Polykernel-Augmented MEMIT+SeqReg: Kernel-weighted sequential regularization.

Uses evaluate_harness.run_experiment() + composable algorithm hooks instead of
exec(compile(evaluate.py)) + exec(compile(memit_main.py)).

Supports multiple base algorithms via --base_alg:
  - MEMIT (default): apply_memit_with_hooks + seqreg_hooks
  - AlphaEdit: apply_alphaedit_with_hooks + seqreg_hooks
  - NSE / MEMIT_rect: vendor apply functions with post-hoc REVIVE wrapper

REVIVE spectral filter composes with any base: compose_hooks(seqreg, revive)

Usage:
    python src/polykernel/polykernel_seqreg_runner.py \
        --seed 42 --dataset_size_limit 2000 --num_edits 100 \
        --lambda_prev 1.0 --lambda_delta 1.0 \
        --kernel_type poly --kernel_degree 2 \
        --fast_checkpoint --save_interval 10

    # With REVIVE:
    python src/polykernel/polykernel_seqreg_runner.py \
        --seed 42 --base_alg MEMIT --revive --revive_tau 0.1 \
        --dataset_size_limit 10000 --num_edits 100 \
        --ordering fb_high_exposure \
        --dataset_override results/matched_ordering/orderings/fb_high_exposure_seed42.json
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

_SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC_DIR / "util"))
sys.path.insert(0, str(_SRC_DIR))

from model_registry import DEFAULT_MODEL
from model_resolve import resolve_model_path
from setup_hparams import link_hparams
from eval_config import hash_eval_config
from paths import get_project_root, get_alphaedit_root, get_result_root, get_checkpoint_root
from evaluate_harness import (
    run_experiment, load_model_and_tok, load_dataset,
    ExperimentHooks,
)
from checkpoint_io import (
    save_checkpoint, load_checkpoint, find_latest_checkpoint,
    should_save, should_skip, validate_checkpoint_path,
)
from mega_batch_eval import get_mega_batch_eval_source
from experiment_config import ExperimentConfig
from algorithms.hooks import AlgorithmHooks, compose_hooks
from algorithms.hook_presets import seqreg_hooks, revive_hooks, polykernel_hooks
from algorithms.memit_with_hooks import apply_memit_with_hooks
from algorithms.alphaedit_with_hooks import apply_alphaedit_with_hooks


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _make_eval_fn(fast_mode: bool = False):
    ns = {}
    exec(get_mega_batch_eval_source(), ns)
    mbe_fn = ns["_mega_batch_eval"]

    if fast_mode:
        def fast_eval(model, tok, records, template, num_edits, case_ids, exec_time):
            batch_records = [r for r in records if r["case_id"] in case_ids[-num_edits:]]
            mbe_fn(model, tok, batch_records, template, num_edits, case_ids, exec_time, batch_size=2)
        return fast_eval
    return mbe_fn


def run(args: argparse.Namespace) -> None:
    alphaedit_root = get_alphaedit_root()
    if not alphaedit_root.exists():
        print(f"ERROR: AlphaEdit not found at {alphaedit_root}")
        sys.exit(1)

    link_hparams()
    _seed_everything(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Parse cache_max
    cache_max = None if args.cache_max == "none" else int(args.cache_max)
    ordering = getattr(args, 'ordering', None)

    # Build config — SINGLE source of truth for variant_name and paths
    exp_config = ExperimentConfig(
        base_alg=args.base_alg, seed=args.seed, ordering=ordering,
        model_name=args.model_name,
        lambda_prev=args.lambda_prev, lambda_delta=args.lambda_delta,
        cache_max=cache_max, kernel_type=args.kernel_type,
        kernel_degree=args.kernel_degree, kernel_sigma=args.kernel_sigma,
        kernel_prev=args.kernel_prev,
        revive=args.revive, revive_tau=args.revive_tau,
        experiment_type="polykernel_seqreg",
    )
    variant_name = exp_config.variant_name

    # Checkpoint directory
    if args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir)
    else:
        ckpt_dir = exp_config.checkpoint_dir()
    validate_checkpoint_path(str(ckpt_dir), args.base_alg)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Results directory
    model_name = resolve_model_path(args.model_name)
    _mn = (args.model_name or "").lower()
    if "gpt-j" in _mn:
        _exp_name = "failure_curve_gptj"
    elif "qwen2.5-7b" in _mn:
        _exp_name = "failure_curve_qwen"
    else:
        _exp_name = "failure_curve_checkpointed"

    if ordering:
        results_dir = (
            get_result_root() / "matched_ordering" / ordering
            / f"seed{args.seed}" / f"{args.dataset_size_limit}edits"
        )
    else:
        results_dir = (
            get_result_root() / _exp_name
            / f"seed{args.seed}" / f"{args.dataset_size_limit}edits"
        )
    results_dir.mkdir(parents=True, exist_ok=True)
    variant_dir = results_dir / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_jsonl = variant_dir / f"log_seed{args.seed}_{timestamp}.jsonl"

    # Auto-detect resume point
    total_batches = args.dataset_size_limit // args.num_edits
    start_from_batch = args.start_from_batch
    if start_from_batch < 0:
        latest = find_latest_checkpoint(ckpt_dir)
        if latest:
            start_from_batch = latest[0] + 1
            if start_from_batch >= total_batches:
                start_from_batch = total_batches
                print(f"  Auto-detected: checkpoint at batch {latest[0]} covers all {total_batches} batches. Will run eval only.")
            else:
                print(f"  Auto-detected: resume from batch {start_from_batch} (checkpoint at batch {latest[0]})")
        else:
            start_from_batch = 0
            print("  No existing checkpoints found. Starting from batch 0.")

    # Print config
    eval_mode = "End only" if args.eval_at_end_only else ("Milestone" if args.eval_at_checkpoints_only else ("Fast" if args.fast_checkpoint else "Full"))
    kernel_mode = "HYBRID (kernel current only, linear K_prev)" if not args.kernel_prev else "full (kernel both)"
    print(f"\n{'=' * 70}")
    print("Polykernel+SeqReg Runner (harness-based)")
    print(f"  Seed:           {args.seed}")
    print(f"  Base algorithm: {args.base_alg}")
    print(f"  Kernel:         {args.kernel_type} (degree={args.kernel_degree}, sigma={args.kernel_sigma})")
    print(f"  Kernel mode:    {kernel_mode}")
    print(f"  lambda_prev:    {args.lambda_prev}")
    print(f"  lambda_delta:   {args.lambda_delta}")
    print(f"  Cache strategy: {args.cache_strategy}")
    print(f"  Cache max:      {cache_max}")
    print(f"  Dataset:        {args.ds_name} (limit={args.dataset_size_limit})")
    print(f"  Num edits:      {args.num_edits}")
    print(f"  Total batches:  {total_batches}")
    print(f"  Resume from:    batch {start_from_batch} ({start_from_batch * args.num_edits} edits)")
    print(f"  Save interval:  every {args.save_interval} batches")
    print(f"  Evaluation:     {eval_mode}")
    print(f"  Checkpoint dir: {ckpt_dir}")
    print(f"  Model:          {args.model_name}")
    if args.dataset_override:
        print(f"  Dataset override: {args.dataset_override}")
    if args.revive:
        print(f"  REVIVE:         ENABLED (tau={args.revive_tau}, mode={args.revive_mode})")
        print(f"  REVIVE SVD:     device={args.revive_svd_device}, dtype={args.revive_svd_dtype}")
    print(f"  Variant:        {variant_name}")
    print(f"  Output:         {output_jsonl}")
    print(f"  Started:        {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 70}")

    # Load model + tokenizer
    model, tok = load_model_and_tok(args.model_name)

    # Load dataset
    dataset = load_dataset(
        args.ds_name, args.dataset_size_limit, alphaedit_root,
        dataset_override=args.dataset_override,
    )

    # Shuffle if order_id > 0
    if args.order_id > 0:
        _r = random.Random(args.order_id)
        _r.shuffle(dataset)
        print(f"  Dataset shuffled with order_id={args.order_id}")

    # Pre-populate vendor globals before importing vendor modules
    from evaluate_harness import _ensure_vendor_globals
    _ensure_vendor_globals(alphaedit_root)

    # Load hparams — each base algorithm has its own HyperParams class and directory
    sys.path.insert(0, str(alphaedit_root))
    if args.base_alg == "AlphaEdit":
        from AlphaEdit import AlphaEditHyperParams as HParams
        alg_for_hparams = "AlphaEdit"
    elif args.base_alg == "NSE":
        from nse import NSEHyperParams as HParams
        alg_for_hparams = "NSE"
    elif args.base_alg == "MEMIT_rect":
        from memit import MEMITHyperParams as HParams
        alg_for_hparams = "MEMIT"
    else:
        from memit import MEMITHyperParams as HParams
        alg_for_hparams = "MEMIT"
    hparams_path = alphaedit_root / "hparams" / alg_for_hparams / args.hparams_fname
    hparams = HParams.from_json(hparams_path)

    # Build algorithm hooks — compose seqreg + optional revive + optional polykernel
    hook_list = []

    # SeqReg augmentation (K_prev regularization + key caching)
    hook_list.append(seqreg_hooks(
        lambda_prev=args.lambda_prev,
        lambda_delta=args.lambda_delta,
        cache_strategy=args.cache_strategy,
        cache_max=cache_max,
        kernel_type=args.kernel_type,
        kernel_degree=args.kernel_degree,
        kernel_sigma=args.kernel_sigma,
        kernel_prev=args.kernel_prev,
    ))

    # REVIVE spectral filter
    _revive_hook = None
    if args.revive:
        _revive_hook = revive_hooks(
            revive_tau=args.revive_tau,
            revive_svd_device=args.revive_svd_device,
            revive_svd_dtype=args.revive_svd_dtype,
        )
        hook_list.append(_revive_hook)

    algo_hooks = compose_hooks(*hook_list)
    algo_state = algo_hooks.get_state()

    # Load checkpoint if resuming
    cache_c = None
    error_cache = None
    if start_from_batch > 0:
        extra_keys = ["prev_cache.pt", "mechanism_log.jsonl"]
        if args.base_alg in ("AlphaEdit", "MEMIT"):
            extra_keys.append("cache_c.pt")
        if args.base_alg == "MEMIT_rect":
            extra_keys.append("error_cache.pt")
        ckpt_result = load_checkpoint(
            model, hparams, str(ckpt_dir), start_from_batch - 1,
            extra_state_keys=extra_keys,
        )
        if ckpt_result.get("prev_cache.pt") is not None:
            algo_state["prev_cache"] = ckpt_result["prev_cache.pt"]
            total_keys = sum(sum(k.shape[1] for k in v) for v in algo_state["prev_cache"].values())
            print(f"  [CHECKPOINT] Loaded prev_cache ({len(algo_state['prev_cache'])} layers, {total_keys} total keys)")
        if ckpt_result.get("mechanism_log.jsonl") is not None:
            algo_state["mechanism_log"] = ckpt_result["mechanism_log.jsonl"]
            print(f"  [CHECKPOINT] Loaded {len(algo_state['mechanism_log'])} log entries")
        if ckpt_result.get("cache_c.pt") is not None:
            cache_c = ckpt_result["cache_c.pt"]
            print(f"  [CHECKPOINT] Loaded cache_c (shape: {cache_c.shape})")
        if ckpt_result.get("error_cache.pt") is not None:
            error_cache = ckpt_result["error_cache.pt"]
            print(f"  [CHECKPOINT] Loaded error_cache (shape: {error_cache.shape})")
        algo_state["batch_idx"] = [start_from_batch]

    # Select apply function based on base_alg
    if args.base_alg == "MEMIT":
        base_apply = apply_memit_with_hooks
    elif args.base_alg == "AlphaEdit":
        base_apply = apply_alphaedit_with_hooks
    elif args.base_alg == "NSE":
        from nse.nse_main import apply_nse_to_model
        base_apply = apply_nse_to_model
    elif args.base_alg == "MEMIT_rect":
        # RECT module uses relative imports (from .compute_ks) so we must import
        # it as part of the baselines memit package, not standalone
        baselines_root = str(get_project_root() / "baselines" / "EvoEdit")
        old_memit = sys.modules.pop("memit", None)
        sys.path.insert(0, baselines_root)
        from memit.memit_seq_rect_main import apply_memit_seq_rect_to_model
        base_apply = apply_memit_seq_rect_to_model
        sys.path.remove(baselines_root)
        if old_memit is not None:
            sys.modules["memit"] = old_memit
    else:
        raise ValueError(f"Unknown base_alg: {args.base_alg}")

    # Initialize cache_c and P for algorithms that need them
    cache_c = None
    P = None
    error_cache = None
    n_layers = len(hparams.layers)

    if args.base_alg == "AlphaEdit":
        # AlphaEdit needs both P (null-space projection) and cache_c.
        # link_stats.sh copies P to alphaedit_root/null_space_project.pt (local, not S3).
        p_path = alphaedit_root / "null_space_project.pt"
        if p_path.exists():
            P = torch.load(str(p_path), map_location="cpu")
            print(f"  [AlphaEdit] Loaded P matrix from {p_path} (shape: {P.shape})")
            d = P.shape[-1]
            cache_c = torch.zeros(n_layers, d, d)
        else:
            raise FileNotFoundError(
                f"P matrix not found at {p_path}. Run link_stats.sh first."
            )

    elif args.base_alg == "NSE":
        # NSE needs cache_c but NOT P. NSE indexes cache_c with neuron indices
        # which correspond to the INPUT dimension of down_proj (shape[1]), not output (shape[0])
        sample_layer = hparams.layers[0]
        weight_name = f"{hparams.rewrite_module_tmp.format(sample_layer)}.weight"
        d = dict(model.named_parameters())[weight_name].shape[1]  # INPUT dim for NSE
        cache_c = torch.zeros(n_layers, d, d)
        print(f"  [NSE] Initialized cache_c: ({n_layers}, {d}, {d})")

    elif args.base_alg == "MEMIT_rect":
        # RECT's execute_memit adds cache_c and error_cache to cov + K@K^T.
        # Both cov and K are in the INPUT dimension of down_proj.
        sample_layer = hparams.layers[0]
        weight_name = f"{hparams.rewrite_module_tmp.format(sample_layer)}.weight"
        w = dict(model.named_parameters())[weight_name]
        d_in = w.shape[1]  # INPUT dim for cache_c (cov + K@K^T space)
        cache_c = torch.zeros(n_layers, d_in, d_in)
        # error_cache accumulates rectification residuals — same shape as upd_matrix
        # upd_matrix shape depends on weight shape: [d_out, d_in] or [d_in, d_out]
        error_cache = torch.zeros(n_layers, w.shape[0], w.shape[1])
        print(f"  [RECT] Initialized cache_c: ({n_layers}, {d_in}, {d_in}), "
              f"error_cache: ({n_layers}, {w.shape[0]}, {w.shape[1]})")

    # MEMIT and AlphaEdit accept hooks= kwarg and call post_solve internally.
    # NSE and RECT are vendor functions that silently ignore hooks via **_kwargs.
    # For REVIVE on NSE/RECT, we apply the spectral filter post-hoc on weight deltas.
    _hooks_aware = args.base_alg in ("MEMIT", "AlphaEdit")

    # NSE uses precomputed v_star from W₀. Without these, compute_z does 25 gradient
    # steps per edit (~1.5s each), making runs 5x slower. Extract from S3 tar if needed.
    _nse_cache_template = None
    if args.base_alg == "NSE":
        _kvs_dir = alphaedit_root / "share" / "projects" / "rewriting-knowledge" / "kvs"
        if not _kvs_dir.exists():
            _kvs_dir = get_project_root() / "baselines" / "EvoEdit" / "share" / "projects" / "rewriting-knowledge" / "kvs"

        _nse_model_dir = _kvs_dir / f"{args.model_name.replace('/', '_')}_NSE"
        _nse_npz_count = len(list(_nse_model_dir.glob("*.npz"))) if _nse_model_dir.exists() else 0

        if _nse_npz_count < 100:
            _tar_dir = Path("/s3-data/continual-learning/alphaedit/nse_kv_cache")
            if not _tar_dir.exists():
                _tar_dir = get_project_root() / "data" / "nse_kv_cache"
            if _tar_dir.exists():
                import tarfile
                _kvs_dir.mkdir(parents=True, exist_ok=True)
                for _tar_path in sorted(_tar_dir.glob("*.tar")):
                    print(f"  [NSE] Extracting KV cache from {_tar_path.name}...")
                    with tarfile.open(str(_tar_path), "r") as tf:
                        tf.extractall(str(_kvs_dir))
                # Symlink model name variants so canonical name resolves
                _model_canonical = args.model_name.replace("/", "_") + "_NSE"
                if not (_kvs_dir / _model_canonical).exists():
                    for _d in _kvs_dir.iterdir():
                        if _d.is_dir() and _d.name.endswith("_NSE") and _d.name != _model_canonical:
                            (_kvs_dir / _model_canonical).symlink_to(_d.name)
                            break
                _nse_npz_count = len(list(_nse_model_dir.glob("*.npz"))) if _nse_model_dir.exists() else 0
                print(f"  [NSE] KV cache: {_nse_npz_count} files extracted")
            else:
                print(f"  [NSE] WARNING: No KV cache tar at {_tar_dir}. "
                      f"compute_z will use slow gradient optimization (25 steps/edit).")

        _nse_cache_template = str(
            _nse_model_dir / f"{args.ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
        )
        print(f"  [NSE] Cache template: {_nse_cache_template} ({_nse_npz_count} cached)")

    def apply_fn(model, tok, requests, hparams, **kwargs):
        nonlocal error_cache
        extra = {}
        if cache_c is not None:
            extra["cache_c"] = cache_c
        if P is not None:
            extra["P"] = P
        if _nse_cache_template is not None:
            extra["cache_template"] = _nse_cache_template
        if error_cache is not None:
            extra["error_cache"] = error_cache

        if _hooks_aware:
            if args.revive:
                weights_dict = {}
                for layer in hparams.layers:
                    wn = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
                    param = dict(model.named_parameters()).get(wn)
                    if param is not None:
                        weights_dict[wn] = param.data.detach().clone()
                algo_state["_current_weights"] = weights_dict

            return base_apply(
                model, tok, requests, hparams,
                hooks=algo_hooks, state=algo_state,
                **extra, **kwargs,
            )

        # Non-hook path: NSE / RECT vendor functions
        if args.revive:
            pre_weights = {}
            params = dict(model.named_parameters())
            for layer in hparams.layers:
                wn = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
                if wn in params:
                    pre_weights[wn] = params[wn].data.detach().clone()

        result = base_apply(model, tok, requests, hparams, **extra, **kwargs)

        # RECT returns (model, cache_c, error_cache) — capture error_cache
        if args.base_alg == "MEMIT_rect" and isinstance(result, tuple) and len(result) >= 3:
            error_cache = result[2]
            result = (result[0], result[1])  # normalize to 2-tuple for harness

        if args.revive and pre_weights:
            params = dict(model.named_parameters())
            _rv_applied = 0
            for layer in hparams.layers:
                wn = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
                if wn not in params or wn not in pre_weights:
                    continue
                delta = params[wn].data.double() - pre_weights[wn].double()
                dnorm = delta.norm().item()
                if dnorm < 1e-10:
                    continue
                state_rv = {"_current_weights": {wn: pre_weights[wn]}}
                filtered = _revive_hook.post_solve(layer, delta, None, None, wn, state_rv)
                with torch.no_grad():
                    params[wn].data.copy_(pre_weights[wn] + filtered.to(params[wn].dtype))
                _rv_applied += 1
            if _rv_applied == 0:
                print(f"  [REVIVE] WARNING: post-hoc filter applied to 0 layers "
                      f"(all deltas < 1e-10). base_alg={args.base_alg}", flush=True)

        return result

    # Build eval function
    mbe_fn = _make_eval_fn(fast_mode=args.fast_checkpoint)

    # Build experiment hooks
    def after_edit(batch_idx, model, records, hparams, edit_extra, exec_time):
        nonlocal cache_c
        algo_state["batch_idx"] = [batch_idx + 1]

        # AlphaEdit, NSE, and RECT all return cache_c — capture the updated value
        if edit_extra is not None and args.base_alg in ("AlphaEdit", "NSE", "MEMIT_rect"):
            cache_c = edit_extra

        if should_save(batch_idx, args.save_interval):
            extra_state = {
                "prev_cache.pt": algo_state.get("prev_cache", {}),
                "mechanism_log.jsonl": algo_state.get("mechanism_log", []),
            }
            if cache_c is not None:
                extra_state["cache_c.pt"] = cache_c
            if error_cache is not None:
                extra_state["error_cache.pt"] = error_cache
            save_checkpoint(
                batch_idx, model, hparams, str(ckpt_dir), args.num_edits,
                extra_state=extra_state,
                metadata={
                    "lambda_prev": args.lambda_prev,
                    "lambda_delta": args.lambda_delta,
                    "cache_strategy": args.cache_strategy,
                    "kernel_type": args.kernel_type,
                    "kernel_degree": args.kernel_degree,
                    "kernel_sigma": args.kernel_sigma,
                    "kernel_prev": args.kernel_prev,
                    "base_alg": args.base_alg,
                    "revive": args.revive,
                    "revive_tau": args.revive_tau if args.revive else None,
                    "batch_idx_counter": algo_state["batch_idx"][0],
                },
                base_alg=args.base_alg,
            )

        if (batch_idx + 1) % 10 == 0:
            total_edits = (batch_idx + 1) * args.num_edits
            print(f"{'=' * 30}{total_edits}_edit{'=' * 30}", flush=True)
        print(f"Execution took {exec_time}", flush=True)

    total_batches = len(dataset) // args.num_edits

    def should_eval_fn(batch_idx):
        if args.eval_at_end_only:
            return batch_idx == total_batches - 1
        if args.eval_at_checkpoints_only:
            return should_save(batch_idx, args.save_interval)
        return True

    hooks = ExperimentHooks(
        after_edit=after_edit,
        should_eval=should_eval_fn,
        eval_fn=mbe_fn,
    )

    alg_results_dir = variant_dir

    # Run
    summary = run_experiment(
        model=model,
        tok=tok,
        hparams=hparams,
        dataset=dataset,
        apply_fn=apply_fn,
        alg_name=args.base_alg,
        num_edits=args.num_edits,
        results_dir=alg_results_dir,
        ds_name=args.ds_name,
        conserve_memory=args.conserve_memory,
        hooks=hooks,
    )

    # Write mechanism log (re-ensure parent dir exists — S3 FUSE may drop empty dirs)
    log_entries = algo_state.get("mechanism_log", [])
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w") as f:
        for entry in log_entries:
            f.write(json.dumps(entry) + "\n")
    print(f"\n[SeqReg] Log written: {output_jsonl} ({len(log_entries)} entries)")

    # Save metadata
    metadata = {
        "experiment": "polykernel_seqreg",
        "seed": args.seed,
        "order_id": args.order_id,
        "lambda_prev": args.lambda_prev,
        "lambda_delta": args.lambda_delta,
        "cache_strategy": args.cache_strategy,
        "cache_max": cache_max,
        "kernel_type": args.kernel_type,
        "kernel_degree": args.kernel_degree,
        "kernel_sigma": args.kernel_sigma,
        "kernel_prev": args.kernel_prev,
        "base_alg": args.base_alg,
        "revive": args.revive,
        "revive_tau": args.revive_tau if args.revive else None,
        "revive_mode": args.revive_mode if args.revive else None,
        "model_name": args.model_name,
        "hparams_fname": args.hparams_fname,
        "ds_name": args.ds_name,
        "dataset_size_limit": args.dataset_size_limit,
        "num_edits": args.num_edits,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "alphaedit_commit": "b84624f",
        "eval_config_hash": hash_eval_config(),
        "runner": "polykernel_seqreg_runner (harness-based)",
        "variant_name": variant_name,
    }
    meta_path = variant_dir / f"metadata_seed{args.seed}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 70}")
    print("Polykernel+SeqReg completed.")
    print(f"  Finished:  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Batches:   {summary['batches_run']}")
    print(f"  Edits:     {summary['total_edits']}")
    print(f"  Log:       {output_jsonl}")
    print(f"  Metadata:  {meta_path}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Polykernel+SeqReg: kernel-weighted sequential regularization for MEMIT"
    )

    # Seed and hardware
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cuda_device", default="0")

    # Model and data
    parser.add_argument("--model_name", default=os.environ.get("MODEL_NAME", DEFAULT_MODEL))
    parser.add_argument("--hparams_fname", default="Llama3-8B.json")
    parser.add_argument("--ds_name", default="mcf", choices=["mcf", "cf", "zsre"])
    parser.add_argument("--dataset_size_limit", type=int, default=2000)
    parser.add_argument("--num_edits", type=int, default=100)
    parser.add_argument("--downstream_eval_steps", type=int, default=10)
    parser.add_argument("--conserve_memory", action="store_true", default=True)

    # Base algorithm
    parser.add_argument("--base_alg", default="MEMIT", choices=["MEMIT", "AlphaEdit", "NSE", "MEMIT_rect"],
                        help="Base editing algorithm (default: MEMIT)")

    # SeqReg parameters
    parser.add_argument("--lambda_prev", type=float, default=1.0)
    parser.add_argument("--lambda_delta", type=float, default=1.0)
    parser.add_argument("--cache_strategy", default="all", choices=["recent", "all"])
    parser.add_argument("--cache_max", default="none")

    # Kernel parameters
    parser.add_argument("--kernel_type", choices=["poly", "rbf"], default="poly")
    parser.add_argument("--kernel_degree", type=int, default=2)
    parser.add_argument("--kernel_sigma", default="median")
    parser.add_argument("--no_kernel_prev", dest="kernel_prev", action="store_false")
    parser.set_defaults(kernel_prev=True)

    # REVIVE spectral subspace filter
    parser.add_argument("--revive", action="store_true")
    parser.add_argument("--revive_tau", type=float, default=0.1)
    parser.add_argument("--revive_svd_device", default="cuda")
    parser.add_argument("--revive_svd_dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--revive_cache_dir", default=None)
    parser.add_argument("--revive_log_interval", type=int, default=1)
    parser.add_argument("--revive_mode", default="hard", choices=["hard"])

    # Checkpoint and resume
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--start_from_batch", type=int, default=-1)

    # Evaluation modes
    eval_group = parser.add_mutually_exclusive_group()
    eval_group.add_argument("--fast_checkpoint", action="store_true")
    eval_group.add_argument("--eval_at_checkpoints_only", action="store_true")
    eval_group.add_argument("--eval_at_end_only", action="store_true",
                            help="Only evaluate at the final batch (saves checkpoints normally)")

    # Dataset override and ordering
    parser.add_argument("--dataset_override", type=str, default=None)
    parser.add_argument("--order_id", type=int, default=0)
    parser.add_argument("--ordering", type=str, default=None)

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
