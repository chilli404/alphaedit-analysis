"""Batched multi-token evaluation for MCF and zsRE datasets.

Replaces the vendor's per-record evaluation loop with batched forward passes.
~4-10x faster than the vendor loop which does 26 separate forward passes per record.

This is the SINGLE authoritative copy — all runners should use this instead of
their own inline copy. Produces both prob-pref and argmax metrics per record.

Supports:
  - MCF (MultiCounterFact): pairwise NLL comparison (efficacy, paraphrase, neighborhood)
  - zsRE: per-token argmax accuracy + pairwise NLL

Usage (inside injected evaluate.py code):
    _mega_batch_eval(model, tok, records, case_result_template, num_edits, case_ids, exec_time)

Usage (standalone):
    from util.mega_batch_eval import mega_batch_eval
    mega_batch_eval(model, tok, records, output_dir, num_edits=100, ds_name="mcf")
"""

MEGA_BATCH_EVAL_SOURCE = r'''
def _mega_batch_eval(model, tok, records, case_result_template, num_edits, case_ids, exec_time, batch_size=4, ds_name="mcf"):
    """Batched multi-token scoring with prob-pref + argmax dual metrics.

    Evaluates `records` in batches of `batch_size`, writing per-case JSON files.
    Each JSON contains both prob-pref (official metric) and argmax metrics.
    """
    import torch as _mbe_torch
    import numpy as _mbe_np
    import json as _mbe_json
    from itertools import chain as _mbe_chain
    from time import time as _mbe_time
    from pathlib import Path as _mbe_Path

    _is_llama = 'llama' in model.config._name_or_path.lower()
    _mbe_start = _mbe_time()
    _mbe_total = len(records)
    _mbe_done = 0
    _mbe_skipped = 0

    for batch_start in range(0, _mbe_total, batch_size):
        batch_records = records[batch_start:batch_start + batch_size]

        # --- Phase 1: Collect all sequences for this batch ---
        all_sequences = []
        record_meta = []

        for record in batch_records:
            out_file = _mbe_Path(case_result_template.format(num_edits, record["case_id"]))
            if out_file.exists():
                record_meta.append(None)
                _mbe_skipped += 1
                continue

            subject = record["requested_rewrite"]["subject"]
            target_new = record["requested_rewrite"]["target_new"]["str"]
            target_true = record["requested_rewrite"]["target_true"]["str"]

            rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
            paraphrase_prompts = record.get("paraphrase_prompts", [])
            neighborhood_prompts = record.get("neighborhood_prompts", [])

            prefixes = rewrite_prompts + paraphrase_prompts + neighborhood_prompts
            which_correct = (
                [0] * len(rewrite_prompts)
                + [0] * len(paraphrase_prompts)
                + [1] * len(neighborhood_prompts)
            )

            a_tok = tok(f" {target_new}")["input_ids"]
            b_tok = tok(f" {target_true}")["input_ids"]
            if _is_llama:
                a_tok = a_tok[1:]
                b_tok = b_tok[1:]

            prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
            if _is_llama:
                prefix_lens = [l - 1 for l in prefix_lens]

            seqs = [f"{prefix} {suffix}" for prefix in prefixes for suffix in [target_new, target_true]]
            seq_idx_start = len(all_sequences)
            all_sequences.extend(seqs)

            record_meta.append({
                "record": record,
                "out_file": str(out_file),
                "n_prefixes": len(prefixes),
                "prefix_lens": prefix_lens,
                "which_correct": which_correct,
                "a_tok": a_tok,
                "b_tok": b_tok,
                "seq_idx_start": seq_idx_start,
                "n_seqs": len(seqs),
                "target_new": target_new,
                "target_true": target_true,
            })

        if not all_sequences:
            _mbe_done += len(batch_records)
            continue

        # --- Phase 2: Batched forward pass ---
        tok_out = tok(all_sequences, padding=True, return_tensors="pt").to(model.device)
        with _mbe_torch.no_grad():
            logits = model(**tok_out).logits

        # --- Phase 3: Score each record ---
        for meta in record_meta:
            if meta is None:
                continue

            s = meta["seq_idx_start"]
            n = meta["n_seqs"]
            n_pref = meta["n_prefixes"]

            probs = _mbe_np.zeros(n, dtype=_mbe_np.float32)
            targets_correct = []

            for i in range(n):
                seq_logits = logits[s + i]
                is_a = (i % 2 == 0)
                tgt_tok = meta["a_tok"] if is_a else meta["b_tok"]
                pref_idx = i // 2
                pref_len = meta["prefix_lens"][pref_idx]
                cur_len = len(tgt_tok)

                nll = 0.0
                for j in range(cur_len):
                    nll += -_mbe_torch.nn.functional.log_softmax(
                        seq_logits[pref_len + j - 1], dim=0
                    )[tgt_tok[j]].item()
                probs[i] = nll / cur_len

                # Argmax correctness
                wc = meta["which_correct"][pref_idx]
                if (wc == 0 and is_a) or (wc == 1 and not is_a):
                    correct = True
                    for j in range(cur_len):
                        if seq_logits[pref_len + j - 1, :].argmax().item() != tgt_tok[j]:
                            correct = False
                            break
                    targets_correct.append(correct)
                else:
                    targets_correct.append(False)

            # Build per-prompt prob dicts and correctness
            ret_probs = []
            ret_corrects = []
            for p_idx in range(n_pref):
                base = p_idx * 2
                ret_probs.append({
                    "target_new": float(probs[base]),
                    "target_true": float(probs[base + 1]),
                })
                ret_corrects.append(targets_correct[base] if base < len(targets_correct) else False)

            # Split by prompt type
            n_rw = 1
            n_para = len(meta["record"].get("paraphrase_prompts", []))
            n_neigh = len(meta["record"].get("neighborhood_prompts", []))

            rw_probs = ret_probs[:n_rw]
            rw_correct = ret_corrects[:n_rw]
            para_probs = ret_probs[n_rw:n_rw+n_para]
            para_correct = ret_corrects[n_rw:n_rw+n_para]
            neigh_probs = ret_probs[n_rw+n_para:]
            neigh_correct = ret_corrects[n_rw+n_para:]

            post = {
                "rewrite_prompts_probs": rw_probs,
                "paraphrase_prompts_probs": para_probs,
                "neighborhood_prompts_probs": neigh_probs,
                "rewrite_prompts_correct": rw_correct,
                "paraphrase_prompts_correct": para_correct,
                "neighborhood_prompts_correct": neigh_correct,
            }

            metrics = {
                "case_id": meta["record"]["case_id"],
                "grouped_case_ids": case_ids,
                "num_edits": num_edits,
                "requested_rewrite": meta["record"]["requested_rewrite"],
                "time": exec_time,
                "post": post,
            }
            with open(meta["out_file"], "w") as f:
                _mbe_json.dump(metrics, f, indent=1)

        del tok_out, logits
        _mbe_torch.cuda.empty_cache()

        _mbe_done += len(batch_records)
        _elapsed = _mbe_time() - _mbe_start
        _rate = _mbe_done / _elapsed if _elapsed > 0 else 0
        if _mbe_done % (batch_size * 4) < batch_size or _mbe_done >= _mbe_total:
            print(f"  [MEGA-BATCH EVAL] {_mbe_done}/{_mbe_total} records "
                  f"({_mbe_skipped} skipped, {_rate:.1f} rec/s, "
                  f"{_elapsed:.0f}s elapsed)")

    print(f"  [MEGA-BATCH EVAL] Complete: {_mbe_total} records in {_mbe_time() - _mbe_start:.1f}s")
'''


def get_mega_batch_eval_source() -> str:
    """Return the mega_batch_eval function as injectable source code."""
    return MEGA_BATCH_EVAL_SOURCE
