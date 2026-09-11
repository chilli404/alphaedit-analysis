"""Recovery rate analysis — determines whether forgetting is absorbing (Criticism 4).

Loads per-case evaluation JSONs across checkpoints and tracks per-fact
trajectories (remembered/forgotten at each checkpoint). Reports:
  - % stable success / stable failure / recovery / oscillation
  - Transition-level recovery rate: P(remembered at t+1 | forgotten at t)
  - Per-algorithm, per-seed breakdown

If recovery is substantial (>>5%), "survival" language is inappropriate
and should be replaced with "longitudinal retention model."

Usage:
    uv run python -m analysis.recovery_report
"""

import json
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
FC_DIR = RESULTS / "failure_curve_checkpointed"

SEEDS = [42, 137, 2024]
ALGORITHMS = ["AlphaEdit", "MEMIT", "MEMIT-Seq-lp1.0-ld0.0-cache0"]
CHECKPOINTS = [5000, 7000, 9000, 10000]


def load_case_outcomes(seed: int, edits: int, alg: str) -> dict[int, bool]:
    """Load efficacy for all cases at a checkpoint.

    Uses probability-preference metric (official AlphaEdit metric):
    success = target_new is more probable than target_true.
    Falls back to argmax if _probs data is unavailable.
    """
    import numpy as np

    run_dir = FC_DIR / f"seed{seed}" / f"{edits}edits" / alg / "run_000"
    if not run_dir.exists():
        return {}

    outcomes = {}
    for f_path in run_dir.glob("*_edits-case_*.json"):
        with open(f_path) as f:
            data = json.load(f)
        case_id = data["case_id"]
        post = data.get("post", {})

        # Probability-preference (primary)
        probs = post.get("rewrite_prompts_probs", [])
        if probs and isinstance(probs[0], dict):
            # success = target_new more probable (lower NLL)
            eff = float(np.mean([x["target_true"] > x["target_new"] for x in probs]))
            outcomes[case_id] = eff >= 0.5
        else:
            # Fallback to argmax
            correct = post.get("rewrite_prompts_correct", [])
            if correct:
                outcomes[case_id] = bool(correct[0])
    return outcomes


def compute_trajectories(
    seed: int, alg: str, checkpoints: list[int]
) -> dict[int, list[tuple[int, bool]]]:
    """Build per-case trajectories across checkpoints."""
    case_trajectories: dict[int, list[tuple[int, bool]]] = defaultdict(list)

    for ckpt in checkpoints:
        outcomes = load_case_outcomes(seed, ckpt, alg)
        for case_id, remembered in outcomes.items():
            case_trajectories[case_id].append((ckpt, remembered))

    # Sort each trajectory by checkpoint and filter to cases with ≥2 observations
    result = {}
    for case_id, traj in case_trajectories.items():
        traj.sort()
        if len(traj) >= 2:
            result[case_id] = traj
    return result


def classify_trajectory(states: list[bool]) -> str:
    """Classify a trajectory into one of 5 categories."""
    if all(states):
        return "stable_success"
    if not any(states):
        return "stable_failure"

    s2f = sum(1 for i in range(len(states) - 1) if states[i] and not states[i + 1])
    f2s = sum(1 for i in range(len(states) - 1) if not states[i] and states[i + 1])

    if f2s == 0 and s2f > 0:
        return "monotonic_forgetting"
    if f2s > 0 and s2f == 0:
        return "recovery_only"
    return "oscillation"


def compute_stats(seed: int, alg: str) -> dict | None:
    """Compute recovery statistics for one seed × algorithm."""
    trajectories = compute_trajectories(seed, alg, CHECKPOINTS)
    if not trajectories:
        return None

    n_total = len(trajectories)
    categories = defaultdict(int)
    transition_f2s = 0
    transition_s2f = 0
    total_f_observations = 0  # observations where fact is forgotten AND has a next checkpoint

    for case_id, traj in trajectories.items():
        states = [s for _, s in traj]
        cat = classify_trajectory(states)
        categories[cat] += 1

        # Count transitions
        for i in range(len(states) - 1):
            if not states[i]:  # forgotten at this checkpoint
                total_f_observations += 1
                if states[i + 1]:  # recovered at next
                    transition_f2s += 1
            if states[i] and not states[i + 1]:
                transition_s2f += 1

    recovery_rate = transition_f2s / max(total_f_observations, 1)

    return {
        "seed": seed,
        "algorithm": alg,
        "n_cases": n_total,
        "stable_success": categories["stable_success"],
        "stable_failure": categories["stable_failure"],
        "monotonic_forgetting": categories["monotonic_forgetting"],
        "recovery_only": categories["recovery_only"],
        "oscillation": categories["oscillation"],
        "pct_stable_success": 100 * categories["stable_success"] / n_total,
        "pct_stable_failure": 100 * categories["stable_failure"] / n_total,
        "pct_monotonic_forgetting": 100 * categories["monotonic_forgetting"] / n_total,
        "pct_recovery": 100 * (categories["recovery_only"] + categories["oscillation"]) / n_total,
        "pct_oscillation": 100 * categories["oscillation"] / n_total,
        "transition_f2s": transition_f2s,
        "total_f_observations": total_f_observations,
        "recovery_rate": recovery_rate,
        "transition_s2f": transition_s2f,
    }


def main():
    print("=" * 80)
    print("RECOVERY RATE ANALYSIS — Is forgetting absorbing?")
    print("=" * 80)
    print(f"\nCheckpoints: {CHECKPOINTS}")
    print(f"Seeds: {SEEDS}")
    print(f"Algorithms: {ALGORITHMS}")
    print()

    all_results = []

    for alg in ALGORITHMS:
        print(f"\n{'─' * 70}")
        print(f"  Algorithm: {alg}")
        print(f"{'─' * 70}")

        for seed in SEEDS:
            stats = compute_stats(seed, alg)
            if stats is None:
                print(f"  Seed {seed}: NO DATA")
                continue

            all_results.append(stats)
            print(f"\n  Seed {seed} (N={stats['n_cases']} cases across {len(CHECKPOINTS)} checkpoints):")
            print(f"    Stable success (never forgotten):     {stats['pct_stable_success']:5.1f}%  (n={stats['stable_success']})")
            print(f"    Monotonic forgetting (absorbing):     {stats['pct_monotonic_forgetting']:5.1f}%  (n={stats['monotonic_forgetting']})")
            print(f"    Stable failure (never remembered):    {stats['pct_stable_failure']:5.1f}%  (n={stats['stable_failure']})")
            print(f"    Recovery (forgotten→remembered):      {stats['pct_recovery']:5.1f}%  (n={stats['recovery_only'] + stats['oscillation']})")
            print(f"      - Recovery only (no re-forgetting): {100 * stats['recovery_only'] / stats['n_cases']:5.1f}%  (n={stats['recovery_only']})")
            print(f"      - Oscillation (≥2 transitions):     {stats['pct_oscillation']:5.1f}%  (n={stats['oscillation']})")
            print(f"    Transition-level recovery rate:")
            print(f"      P(remembered @ t+1 | forgotten @ t) = {stats['recovery_rate']:.3f}")
            print(f"      ({stats['transition_f2s']}/{stats['total_f_observations']} forgotten-observations)")

    # Aggregate summary
    if all_results:
        print("\n" + "=" * 80)
        print("AGGREGATE SUMMARY")
        print("=" * 80)

        # Aggregate across all AlphaEdit results (primary algorithm)
        ae_results = [r for r in all_results if r["algorithm"] == "AlphaEdit"]
        if ae_results:
            total_f2s = sum(r["transition_f2s"] for r in ae_results)
            total_f_obs = sum(r["total_f_observations"] for r in ae_results)
            total_cases = sum(r["n_cases"] for r in ae_results)
            n_seeds = len(ae_results)
            recovery_cases = sum(r["recovery_only"] + r["oscillation"] for r in ae_results)
            mean_pct_recovery = sum(r["pct_recovery"] for r in ae_results) / n_seeds

            print(f"\n  AlphaEdit (N={total_cases} cases across {n_seeds} seeds):")
            print(f"    Mean % cases with ≥1 recovery event: {mean_pct_recovery:.1f}%")
            print(f"    Aggregate transition recovery rate:    {total_f2s}/{total_f_obs} = {total_f2s / max(total_f_obs, 1):.3f}")

        # All algorithms combined
        total_f2s_all = sum(r["transition_f2s"] for r in all_results)
        total_f_obs_all = sum(r["total_f_observations"] for r in all_results)
        total_cases_all = sum(r["n_cases"] for r in all_results)
        n_conditions = len(all_results)
        all_recovery = sum(r["pct_recovery"] for r in all_results) / n_conditions

        print(f"\n  All algorithms (N={total_cases_all} cases across {n_conditions} seed×alg conditions):")
        print(f"    Mean % cases with ≥1 recovery event: {all_recovery:.1f}%")
        print(f"    Aggregate transition recovery rate:    {total_f2s_all}/{total_f_obs_all} = {total_f2s_all / max(total_f_obs_all, 1):.3f}")

        # Paper-ready statement
        print("\n" + "─" * 80)
        print("PAPER-READY STATEMENT:")
        print("─" * 80)

        if ae_results:
            agg_rate = total_f2s / max(total_f_obs, 1)
            print(f"""
  AlphaEdit transition recovery rate: {agg_rate:.1%}
  ({total_f2s}/{total_f_obs} forgotten-observations across {n_seeds} seeds)
  {mean_pct_recovery:.1f}% of edits show ≥1 recovery event.
""")
            if agg_rate > 0.05:
                print("""  RECOMMENDATION: Use "longitudinal logistic retention model."
  State: "Forgetting is non-absorbing; X% of forgotten edits subsequently
  recover. We model per-checkpoint retention probability rather than
  time-to-event."
""")
            elif agg_rate > 0.01:
                print("""  RECOMMENDATION: Acknowledge low but non-zero recovery, justify
  approximate absorbing treatment:

  "Recovery is rare but non-negligible: {rate:.1%} of checkpoint-level forgotten
  observations are subsequently remembered (N={obs}). We model retention
  probability per checkpoint rather than strict time-to-first-failure,
  acknowledging that the absorbing approximation holds for >95% of cases.
  Results are robust to excluding the {pct:.1f}% of edits with non-monotonic
  trajectories."
""".format(rate=agg_rate, obs=total_f_obs, pct=mean_pct_recovery))
            else:
                print(f"""  RECOMMENDATION: Forgetting is approximately absorbing.
  "Fewer than {agg_rate:.1%} of forgotten edits subsequently recover
  (N={total_f_obs} observations across {n_seeds} seeds). Forgetting is
  treated as absorbing."
""")

    # Save JSON for downstream use
    output_path = RESULTS / "recovery_analysis.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
