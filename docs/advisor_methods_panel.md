# Advisor: Cross-Method Panel Recommendation (2026-09-09)

## Target ICLR Method Panel

| Method | Mechanism | Priority | Status |
|---|---|---|---|
| **AlphaEdit** | Fixed null-space projection | Required | Done (3 seeds) |
| **MEMIT-Seq** | Accumulated-history regularization | Required | Done (s42), running (s2024/s137) |
| **EvoEdit** | Evolving sequential null-space alignment | Required | Done (s42), running (s2024/s137) |
| **NSE** | Neuron-level selection + original-weight optimization | **Add now** | Not started |
| **REVIVE or SpecEdit** | Spectral preservation/decoupling | Keep one | REVIVE has checkpoint issues |
| **QueueEDIT or AlphaEdit+** | Active historical realignment / conflict-aware | Add if feasible | Not started |

**Do NOT include PathGuard or Poly2-Hybrid** — those go in the separate methods paper.

## Why NSE Is Highest Priority

- Specifically designed for sequential editing (ACL 2025 Main)
- Methodologically distinct: neuron-level selection, not null-space projection
- Official repo supports Llama-3-8B, MCF, batch_size=100
- Our 10K stress test would be a NEW extension (their paper uses 2K)
- Tests whether path fragility persists beyond null-space method family

NSE repo: https://github.com/jianghoucheng/NSE

## Minimum Experiment Per New Method

- HIGH and RANDOM0 orderings
- Seeds 42 and 2024
- 10K edits
- Same evaluator (prob-pref v2)
- Same checkpoint intervals

Primary statistic: Δ_path = Efficacy_HIGH - Efficacy_RANDOM

## Hyperparameter Rules

1. Use authors' released hyperparameters unchanged
2. If tuning needed, tune on seed-42 RANDOM only, then freeze
3. Evaluate HIGH on seeds 2024/137 as holdout

## What The Additional Methods Would Let Us Claim

Current (3 methods): "Temporal-path fragility is method-dependent and affects more than AlphaEdit"

With NSE + QueueEDIT/AlphaEdit+ (6 methods): "Temporal order is a consequential and previously underreported benchmark axis across distinct sequential-editing design families, including fixed projection, evolving projection, neuron-level editing, spectral preservation, and explicit historical realignment"

## Priority Order

1. NSE (highest value, most distinct)
2. QueueEDIT (explicit self-correction) or AlphaEdit+ (easier, less distinct)
3. SpecEdit only if REVIVE unreliable
4. DipEdit if time remains
5. Stop — do NOT add TamEdit, HiEdit, or external-memory editors

## Do NOT turn into a leaderboard
The paper is about the scientific finding (temporal order matters), not about ranking editors.
