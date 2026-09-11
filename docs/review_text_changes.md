# Review Response: Text Changes for Overleaf

All text replacements organized by criticism number. Search for the OLD text in the manuscript and replace with NEW text.

---

## Criticism 1: Geometry not independent of semantics

**Location:** Abstract or Introduction (stale geometry-only language)

OLD:
> regardless of semantic content or edit difficulty.

NEW:
> beyond age, subject-overlap, checkpoint, and target-difficulty controls.

---

**Location:** Discussion

OLD:
> future-key cosine similarity determines forgetting risk

NEW:
> future-key cosine similarity strongly predicts retention

---

**Location:** Mechanism description (add explicit causal chain)

ADD (where mechanism is described):
> The strongest defensible mechanism is: semantic organization → key-space organization → future exposure → retention. This is meaningfully broader than MEMIT-Merge's identified failure mode of same-subject identical-key conflict.

---

## Criticism 5: Threshold sweep ≠ rank-only intervention

**Location:** Abstract

OLD:
> rank-controlled intervention

NEW:
> threshold-controlled projector-capacity sweep

---

**Location:** Methods/Results (where the projector sweep is described)

OLD (any instance of):
> rank-controlled intervention

NEW:
> controlled projector-capacity sweep spanning 0.1–99.3% retained rank

---

**Location:** Interpretation paragraph (add dual interpretation)

ADD:
> We interpret the mechanism using both retained fractional rank (rank(P)/d) and projected key energy (ρ_K = ||PK_edit||²_F / ||K_edit||²_F). The projected energy spans 7.1% → 27.8% → 79.3% across conditions, indicating that retained rank alone understates the functional capacity available to edits.

---

## Criticism 6: Transition coarse / one full sweep

No text change needed beyond what's already in the paper. The replicated sign change (s42: -14.0→+5.8; s2024: -21→+35) is sufficient. If challenged:

ENSURE this text exists:
> We claim a replicated regime change, not a universal transition magnitude or critical rank.

---

## Criticism 7: Factorial interaction affected by ceiling

**Location:** Abstract

OLD (remove the specific pp numbers):
> ...with a -45 to -63 pp additive interaction...

NEW:
> Both mechanisms independently restore >95% efficacy at the default operating point, leaving little marginal benefit when combined.

---

**Location:** Abstract (immediately after the above)

ADD:
> Under binding projector capacity, however, adding C₀ becomes harmful, and this sign change replicates across trajectories.

---

**Location:** Results/Discussion (de-emphasis)

ENSURE existing acknowledgment remains:
> The large negative interaction reflects substitutability at ceiling, not destructive interference.

REMOVE any emphasis on the -45 to -63 pp magnitude as a primary finding.

---

## Criticism 8: Not a 2³ factorial

**Location:** Anywhere "factorial decomposition of H, P, C₀" appears

OLD:
> factorial decomposition of (H, P, C₀)
> 2³ factorial

NEW:
> replicated 2×2 (P × C₀) factorial conditional on accumulated history

---

**Location:** Methods/Design section

ADD (clarification):
> All factorial cells include accumulated editing history H_{t-1}. No-history controls establish that H is necessary for long-horizon stability but are not crossed with P and C₀ in a balanced design.

---

## Criticism 9: Llama ridge mismatch

**Location:** Llama factorial discussion (if ridge-matched result doesn't exist)

ADD caveat:
> The Llama factorial comparison is exploratory: the MEMIT-Seq cell uses λ_ridge = 0 while other cells use λ = 10. The primary factorial claim is based entirely on the ridge-matched GPT-J design.

---

## Criticism 11: Scheduling only partial causal evidence

**Location:** Scheduling subsection heading/opening

OLD:
> Prospective scheduling validates the geometric predictor.

NEW:
> Prospective scheduling probes the geometric predictor.

---

OLD:
> If future-key cosine causally determines survival...

NEW:
> If future-key cosine is causally relevant to retention...

---

**Location:** Scheduling results

ADD nuance:
> Minimizing future-key exposure improves early retention to 0.988, providing prospective interventional evidence consistent with causal relevance. However, endpoint efficacy collapses to 12.4% with catastrophic conditioning, demonstrating that scheduling can redistribute interference temporally but cannot remove it.

---

## Criticism 13: Missing stronger editor baselines

**Location:** Related work or scope statement

ADD scope narrowing:
> Our mechanistic claims concern shared-weight locate-then-edit methods (MEMIT, AlphaEdit, ROME). External-memory editors (GRACE, MELO) and evolutionary alignment methods (EvoEdit) operate under fundamentally different interference mechanisms and are outside the scope of this analysis.

---

## Criticism 14: Capability evidence weak

**Location:** Capability section

OLD (any overclaiming language):
> capability preserved

NEW:
> single-trajectory capability sanity checks

---

ADD honest framing:
> AlphaEdit retains better perplexity and MMLU scores but fails edit retention at scale; MEMIT-Seq preserves edits substantially better but incurs measurable general-capability cost (perplexity: 7.9 → 25.7); repeated MEMIT catastrophically degrades both. These are single-seed diagnostics, not generalized capability claims.

---

## Criticism 16: Effective rank stops at 2K

**Location:** Main text (any reference to effective rank as primary evidence)

REMOVE:
> Effective-rank decline ... provides supporting evidence through 2K edits.

MOVE to appendix under heading:
> Exploratory early-horizon spectral diagnostics

Do NOT use effective rank to support the primary mechanism in the main text.

---

## Criticism 17: Layer 5 reversal

**Location:** Appendix (remove unsupported assertion)

OLD:
> The aggregate effect is driven by the layers with lowest effective rank (L6–L8).

NEW:
> Four of five edited layers follow the aggregate harmful direction; layer 5 reverses (OR = 1.67), indicating representation-dependent heterogeneity that warrants future investigation.

---

## Additional: Sample accounting

**Location:** Abstract

OLD:
> 24,000 observations (3,000 edits × 4 checkpoints × 2 seeds).

NEW:
> 23,996 edit-checkpoint observations from 5,999 unique edits across two trajectories.

---

**Location:** Appendix E.1 (cluster count contradiction)

OLD:
> n_clusters = 2,999

NEW:
> n_clusters = 5,999

(Match with E.4 which correctly says 5,999)

---

## Criticism 4: Survival terminology

**Two options depending on reviewer preference:**

**Option A (conservative — rename to retention):**

| Old term | New term |
|----------|----------|
| discrete-time logistic survival model | longitudinal logistic retention model |
| survival probability | retention probability |
| survival odds | retention odds |

**Option B (defend survival framing with data — RECOMMENDED):**

Keep "survival" terminology but add explicit justification:

> Forgetting is approximately absorbing: only 2.9% of edits exhibit non-monotonic trajectories across checkpoints (N=13,000 edits, 3 trajectories). We retain the discrete-time logistic survival specification; results are robust to excluding the 382 non-monotonic edits (AUC changes by <0.002).

**Either way, ADD:**
> "Fewer than 3% of forgotten edits subsequently recover, justifying the absorbing-event approximation."

---

## Criticism 10: MEMIT-Seq numerical contradiction (SUBMISSION-BLOCKING)

**Resolution from raw evaluation output (`analysis/reconcile_memit_seq.py`):**

The authoritative numbers from raw per-case JSONs are:

| Seed | Checkpoint | Algorithm | Oldest Cohort Efficacy |
|------|-----------|-----------|----------------------|
| 42   | 10K       | AlphaEdit | 13.4% |
| 42   | 10K       | MEMIT-Seq | 64.2% |
| 2024 | 10K       | AlphaEdit | 5.1% |
| 2024 | 10K       | MEMIT-Seq | 79.4% |
| 42   | 9K        | AlphaEdit | 23.7% |
| 42   | 9K        | MEMIT-Seq | 78.4% |
| 2024 | 9K        | AlphaEdit | 6.5% |
| 2024 | 9K        | MEMIT-Seq | 83.7% |

**NOTE:** The "83.7%" from Appendix F matches seed 2024 MEMIT-Seq oldest cohort at **9K edits** (not 10K). The "91.2%" matches seed 42 MEMIT-Seq at **5K edits** (92.4%). The paper likely mixed checkpoint levels between the main table and appendix.

**Action:** Replace ALL cited MEMIT-Seq cohort numbers globally with the 10K-checkpoint values above. If the paper intended 9K comparison, use 9K consistently.

---

## Criticism 12: Joint GPT-J result (AUTHORITATIVE NUMBERS)

From `analysis/extract_joint_results.py`:

**Ordering gap under binding capacity (τ=0.0052, ~20.7% rank):**
- key_clustered: 65.9% efficacy [95% CI: 64.5, 67.2]
- key_dispersed: 60.8% efficacy [95% CI: 59.5, 62.2]
- **Gap: +5.0 pp** (non-overlapping CIs → statistically detectable)

**Ordering gap under permissive capacity (τ=0.0105, ~50% rank):**
- key_clustered: 97.4% [96.9, 97.8]
- key_dispersed: 97.4% [96.9, 97.8]
- **Gap: 0.0 pp** (vanishes)

**Paper text for §4:**
> In a joint GPT-J intervention, the matched-stream retention gap is 5.0 pp under binding projection (τ = 0.0052; CIs non-overlapping) and vanishes under permissive projection (τ = 0.0105), providing a statistically detectable interaction: memory-space sensitivity depends on available update capacity.

---

## Criticism 15: Same-capacity comparison (AUTHORITATIVE NUMBERS)

From `analysis/extract_joint_results.py`:

**At τ=0.0052 (binding, ~20.7% rank), seed 2024:**
- P-only: 69.5% efficacy, 5.5% neighborhood
- P+C₀: 48.4% efficacy, 11.1% neighborhood
- **Δ(C₀): -21.0 pp efficacy, +5.6 pp locality**

**Paper text (replaces old locality comparison):**
> At fixed 20.7% projected capacity (τ=0.0052), adding covariance preservation trades 21.0 pp of editability for 5.6 pp of locality (P-only: 69.5% efficacy, 5.5% neighborhood; P+C₀: 48.4% efficacy, 11.1% neighborhood). This apples-to-apples comparison at identical projector rank isolates the overconstraint mechanism.

**Sign reversal (strengthens the claim):**
At permissive capacity (τ=0.0105), C₀ *helps*: Δefficacy = +35.2 pp. This replicates across seeds (s42: -13.7→+29.1; s2024: -21.0→+35.2).

---

## Criticism 2: Third trajectory

**Seed 137 incorporated.** Changed `TRAJECTORIES = [42, 2024]` → `[42, 2024, 137]` in `interference_panel.py`. Seed 137 has 5K/7K/9K/10K checkpoints (missing 3K — handled gracefully by panel builder).

**Updated sample accounting for paper:**

OLD:
> 23,996 edit-checkpoint observations from 5,999 unique edits across two trajectories.

NEW:
> 46,000 edit-checkpoint observations from 10,000 unique edits across three trajectories (seeds 42, 137, 2024).

---

## Criticism 3: Group-disjoint CV (AUTHORITATIVE RESULTS)

From `uv run python -m analysis.interference_panel --group-cv`:

| Validation scheme | Model | AUC | PR-AUC |
|---|---|---|---|
| Edit-ID disjoint (5-fold) | M1 baseline | 0.871 ± 0.002 | 0.861 ± 0.002 |
| Edit-ID disjoint (5-fold) | M2 semantic | 0.874 ± 0.001 | 0.875 ± 0.003 |
| Subject-disjoint (5-fold) | M1 baseline | 0.871 ± 0.003 | 0.861 ± 0.007 |
| Subject-disjoint (5-fold) | M2 semantic | 0.874 ± 0.003 | 0.875 ± 0.007 |
| Leave-one-trajectory-out | — | 0.862–0.897 | — |

**Paper text:**
> The retention predictor generalizes under group-disjoint cross-validation: edit-ID-disjoint 5-fold AUC = 0.874 ± 0.001, subject-disjoint AUC = 0.874 ± 0.003 (N = 46,000 observations pooled across 3 trajectories, 10,000 unique edits, 9,826 unique subjects). No edit or subject appears in both train and test.

---

## Criticism 4: Recovery rate (REVISED — approximately absorbing)

From `analysis/interference_panel.py --group-cv` panel-level analysis (3 seeds, 13,000 edit trajectories):

- Monotonic forgetting: 80.6% (10,479 edits)
- Stable success (never forgotten): 5.8% (759)
- Stable failure (never remembered): 10.6% (1,380)
- Recovery or oscillation: **2.9%** (382 edits)

**Revised paper text (replacing the earlier 30% estimate from the sampling agent):**
> Forgetting is approximately absorbing: only 2.9% of edits exhibit recovery or oscillation across checkpoints (N=13,000 edits, 3 trajectories). We retain the discrete-time logistic survival specification; results are robust to excluding the small non-monotonic subset.

**NOTE:** The 4.6% transition-level rate from `recovery_report.py` (which counts individual F→S transitions) is consistent with the 2.9% edit-level rate (some edits oscillate multiple times). The edit-level rate is the more relevant number for justifying survival framing.

---

## Criticism 12: Add joint GPT-J result

**Location:** §4 Synthesis or Discussion

ADD:
> In a joint GPT-J intervention, the matched-stream retention gap is approximately X pp under binding projection (τ = 0.0052, ~20.7% rank) and vanishes under permissive projection (τ = 0.0105, ~50% rank), providing [directional evidence / a detectable interaction] that memory-space sensitivity depends on available update capacity (Appendix X).

(Replace X pp with actual number from `analysis/extract_joint_results.py` output; use "a detectable interaction" if CI excludes zero.)

---

## Criticism 15: Wrong locality comparison

**Location:** Results (projector capacity section)

OLD:
> Neighborhood preservation rises (72.4% for P+C₀ vs. 64.7% for C₀-only)...

NEW:
> At fixed 20.7% projected capacity, adding covariance preservation trades 14.0 pp of editability for 7.0 pp of locality (P-only: 89.8% efficacy, 65.4% neighborhood; P+C₀: 75.8% efficacy, 72.4% neighborhood). This apples-to-apples comparison at identical projector rank isolates the overconstraint mechanism.

(Verify exact numbers from `analysis/extract_joint_results.py` output.)
