"""
PathGuard Woodbury solver, hazard selector, and displacement computation.

Pure functions — no source injection, no vendor code dependency.
These are inlined into the pathguard_runner.py source injection script
for use inside the exec'd memit_main.py namespace.

The Woodbury identity enables efficient lambda search for PathGuard's
risk-weighted historical covariance G_t = K_S @ diag(h) @ K_S^T:

    (A + λ K_S W K_S^T)^{-1} = A^{-1} - A^{-1} K_S (λ^{-1} W^{-1} + K_S^T A^{-1} K_S)^{-1} K_S^T A^{-1}

By solving A @ [K_new | K_S] in one call (single factorization), the lambda
search requires only M×M inversions — negligible cost compared to the d×d base.
"""

import torch


def select_vulnerable(
    hist_keys: torch.Tensor,
    batch_keys: torch.Tensor,
    M: int,
) -> tuple[list[int], list[float]]:
    """Select top-M vulnerable historical edits by max-cosine exposure.

    Args:
        hist_keys: [d, n_hist] historical edit keys (any precision)
        batch_keys: [d, n_batch] current batch keys
        M: maximum number of vulnerable edits to return

    Returns:
        indices: list of int, indices into hist_keys columns, sorted by exposure descending
        hazard: list of float, hazard weights (= max cosine exposure), same order
    """
    n_hist = hist_keys.shape[1]
    if n_hist == 0:
        return [], []

    M = min(M, n_hist)

    hist_normed = hist_keys / torch.clamp(
        torch.linalg.norm(hist_keys, dim=0, keepdim=True), min=1e-8
    )
    batch_normed = batch_keys / torch.clamp(
        torch.linalg.norm(batch_keys, dim=0, keepdim=True), min=1e-8
    )

    cos_matrix = hist_normed.T @ batch_normed  # [n_hist, n_batch]
    max_cos = cos_matrix.max(dim=1).values  # [n_hist]

    topk_vals, topk_idx = torch.topk(max_cos, M)

    indices = topk_idx.tolist()
    hazard = topk_vals.tolist()
    return indices, hazard


def woodbury_solve(
    A: torch.Tensor,
    K_new: torch.Tensor,
    K_S: torch.Tensor,
    h: torch.Tensor,
    lambda_candidates: list[float],
    epsilon_t: float,
    E_max: float,
    resid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float, dict]:
    """Efficient PathGuard solve using Woodbury identity.

    Solves (A + lambda * K_S @ diag(h) @ K_S^T) x = K_new for the smallest
    lambda in lambda_candidates satisfying displacement <= epsilon_t.

    Args:
        A: [d, d] base LHS matrix (double precision)
        K_new: [d, n_new] current batch keys
        K_S: [d, M] vulnerable historical keys (can be empty)
        h: [M] hazard weights (positive)
        lambda_candidates: lambda values to try (sorted ascending recommended)
        epsilon_t: displacement budget
        E_max: maximum editability residual threshold
        resid: [d, n_new] target residuals (needed for displacement computation)

    Returns:
        adj_k: [d, n_new] solution
        lambda_used: the lambda that was selected
        info: dict with solve metadata
    """
    M = K_S.shape[1] if K_S.dim() == 2 else 0

    if M == 0:
        adj_k = torch.linalg.solve(A, K_new)
        return adj_k, 0.0, {"fallback": True, "reason": "empty_K_S"}

    candidates = sorted(lambda_candidates)

    # Step 1: single factorization — solve A @ [K_new | K_S]
    combined = torch.cat([K_new, K_S], dim=1)
    solutions = torch.linalg.solve(A, combined)
    adj_k_base = solutions[:, :K_new.shape[1]]
    A_inv_KS = solutions[:, K_new.shape[1]:]

    # Step 2: M×M core for Woodbury
    core = K_S.T @ A_inv_KS  # [M, M]
    KS_adj_k = K_S.T @ adj_k_base  # [M, n_new]

    best_adj_k = adj_k_base
    best_lambda = 0.0
    info = {"candidates_tried": 0}

    for lam in candidates:
        info["candidates_tried"] += 1

        if lam == 0.0:
            adj_k_candidate = adj_k_base
        else:
            W_inv_diag = 1.0 / torch.clamp(h, min=1e-10)
            inner = (1.0 / lam) * torch.diag(W_inv_diag) + core  # [M, M]
            correction = A_inv_KS @ torch.linalg.solve(inner, KS_adj_k)
            adj_k_candidate = adj_k_base - correction

        if resid is not None:
            D = compute_displacement(adj_k_candidate, K_S, h, resid)
        else:
            D = 0.0

        if D <= epsilon_t:
            best_adj_k = adj_k_candidate
            best_lambda = lam
            info["displacement"] = D
            break

        best_adj_k = adj_k_candidate
        best_lambda = lam
        info["displacement"] = D

    return best_adj_k, best_lambda, info


def compute_displacement(
    adj_k: torch.Tensor,
    K_S: torch.Tensor,
    h: torch.Tensor,
    resid: torch.Tensor,
) -> float:
    """Compute displacement D_t = ||resid @ adj_k^T @ K_S @ diag(sqrt(h))||_F^2.

    This measures how much the proposed update ΔW = resid @ adj_k^T
    displaces the vulnerable historical keys, weighted by hazard.

    Args:
        adj_k: [d, n_new] solution from solve
        K_S: [d, M] vulnerable historical keys
        h: [M] hazard weights
        resid: [d, n_new] target residuals

    Returns:
        D_t: scalar displacement (non-negative)
    """
    if K_S.shape[1] == 0:
        return 0.0

    h_sqrt = torch.sqrt(h)
    delta_W_KS = resid @ (adj_k.T @ K_S)  # [d, M]
    weighted = delta_W_KS * h_sqrt.unsqueeze(0)  # [d, M]
    return torch.sum(weighted ** 2).item()


def adapt_epsilon(
    displacement_history: list[float],
    epsilon_base: float,
    warmup_batches: int = 10,
    window: int = 5,
    tighten_factor: float = 0.5,
    relax_threshold: float = 1.5,
) -> float:
    """Adapt displacement budget based on recent displacement history.

    During warmup: return epsilon_base.
    After warmup: compare recent displacement to the warmup baseline.
    If recent >> baseline, tighten. If recent ~= baseline, keep base.

    Args:
        displacement_history: list of D_t values from all previous batches
        epsilon_base: permissive baseline budget
        warmup_batches: number of initial batches used to establish baseline
        window: number of recent batches to average for comparison
        tighten_factor: multiply epsilon by this when tightening (must be in (0, 1))
        relax_threshold: ratio of recent/baseline above which to tighten

    Returns:
        epsilon_t: adapted budget (always positive)
    """
    if not displacement_history or len(displacement_history) < warmup_batches:
        return epsilon_base

    baseline = sum(displacement_history[:warmup_batches]) / warmup_batches
    if baseline <= 0:
        return epsilon_base

    recent = displacement_history[-window:] if len(displacement_history) >= window else displacement_history
    recent_mean = sum(recent) / len(recent)

    ratio = recent_mean / baseline

    if ratio > relax_threshold:
        n_violations = sum(1 for d in recent if d / baseline > relax_threshold)
        scale = tighten_factor ** n_violations
        return max(epsilon_base * scale, epsilon_base * 0.01)

    return epsilon_base
