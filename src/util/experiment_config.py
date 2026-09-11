"""Centralized experiment configuration and path construction.

Single source of truth for variant names, checkpoint paths, and result paths.
Replaces the 3 separate resolve_checkpoint_dir implementations that diverged
and caused the MEMIT-Seq hardcoded variant_name bug.
"""

from dataclasses import dataclass, field
from pathlib import Path

from paths import get_checkpoint_root, get_result_root


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run.

    This is the ONLY place variant_name and paths should be computed.
    """
    base_alg: str
    seed: int
    ordering: str | None = None
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"

    # Algorithm params
    lambda_prev: float = 0.0
    lambda_delta: float = 0.0
    cache_max: int | None = None
    cache_strategy: str = "all"

    # Kernel params
    kernel_type: str = "poly"
    kernel_degree: int = 1
    kernel_sigma: str = "median"
    kernel_prev: bool = True

    # REVIVE params
    revive: bool = False
    revive_tau: float = 0.1

    @property
    def base_prefix(self) -> str:
        return "MEMIT-Seq" if self.base_alg == "MEMIT" else self.base_alg

    @property
    def kernel_tag(self) -> str:
        tag = f"poly{self.kernel_degree}" if self.kernel_type == "poly" else f"rbf_{self.kernel_sigma}"
        if not self.kernel_prev:
            tag += "-hybrid"
        if self.revive:
            tag += f"-REVIVE-tau{self.revive_tau}"
        return tag

    @property
    def variant_name(self) -> str:
        cache_str = str(self.cache_max) if self.cache_max is not None else "0"
        return f"{self.base_prefix}-{self.kernel_tag}-lp{self.lambda_prev}-ld{self.lambda_delta}-cache{cache_str}"

    @property
    def model_tag(self) -> str:
        mn = (self.model_name or "").lower()
        default = "meta-llama/meta-llama-3-8b-instruct"
        if not mn or mn == default or mn.endswith("meta-llama-3-8b-instruct"):
            return ""
        if "gpt-j" in mn:
            return "gpt-j-6b"
        if "qwen2.5-7b" in mn:
            return "qwen2.5-7b"
        return mn.rsplit("/", 1)[-1]

    def checkpoint_dir(self, root: Path | None = None) -> Path:
        base = (root or get_checkpoint_root()) / "polykernel_seqreg"
        if self.model_tag:
            base = base / self.model_tag
        if self.ordering:
            return base / self.variant_name / self.ordering / f"seed{self.seed}"
        return base / self.variant_name / f"seed{self.seed}"

    def results_dir(self, root: Path | None = None, experiment_name: str | None = None) -> Path:
        result_root = root or get_result_root()
        if not experiment_name:
            if self.model_tag == "gpt-j-6b":
                experiment_name = "failure_curve_gptj"
            elif self.model_tag == "qwen2.5-7b":
                experiment_name = "failure_curve_qwen"
            else:
                experiment_name = "failure_curve_checkpointed"

        if self.ordering:
            return result_root / "matched_ordering" / self.ordering / f"seed{self.seed}"
        return result_root / experiment_name / f"seed{self.seed}"

    def validate(self) -> None:
        """Raise if configuration is inconsistent."""
        from checkpoint_io import validate_checkpoint_path
        validate_checkpoint_path(str(self.checkpoint_dir()), self.base_alg)
