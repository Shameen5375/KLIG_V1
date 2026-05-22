"""
DDiffusionPath: KL-IG path that follows a diffusion noise schedule.

Instead of the linear interpolation used by LinearPath, the path tracks:

    μ(s)      = √ᾱ(s) · μ_final
    logvar(s) = log(1 - ᾱ(s) + ε)          (scalar, same for every dim)

where s ∈ [0, 1] and ᾱ(s) is a noise schedule with ᾱ(0) = 0, ᾱ(1) = 1.

Every intermediate (μ(s), logvar(s)) is a valid diffusion noisy state of
the target image — gradients are evaluated at plausible noisy versions of x,
keeping the integration path on the data manifold.

At s = 0: N(0, I)         ← standard prior, same as LinearPath
At s = 1: N(μ_final, ε)   ← near-deterministic target

Completeness holds because the integral telescopes from prior to target
just as in standard KL-IG.

Reference: DDPath-IG  https://openreview.net/pdf?id=bSv0MBDBF2

Available schedules
-------------------
"cosine"    ᾱ(s) = sin²(πs/2)   — smooth, no singularity at s=0
"linear"    ᾱ(s) = s             — simplest, same pacing as LinearPath μ
"quadratic" ᾱ(s) = s²            — slow start, similar to PowerPath(γ=2)
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

from klig.core.path import DistributionPath


class DDiffusionPath(DistributionPath):
    """
    KL-IG path parameterised by a diffusion noise schedule.

    Parameters
    ----------
    schedule : {"cosine", "linear", "quadratic"}
        Noise schedule that determines how quickly ᾱ rises from 0 to 1.
    eps : float
        Numerical floor added inside sqrt and log for stability at the
        endpoints (s=0 and s=1).
    """

    def __init__(self, schedule: str = "cosine", eps: float = 1e-6) -> None:
        if schedule not in ("cosine", "linear", "quadratic"):
            raise ValueError(
                f"Unknown schedule '{schedule}'. Choose cosine / linear / quadratic."
            )
        self.schedule = schedule
        self.eps = eps

    # ── noise schedule helpers ────────────────────────────────────────────────

    def _ab(self, s: float) -> float:
        """ᾱ(s) ∈ [0, 1]: monotone noise schedule."""
        if self.schedule == "cosine":
            return math.sin(math.pi / 2.0 * s) ** 2
        elif self.schedule == "linear":
            return float(s)
        else:                           # quadratic
            return float(s) ** 2

    def _dab(self, s: float) -> float:
        """dᾱ/ds."""
        if self.schedule == "cosine":
            return math.pi / 2.0 * math.sin(math.pi * s)
        elif self.schedule == "linear":
            return 1.0
        else:                           # quadratic
            return 2.0 * float(s)

    # ── DistributionPath interface ────────────────────────────────────────────

    def at(
        self,
        s: float,
        mu_final: torch.Tensor,
        logvar_final: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (μ(s), logvar(s)) at path position s ∈ [0, 1]."""
        ab   = self._ab(s)
        mu_s = math.sqrt(ab + self.eps) * mu_final
        lv_s = math.log(1.0 - ab + self.eps) * torch.ones_like(mu_final)
        return mu_s, lv_s

    def derivatives(
        self,
        s: float,
        mu_final: torch.Tensor,
        logvar_final: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (dμ/ds, dlogvar/ds) at path position s."""
        ab  = self._ab(s)
        dab = self._dab(s)
        # dμ/ds = (dᾱ/ds) / (2√ᾱ) · μ_final
        dmu = (dab / (2.0 * math.sqrt(ab + self.eps))) * mu_final
        # dlogvar/ds = -(dᾱ/ds) / (1 - ᾱ)
        dlv = (-dab / (1.0 - ab + self.eps)) * torch.ones_like(mu_final)
        return dmu, dlv

    def steps(self, n: int) -> torch.Tensor:
        # midpoint rule avoids s=0 (sqrt singularity) and s=1 (log singularity)
        return torch.linspace(0.5 / n, 1.0 - 0.5 / n, n)
