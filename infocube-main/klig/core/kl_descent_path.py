"""
KLDescentPath: Distribution-space adaptive path for KL-IG.

Instead of a fixed parametric schedule in (μ, logvar) space (LinearPath,
DDiffusionPath, PowerPath …), this path *discovers itself* by gradient
descent on the KL divergence between two diagonal Gaussians:

    minimize  KL( N(μ_curr, exp(lv_curr)·I)  ‖  N(μ_cf, exp(lv_cf)·I) )
    over     (μ_curr, lv_curr)
    starting from  (μ_explicand, lv_explicand)
    toward         (μ_cf,         lv_cf)

The KL between two diagonal Gaussians has a closed form, so the descent is
purely geometric in (μ, lv) space — **no model autograd is needed to build
the path**.  The model is only invoked during integration (KL-IG's standard
df/dμ, df/dlv MC estimator).

    KL = ½ · Σ_i  [ exp(lv1−lv2) + (μ1−μ2)² · exp(−lv2) − 1 + (lv2 − lv1) ]

Closed-form gradients:

    ∂KL/∂μ1_i  =  (μ1_i − μ2_i) · exp(−lv2_i)
    ∂KL/∂lv1_i =  ½ · ( exp(lv1_i − lv2_i) − 1 )

After descent the trajectory is *reversed*: s=0 ↦ near-counterfactual end,
s=1 ↦ explicand — matching KL-IG convention that attribution accumulates
from baseline to explicand.

This is the distribution-space sibling of KLIG2 (pixel-space).  Where KLIG2
walks a single image, KLDescentPath walks a Gaussian — both μ and σ carry
semantic information about the counterfactual contrast.

References
----------
"Distribution-space IG²" design note (this repo).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

from klig.core.path import DistributionPath


class KLDescentPath(DistributionPath):
    """
    KL-descent path in distribution space.

    Parameters
    ----------
    mu_cf      : Counterfactual mean, broadcastable to the explicand shape.
                 Common choices:
                   • another class's image          → sharp contrast
                   • zeros                          → max-entropy / "neutral"
    lv_cf      : Counterfactual log-variance (scalar or tensor).
                 Common choices:
                   • small (e.g. 2·log(1/256))      → sharp counterfactual
                   • large (e.g. 0.0 = N(·, 1))     → fuzzy / class neighborhood
    T          : Maximum descent steps.
    step_size  : Learning rate for the natural gradient step (lerp fraction).
                 Typical range: 0.05–0.2.  With step_size=0.1, convergence
                 in ~50 steps to within 0.5% of the counterfactual.
    kl_stop    : Early-stop threshold on KL value (sum over dims).
    """

    def __init__(
        self,
        mu_cf: torch.Tensor,
        lv_cf: float | torch.Tensor = 0.0,
        T: int = 50,
        step_size: float = 0.01,
        kl_stop: float = 1e-3,
    ) -> None:
        self.mu_cf = mu_cf
        self.lv_cf = lv_cf
        self.T = T
        self.step_size = step_size
        self.kl_stop = kl_stop

        # Filled in by _build() — keyed on id(mu_final) so the integrator's
        # repeated at()/derivatives() calls share one trajectory per attribute().
        self._traj_mu: list[torch.Tensor] | None = None
        self._traj_lv: list[torch.Tensor] | None = None
        self._kl_traj: list[float] | None = None
        self._cached_id: int | None = None

    # ── trajectory builder ────────────────────────────────────────────────────

    def _resolve_cf(self, mu_final: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu_cf = self.mu_cf.to(mu_final.device).expand_as(mu_final).contiguous()
        if isinstance(self.lv_cf, torch.Tensor):
            lv_cf = self.lv_cf.to(mu_final.device).expand_as(mu_final).contiguous()
        else:
            lv_cf = torch.full_like(mu_final, float(self.lv_cf))
        return mu_cf, lv_cf

    def _build(self, mu_final: torch.Tensor, lv_final: torch.Tensor) -> None:
        """Run KL descent from explicand toward counterfactual, then reverse."""
        mu_cf, lv_cf = self._resolve_cf(mu_final)
        inv_var_cf = (-lv_cf).exp()

        mu_curr = mu_final.detach().clone()
        lv_curr = lv_final.detach().clone()

        traj_mu = [mu_curr.clone()]
        traj_lv = [lv_curr.clone()]
        kl_traj: list[float] = []

        for _ in range(self.T):
            # closed-form per-element KL gradients
            g_mu = (mu_curr - mu_cf) * inv_var_cf
            g_lv = 0.5 * ((lv_curr - lv_cf).exp() - 1.0)

            # scalar KL value (sum over dims)
            kl_val = 0.5 * (
                (lv_curr - lv_cf).exp()
                + (mu_curr - mu_cf).pow(2) * inv_var_cf
                - 1.0
                + (lv_cf - lv_curr)
            ).sum().item()
            kl_traj.append(kl_val)

            if kl_val < self.kl_stop:
                break

            # Direct lerp toward counterfactual — equivalent to the natural
            # gradient step for μ (raw g_μ * exp(lv_cf) = μ-μ_cf) and the
            # only numerically stable choice for lv across large logvar gaps
            # (raw Hessian ≈ 0 there, making Newton/natural steps blow up).
            # Guaranteed monotone KL decrease; converges in O(log(1/ε)/lr) steps.
            mu_curr = mu_curr - self.step_size * (mu_curr - mu_cf)    # lerp toward μ_cf
            lv_curr = lv_curr - self.step_size * (lv_curr - lv_cf)    # lerp toward lv_cf

            traj_mu.append(mu_curr.clone())
            traj_lv.append(lv_curr.clone())

        # Reverse so s=0 ↦ counterfactual end, s=1 ↦ explicand.
        traj_mu.reverse()
        traj_lv.reverse()

        self._traj_mu = traj_mu
        self._traj_lv = traj_lv
        self._kl_traj = kl_traj
        self._cached_id = id(mu_final)

    def _ensure(self, mu_final: torch.Tensor, lv_final: torch.Tensor) -> None:
        if self._traj_mu is None or self._cached_id != id(mu_final):
            self._build(mu_final, lv_final)

    # ── DistributionPath interface ────────────────────────────────────────────

    def at(
        self,
        s: float,
        mu_final: torch.Tensor,
        logvar_final: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._ensure(mu_final, logvar_final)
        n = len(self._traj_mu) - 1
        pos = max(0.0, min(float(s), 1.0)) * n
        i0 = int(math.floor(pos))
        i0 = min(n, max(0, i0))
        i1 = min(n, i0 + 1)
        frac = pos - i0
        mu = (1.0 - frac) * self._traj_mu[i0] + frac * self._traj_mu[i1]
        lv = (1.0 - frac) * self._traj_lv[i0] + frac * self._traj_lv[i1]
        return mu, lv

    def derivatives(
        self,
        s: float,
        mu_final: torch.Tensor,
        logvar_final: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Local secant derivative scaled to ds units.  Path index runs 0..n,
        so dμ/ds = (μ_{i+1} − μ_i) · n.
        """
        self._ensure(mu_final, logvar_final)
        n = len(self._traj_mu) - 1
        pos = max(0.0, min(float(s), 1.0)) * n
        i0 = int(math.floor(pos))
        i0 = min(n - 1, max(0, i0))
        i1 = i0 + 1
        dmu = (self._traj_mu[i1] - self._traj_mu[i0]) * n
        dlv = (self._traj_lv[i1] - self._traj_lv[i0]) * n
        return dmu, dlv

    def steps(self, n: int) -> torch.Tensor:
        return torch.linspace(0.5 / n, 1.0 - 0.5 / n, n)

    # ── diagnostics ───────────────────────────────────────────────────────────

    @property
    def kl_trajectory(self) -> list[float] | None:
        """KL value at each descent step (pre-reversal)."""
        return self._kl_traj

    @property
    def path_length(self) -> int:
        """Number of stored trajectory points (T+1 or fewer if early-stopped)."""
        return 0 if self._traj_mu is None else len(self._traj_mu)
