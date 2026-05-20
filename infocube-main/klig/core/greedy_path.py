"""
Greedy path variants for KL Integrated Gradients.

Three variants in increasing complexity:

  SortedDimPath          -- deterministic, ~zero overhead.
                            Assigns each dimension a per-dim power exponent γ_i
                            based on its gradient magnitude at the prior.
                            High-gradient dims get γ < 1 (fast start / move early);
                            low-gradient dims get γ > 1 (slow start / move late).
                            Drops directly into the existing KLIntegratedGradients
                            framework as a DistributionPath.

  GreedyMuAttributor     -- greedy search over μ at every step.
                            At each integration step the gradient w.r.t. μ is
                            evaluated at the current position and dimensions are
                            advanced proportionally to their gradient magnitude,
                            concentrating step budget where the model is most
                            sensitive.  σ (logvar) follows a fixed linear schedule.

  GreedyJointAttributor  -- same as GreedyMuAttributor but extends the greedy
                            weighting to the joint (μ, logvar) space so that both
                            parameters are advanced along the high-sensitivity
                            direction at each step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from klig.core.integrator import AttributionResult, KLIntegratedGradients
from klig.core.kl import gaussian_kl, kl_delta
from klig.core.path import DistributionPath, LinearPath


# ──────────────────────────────────────────────────────────────────────────────
# 1.  SortedDimPath
# ──────────────────────────────────────────────────────────────────────────────

class SortedDimPath(DistributionPath):
    """
    Per-dimension power-law path sorted by gradient magnitude at the prior.

    Each dimension i gets its own exponent γ_i in [gamma_lo, gamma_hi]:
        γ_i = gamma_lo + (gamma_hi - gamma_lo) * rank_i / (D - 1)
    where rank_i = 0 for the highest-gradient dimension and rank_i = D-1 for
    the lowest, so high-sensitivity dims move early (γ < 1) and low-sensitivity
    dims move late (γ > 1).

        mu_i(t)     = mu_final_i  * t^γ_i
        logvar_i(t) = logvar_final_i * t^γ_i

    Because only a one-time forward+backward pass at the prior is needed to
    compute the γ values, the overhead vs LinearPath is negligible.
    """

    def __init__(self, gamma: torch.Tensor, eps: float = 1e-7) -> None:
        self._gamma = gamma   # (D,) or any shape matching x
        self._eps = eps

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_model_and_input(
        cls,
        model: nn.Module,
        x: torch.Tensor,
        target: int,
        n_samples: int = 32,
        gamma_lo: float = 0.25,
        gamma_hi: float = 4.0,
    ) -> "SortedDimPath":
        """
        Build a SortedDimPath by evaluating |∂f/∂μ_i| at the prior N(0, 1).

        Args:
            model:     nn.Module (eval mode recommended).
            x:         Input tensor (C,H,W) or (1,C,H,W) – only its shape and
                       device are used here.
            target:    Class index for the objective f(x)[target].
            n_samples: MC samples to estimate the gradient expectation.
            gamma_lo:  γ assigned to the highest-gradient dimension.
            gamma_hi:  γ assigned to the lowest-gradient dimension.

        Returns:
            SortedDimPath ready to be passed to KLIntegratedGradients.
        """
        device = next(model.parameters()).device
        x = x.to(device)
        if x.dim() > 1 and x.shape[0] == 1:
            x = x.squeeze(0)
        x_shape = x.shape

        model.eval()
        saved = [p.requires_grad for p in model.parameters()]
        for p in model.parameters():
            p.requires_grad_(False)

        try:
            mu_t = torch.zeros(x_shape, device=device, requires_grad=True)
            logvar_t = torch.zeros(x_shape, device=device, requires_grad=False)

            eps_samp = torch.randn(n_samples, *x_shape, device=device)
            # at prior logvar=0 → std=1
            x_samp = mu_t.unsqueeze(0) + eps_samp

            out = model(x_samp)
            obj = out[:, target].mean()
            obj.backward()

            grad_mag = mu_t.grad.abs().detach()
        finally:
            for p, s in zip(model.parameters(), saved):
                p.requires_grad_(s)

        return cls._build(grad_mag, gamma_lo, gamma_hi)

    @classmethod
    def _build(
        cls,
        grad_magnitudes: torch.Tensor,
        gamma_lo: float,
        gamma_hi: float,
    ) -> "SortedDimPath":
        flat = grad_magnitudes.flatten().float()
        n = len(flat)
        order = torch.argsort(flat, descending=True)
        rank = torch.empty(n, device=flat.device)
        rank[order] = torch.linspace(0.0, 1.0, n, device=flat.device)
        gamma = (gamma_lo + (gamma_hi - gamma_lo) * rank).reshape(grad_magnitudes.shape)
        return cls(gamma)

    # ------------------------------------------------------------------
    # DistributionPath interface
    # ------------------------------------------------------------------

    def at(
        self,
        t: float,
        mu_final: torch.Tensor,
        logvar_final: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t_eff = max(float(t), self._eps)
        scale = t_eff ** self._gamma
        return mu_final * scale, logvar_final * scale

    def derivatives(
        self,
        t: float,
        mu_final: torch.Tensor,
        logvar_final: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t_eff = max(float(t), self._eps)
        rate = self._gamma * (t_eff ** (self._gamma - 1.0))
        return mu_final * rate, logvar_final * rate

    def steps(self, n: int) -> torch.Tensor:
        return torch.linspace(0.5 / n, 1.0 - 0.5 / n, n)


# ──────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ──────────────────────────────────────────────────────────────────────────────

def _get_gradients_at(
    model: nn.Module,
    mu_curr: torch.Tensor,
    logvar_curr: torch.Tensor,
    x_shape: torch.Size,
    objective_fn: Callable,
    n_samples: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MC estimate of E[∂f/∂μ] and E[∂f/∂logvar] at a single (μ, logvar) point."""
    mu_t = mu_curr.detach().requires_grad_(True)
    logvar_t = logvar_curr.detach().requires_grad_(True)

    eps = torch.randn(n_samples, *x_shape, device=device)
    std_t = (0.5 * logvar_t).exp()
    x_samp = mu_t.unsqueeze(0) + std_t.unsqueeze(0) * eps

    obj = objective_fn(x_samp)
    obj.backward()

    return mu_t.grad.clone(), logvar_t.grad.clone()


def _resolve_target(
    model: nn.Module,
    x: torch.Tensor,
    target,
    device: torch.device,
) -> Tuple[int, Callable]:
    if callable(target):
        return -1, lambda xs: target(model(xs)).mean()

    if target is None:
        with torch.no_grad():
            out = model(x.unsqueeze(0))
            target = int(out.argmax(dim=-1).item())

    idx = int(target)
    return idx, lambda xs: model(xs)[:, idx].mean()


# ──────────────────────────────────────────────────────────────────────────────
# Extended result dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GreedyAttributionResult:
    """Attribution result from a greedy-path integrator, with diagnostic fields."""

    attr: torch.Tensor
    attr_mu: torch.Tensor
    attr_logvar: torch.Tensor
    kl_final: torch.Tensor
    target: int

    # per-step diagnostics (length == n_steps)
    waypoints_mu: List[torch.Tensor] = field(default_factory=list)
    waypoints_logvar: List[torch.Tensor] = field(default_factory=list)
    step_grad_signal: List[float] = field(default_factory=list)

    def completeness_check(self) -> float:
        return float(self.attr.sum().item())


# ──────────────────────────────────────────────────────────────────────────────
# 2.  GreedyMuAttributor
# ──────────────────────────────────────────────────────────────────────────────

class GreedyMuAttributor:
    """
    Greedy path search over μ; logvar follows a fixed linear schedule.

    At each integration step k the gradient ∂f/∂μ is evaluated at the current
    (μ_k, logvar_k) and used to compute a gradient-weighted advance:

        w_i     = D · |g_μ_i| / Σ_j |g_μ_j|       (D = total dims, sums to D)
        Δμ_i    = w_i · (μ_final_i − μ_k_i) / (n_steps − k)

    Dimensions with large gradients advance faster, concentrating integration
    effort where the model is most sensitive.  The last step always forces
    μ to exactly μ_final so completeness is not degraded.

    Attribution accumulates via a standard Riemann sum over the adaptive steps:
        attr_i += g_μ_i · Δμ_i + g_logvar_i · Δlogvar_i
    """

    def __init__(
        self,
        model: nn.Module,
        n_steps: int = 50,
        n_samples: int = 10,
        sigma_final: float = 1.0 / 256.0,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.n_steps = n_steps
        self.n_samples = n_samples
        self.logvar_final_val = 2.0 * math.log(sigma_final)
        self.device = device or next(model.parameters()).device

    def attribute(
        self,
        x: torch.Tensor,
        target=None,
        show_progress: bool = False,
    ) -> GreedyAttributionResult:
        self.model.eval()
        x = x.to(self.device)
        if x.dim() > 1 and x.shape[0] == 1:
            x = x.squeeze(0)
        x_shape = x.shape
        D = x.numel()

        mu_final = x.detach()
        logvar_final = torch.full_like(mu_final, self.logvar_final_val)

        target_idx, objective_fn = _resolve_target(self.model, x, target, self.device)

        saved = [p.requires_grad for p in self.model.parameters()]
        for p in self.model.parameters():
            p.requires_grad_(False)

        mu_curr = torch.zeros_like(mu_final)
        attr_mu_sum = torch.zeros_like(mu_final)
        attr_logvar_sum = torch.zeros_like(mu_final)
        waypoints_mu: List[torch.Tensor] = []
        waypoints_logvar: List[torch.Tensor] = []
        step_signal: List[float] = []

        steps_iter = range(self.n_steps)
        if show_progress:
            steps_iter = tqdm(steps_iter, desc="GreedyMu-KL-IG")

        try:
            for k in steps_iter:
                t_logvar = (k + 0.5) / self.n_steps
                logvar_curr = t_logvar * logvar_final

                waypoints_mu.append(mu_curr.clone())
                waypoints_logvar.append(logvar_curr.clone())

                g_mu, g_logvar = _get_gradients_at(
                    self.model, mu_curr, logvar_curr,
                    x_shape, objective_fn, self.n_samples, self.device,
                )

                remaining_steps = self.n_steps - k
                if k < self.n_steps - 1:
                    # gradient-weighted advance in μ
                    weights = D * g_mu.abs() / (g_mu.abs().sum() + 1e-12)
                    delta_mu = weights * (mu_final - mu_curr) / remaining_steps
                else:
                    # final step: close any residual gap exactly
                    delta_mu = mu_final - mu_curr

                # logvar advances uniformly
                delta_logvar = logvar_final / self.n_steps

                attr_mu_sum.add_(g_mu * delta_mu)
                attr_logvar_sum.add_(g_logvar * delta_logvar)
                step_signal.append(float((g_mu * delta_mu).abs().sum().item()))

                mu_curr = (mu_curr + delta_mu).detach()
        finally:
            for p, s in zip(self.model.parameters(), saved):
                p.requires_grad_(s)

        attr_mu = attr_mu_sum
        attr_logvar = attr_logvar_sum
        attr = attr_mu + attr_logvar
        kl = kl_delta(mu_final, logvar_final)

        return GreedyAttributionResult(
            attr=attr,
            attr_mu=attr_mu,
            attr_logvar=attr_logvar,
            kl_final=kl,
            target=target_idx,
            waypoints_mu=waypoints_mu,
            waypoints_logvar=waypoints_logvar,
            step_grad_signal=step_signal,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 3.  GreedyJointAttributor
# ──────────────────────────────────────────────────────────────────────────────

class GreedyJointAttributor:
    """
    Greedy path search over the joint (μ, logvar) space.

    Both μ and logvar are advanced with gradient-weighted steps:

        joint_score_i = |g_μ_i| + |g_logvar_i|
        w_i           = D · joint_score_i / Σ_j joint_score_j
        Δμ_i          = w_i · (μ_final_i − μ_k_i)    / (n_steps − k)
        Δlogvar_i     = w_i · (logvar_final_i − lv_k_i) / (n_steps − k)

    This concentrates integration budget in dimensions where EITHER the mean OR
    the variance transition contributes most to the model output change.
    """

    def __init__(
        self,
        model: nn.Module,
        n_steps: int = 50,
        n_samples: int = 10,
        sigma_final: float = 1.0 / 256.0,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.n_steps = n_steps
        self.n_samples = n_samples
        self.logvar_final_val = 2.0 * math.log(sigma_final)
        self.device = device or next(model.parameters()).device

    def attribute(
        self,
        x: torch.Tensor,
        target=None,
        show_progress: bool = False,
    ) -> GreedyAttributionResult:
        self.model.eval()
        x = x.to(self.device)
        if x.dim() > 1 and x.shape[0] == 1:
            x = x.squeeze(0)
        x_shape = x.shape
        D = x.numel()

        mu_final = x.detach()
        logvar_final = torch.full_like(mu_final, self.logvar_final_val)

        target_idx, objective_fn = _resolve_target(self.model, x, target, self.device)

        saved = [p.requires_grad for p in self.model.parameters()]
        for p in self.model.parameters():
            p.requires_grad_(False)

        mu_curr = torch.zeros_like(mu_final)
        logvar_curr = torch.zeros_like(logvar_final)
        attr_mu_sum = torch.zeros_like(mu_final)
        attr_logvar_sum = torch.zeros_like(mu_final)
        waypoints_mu: List[torch.Tensor] = []
        waypoints_logvar: List[torch.Tensor] = []
        step_signal: List[float] = []

        steps_iter = range(self.n_steps)
        if show_progress:
            steps_iter = tqdm(steps_iter, desc="GreedyJoint-KL-IG")

        try:
            for k in steps_iter:
                waypoints_mu.append(mu_curr.clone())
                waypoints_logvar.append(logvar_curr.clone())

                g_mu, g_logvar = _get_gradients_at(
                    self.model, mu_curr, logvar_curr,
                    x_shape, objective_fn, self.n_samples, self.device,
                )

                remaining_steps = self.n_steps - k
                joint_score = g_mu.abs() + g_logvar.abs()
                raw_weights = D * joint_score / (joint_score.sum() + 1e-12)
                # cap: w_i > remaining_steps causes overshoot → oscillation → NaN
                weights = raw_weights.clamp(max=float(remaining_steps))
                if k < self.n_steps - 1:
                    delta_mu     = weights * (mu_final     - mu_curr)     / remaining_steps
                    delta_logvar = weights * (logvar_final - logvar_curr) / remaining_steps
                else:
                    delta_mu     = mu_final     - mu_curr
                    delta_logvar = logvar_final - logvar_curr

                attr_mu_sum.add_(g_mu * delta_mu)
                attr_logvar_sum.add_(g_logvar * delta_logvar)

                sig_dm  = weights * (mu_final     - mu_curr)     / remaining_steps
                sig_dlv = weights * (logvar_final - logvar_curr) / remaining_steps
                step_signal.append(float(sig_dm.abs().sum() + sig_dlv.abs().sum()))

                mu_curr     = (mu_curr     + delta_mu).detach()
                logvar_curr = (logvar_curr + delta_logvar).detach()
        finally:
            for p, s in zip(self.model.parameters(), saved):
                p.requires_grad_(s)

        attr_mu = attr_mu_sum
        attr_logvar = attr_logvar_sum
        attr = attr_mu + attr_logvar
        kl = kl_delta(mu_final, logvar_final)

        return GreedyAttributionResult(
            attr=attr,
            attr_mu=attr_mu,
            attr_logvar=attr_logvar,
            kl_final=kl,
            target=target_idx,
            waypoints_mu=waypoints_mu,
            waypoints_logvar=waypoints_logvar,
            step_grad_signal=step_signal,
        )
