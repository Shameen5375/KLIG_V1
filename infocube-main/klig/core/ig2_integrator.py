"""
KL-IG²: Distribution-space IG² integrator.

IG² (Zhuo & Ge, TPAMI 2024) uses two gradient signals:
  1. Explicand class gradient:   E[∂F_target/∂(μ,lv)]
  2. Counterfactual path gradient (encoded in displacement): sign(∂L_repdist/∂(μ,lv))

In pixel space IG² computes:
    attr_i = Σ_k  ∂F(x_k)/∂x_i  ·  (x_k − x_{k+1})_i

where {x_k} is GradPath — built by descending
    L(x) = ‖φ(x) − φ(x_cf)‖²
and GradCF = x_T (the endpoint) is the model-derived baseline.

This file lifts the concept to (μ, logvar) space:
    L(μ,lv) = E_{x ~ N(μ, exp(lv)·I)} [ ‖φ(x) − φ(x_cf)‖² ]

Attribution:
    attr_i = Σ_k  E[∂F_target/∂μ_i | μ_k, lv_k]  ·  (μ_{k+1} − μ_k)_i
           + Σ_k  E[∂F_target/∂lv_i | μ_k, lv_k]  ·  (lv_{k+1} − lv_k)_i

Completeness:
    Σ_i attr_i  ≈  E[F_target(explicand)] − E[F_target(GradCF)]

The baseline GradCF is model-derived — wherever the representation descent lands —
not an arbitrary choice like a black image or Gaussian prior.

Difference from KLIntegratedGradients + RepDescentPath
-------------------------------------------------------
RepDescentPath reverses the trajectory so the existing integrator can walk
s=0→1 from cf-end to explicand. KLIGSquared integrates directly in the natural
descent direction (explicand → GradCF), making the two-gradient structure and
the completeness identity explicit without a reversal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class KLIGSquaredResult:
    """Attribution result from KLIGSquared.attribute()."""

    attr: torch.Tensor         # total attribution = attr_mu + attr_lv
    attr_mu: torch.Tensor      # μ component
    attr_lv: torch.Tensor      # lv component

    # Waypoints: index 0 = explicand, index -1 = GradCF
    traj_mu: List[torch.Tensor] = field(default_factory=list)
    traj_lv: List[torch.Tensor] = field(default_factory=list)

    # Per-step |g_mu*dmu + g_lv*dlv| signal (length = len(traj) - 1)
    step_grad_signal: List[float] = field(default_factory=list)

    loss_trajectory: List[float] = field(default_factory=list)
    target: int = -1

    def completeness_check(self) -> float:
        """sum(attr) ≈ E[F(explicand)] - E[F(GradCF)]."""
        return float(self.attr.sum().item())


class KLIGSquared:
    """
    Distribution-space IG² integrator.

    Builds GradPath by descending L(μ,lv) = E[‖φ(x)−φ(x_cf)‖²] from the
    explicand, then integrates ∂F_target/∂(μ,lv) along the descent trajectory.

    The baseline is GradCF — the endpoint of descent — not the N(0,I) prior.
    Completeness: Σ attr ≈ E[F_target(explicand)] − E[F_target(GradCF)].

    Args:
        model:        nn.Module → (B, n_classes) logits.
        phi:          Representation extractor (B, *x_shape) → (B, R).
                      Build with make_phi_from_layer() from rep_descent_path.
        x_cf:         Counterfactual reference image (no batch dim).
        T:            Max descent steps. Default 50.
        lr_mu:        Sign-normalised μ step size. Default 0.05.
        lr_lv:        Sign-normalised lv step size. Default 0.10.
        n_mc_path:    MC samples per descent step. Default 16.
        n_mc_grad:    MC samples per integration gradient estimate. Default 10.
        sigma_start:  Starting σ at the explicand end (logvar = 2·log(σ)).
                      Default 1/256. Pass find_sigma_stop(...) for adaptive.
        loss_stop:    Early-stop threshold for descent. Default 1e-3.
        lv_floor:     Lower lv clamp during descent. Default 2·log(1/256).
        lv_ceil:      Upper lv clamp. Default 4.0.
        mu_min/max:   Pixel-range clamps (ImageNet: ±2.64). Default ±2.64.
        clamp_samples: Clamp MC samples to [mu_min, mu_max]. Default True.
        device:       Torch device. Default: inferred from model.
    """

    _LV_FLOOR_DEFAULT = 2.0 * math.log(1.0 / 256.0)   # ≈ −11.09

    def __init__(
        self,
        model: nn.Module,
        phi: Callable[[torch.Tensor], torch.Tensor],
        x_cf: torch.Tensor,
        T: int = 50,
        lr_mu: float = 0.05,
        lr_lv: float = 0.10,
        n_mc_path: int = 16,
        n_mc_grad: int = 10,
        sigma_start: float = 1.0 / 256.0,
        loss_stop: float = 1e-3,
        lv_floor: float | None = None,
        lv_ceil: float = 4.0,
        mu_min: float | None = -2.64,
        mu_max: float | None = 2.64,
        clamp_samples: bool = True,
        device: torch.device | None = None,
    ) -> None:
        self.model         = model
        self.phi           = phi
        self.x_cf          = x_cf
        self.T             = T
        self.lr_mu         = lr_mu
        self.lr_lv         = lr_lv
        self.n_mc_path     = n_mc_path
        self.n_mc_grad     = n_mc_grad
        self.lv_start      = 2.0 * math.log(sigma_start)
        self.loss_stop     = loss_stop
        self.lv_floor      = lv_floor if lv_floor is not None else self._LV_FLOOR_DEFAULT
        self.lv_ceil       = lv_ceil
        self.mu_min        = mu_min
        self.mu_max        = mu_max
        self.clamp_samples = clamp_samples
        self.device        = device or next(model.parameters()).device

    # ── public API ────────────────────────────────────────────────────────────

    def attribute(
        self,
        x: torch.Tensor,
        target: int | Callable | None = None,
        show_progress: bool = False,
    ) -> KLIGSquaredResult:
        """
        Compute KL-IG² attributions for input x.

        Args:
            x:             (C,H,W) or (1,C,H,W) input tensor.
            target:        int class index, callable objective, or None (argmax).
            show_progress: Show tqdm bar over integration steps.

        Returns:
            KLIGSquaredResult with attr shape matching x (no batch dim).
        """
        self.model.eval()
        x = x.to(self.device)
        if x.dim() > 1 and x.shape[0] == 1:
            x = x.squeeze(0)
        x_shape = x.shape

        target_idx, objective_fn = self._resolve_target(x, target)

        # ── Phase 1: GradPath descent ─────────────────────────────────────────
        traj_mu, traj_lv, loss_traj = self._build_gradpath(x, x_shape)

        # ── Phase 2: Integrate along GradPath ────────────────────────────────
        saved = [p.requires_grad for p in self.model.parameters()]
        for p in self.model.parameters():
            p.requires_grad_(False)

        attr_mu_sum     = torch.zeros_like(x)
        attr_lv_sum     = torch.zeros_like(x)
        step_grad_signal: List[float] = []

        K        = len(traj_mu) - 1
        iterator = range(K)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="KL-IG²", unit="step")

        try:
            for k in iterator:
                mu_k  = traj_mu[k]
                lv_k  = traj_lv[k]
                dmu_k = traj_mu[k + 1] - traj_mu[k]   # descent displacement
                dlv_k = traj_lv[k + 1] - traj_lv[k]

                g_mu, g_lv = self._eval_gradients(mu_k, lv_k, x_shape, objective_fn)

                with torch.no_grad():
                    contrib_mu = g_mu * dmu_k
                    contrib_lv = g_lv * dlv_k
                    attr_mu_sum.add_(contrib_mu)
                    attr_lv_sum.add_(contrib_lv)
                    step_grad_signal.append(
                        float((contrib_mu + contrib_lv).abs().sum().item())
                    )
        finally:
            for p, s in zip(self.model.parameters(), saved):
                p.requires_grad_(s)

        with torch.no_grad():
            attr = attr_mu_sum + attr_lv_sum

        return KLIGSquaredResult(
            attr=attr,
            attr_mu=attr_mu_sum,
            attr_lv=attr_lv_sum,
            traj_mu=traj_mu,
            traj_lv=traj_lv,
            step_grad_signal=step_grad_signal,
            loss_trajectory=loss_traj,
            target=target_idx,
        )

    # ── internals ─────────────────────────────────────────────────────────────

    def _resolve_target(
        self,
        x: torch.Tensor,
        target: int | Callable | None,
    ) -> Tuple[int, Callable]:
        if callable(target):
            return -1, lambda xs: target(self.model(xs)).mean()
        if target is None:
            with torch.no_grad():
                target = int(self.model(x.unsqueeze(0)).argmax(-1).item())
        idx = int(target)
        return idx, lambda xs: self.model(xs)[:, idx].mean()

    def _build_gradpath(
        self,
        x: torch.Tensor,
        x_shape: torch.Size,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float]]:
        """
        Descend L(μ,lv) = E[‖φ(x)−φ(x_cf)‖²] from the explicand.

        Starts at (μ=x, lv=lv_start). Each step:
          1. Reparameterise: x_n = μ + exp(lv/2) · ε_n
          2. Compute L = mean_n ‖φ(x_n) − φ(x_cf)‖²
          3. Sign-normalised update on (μ, lv)
        """
        x_cf = self.x_cf.to(self.device)
        if x_cf.dim() == x.dim() + 1 and x_cf.shape[0] == 1:
            x_cf = x_cf.squeeze(0)

        with torch.no_grad():
            phi_cf_flat = self.phi(x_cf.unsqueeze(0)).detach().flatten(1)

        mu_curr = x.detach().clone()
        lv_curr = torch.full_like(mu_curr, self.lv_start)

        traj_mu:   List[torch.Tensor] = [mu_curr.clone()]
        traj_lv:   List[torch.Tensor] = [lv_curr.clone()]
        loss_traj: List[float]        = []

        for _ in range(self.T):
            mu_p = mu_curr.detach().requires_grad_(True)
            lv_p = lv_curr.detach().requires_grad_(True)

            eps    = torch.randn(self.n_mc_path, *x_shape, device=self.device)
            std    = (0.5 * lv_p).exp()
            x_samp = mu_p.unsqueeze(0) + std.unsqueeze(0) * eps

            if self.clamp_samples and self.mu_min is not None and self.mu_max is not None:
                x_samp = x_samp.clamp(self.mu_min, self.mu_max)

            phi_samp = self.phi(x_samp).flatten(1)
            loss     = (phi_samp - phi_cf_flat).pow(2).sum(dim=-1).mean()

            loss_val = float(loss.item())
            loss_traj.append(loss_val)
            if loss_val < self.loss_stop:
                break

            g_mu, g_lv = torch.autograd.grad(loss, [mu_p, lv_p])

            mu_curr = (mu_curr - self.lr_mu * g_mu.sign()).detach()
            lv_curr = (lv_curr - self.lr_lv * g_lv.sign()).detach()

            if self.mu_min is not None and self.mu_max is not None:
                mu_curr = mu_curr.clamp(self.mu_min, self.mu_max)
            lv_curr = lv_curr.clamp(self.lv_floor, self.lv_ceil)

            traj_mu.append(mu_curr.clone())
            traj_lv.append(lv_curr.clone())

        return traj_mu, traj_lv, loss_traj

    def _eval_gradients(
        self,
        mu_k: torch.Tensor,
        lv_k: torch.Tensor,
        x_shape: torch.Size,
        objective_fn: Callable,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """MC estimate of E[∂F_target/∂μ] and E[∂F_target/∂lv] at (μ_k, lv_k)."""
        mu_t  = mu_k.detach().requires_grad_(True)
        lv_t  = lv_k.detach().requires_grad_(True)

        eps    = torch.randn(self.n_mc_grad, *x_shape, device=self.device)
        x_samp = mu_t.unsqueeze(0) + (0.5 * lv_t).exp().unsqueeze(0) * eps

        objective_fn(x_samp).backward()

        return mu_t.grad.clone(), lv_t.grad.clone()
