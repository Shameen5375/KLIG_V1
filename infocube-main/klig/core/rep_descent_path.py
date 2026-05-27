"""
RepDescentPath: distribution-space IG² (KL-IG²).

Lifts the IG² pixel-space descent objective into (μ, logvar) space by
descending the expected representation distance between samples from the
current Gaussian and a fixed counterfactual point:

    L(μ, lv) = E_{x ~ N(μ, exp(lv)·I)} [ ‖φ(x) − φ(x_cf)‖² ]

where φ is a hidden-layer representation of the classifier.

Gradients w.r.t. (μ, lv) come from the reparameterisation trick:

    x_n = μ + exp(lv/2) ⊙ ε_n,    ε_n ~ N(0, I)

with ε_n held fixed across the gradient computation so the mapping
(μ, lv) → x_n is differentiable.

Gradient structure (named for clarity):

    let δ_n = φ(x_n) − φ(x_cf)
    ∂L/∂μ  = (2/N) · Σ_n  Jφ(x_n)ᵀ δ_n            (standard rep-dist grad)
    ∂L/∂lv = (1/N) · Σ_n  Jφ(x_n)ᵀ δ_n ⊙ (exp(lv/2)⊙ε_n / 2)

The lv gradient is the μ-gradient weighted by how far each sample deviated
from the mean — samples drawn far from μ (large |ε_n|) dominate, which is
exactly the variance sensitivity we want.

Update rule:
    μ  ← μ  − lr_μ  · sign(∂L/∂μ)
    lv ← lv − lr_lv · sign(∂L/∂lv)

Sign normalisation keeps step magnitudes predictable regardless of how
the raw gradient magnitude varies along the path (matches the original
IG² convention of fixed-magnitude pixel steps).

After T steps the trajectory is reversed so s=0 ↦ near-counterfactual
end and s=1 ↦ explicand, matching the KL-IG convention.

Why this is KL-IG²
------------------
The integration step accumulates:

    attribution += (∂F_target/∂(μ,lv)) · (dμ, dlv)

where dμ ≈ −lr_μ · sign(∂L_repdist/∂μ).  The displacement is, up to sign
and scale, the counterfactual representation distance gradient — so each
integration step is implicitly the product of two gradient signals, one
from the explicand class output and one from the counterfactual contrast.
That is the defining structure of IG², lifted to distribution space.

Difference vs. KLDescentPath
-----------------------------
KLDescentPath descends a closed-form Gaussian KL (no model calls).
RepDescentPath queries the classifier every descent step — model
representation gradients shape the path itself.  The variance dimension
moves non-trivially: ∂L/∂lv is not zero, so lv does not just follow a
fixed schedule.

References
----------
- IG²: Zhuo & Ge, "IG²: Integrated Gradient on Iterative Gradient Path"
  https://arxiv.org/abs/2308.05858
- This file: distribution-space generalisation (Option 2a).
"""

from __future__ import annotations

import math
from typing import Callable, Tuple

import torch
import torch.nn as nn

from klig.core.path import DistributionPath


# ── utility: hook a named layer to build a φ-callable ─────────────────────────


def make_phi_from_layer(
    model: nn.Module,
    layer: str | nn.Module,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Wrap (model, layer) into a φ(x) callable that returns the layer's output.

    The returned callable runs a full forward pass and captures the activation
    at `layer` via a forward hook.  Gradients flow through normally (the hook
    captures the tensor itself, not a copy).

    Parameters
    ----------
    model : nn.Module
        The classifier.
    layer : str | nn.Module
        Named module string (e.g. ``"layer4"`` for ResNet) **or** a direct
        ``nn.Module`` reference.

    Returns
    -------
    phi : callable  (x: Tensor) -> Tensor
        Accepts a batched tensor (B, ...) and returns the layer output with
        the same leading batch dimension.
    """
    if isinstance(layer, str):
        mods = dict(model.named_modules())
        if layer not in mods:
            raise ValueError(
                f"Layer '{layer}' not found in model.  "
                f"Available: {list(mods)[:10]} ..."
            )
        target = mods[layer]
    else:
        target = layer

    cache: dict = {}

    def _hook(_m: nn.Module, _inp, out: torch.Tensor) -> None:
        cache["out"] = out

    handle = target.register_forward_hook(_hook)

    def phi(x: torch.Tensor) -> torch.Tensor:
        cache.pop("out", None)
        _ = model(x)
        if "out" not in cache:
            raise RuntimeError(
                "Forward hook did not fire.  Make sure the chosen layer is "
                "executed by model(x) (not a no-op branch)."
            )
        return cache["out"]

    # Keep handle alive on the closure so it isn't GC'd.
    phi._hook_handle = handle  # type: ignore[attr-defined]

    def remove_hook() -> None:
        handle.remove()

    phi.remove_hook = remove_hook  # type: ignore[attr-defined]

    return phi


# ── the path ──────────────────────────────────────────────────────────────────


class RepDescentPath(DistributionPath):
    """
    Distribution-space IG² path via expected representation descent.

    Parameters
    ----------
    phi : callable
        Representation extractor.  Accepts (n_mc, *x_shape) → tensor with
        leading n_mc dimension.  Built by ``make_phi_from_layer`` or any
        callable with the same signature.
    x_cf : torch.Tensor
        Counterfactual reference image (no batch dim, or leading singleton).
        The descent drives the current distribution's expected representation
        toward φ(x_cf).
    T : int
        Maximum descent steps.  With sign-normalised steps and typical
        learning rates, 50–200 steps are sufficient.
    lr_mu : float
        Step size for μ in normalised pixel / representation units.
        Typical: 0.05.  Controls how far μ moves per step in pixel space.
    lr_lv : float
        Step size for lv (log-variance axis).  Typical: 0.10.
        Must be tuned separately from lr_mu because lv lives in log-space
        and small changes correspond to large variance shifts.
    n_mc : int
        Monte-Carlo samples per descent step for the expected loss.
        16–32 is typical; more reduces gradient variance at the cost of
        n_mc forward+backward passes per step.
    loss_stop : float
        Early-stop threshold.  Descent halts when L < loss_stop.
        Default 1e-3.  If most paths hit T (the cap), loosen this or
        increase T.
    lv_floor : float | None
        Lower clamp on lv.  Prevents exp(lv/2) from underflowing to zero
        and killing the variance gradient.  Default: 2·log(1/256) ≈ −11.09
        (one 8-bit step), matching the ``sigma_final`` used by the integrator.
    lv_ceil : float
        Upper clamp on lv.  Prevents MC samples from exploding in magnitude.
        Default 4.0 (σ ≈ 7.4), well above any plausible image variance.
    mu_min, mu_max : float | None
        Optional pixel-range clamps on μ after each step.  Keeps the path
        on the data manifold.  For ImageNet-normalised inputs roughly
        [−2.64, 2.64]; for [0,1]-normalised inputs [0,1].  None = unclamped.
    clamp_samples : bool
        If True, clamp each MC sample x_n to [mu_min, mu_max] before the
        forward pass.  Prevents φ from seeing out-of-distribution pixels
        whose gradients are meaningless.  Default True.
    """

    _LV_FLOOR_DEFAULT = 2.0 * math.log(1.0 / 256.0)   # ≈ −11.09

    def __init__(
        self,
        phi: Callable[[torch.Tensor], torch.Tensor],
        x_cf: torch.Tensor,
        T: int = 50,
        lr_mu: float = 0.05,
        lr_lv: float = 0.10,
        n_mc: int = 16,
        loss_stop: float = 1e-3,
        lv_floor: float | None = None,
        lv_ceil: float = 4.0,
        mu_min: float | None = None,
        mu_max: float | None = None,
        clamp_samples: bool = True,
    ) -> None:
        self.phi          = phi
        self.x_cf         = x_cf
        self.T            = T
        self.lr_mu        = lr_mu
        self.lr_lv        = lr_lv
        self.n_mc         = n_mc
        self.loss_stop    = loss_stop
        self.lv_floor     = lv_floor if lv_floor is not None else self._LV_FLOOR_DEFAULT
        self.lv_ceil      = lv_ceil
        self.mu_min       = mu_min
        self.mu_max       = mu_max
        self.clamp_samples = clamp_samples

        # cached trajectory — keyed on id(mu_final)
        self._traj_mu:   list[torch.Tensor] | None = None
        self._traj_lv:   list[torch.Tensor] | None = None
        self._loss_traj: list[float]         | None = None
        self._cached_id: int                 | None = None

    # ── path builder ─────────────────────────────────────────────────────────

    def _resolve_cf(self, mu_final: torch.Tensor) -> torch.Tensor:
        x_cf = self.x_cf.to(mu_final.device)
        if x_cf.dim() == mu_final.dim() + 1 and x_cf.shape[0] == 1:
            x_cf = x_cf.squeeze(0)
        return x_cf.contiguous()

    def _build(self, mu_final: torch.Tensor, lv_final: torch.Tensor) -> None:
        """
        Descend  L(μ,lv) = E[‖φ(x) − φ(x_cf)‖²]  from explicand to cf.

        Algorithm
        ---------
        1. Cache φ(x_cf) once (no gradient needed).
        2. For each step:
           a. Reparameterise: x_n = μ + exp(lv/2) ⊙ ε_n
           b. (Optionally) clamp x_n to valid pixel range.
           c. Compute L = mean_n ‖φ(x_n) − φ_cf‖²
           d. Backprop to get (g_μ, g_lv).
           e. Sign-normalised update; clamp (μ, lv) to valid ranges.
        3. Reverse trajectory so s=0 ↦ cf end, s=1 ↦ explicand.
        """
        x_cf = self._resolve_cf(mu_final)

        with torch.no_grad():
            phi_cf = self.phi(x_cf.unsqueeze(0)).detach()   # (1, R*)
            phi_cf_flat = phi_cf.flatten(1)                 # (1, R)

        mu_curr = mu_final.detach().clone()
        lv_curr = lv_final.detach().clone()

        traj_mu:   list[torch.Tensor] = [mu_curr.clone()]
        traj_lv:   list[torch.Tensor] = [lv_curr.clone()]
        loss_traj: list[float]        = []

        for _ in range(self.T):
            mu_p = mu_curr.detach().requires_grad_(True)
            lv_p = lv_curr.detach().requires_grad_(True)

            # Step 1: reparameterised samples
            eps    = torch.randn(self.n_mc, *mu_p.shape, device=mu_p.device)
            std    = (0.5 * lv_p).exp()                             # (D,)
            x_samp = mu_p.unsqueeze(0) + std.unsqueeze(0) * eps    # (n_mc, D)

            # Step 2 (optional): clamp samples to valid pixel range
            if self.clamp_samples and self.mu_min is not None and self.mu_max is not None:
                x_samp = x_samp.clamp(self.mu_min, self.mu_max)

            # Step 3: expected representation distance
            phi_samp      = self.phi(x_samp)                        # (n_mc, R*)
            phi_samp_flat = phi_samp.flatten(1)                     # (n_mc, R)
            diff          = phi_samp_flat - phi_cf_flat             # (n_mc, R)
            loss          = diff.pow(2).sum(dim=-1).mean()

            loss_val = float(loss.item())
            loss_traj.append(loss_val)

            if loss_val < self.loss_stop:
                break

            # Step 4: gradients via autograd through the reparameterisation
            g_mu, g_lv = torch.autograd.grad(loss, [mu_p, lv_p])

            # Step 5: sign-normalised update (scale-free, matches IG² convention)
            mu_curr = (mu_curr - self.lr_mu  * g_mu.sign()).detach()
            lv_curr = (lv_curr - self.lr_lv  * g_lv.sign()).detach()

            # Step 6: clamp to valid ranges
            if self.mu_min is not None and self.mu_max is not None:
                mu_curr = mu_curr.clamp(self.mu_min, self.mu_max)
            lv_curr = lv_curr.clamp(self.lv_floor, self.lv_ceil)

            traj_mu.append(mu_curr.clone())
            traj_lv.append(lv_curr.clone())

        # Reverse: s=0 ↦ counterfactual end, s=1 ↦ explicand.
        traj_mu.reverse()
        traj_lv.reverse()

        self._traj_mu   = traj_mu
        self._traj_lv   = traj_lv
        self._loss_traj = loss_traj
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
        n   = len(self._traj_mu) - 1
        pos = max(0.0, min(float(s), 1.0)) * n
        i0  = min(n, max(0, int(math.floor(pos))))
        i1  = min(n, i0 + 1)
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
        """Secant derivative scaled to ds units (n segments → n factor)."""
        self._ensure(mu_final, logvar_final)
        n   = len(self._traj_mu) - 1
        pos = max(0.0, min(float(s), 1.0)) * n
        i0  = min(n - 1, max(0, int(math.floor(pos))))
        i1  = i0 + 1
        dmu = (self._traj_mu[i1] - self._traj_mu[i0]) * n
        dlv = (self._traj_lv[i1] - self._traj_lv[i0]) * n
        return dmu, dlv

    def steps(self, n: int) -> torch.Tensor:
        return torch.linspace(0.5 / n, 1.0 - 0.5 / n, n)

    # ── diagnostics ───────────────────────────────────────────────────────────

    @property
    def loss_trajectory(self) -> list[float] | None:
        """Descent loss at each step (pre-reversal); None before first build."""
        return self._loss_traj

    @property
    def path_length(self) -> int:
        """Number of stored waypoints (T+1 or fewer if early-stopped)."""
        return 0 if self._traj_mu is None else len(self._traj_mu)

    @property
    def traj_mu(self) -> list[torch.Tensor] | None:
        """μ waypoints (reversed, s=0 first).  None before first build."""
        return self._traj_mu

    @property
    def traj_lv(self) -> list[torch.Tensor] | None:
        """lv waypoints (reversed, s=0 first).  None before first build."""
        return self._traj_lv
