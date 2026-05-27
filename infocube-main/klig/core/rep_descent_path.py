"""
RepDescentPath: distribution-space IG² (Option 2a).

Lifts the IG² pixel-space descent objective into (μ, logvar) space by
descending the *expected representation distance* between samples from the
current Gaussian and a fixed counterfactual point:

    L(μ, lv) = E_{x ~ N(μ, exp(lv)·I)} [ ‖φ(x) − φ(x_cf)‖² ]

where φ is a hidden-layer representation of the classifier.  Gradients w.r.t.
(μ, lv) are obtained via the reparameterisation trick:

    x_n = μ + exp(lv/2) ⊙ ε_n,    ε_n ~ N(0, I)

Each descent step:

    g_μ  = ∂L/∂μ     (via autograd through reparameterised x_n)
    g_lv = ∂L/∂lv
    μ   ← μ  − lr_μ  · normalize(g_μ)
    lv  ← lv − lr_lv · normalize(g_lv)

The trajectory is reversed after descent (s=0 ↦ near-counterfactual end,
s=1 ↦ explicand), matching the KL-IG convention.

Difference vs. KLDescentPath
----------------------------
KLDescentPath descends a *closed-form* KL between two Gaussians (no model
calls — purely geometric in (μ, lv) space).
RepDescentPath descends an objective that *queries the classifier* through
a representation φ.  This is the actual "²" — model gradients shape the
path itself, not only the attribution integral.

The two share the same `Path` interface and the same trajectory storage,
so either can be passed to `KLIntegratedGradients(path=...)`.

References
----------
- Original IG² (pixel-space):
  https://arxiv.org/abs/2308.05858
- This file: distribution-space generalisation (Option 2a in the design note).
"""

from __future__ import annotations

import math
from typing import Callable, Tuple

import torch
import torch.nn as nn

from klig.core.path import DistributionPath


# ── utility: build a φ-callable from (model, layer) via a forward hook ────────


def make_phi_from_layer(
    model: nn.Module,
    layer: str | nn.Module,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Wrap a (model, layer) pair into a callable φ(x) → activation tensor.

    The returned callable runs a forward pass through `model` and returns the
    output of `layer`.  Gradients flow through normally because the hook just
    captures the activation tensor in-place.

    Parameters
    ----------
    model : nn.Module
        The classifier (or any nn.Module).
    layer : str | nn.Module
        Either a named module in `model` (e.g. "layer4.1.conv2") or an
        already-resolved nn.Module reference.

    Returns
    -------
    phi : callable
        phi(x) where x has shape (B, ...) and the return shape matches the
        chosen layer's output shape, with leading batch dim preserved.
    """
    if isinstance(layer, str):
        mods = dict(model.named_modules())
        if layer not in mods:
            raise ValueError(f"Layer '{layer}' not found in model.")
        target = mods[layer]
    else:
        target = layer

    cache: dict = {}

    def _hook(_m, _inp, out):
        cache["out"] = out

    handle = target.register_forward_hook(_hook)

    def phi(x: torch.Tensor) -> torch.Tensor:
        _ = model(x)
        if "out" not in cache:
            raise RuntimeError(
                "Forward hook did not fire — check that the chosen layer is "
                "actually executed by model(x)."
            )
        return cache["out"]

    # Keep handle alive on the closure so it isn't GC'd while phi is in use.
    phi._handle = handle  # type: ignore[attr-defined]
    return phi


# ── the path ──────────────────────────────────────────────────────────────────


class RepDescentPath(DistributionPath):
    """
    Distribution-space IG² path: descends expected representation distance.

    Parameters
    ----------
    phi : callable
        Representation extractor.  Must accept a batched input tensor
        (n_mc, *x_shape) and return a tensor whose leading dim is `n_mc`.
        See `make_phi_from_layer` for a convenient default.
    x_cf : torch.Tensor
        Counterfactual *point* (not a distribution) used as the descent
        anchor.  Shape must match the explicand x (no batch dim, or a
        leading singleton batch).
    T : int
        Maximum descent steps.
    step_size_mu : float
        Learning rate applied to normalised ∂L/∂μ.
    step_size_lv : float
        Learning rate applied to normalised ∂L/∂lv.
    n_mc : int
        Monte-Carlo samples per descent step (reparameterisation).
    loss_stop : float
        Early-stop threshold on the descent loss.
    normalize_grad : bool
        Whether to L2-normalise the (μ, lv) gradients before stepping.
        Normalisation keeps step magnitudes scale-free w.r.t. φ.
    """

    def __init__(
        self,
        phi: Callable[[torch.Tensor], torch.Tensor],
        x_cf: torch.Tensor,
        T: int = 30,
        step_size_mu: float = 0.05,
        step_size_lv: float = 0.05,
        n_mc: int = 8,
        loss_stop: float = 1e-3,
        normalize_grad: bool = True,
    ) -> None:
        self.phi = phi
        self.x_cf = x_cf
        self.T = T
        self.step_size_mu = step_size_mu
        self.step_size_lv = step_size_lv
        self.n_mc = n_mc
        self.loss_stop = loss_stop
        self.normalize_grad = normalize_grad

        # cached trajectory (keyed on id(mu_final))
        self._traj_mu: list[torch.Tensor] | None = None
        self._traj_lv: list[torch.Tensor] | None = None
        self._loss_traj: list[float] | None = None
        self._cached_id: int | None = None

    # ── trajectory builder ────────────────────────────────────────────────────

    def _resolve_cf(self, mu_final: torch.Tensor) -> torch.Tensor:
        x_cf = self.x_cf.to(mu_final.device)
        if x_cf.dim() == mu_final.dim() + 1 and x_cf.shape[0] == 1:
            x_cf = x_cf.squeeze(0)
        return x_cf.expand_as(mu_final).contiguous()

    def _build(self, mu_final: torch.Tensor, lv_final: torch.Tensor) -> None:
        """Run reparameterised descent on E[‖φ(x) − φ(x_cf)‖²]."""
        x_cf = self._resolve_cf(mu_final)

        # Anchor representation — fixed for the entire descent.
        with torch.no_grad():
            phi_cf = self.phi(x_cf.unsqueeze(0))           # (1, R*)
            phi_cf = phi_cf.detach()

        mu_curr = mu_final.detach().clone()
        lv_curr = lv_final.detach().clone()

        traj_mu = [mu_curr.clone()]
        traj_lv = [lv_curr.clone()]
        loss_traj: list[float] = []

        for _ in range(self.T):
            mu_p = mu_curr.detach().requires_grad_(True)
            lv_p = lv_curr.detach().requires_grad_(True)

            eps      = torch.randn(self.n_mc, *mu_p.shape, device=mu_p.device)
            std      = (0.5 * lv_p).exp()
            x_samp   = mu_p.unsqueeze(0) + std.unsqueeze(0) * eps   # (n_mc, ...)

            phi_samp = self.phi(x_samp)                              # (n_mc, R*)
            diff     = phi_samp - phi_cf                             # broadcast (1, R*)
            loss     = diff.flatten(1).pow(2).sum(dim=-1).mean()

            loss_traj.append(float(loss.item()))
            if loss.item() < self.loss_stop:
                break

            g_mu, g_lv = torch.autograd.grad(loss, [mu_p, lv_p])

            if self.normalize_grad:
                g_mu = g_mu / (g_mu.norm() + 1e-12)
                g_lv = g_lv / (g_lv.norm() + 1e-12)

            mu_curr = (mu_curr - self.step_size_mu * g_mu).detach()
            lv_curr = (lv_curr - self.step_size_lv * g_lv).detach()

            traj_mu.append(mu_curr.clone())
            traj_lv.append(lv_curr.clone())

        # Reverse so s=0 ↦ near-cf, s=1 ↦ explicand (KL-IG convention).
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
        """Local secant scaled to ds units."""
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
        """Descent loss value at each step (pre-reversal)."""
        return self._loss_traj

    @property
    def path_length(self) -> int:
        return 0 if self._traj_mu is None else len(self._traj_mu)
