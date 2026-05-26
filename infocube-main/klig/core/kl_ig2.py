"""
KL-IG²: KL-descent Integrated Gradients.

Rather than a fixed parametric path in (μ, σ) space, KL-IG² builds the
integration path adaptively by descending KL(p(y|x) ‖ q) in pixel space.
Attribution is the standard path integral of the KL gradient along this path.

    attr_i = Σ_t  ∂KL/∂x_i(x_t) · Δx_{t,i}

Completeness: Σ_i attr_i ≈ KL_0 − KL_T  (initial minus final KL)

Key design choices:
  • Normalized gradient steps (grad / ‖grad‖) — step size stays in
    pixel-space units regardless of KL magnitude.
  • Optional gradient norm clipping to prevent early explosion on
    very confident models.
  • KL smoothing: p ← (1−ε)p + ε/K avoids log(0).
  • Early stop when KL < kl_stop (no point integrating noise).
  • Also accumulates squared attribution: Σ (∂KL/∂x_i)² · step_size,
    which is an unsigned importance map (Approach A from the derivation).

Reference: "KL-Descent IG" pilot plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F


@dataclass
class KLIG2Result:
    """Output of KLIG2Attributor.attribute()."""

    # Signed path integral: Σ_t ∇KL(x_t) · Δx_t  (shape = x.shape)
    attr: torch.Tensor

    # Unsigned importance: Σ_t (∇KL)² · step_size  (shape = x.shape)
    attr_sq: torch.Tensor

    # KL value at every step taken (length = path_len)
    kl_traj: list[float]

    # Number of path steps actually taken (≤ T, early-stop may reduce it)
    path_len: int

    # Model's top-1 prediction at the original input
    target: int

    # q_mode used to construct q
    q_mode: str

    def completeness_check(self) -> float:
        """Σ attr ≈ KL_0 − KL_T (signed completeness)."""
        return float(self.attr.sum().item())

    def kl_drop(self) -> float:
        """Total KL drop along the path."""
        if not self.kl_traj:
            return 0.0
        return float(self.kl_traj[0] - self.kl_traj[-1])


class KLIG2Attributor:
    """
    KL-descent Integrated Gradients (KL-IG²).

    Builds an adaptive integration path by iteratively descending
    KL(p(y|x) ‖ q) in pixel space, then accumulates ∇KL · Δx as
    the attribution.

    Parameters
    ----------
    model      : nn.Module in eval mode.
    T          : Maximum path steps.
    step_size  : Step size for normalized gradient descent (pixel units).
    kl_eps     : Smoothing floor; p ← (1−kl_eps)·p + kl_eps/K.
    kl_stop    : Stop early if KL drops below this threshold.
    grad_clip  : Clip pixel-space gradient norm to this value before
                 normalization. None disables clipping.
    clip_pixels: Clamp x_t to [0, 1] after each step. Useful for
                 unnormalized images; leave False for ImageNet-normalized.
    device     : Defaults to first model parameter's device.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        T: int = 50,
        step_size: float = 0.01,
        kl_eps: float = 1e-6,
        kl_stop: float = 1e-3,
        grad_clip: float | None = 1.0,
        clip_pixels: bool = False,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.T = T
        self.step_size = step_size
        self.kl_eps = kl_eps
        self.kl_stop = kl_stop
        self.grad_clip = grad_clip
        self.clip_pixels = clip_pixels
        self.device = device or next(model.parameters()).device

    # ── public ────────────────────────────────────────────────────────────────

    def attribute(
        self,
        x: torch.Tensor,
        q_mode: Literal["second_class", "uniform", "top_class"] = "second_class",
        q_target: torch.Tensor | None = None,
    ) -> KLIG2Result:
        """
        Compute KL-IG² attributions for input x.

        Parameters
        ----------
        x        : (C,H,W) or (1,C,H,W) input tensor.
        q_mode   : target distribution construction:
                   "second_class" — one-hot at runner-up class  (default)
                   "uniform"      — 1/K at every class
                   "top_class"    — one-hot at model's top prediction
        q_target : override q_mode with a custom (1, K) tensor.

        Returns
        -------
        KLIG2Result with attr / attr_sq of shape (C, H, W).
        """
        self.model.eval()
        x = x.to(self.device)
        if x.dim() == 4 and x.shape[0] == 1:
            x = x.squeeze(0)

        x_4d = x.unsqueeze(0)

        with torch.no_grad():
            logits = self.model(x_4d)
            top_idx = int(logits.argmax(-1).item())

        q = q_target if q_target is not None else self._make_q(logits, q_mode)

        path, kl_traj = self._build_path(x_4d, q)
        attr, attr_sq = self._integrate(path, q)

        return KLIG2Result(
            attr=attr.squeeze(0),
            attr_sq=attr_sq.squeeze(0),
            kl_traj=kl_traj,
            path_len=len(path),
            target=top_idx,
            q_mode=q_mode,
        )

    # ── internals ─────────────────────────────────────────────────────────────

    def _make_q(self, logits: torch.Tensor, mode: str) -> torch.Tensor:
        q = torch.zeros_like(logits)
        if mode == "top_class":
            q[0, int(logits.argmax(-1))] = 1.0
        elif mode == "second_class":
            top2 = logits.topk(2, dim=-1).indices[0]
            q[0, int(top2[1])] = 1.0
        elif mode == "uniform":
            q[:] = 1.0 / logits.shape[-1]
        else:
            raise ValueError(f"Unknown q_mode '{mode}'")
        return q

    def _kl(self, x_t: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Scalar KL(p(y|x_t) ‖ q) with smoothed p."""
        logits = self.model(x_t)
        p = F.softmax(logits, dim=-1)
        p_s = (1.0 - self.kl_eps) * p + self.kl_eps / p.shape[-1]
        # F.kl_div(log_input, target) computes Σ target*(log_target - log_input)
        # so kl_div(q.log(), p_s) = Σ p_s*(log p_s - log q) = KL(p_s ‖ q)
        return F.kl_div(q.clamp(min=1e-12).log(), p_s, reduction="sum")

    def _build_path(
        self,
        x_4d: torch.Tensor,
        q: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[float]]:
        path = [x_4d.clone().detach()]
        kl_traj: list[float] = []
        x_t = x_4d.clone().detach()

        saved = [p.requires_grad for p in self.model.parameters()]
        for p in self.model.parameters():
            p.requires_grad_(False)
        try:
            for _ in range(self.T):
                x_t = x_t.requires_grad_(True)
                kl = self._kl(x_t, q)
                kl_val = kl.item()
                kl_traj.append(kl_val)

                if kl_val < self.kl_stop:
                    break

                grad = torch.autograd.grad(kl, x_t)[0]

                if self.grad_clip is not None:
                    n = grad.norm()
                    if n > self.grad_clip:
                        grad = grad * (self.grad_clip / (n + 1e-12))

                grad_norm = grad / (grad.norm() + 1e-12)
                x_t = (x_t - self.step_size * grad_norm).detach()

                if self.clip_pixels:
                    x_t = x_t.clamp(0.0, 1.0)

                path.append(x_t.clone())
        finally:
            for p, s in zip(self.model.parameters(), saved):
                p.requires_grad_(s)

        return path, kl_traj

    def _integrate(
        self,
        path: list[torch.Tensor],
        q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attr = torch.zeros_like(path[0])
        attr_sq = torch.zeros_like(path[0])

        saved = [p.requires_grad for p in self.model.parameters()]
        for p in self.model.parameters():
            p.requires_grad_(False)
        try:
            for t in range(len(path) - 1):
                x_t = path[t].clone().requires_grad_(True)
                kl = self._kl(x_t, q)
                grad = torch.autograd.grad(kl, x_t)[0].detach()
                dx = (path[t + 1] - path[t]).detach()
                attr += grad * dx
                attr_sq += (grad ** 2) * self.step_size
        finally:
            for p, s in zip(self.model.parameters(), saved):
                p.requires_grad_(s)

        return attr, attr_sq
