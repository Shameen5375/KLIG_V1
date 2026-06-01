#!/usr/bin/env python
"""
Scalar heatmap of (A_x − A_y) — first-component minus second-component
attribution — on a 100×100 grid for ∇f, IG, KLIG-Adapt, and KL-IG²-Adapt.

Functions: xor, checkerboard, diagonal_ckb, radial, flat_far_field

KLIG-Adapt:  KLIntegratedGradients + LinearPath, per-query adaptive σ
             (same algorithm as the image pipeline, SIGMA_F = 0.05 fallback)
KL-IG²-Adapt: KLIGSquared, φ=identity, x_cf=origin, per-query adaptive σ.

Outputs:
    results/toy_heatmap_attributions.png   (5 functions × 5 columns)
    results/toy_heatmap_attributions.svg   (with --svg)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXTENT           = 2.0
GRID_RESOLUTION  = 100       # attribution grid resolution
HEAT_N           = 240       # function heatmap resolution (background panel)

DEFAULT_PERCENTILE = 100.0

# ── Library path setup ────────────────────────────────────────────────────────
import os, sys
for _p in [str(Path(__file__).parent), "infocube-main", "."]:
    if os.path.isdir(os.path.join(_p, "klig")) and _p not in sys.path:
        sys.path.insert(0, _p)

from klig import KLIntegratedGradients, KLIGSquared
from klig.core.path import LinearPath
from klig.image.stopping import find_sigma_stop

# ── KLIG-Adapt / KL-IG²-Adapt hyperparams ────────────────────────────────────
SIGMA_FIXED       = 0.05     # reference notebook SIGMA_F — do NOT use ImageNet 0.25
ADAPTIVE_TAU      = 0.95
ADAPTIVE_N_SAMPLES= 32
ADAPTIVE_N_ITER   = 12
ADAPTIVE_FALLBACK = SIGMA_FIXED
ADAPTIVE_ZERO_THR = 0.05     # |f(x)| below this → use fallback σ

KLIG_N_STEPS     = 30
KLIG_N_MC        = 32        # MC samples per KLIG-Adapt step
KLIG_CONT_N_MC   = 1365      # continuous_klig: matches reference (30×1365≈40k evals)

IG2_T            = 15        # max gradient-descent steps for KL-IG²
IG2_LR_MU        = 0.05
IG2_LR_LV        = 0.10
IG2_N_MC_PATH    = 8
IG2_N_MC_GRAD    = 16
IG2_LOSS_STOP    = 1e-3
IG2_LV_CEIL      = 2.0 * math.log(EXTENT * 2 + 1e-9)


# ── Toy functions ─────────────────────────────────────────────────────────────

class XOR(nn.Module):
    """Smooth XOR: tanh(s·x) · tanh(s·y).  Origin is the saddle point."""
    def __init__(self, sharpness: float = 5.0):
        super().__init__()
        self.sharpness = sharpness

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.tanh(self.sharpness * x[..., 0])
                * torch.tanh(self.sharpness * x[..., 1]))


class Checkerboard(nn.Module):
    """cos·cos checkerboard, origin at the centre of a +1 square."""
    def __init__(self, period: float = 1.0, sharpness: float = 15.0):
        super().__init__()
        self.period    = period
        self.sharpness = sharpness

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = torch.cos(math.pi * x[..., 0] / self.period)
        v = torch.cos(math.pi * x[..., 1] / self.period)
        return torch.tanh(self.sharpness * u * v)


class DiagonalCheckerboard(nn.Module):
    """45°-rotated checkerboard: boundaries along x+y and x-y diagonals."""
    def __init__(self, period: float = 1.0, sharpness: float = 15.0):
        super().__init__()
        self.period    = period
        self.sharpness = sharpness

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u  = (x[..., 0] + x[..., 1]) / math.sqrt(2)
        v  = (x[..., 0] - x[..., 1]) / math.sqrt(2)
        cu = torch.cos(math.pi * u / self.period)
        cv = torch.cos(math.pi * v / self.period)
        return torch.tanh(self.sharpness * cu * cv)


class RadialRings(nn.Module):
    """Cosine of radius, attenuated by a Gaussian envelope."""
    def __init__(self, k_rings: float = 1.0, env_sigma: float = 1.5):
        super().__init__()
        self.k         = k_rings
        self.env_sigma = env_sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r   = torch.norm(x, dim=-1)
        env = torch.exp(-r ** 2 / (2 * self.env_sigma ** 2))
        return env * torch.cos(2 * math.pi * self.k * r)


class FlatFarFieldBumps(nn.Module):
    """≈0 everywhere except narrow, high-amplitude Gaussian bumps near origin."""
    def __init__(self,
                 centers=((0.15, -0.10), (-0.12, 0.18), (0.05, 0.05)),
                 amplitudes=(1.0, -0.85, 0.70),
                 sigma: float = 0.12):
        super().__init__()
        self.register_buffer("centers",    torch.tensor(centers,    dtype=torch.float32))
        self.register_buffer("amplitudes", torch.tensor(amplitudes, dtype=torch.float32))
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff  = x.unsqueeze(-2) - self.centers
        r2    = (diff ** 2).sum(-1)
        bumps = torch.exp(-r2 / (2 * self.sigma ** 2))
        return (bumps * self.amplitudes).sum(-1)


FUNCS = {
    "xor":            XOR(sharpness=5.0),
    "checkerboard":   Checkerboard(period=1.0, sharpness=15.0),
    "diagonal_ckb":   DiagonalCheckerboard(period=1.0, sharpness=15.0),
    "radial":         RadialRings(k_rings=1.0, env_sigma=1.5),
    "flat_far_field": FlatFarFieldBumps(),
}
FUNC_NAMES = list(FUNCS.keys())


# ── ToyClassifierWrapper ──────────────────────────────────────────────────────
# Wraps scalar fn → (B,2) logits for KLIGSquared and find_sigma_stop.

class ToyClassifierWrapper(nn.Module):
    """scalar 2D fn → (B, 2) logits  [f(x)·scale, −f(x)·scale]."""
    def __init__(self, fn: nn.Module, scale: float = 5.0):
        super().__init__()
        self.fn    = fn
        self.scale = scale
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logit = self.fn(x) * self.scale
        return torch.stack([logit, -logit], dim=1)


_SCALAR_TARGET = lambda y: y  # KLIntegratedGradients target for raw scalar fn


def _target_for_point(fn: nn.Module, x: torch.Tensor) -> int:
    with torch.no_grad():
        f_val = fn(x.unsqueeze(0)).item()
    return 0 if f_val >= 0 else 1


def _get_sigma_adaptive(clf: ToyClassifierWrapper, fn: nn.Module,
                         x: torch.Tensor) -> float:
    with torch.no_grad():
        f_val = fn(x.unsqueeze(0)).item()
    if abs(f_val) < ADAPTIVE_ZERO_THR:
        return ADAPTIVE_FALLBACK
    target = 0 if f_val >= 0 else 1
    return find_sigma_stop(
        clf, x, target=target,
        tau=ADAPTIVE_TAU,
        n_samples=ADAPTIVE_N_SAMPLES,
        n_iter=ADAPTIVE_N_ITER,
        sigma_hi=1.0,
    )


# ── Grid helpers ──────────────────────────────────────────────────────────────

def grid_points(n: int, extent: float = EXTENT) -> np.ndarray:
    xs = np.linspace(-extent, extent, n)
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    return np.stack([X.flatten(), Y.flatten()], axis=1).astype(np.float32)


def evaluate_heat(fn: nn.Module, n: int = HEAT_N,
                  extent: float = EXTENT) -> np.ndarray:
    fn = fn.to(DEVICE)
    xs = torch.linspace(-extent, extent, n)
    X, Y = torch.meshgrid(xs, xs, indexing="xy")
    pts = torch.stack([X.flatten(), Y.flatten()], dim=1).to(DEVICE)
    with torch.no_grad():
        return fn(pts).cpu().numpy().reshape(n, n)


# ── Attribution methods ───────────────────────────────────────────────────────

def gradient_field(fn: nn.Module, points: np.ndarray) -> np.ndarray:
    """∇f at all query points — single batched autograd call."""
    pts = torch.tensor(points, dtype=torch.float32, device=DEVICE).requires_grad_(True)
    y   = fn(pts)
    g   = torch.autograd.grad(y.sum(), pts)[0]
    return g.detach().cpu().numpy()


def integrated_gradients(fn: nn.Module, points: np.ndarray,
                          n_steps: int = 64) -> np.ndarray:
    """IG from (0,0) baseline — midpoint rule, batched over query points."""
    pts      = torch.tensor(points, dtype=torch.float32, device=DEVICE)
    B, D     = pts.shape
    baseline = torch.zeros(D, device=DEVICE)
    delta    = pts - baseline
    attr     = torch.zeros(B, D, device=DEVICE)
    for k in range(n_steps):
        alpha = (k + 0.5) / n_steps
        x_a   = (baseline + alpha * delta).requires_grad_(True)
        y     = fn(x_a)
        g     = torch.autograd.grad(y.sum(), x_a)[0]
        attr += g.detach() * delta / n_steps
    return attr.detach().cpu().numpy()


def continuous_klig(fn: nn.Module, points: np.ndarray,
                    n_steps: int = KLIG_N_STEPS,
                    n_mc:    int = KLIG_CONT_N_MC,
                    extent:  float = EXTENT,
                    eps:     float = 0.02,
                    seed:    int   = 0) -> np.ndarray:
    """Continuous uniform-path KLIG, fully batched over all query points.

    Path (per query point x_q, independent per-feature uniform):
        μ_t  = t · x_q                            (centre: 0 → x_q)
        hw_t = extent·(1 − t) + eps·t             (half-width: EXTENT → eps)
    At t=0: U(−extent, extent)²  — broad background
    At t=1: U(x_q − eps, x_q + eps)²  — near-delta at x_q
    """
    torch.manual_seed(seed)
    pts  = torch.tensor(points, dtype=torch.float32, device=DEVICE)
    B, D = pts.shape
    dt   = 1.0 / n_steps
    attr = torch.zeros(B, D, device=DEVICE)

    for k in range(n_steps):
        t    = (k + 0.5) * dt
        mu_t = t * pts
        hw_t = extent * (1.0 - t) + eps * t

        u      = torch.rand(B, n_mc, D, device=DEVICE)
        x_samp = mu_t.unsqueeze(1) + hw_t * (2.0 * u - 1.0)
        x_flat = x_samp.reshape(B * n_mc, D).requires_grad_(True)

        y     = fn(x_flat)
        grads = torch.autograd.grad(y.sum(), x_flat)[0]
        grads = grads.reshape(B, n_mc, D).detach()

        dE_dmu = grads.mean(1)
        dE_dhw = ((2.0 * u.detach() - 1.0) * grads).mean(1)
        attr  += (dE_dmu * pts + dE_dhw * (eps - extent)) * dt

    return attr.detach().cpu().numpy()


def continuous_klig2(fn: nn.Module, points: np.ndarray,
                     n_steps: int  = KLIG_N_STEPS,
                     n_mc:    int  = KLIG_CONT_N_MC,
                     sigma:   float = SIGMA_FIXED,
                     seed:    int   = 0) -> np.ndarray:
    """Continuous KL-IG², fully batched — mirrors continuous_klig structure.

    Path (linear in μ-space, fixed σ):
        μ_t  = (1 − t) · x_q        (explicand → origin as t: 0 → 1)
        σ_t  = sigma                 (constant — same as SIGMA_FIXED)
    Samples: x_samp = μ_t + σ · ε,  ε ~ N(0, I)

    Attribution (backward-displacement chain rule):
        attr_i = Σ_k  E[∂f/∂x_i(x_samp_k)]  ·  dμ_k_i
    where  dμ_k_i = μ_{t_k}_i − μ_{t_{k+1}}_i = x_q_i / n_steps

    Completeness: Σ attr_i ≈ f(x_q) − f(0)  as σ → 0.
    """
    torch.manual_seed(seed)
    pts  = torch.tensor(points, dtype=torch.float32, device=DEVICE)
    B, D = pts.shape
    dt   = 1.0 / n_steps
    attr = torch.zeros(B, D, device=DEVICE)

    for k in range(n_steps):
        t      = (k + 0.5) * dt
        mu_t   = (1.0 - t) * pts                            # (B, D)

        eps    = torch.randn(B, n_mc, D, device=DEVICE)
        x_samp = mu_t.unsqueeze(1) + sigma * eps            # (B, n_mc, D)
        x_flat = x_samp.reshape(B * n_mc, D).requires_grad_(True)

        y     = fn(x_flat)
        grads = torch.autograd.grad(y.sum(), x_flat)[0]
        grads = grads.reshape(B, n_mc, D).detach()

        dE_dmu = grads.mean(1)                              # (B, D)
        attr  += dE_dmu * pts * dt                          # dμ_k = pts * dt

    return attr.detach().cpu().numpy()


def klig_adapt_field(fn: nn.Module, points: np.ndarray,
                     clf: ToyClassifierWrapper) -> np.ndarray:
    """KLIG-Adapt: KLIntegratedGradients + LinearPath + per-point adaptive σ."""
    attr_ = np.zeros_like(points)
    for i in range(len(points)):
        x     = torch.tensor(points[i], dtype=torch.float32, device=DEVICE)
        sigma = _get_sigma_adaptive(clf, fn, x)
        ig    = KLIntegratedGradients(
            fn, n_steps=KLIG_N_STEPS, n_samples=KLIG_N_MC,
            sigma_final=sigma, path=LinearPath(), device=DEVICE,
        )
        r = ig.attribute(x, target=_SCALAR_TARGET)
        attr_[i] = r.attr.detach().cpu().numpy()
    return attr_


def _make_klig2(clf: ToyClassifierWrapper, sigma: float) -> KLIGSquared:
    return KLIGSquared(
        model=clf,
        phi=lambda x: x,
        x_cf=torch.zeros(2, device=DEVICE),
        T=IG2_T,
        lr_mu=IG2_LR_MU,
        lr_lv=IG2_LR_LV,
        n_mc_path=IG2_N_MC_PATH,
        n_mc_grad=IG2_N_MC_GRAD,
        sigma_start=sigma,
        loss_stop=IG2_LOSS_STOP,
        lv_floor=2.0 * math.log(sigma),
        lv_ceil=IG2_LV_CEIL,
        mu_min=-EXTENT,
        mu_max=EXTENT,
        clamp_samples=True,
        device=DEVICE,
    )


def klig2_adapt_field(fn: nn.Module, points: np.ndarray,
                      clf: ToyClassifierWrapper) -> np.ndarray:
    """KL-IG²-Adapt: KLIGSquared + per-point adaptive σ. Returns attr_mu."""
    attr_ = np.zeros_like(points)
    for i in range(len(points)):
        x      = torch.tensor(points[i], dtype=torch.float32, device=DEVICE)
        sigma  = _get_sigma_adaptive(clf, fn, x)
        target = _target_for_point(fn, x)
        ig2    = _make_klig2(clf, sigma)
        r      = ig2.attribute(x, target=target)
        attr_[i] = r.attr_mu.detach().cpu().numpy()
    return attr_


# ── Methods registry ──────────────────────────────────────────────────────────

METHODS = [
    ("∇f",           "grad"),
    ("IG",            "ig"),
    ("KLIG",          "klig"),
    ("KL-IG²",        "klig2_cont"),
    ("KLIG-Adapt",    "klig_adapt"),
    ("KL-IG²-Adapt",  "klig2_adapt"),
]


# ── compute_all ───────────────────────────────────────────────────────────────

def compute_all(n: int = GRID_RESOLUTION) -> dict[str, np.ndarray]:
    points   = grid_points(n)
    clf_dict = {name: ToyClassifierWrapper(fn.to(DEVICE)).to(DEVICE)
                for name, fn in FUNCS.items()}
    data: dict[str, np.ndarray] = {}

    for name, fn in FUNCS.items():
        fn  = fn.to(DEVICE)
        clf = clf_dict[name]
        print(f"[{name}] computing heat + attributions on {n}×{n} grid ...",
              flush=True)
        t0 = time()

        data[f"{name}_heat"]         = evaluate_heat(fn)
        data[f"{name}_grad"]         = gradient_field(fn, points)
        data[f"{name}_ig"]           = integrated_gradients(fn, points)
        data[f"{name}_klig"]         = continuous_klig(fn, points)
        data[f"{name}_klig2_cont"]   = continuous_klig2(fn, points)
        data[f"{name}_klig_adapt"]   = klig_adapt_field(fn, points, clf)
        data[f"{name}_klig2_adapt"]  = klig2_adapt_field(fn, points, clf)

        print(f"  done in {time() - t0:.1f}s")

    return data


# ── render ────────────────────────────────────────────────────────────────────

def render(data: dict[str, np.ndarray],
           out_path: Path,
           n: int               = GRID_RESOLUTION,
           percentile: float    = DEFAULT_PERCENTILE,
           min_ref: float       = 0.1,
           extra_formats: tuple = ()) -> None:
    """One figure: rows = functions, cols = f heatmap + attribution diff panels."""
    nrows = len(FUNC_NAMES)
    ncols = 1 + len(METHODS)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(2.6 * ncols, 2.6 * nrows + 0.3))
    if nrows == 1:
        axes = axes.reshape(1, ncols)

    for ri, name in enumerate(FUNC_NAMES):
        heat = data[f"{name}_heat"]
        vmax = max(1e-6, float(np.abs(heat).max()))

        # col 0: function heatmap
        ax = axes[ri, 0]
        im = ax.imshow(heat, extent=[-EXTENT, EXTENT, -EXTENT, EXTENT],
                       origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax.set_ylabel(name, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        if ri == 0:
            ax.set_title("f", fontsize=11)

        # cols 1+: A_x − A_y diff heatmaps
        for ci, (mname, key) in enumerate(METHODS, start=1):
            attrs = data[f"{name}_{key}"]          # (N², 2)
            diff  = (attrs[:, 0] - attrs[:, 1]).reshape(n, n)
            ref   = (float(np.percentile(np.abs(diff), percentile))
                     if percentile < 100 else float(np.abs(diff).max()))
            ref   = max(ref, min_ref)

            ax = axes[ri, ci]
            ax.imshow(diff, extent=[-EXTENT, EXTENT, -EXTENT, EXTENT],
                      origin="lower", cmap="PuOr", vmin=-ref, vmax=ref)
            if ri == 0:
                ax.set_title(mname, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(
                0.02, 0.97,
                f"max={np.abs(diff).max():.2f}\npct{percentile:g}={ref:.2f}",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          alpha=0.7, edgecolor="none"),
            )

    fig.suptitle(
        f"A_x − A_y attribution diff  ({n}×{n} grid, "
        f"±pct{percentile:g}|A_x−A_y| per panel; "
        f"KLIG/KL-IG² σ_fallback={SIGMA_FIXED})",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.96])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    for fmt in extra_formats:
        alt = out_path.with_suffix(f".{fmt}")
        fig.savefig(alt, bbox_inches="tight")
        print(f"Saved {alt}")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Heatmap attributions for 5 toy 2D functions.")
    p.add_argument("--percentile", type=float, default=DEFAULT_PERCENTILE,
                   help="Per-panel |A_x−A_y| percentile for colour scale "
                        "(100 = max, default)")
    p.add_argument("--min-ref", type=float, default=0.1,
                   help="Floor on per-panel colour range (prevents near-zero "
                        "panels from magnifying noise; default 0.1)")
    p.add_argument("--svg", action="store_true",
                   help="Also save output as SVG")
    p.add_argument("--grid-resolution", type=int, default=GRID_RESOLUTION,
                   help=f"Attribution grid resolution (default {GRID_RESOLUTION})")
    args = p.parse_args()

    data = compute_all(n=args.grid_resolution)

    suffix        = ("" if args.percentile == 100.0
                     else f"_pct{int(args.percentile)}")
    extra_formats = ("svg",) if args.svg else ()
    render(data,
           out_path      = OUT / f"toy_heatmap_attributions{suffix}.png",
           n             = args.grid_resolution,
           percentile    = args.percentile,
           min_ref       = args.min_ref,
           extra_formats = extra_formats)


if __name__ == "__main__":
    main()
