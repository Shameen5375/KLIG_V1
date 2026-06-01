#!/usr/bin/env python
"""
Toy 2D heatmaps — KLIG & KL-IG² all variants.

Renders A_{x1} - A_{x2} attribution difference maps on a GRID_RESOLUTION×GRID_RESOLUTION
grid for 5 scalar 2D functions.  Orange = credits x1, Purple = credits x2.

Methods:
    ∇f            — vanilla gradient (baseline)
    IG            — standard Integrated Gradients (zero baseline, deterministic)
    KLIG-Lin      — KLIntegratedGradients + LinearPath, fixed σ=SIGMA_FIXED
    KLIG-Adapt    — KLIntegratedGradients + LinearPath, per-query adaptive σ
    KL-IG²        — KLIGSquared, φ=identity, x_cf=origin, fixed σ, attr_mu only
    KL-IG²-Adapt  — KLIGSquared, φ=identity, x_cf=origin, adaptive σ, attr_mu only

Output: results/toy_klig_klig2_heatmaps.png
"""

from __future__ import annotations

import math
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

warnings.filterwarnings("ignore")

for _p in ["infocube-main", "."]:
    if os.path.isdir(os.path.join(_p, "klig")) and _p not in sys.path:
        sys.path.insert(0, _p)

from klig import KLIntegratedGradients, KLIGSquared
from klig.core.path import LinearPath
from klig.image.stopping import find_sigma_stop

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)

# ── Config ─────────────────────────────────────────────────────────────────────
EXTENT          = 2.0
GRID_RESOLUTION = 40      # 40×40 = 1600 query points
HEAT_N          = 200     # background f(x) heatmap resolution

N_STEPS         = 24      # integration steps for KLIG variants
N_SAMPLES       = 8       # MC samples for KLIG variants
SIGMA_FIXED     = 0.25    # fixed σ for KLIG-Lin and KL-IG²

IG2_T           = 15      # max descent steps for KL-IG²
IG2_LR_MU       = 0.05
IG2_LR_LV       = 0.10
IG2_N_MC_PATH   = 4       # MC samples per descent step
IG2_N_MC_GRAD   = 4       # MC samples per integration gradient estimate
IG2_LOSS_STOP   = 1e-3
IG2_LV_FLOOR    = 2.0 * math.log(SIGMA_FIXED)          # ≈ −2.77
IG2_LV_CEIL     = 2.0 * math.log(EXTENT * 2 + 1e-9)   # logvar cap

ADAPTIVE_TAU        = 0.95
ADAPTIVE_N_SAMPLES  = 32
ADAPTIVE_N_ITER     = 12
ADAPTIVE_FALLBACK   = SIGMA_FIXED
ADAPTIVE_ZERO_THR   = 0.05   # |f(x)| below this → use fallback σ

# ── Toy 2D scalar functions ────────────────────────────────────────────────────

class XOR(nn.Module):
    def __init__(self, s: float = 5.0):
        super().__init__()
        self.s = s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.s * x[..., 0]) * torch.tanh(self.s * x[..., 1])


class Checkerboard(nn.Module):
    def __init__(self, period: float = 1.0, s: float = 15.0):
        super().__init__()
        self.period = period
        self.s = s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = torch.cos(math.pi * x[..., 0] / self.period)
        v = torch.cos(math.pi * x[..., 1] / self.period)
        return torch.tanh(self.s * u * v)


class DiagonalCheckerboard(nn.Module):
    def __init__(self, period: float = 1.0, s: float = 15.0):
        super().__init__()
        self.period = period
        self.s = s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = (x[..., 0] + x[..., 1]) / math.sqrt(2)
        v = (x[..., 0] - x[..., 1]) / math.sqrt(2)
        cu = torch.cos(math.pi * u / self.period)
        cv = torch.cos(math.pi * v / self.period)
        return torch.tanh(self.s * cu * cv)


class RadialRings(nn.Module):
    def __init__(self, k: float = 1.0, sigma: float = 1.5):
        super().__init__()
        self.k = k
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.norm(x, dim=-1)
        env = torch.exp(-(r ** 2) / (2 * self.sigma ** 2))
        return env * torch.cos(2 * math.pi * self.k * r)


class FlatFarFieldBumps(nn.Module):
    def __init__(
        self,
        centers=((0.15, -0.10), (-0.12, 0.18), (0.05, 0.05)),
        amps=(1.0, -0.85, 0.70),
        sigma: float = 0.12,
    ):
        super().__init__()
        self.register_buffer("centers", torch.tensor(centers, dtype=torch.float32))
        self.register_buffer("amps", torch.tensor(amps, dtype=torch.float32))
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(-2) - self.centers
        r2 = (diff ** 2).sum(-1)
        return (torch.exp(-r2 / (2 * self.sigma ** 2)) * self.amps).sum(-1)


FUNCS = {
    "xor":            XOR(5.0).to(DEVICE),
    "checkerboard":   Checkerboard(1.0, 15.0).to(DEVICE),
    "diagonal_ckb":   DiagonalCheckerboard(1.0, 15.0).to(DEVICE),
    "radial":         RadialRings(1.0, 1.5).to(DEVICE),
    "flat_far_field": FlatFarFieldBumps().to(DEVICE),
}
FUNC_NAMES = list(FUNCS)
print("functions:", FUNC_NAMES)

# ── ToyClassifierWrapper ──────────────────────────────────────────────────────
# Wraps a scalar fn as (B,2) binary logit model so KLIGSquared and
# find_sigma_stop get a proper nn.Module with parameters().


class ToyClassifierWrapper(nn.Module):
    """scalar 2D function → (B, 2) logits  [f(x)·scale, −f(x)·scale]."""

    def __init__(self, fn: nn.Module, scale: float = 5.0):
        super().__init__()
        self.fn    = fn
        self.scale = scale
        self._dev  = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logit = self.fn(x) * self.scale          # (B,)
        return torch.stack([logit, -logit], dim=1)  # (B, 2)


# Scalar target for KLIntegratedGradients — passes fn output directly.
_SCALAR_TARGET = lambda y: y


def _target_for_point(fn: nn.Module, x: torch.Tensor) -> int:
    """Class 0 if f(x) ≥ 0, class 1 otherwise (matches ToyClassifierWrapper)."""
    with torch.no_grad():
        f_val = fn(x.unsqueeze(0)).item()
    return 0 if f_val >= 0 else 1


def _get_sigma_adaptive(clf: ToyClassifierWrapper, fn: nn.Module, x: torch.Tensor) -> float:
    """Per-point adaptive σ via find_sigma_stop (same algorithm as image pipeline)."""
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


def evaluate_heat(fn: nn.Module, n: int = HEAT_N) -> np.ndarray:
    xs = torch.linspace(-EXTENT, EXTENT, n, device=DEVICE)
    X, Y = torch.meshgrid(xs, xs, indexing="xy")
    pts = torch.stack([X.flatten(), Y.flatten()], dim=1)
    with torch.no_grad():
        return fn(pts).cpu().numpy().reshape(n, n)


# ── Attribution methods ───────────────────────────────────────────────────────

def gradient_field(fn: nn.Module, points: np.ndarray) -> np.ndarray:
    """∇f evaluated at each query point (batched)."""
    pts = torch.tensor(points, device=DEVICE).requires_grad_(True)
    y   = fn(pts)
    g   = torch.autograd.grad(y.sum(), pts)[0]
    return g.detach().cpu().numpy()


def integrated_gradients(fn: nn.Module, points: np.ndarray, n_steps: int = 64) -> np.ndarray:
    """Standard IG, zero baseline, deterministic trapezoid (batched over all points)."""
    pts  = torch.tensor(points, device=DEVICE)   # (B, 2)
    attr = torch.zeros_like(pts)
    for k in range(n_steps):
        alpha = (k + 0.5) / n_steps
        xa    = (alpha * pts).requires_grad_(True)
        g     = torch.autograd.grad(fn(xa).sum(), xa)[0]
        attr += g.detach() * pts / n_steps
    return attr.cpu().numpy()


def klig_lin_field(fn: nn.Module, points: np.ndarray) -> np.ndarray:
    """KLIG-Lin: KLIntegratedGradients + LinearPath + fixed σ=SIGMA_FIXED."""
    attr_ = np.zeros_like(points)
    ig    = KLIntegratedGradients(
        fn, n_steps=N_STEPS, n_samples=N_SAMPLES,
        sigma_final=SIGMA_FIXED, path=LinearPath(), device=DEVICE,
    )
    for i in range(len(points)):
        x = torch.tensor(points[i], device=DEVICE)
        r = ig.attribute(x, target=_SCALAR_TARGET)
        attr_[i] = r.attr.detach().cpu().numpy()
    return attr_


def klig_adapt_field(
    fn: nn.Module,
    points: np.ndarray,
    clf: ToyClassifierWrapper,
) -> np.ndarray:
    """KLIG-Adapt: KLIntegratedGradients + LinearPath + per-query adaptive σ."""
    attr_ = np.zeros_like(points)
    for i in range(len(points)):
        x     = torch.tensor(points[i], device=DEVICE)
        sigma = _get_sigma_adaptive(clf, fn, x)
        ig    = KLIntegratedGradients(
            fn, n_steps=N_STEPS, n_samples=N_SAMPLES,
            sigma_final=sigma, path=LinearPath(), device=DEVICE,
        )
        r = ig.attribute(x, target=_SCALAR_TARGET)
        attr_[i] = r.attr.detach().cpu().numpy()
    return attr_


def _make_klig2(clf: ToyClassifierWrapper, sigma: float) -> KLIGSquared:
    """Build a KLIGSquared instance for a 2D toy function."""
    return KLIGSquared(
        model=clf,
        phi=lambda x: x,                          # identity representation
        x_cf=torch.zeros(2, device=DEVICE),       # origin as counterfactual
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


def klig2_field(
    fn: nn.Module,
    points: np.ndarray,
    clf: ToyClassifierWrapper,
) -> np.ndarray:
    """KL-IG²: KLIGSquared, φ=identity, x_cf=origin, fixed σ. Returns attr_mu only."""
    attr_ = np.zeros_like(points)
    ig2   = _make_klig2(clf, SIGMA_FIXED)
    for i in range(len(points)):
        x      = torch.tensor(points[i], device=DEVICE)
        target = _target_for_point(fn, x)
        r      = ig2.attribute(x, target=target)
        attr_[i] = r.attr_mu.detach().cpu().numpy()
    return attr_


def klig2_adapt_field(
    fn: nn.Module,
    points: np.ndarray,
    clf: ToyClassifierWrapper,
) -> np.ndarray:
    """KL-IG²-Adapt: KLIGSquared with per-query adaptive σ. Returns attr_mu only."""
    attr_ = np.zeros_like(points)
    for i in range(len(points)):
        x      = torch.tensor(points[i], device=DEVICE)
        sigma  = _get_sigma_adaptive(clf, fn, x)
        target = _target_for_point(fn, x)
        ig2    = _make_klig2(clf, sigma)
        r      = ig2.attribute(x, target=target)
        attr_[i] = r.attr_mu.detach().cpu().numpy()
    return attr_


# ── Compute attributions for every function × method ─────────────────────────

points   = grid_points(GRID_RESOLUTION)
clf_dict = {fname: ToyClassifierWrapper(fn).to(DEVICE) for fname, fn in FUNCS.items()}
data: dict = {}

METHODS_KEYS = [
    ("∇f",           "grad"),
    ("IG",            "ig"),
    ("KLIG-Lin",      "klig_lin"),
    ("KLIG-Adapt",    "klig_adapt"),
    ("KL-IG²",        "klig2"),
    ("KL-IG²-Adapt",  "klig2_adapt"),
]

for fname, fn in FUNCS.items():
    clf = clf_dict[fname]
    data[f"{fname}_heat"] = evaluate_heat(fn)
    print(f"\n[{fname}]")

    method_runners = [
        ("∇f",           "grad",       lambda: gradient_field(fn, points)),
        ("IG",           "ig",         lambda: integrated_gradients(fn, points)),
        ("KLIG-Lin",     "klig_lin",   lambda: klig_lin_field(fn, points)),
        ("KLIG-Adapt",   "klig_adapt", lambda: klig_adapt_field(fn, points, clf)),
        ("KL-IG²",       "klig2",      lambda: klig2_field(fn, points, clf)),
        ("KL-IG²-Adapt", "klig2_adapt",lambda: klig2_adapt_field(fn, points, clf)),
    ]

    for mname, key, runner in method_runners:
        t0 = time.time()
        data[f"{fname}_{key}"] = runner()
        print(f"  {mname:14s}  {time.time()-t0:.1f}s")

print("\nAll attributions computed.")

# ── Render heatmaps ───────────────────────────────────────────────────────────

CLIP_PCT = 98.0
n        = GRID_RESOLUTION

METHOD_COLORS = {
    "∇f":            "#555555",
    "IG":             "#333333",
    "KLIG-Lin":       "#1f77b4",
    "KLIG-Adapt":     "#2d6a2d",
    "KL-IG²":         "#e41a1c",
    "KL-IG²-Adapt":   "#8b0000",
}

FUNC_LABELS = {
    "xor":            "XOR",
    "checkerboard":   "Checkerboard",
    "diagonal_ckb":   "Diagonal\nCheckerboard",
    "radial":         "Radial Rings",
    "flat_far_field": "Flat Far-Field\nBumps",
}

nrows = len(FUNC_NAMES)
ncols = 1 + len(METHODS_KEYS)

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(2.3 * ncols, 2.5 * nrows),
    facecolor="white",
    gridspec_kw={"wspace": 0.04, "hspace": 0.12},
)
if nrows == 1:
    axes = axes.reshape(1, ncols)

for ri, fname in enumerate(FUNC_NAMES):
    # col 0: function heatmap
    heat = data[f"{fname}_heat"]
    vmax = max(1e-6, float(np.abs(heat).max()))
    ax0  = axes[ri, 0]
    im0  = ax0.imshow(
        heat, extent=[-EXTENT, EXTENT, -EXTENT, EXTENT],
        origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        interpolation="bilinear",
    )
    ax0.set_xticks([]); ax0.set_yticks([])
    ax0.set_ylabel(
        FUNC_LABELS.get(fname, fname), fontsize=9,
        rotation=90, labelpad=4, va="center",
    )
    if ri == 0:
        ax0.set_title("f(x₁, x₂)", fontsize=9, fontweight="bold", pad=5)
    div = make_axes_locatable(ax0)
    cax = div.append_axes("right", size="8%", pad=0.04)
    cb  = plt.colorbar(im0, cax=cax)
    cb.ax.tick_params(labelsize=6)

    # cols 1–N: A_{x1} − A_{x2} attribution diff maps
    for ci, (mname, key) in enumerate(METHODS_KEYS, start=1):
        attrs = data[f"{fname}_{key}"]
        diff  = (attrs[:, 0] - attrs[:, 1]).reshape(n, n)
        ref   = max(float(np.percentile(np.abs(diff), CLIP_PCT)), 1e-3)
        ax    = axes[ri, ci]
        ax.imshow(
            diff, extent=[-EXTENT, EXTENT, -EXTENT, EXTENT],
            origin="lower", cmap="PuOr",
            vmin=-ref, vmax=ref, interpolation="bilinear",
        )
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if ri == 0:
            col = METHOD_COLORS.get(mname, "black")
            ax.set_title(mname, fontsize=8.5, fontweight="bold", color=col, pad=5)

fig.text(0.55, -0.01, "x₁", ha="center", va="bottom", fontsize=10)
fig.suptitle(
    r"$A_{x_1} - A_{x_2}$  attribution difference  —  KLIG & KL-IG² variants"
    f"\n({n}×{n} grid, σ_fixed={SIGMA_FIXED})   "
    "Orange → credits x₁   |   Purple → credits x₂",
    fontsize=11, fontweight="bold", y=1.01,
)

os.makedirs("results", exist_ok=True)
out_path = "results/toy_klig_klig2_heatmaps.png"
plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
print(f"\nSaved {out_path}")
plt.show()
