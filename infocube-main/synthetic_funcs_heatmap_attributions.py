"""
Scalar heatmap of (A_x − A_y) — first-component minus second-component
attribution — on a 100×100 grid for ∇f, IG, KLIG, and SHAP.

Functions: xor, checkerboard, diagonal_ckb, radial, flat_far_field

Background distribution for SHAP: uniform on [-EXTENT, EXTENT]^2.
KLIG uses the same implicit background (uniform path from wide to narrow
around the query point).

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

EXTENT       = 2.0
GRID_RESOLUTION      = 100     # attribution grid resolution
HEAT_N       = 240     # function heatmap resolution (background panel)
BG_PER_DIM   = 100     # SHAP background grid per dimension (= GRID_RESOLUTION)

DEFAULT_PERCENTILE = 100.0


# ── Toy functions ────────────────────────────────────────────────────────

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
    """45°-rotated checkerboard: boundaries along x+y and x-y diagonals.

    Exposes axis-aligned methods (SHAP, IG): marginalising or integrating
    along one axis crosses the same number of sign-changes as the regular
    checkerboard, but the function is only separable in (x+y, x−y), not
    (x, y) — so axis-aligned attributions systematically mis-describe the
    interaction structure.
    """
    def __init__(self, period: float = 1.0, sharpness: float = 15.0):
        super().__init__()
        self.period    = period
        self.sharpness = sharpness

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = (x[..., 0] + x[..., 1]) / math.sqrt(2)
        v = (x[..., 0] - x[..., 1]) / math.sqrt(2)
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
    """≈0 everywhere except narrow, high-amplitude Gaussian bumps near origin.

    IG shadow effect: a straight-line path from a far baseline to a far
    target that transits the bump cluster integrates large bump gradients
    and projects them onto the (target − baseline) direction, giving
    non-zero attribution for features that only describe far-field position.
    KLIG avoids this because its path stays in distribution space.
    """
    def __init__(self,
                 centers=((0.15, -0.10), (-0.12, 0.18), (0.05, 0.05)),
                 amplitudes=(1.0, -0.85, 0.70),
                 sigma: float = 0.12):
        super().__init__()
        self.register_buffer("centers",    torch.tensor(centers,    dtype=torch.float32))
        self.register_buffer("amplitudes", torch.tensor(amplitudes, dtype=torch.float32))
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff  = x.unsqueeze(-2) - self.centers   # (..., K, 2)
        r2    = (diff ** 2).sum(-1)               # (..., K)
        bumps = torch.exp(-r2 / (2 * self.sigma ** 2))
        return (bumps * self.amplitudes).sum(-1)


FUNCS = {
    "xor":          XOR(sharpness=5.0),
    "checkerboard": Checkerboard(period=1.0, sharpness=15.0),
    "diagonal_ckb": DiagonalCheckerboard(period=1.0, sharpness=15.0),
    "radial":       RadialRings(k_rings=1.0, env_sigma=1.5),
    "flat_far_field": FlatFarFieldBumps(),
}

FUNC_NAMES = list(FUNCS.keys())


# ── Grid helpers ─────────────────────────────────────────────────────────

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


# ── Attribution methods ──────────────────────────────────────────────────

def gradient_field(fn: nn.Module, points: np.ndarray) -> np.ndarray:
    """∇f at all query points — single batched autograd call."""
    pts = torch.tensor(points, dtype=torch.float32, device=DEVICE).requires_grad_(True)
    y = fn(pts)
    g = torch.autograd.grad(y.sum(), pts)[0]
    return g.detach().cpu().numpy()


def integrated_gradients(fn: nn.Module, points: np.ndarray,
                          n_steps: int = 64) -> np.ndarray:
    """IG from the (0, 0) baseline — midpoint rule, batched over query points."""
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


# KLIG (KLIG) -----------------------------------

KLIG_N_STEPS = 30
KLIG_N_MC    = 1365   # 30 × 1365 ≈ 40 950 grad evals per query point

def continuous_klig(fn: nn.Module, points: np.ndarray,
                    n_steps: int = KLIG_N_STEPS,
                    n_mc:    int = KLIG_N_MC,
                    extent:  float = EXTENT,
                    eps:     float = 0.02,
                    seed:    int   = 0) -> np.ndarray:
    """Continuous uniform-path KLIG, fully batched over all query points.

    Path (per query point x_q, independent per-feature uniform):
        μ_t  = t · x_q                            (centre: 0 → x_q)
        hw_t = extent·(1 − t) + eps·t             (half-width: EXTENT → eps)
    At t=0: U(−extent, extent)²  — broad background
    At t=1: U(x_q − eps, x_q + eps)²  — near-delta at x_q

    Attribution (chain rule through reparameterisation):
        attr_i = ∫₀¹ [ E[∂f/∂x_i] · x_q_i
                       + E[(2u_i−1)·∂f/∂x_i] · (eps−extent) ] dt

    Completeness: Σ attr_i ≈ f(x_q) − E_{U(−extent,extent)²}[f].
    """
    torch.manual_seed(seed)
    pts  = torch.tensor(points, dtype=torch.float32, device=DEVICE)  # (B, 2)
    B, D = pts.shape
    dt   = 1.0 / n_steps
    attr = torch.zeros(B, D, device=DEVICE)

    for k in range(n_steps):
        t    = (k + 0.5) * dt
        mu_t = t * pts                                # (B, 2)
        hw_t = extent * (1.0 - t) + eps * t          # scalar

        u      = torch.rand(B, n_mc, D, device=DEVICE)
        x_samp = mu_t.unsqueeze(1) + hw_t * (2.0 * u - 1.0)   # (B, n_mc, D)
        x_flat = x_samp.reshape(B * n_mc, D).requires_grad_(True)

        y     = fn(x_flat)
        grads = torch.autograd.grad(y.sum(), x_flat)[0]         # (B*n_mc, D)
        grads = grads.reshape(B, n_mc, D).detach()

        dE_dmu = grads.mean(1)                                   # (B, D)
        dE_dhw = ((2.0 * u.detach() - 1.0) * grads).mean(1)    # (B, D)

        attr += (dE_dmu * pts + dE_dhw * (eps - extent)) * dt

    return attr.detach().cpu().numpy()


# Exact SHAP for 2D with independent uniform background ------------------

def shap_2d(fn: nn.Module, points: np.ndarray,
            bg_n:   int   = BG_PER_DIM,
            extent: float = EXTENT) -> np.ndarray:
    """Exact Shapley values for 2 independent features.

        SHAP_x = ½ [(g_x(x₁) − μ) + (f(x) − g_y(x₂))]
        SHAP_y = ½ [(g_y(x₂) − μ) + (f(x) − g_x(x₁))]

    where μ = E[f], g_x(x₁) = E_{X₂}[f(x₁, X₂)], g_y(x₂) = E_{X₁}[f(X₁, x₂)].
    Background: uniform bg_n × bg_n grid over [−extent, extent]².
    Completeness: SHAP_x + SHAP_y = f(x) − μ.
    """
    bg_xs = torch.linspace(-extent, extent, bg_n, device=DEVICE)
    BX, BY = torch.meshgrid(bg_xs, bg_xs, indexing="xy")
    bg_pts = torch.stack([BX.flatten(), BY.flatten()], dim=1)
    with torch.no_grad():
        bg_vals = fn(bg_pts).reshape(bg_n, bg_n)
    mu      = bg_vals.mean()
    g_x_grid = bg_vals.mean(dim=0)   # marginalise y for each x
    g_y_grid = bg_vals.mean(dim=1)   # marginalise x for each y

    pts = torch.tensor(points, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        fx = fn(pts)

    def nearest(vals, grid):
        return torch.argmin((vals.unsqueeze(1) - grid.unsqueeze(0)).abs(), dim=1)

    i_idx = nearest(pts[:, 0], bg_xs)
    j_idx = nearest(pts[:, 1], bg_xs)
    g_x = g_x_grid[i_idx]
    g_y = g_y_grid[j_idx]

    shap_x = 0.5 * ((g_x - mu) + (fx - g_y))
    shap_y = 0.5 * ((g_y - mu) + (fx - g_x))
    return torch.stack([shap_x, shap_y], dim=1).cpu().numpy()


# ── Compute all attributions ─────────────────────────────────────────────

METHODS = [
    ("∇f",    "grad"),
    ("IG",    "ig"),
    ("KLIG", "klig"),
    ("SHAP",  "shap"),
]


def compute_all(n: int = GRID_RESOLUTION) -> dict[str, np.ndarray]:
    points = grid_points(n)
    data: dict[str, np.ndarray] = {}

    for name, fn in FUNCS.items():
        fn = fn.to(DEVICE)
        print(f"[{name}] computing heat + attributions on {n}×{n} grid ...",
              flush=True)
        t0 = time()

        data[f"{name}_heat"]  = evaluate_heat(fn)
        data[f"{name}_grad"]  = gradient_field(fn, points)
        data[f"{name}_ig"]    = integrated_gradients(fn, points)
        data[f"{name}_klig"] = continuous_klig(fn, points)
        data[f"{name}_shap"]  = shap_2d(fn, points, bg_n=n)

        print(f"  done in {time() - t0:.1f}s")

    return data


# ── Render ───────────────────────────────────────────────────────────────

def render(data: dict[str, np.ndarray],
           out_path: Path,
           n: int               = GRID_RESOLUTION,
           percentile: float    = DEFAULT_PERCENTILE,
           min_ref: float       = 0.1,
           extra_formats: tuple = ()) -> None:
    """One figure: rows = functions, cols = f heatmap + attribution diff panels."""
    nrows  = len(FUNC_NAMES)
    ncols  = 1 + len(METHODS)
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
            attrs = data[f"{name}_{key}"]            # (N², 2)
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
        f"SHAP/KLIG background = Unif[−{EXTENT}, {EXTENT}]²)",
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


# ── Main ─────────────────────────────────────────────────────────────────

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
