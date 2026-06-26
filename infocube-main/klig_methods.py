"""
Unified attribution-method dispatch for the KL-IG² evaluation.

Eleven methods behind one entry point:
  baselines : Vanilla Grad, SmoothGrad, IG-zero, Blur-IG, IDG, Guided IG, ExpGrad
  KLIG fam  : KLIG-Adaptive, KL-IG (linear), KL-IG², KL-IG² (adaptive)

    from klig_methods import attr_map, METHODS, needs_cf, make_phi
    phi = make_phi(model)
    m = attr_map("Blur-IG", model, x, target)               # baselines: no CF
    m = attr_map("KL-IG² (adaptive)", model, x, target, x_cf=cf, phi=phi)

`attr_map(...)` returns an absmax-collapsed **signed** (H, W) torch tensor on CPU.

Baseline code is copied verbatim from evaluation_main_notebook_updated.ipynb (cells 9, 10);
KL-IG² path machinery matches kl_ig2__eval.ipynb (cells 8, 26).
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients, Saliency

from klig import KLIntegratedGradients, make_phi_from_layer, KLIGSquared
from klig.image.stopping import find_sigma_stop
from klig.core.path import LinearPath

# ── config (methods nb cell 4 + eval nb cell 3) ──────────────────────────────
N_STEPS, N_SAMPLES = 50, 10          # KLIG IG quadrature / MC
IG_STEPS = 50                        # captum IG steps
SG_SAMPLES = 50                      # SmoothGrad samples
EG_SAMPLES = 50                      # ExpGrad samples
BLUR_KERNEL, BLUR_SIGMA = 51, 16.0
# KL-IG² (KLIGSquared) hyperparams
SIGMA_FINAL = 0.25
LR_MU, LR_LV = 0.05, 0.10
LOSS_STOP = 1e-3
LV_FLOOR = 2 * math.log(1 / 256)
LV_CEIL = 4.0
MU_MIN, MU_MAX = -2.64, 2.64
T_DESC = 50                          # rep-descent path length
N_MC_DESC = 16                       # MC samples per descent step
N_MC_GRAD = 10                       # MC samples per integration step

M_VG, M_SG, M_IGZ, M_BIG = "Vanilla Grad", "SmoothGrad", "IG-zero", "Blur-IG"
M_IDG, M_GIG, M_EG = "IDG", "Guided IG", "ExpGrad"
M_KLIG, M_LIN = "KLIG-Adaptive", "KL-IG (linear)"
M_KLIG2, M_KLIG2A = "KL-IG²", "KL-IG² (adaptive)"

METHODS = [M_VG, M_SG, M_IGZ, M_BIG, M_IDG, M_GIG, M_EG,
           M_KLIG, M_LIN, M_KLIG2, M_KLIG2A]
NEEDS_CF = {M_KLIG2, M_KLIG2A}

COLORS = {
    M_VG: "#8B4513", M_SG: "#1E90FF", M_IGZ: "#7B68EE", M_BIG: "#20B2AA",
    M_IDG: "#E07B39", M_GIG: "#9B59B6", M_EG: "#DC143C",
    M_KLIG: "#2d6a2d", M_LIN: "#333333", M_KLIG2: "#e41a1c", M_KLIG2A: "#8b0000",
}

def needs_cf(method): return method in NEEDS_CF

# ── shape / collapse helpers (methods nb cell 10) ────────────────────────────
def absmax_collapse(a):
    if a.dim() == 4: a = a.squeeze(0)
    idx = a.abs().argmax(dim=0, keepdim=True)
    return a.gather(0, idx).squeeze(0)

def sum_collapse(a):
    if a.dim() == 4: a = a.squeeze(0)
    return a.sum(dim=0)

def _xb(model, x):
    xb = x.unsqueeze(0) if x.dim() == 3 else x
    return xb.to(next(model.parameters()).device)

def _x1(x):
    """(C,H,W) on whatever device x is on."""
    return x.squeeze(0) if (x.dim() == 4 and x.shape[0] == 1) else x

def make_phi(model, layer="layer4"):
    return make_phi_from_layer(model, layer)

_phi_cache = {}
def _get_phi(model, phi):
    if phi is not None: return phi
    key = id(model)
    if key not in _phi_cache:
        _phi_cache[key] = make_phi_from_layer(model, "layer4")
    return _phi_cache[key]

def clamp_sigma(s):
    """find_sigma_stop is ill-defined for low-logit classes -> guard 2*log(sigma)."""
    if not (isinstance(s, (int, float)) and math.isfinite(s) and s > 0):
        s = SIGMA_FINAL
    return float(min(max(s, 1.0 / 256), 1.0))

def make_blur_baseline(x, kernel_size=BLUR_KERNEL, sigma=BLUR_SIGMA):
    coords = torch.arange(kernel_size, dtype=torch.float32, device=x.device) - kernel_size // 2
    k1d = torch.exp(-0.5 * (coords / sigma) ** 2); k1d = k1d / k1d.sum()
    kh = k1d.view(1, 1, -1, 1).expand(3, -1, -1, -1)
    kw = k1d.view(1, 1, 1, -1).expand(3, -1, -1, -1)
    pad = kernel_size // 2
    out = F.conv2d(x, kh, padding=(pad, 0), groups=3)
    return F.conv2d(out, kw, padding=(0, pad), groups=3)

# ── baseline raw attributions (methods nb cells 9,10) — return (C,H,W) ────────
def raw_vanilla(model, x, target):
    xb = _xb(model, x).requires_grad_(True)
    return Saliency(model).attribute(xb, target=target, abs=False).detach().squeeze(0)

def raw_smoothgrad(model, x, target):
    xb = _xb(model, x)
    std = 0.15 * float((xb.max() - xb.min()).item())
    noisy = (xb + torch.randn(SG_SAMPLES, *xb.shape[1:], device=xb.device) * std).requires_grad_(True)
    return torch.autograd.grad(model(noisy)[:, target].sum(), noisy)[0].detach().mean(dim=0)

def raw_ig_zero(model, x, target):
    xb = _xb(model, x)
    return IntegratedGradients(model).attribute(
        xb, baselines=torch.zeros_like(xb), target=target,
        n_steps=IG_STEPS, method="gausslegendre", internal_batch_size=IG_STEPS).detach().squeeze(0)

def raw_blur_ig(model, x, target):
    xb = _xb(model, x)
    blur = make_blur_baseline(xb if xb.dim() == 4 else xb.unsqueeze(0))
    if blur.dim() == 3: blur = blur.unsqueeze(0)
    return IntegratedGradients(model).attribute(
        xb, baselines=blur.to(xb.device), target=target,
        n_steps=IG_STEPS, method="gausslegendre", internal_batch_size=IG_STEPS).detach().squeeze(0)

def raw_idg(model, x, target):
    xb = _xb(model, x).requires_grad_(True)
    g = torch.autograd.grad(model(xb)[:, target].sum(), xb)[0]
    return (xb * g).detach().squeeze(0)

def raw_expgrad(model, x, target):
    xb = _xb(model, x)
    bg = torch.randn(EG_SAMPLES, *xb.shape[1:], device=xb.device)
    alpha = torch.rand(EG_SAMPLES, 1, 1, 1, device=xb.device)
    interp = (bg + alpha * (xb - bg)).requires_grad_(True)
    grads = torch.autograd.grad(model(interp)[:, target].sum(), interp)[0]
    return (grads.detach() * (xb - bg)).mean(dim=0)

def _attr_guided_ig(model, x, target, steps=50, fraction=0.25, max_dist_frac=0.1):
    dev = next(model.parameters()).device
    x0 = (x if x.dim() == 4 else x.unsqueeze(0)).to(dev).detach()[0]
    x_curr = torch.zeros_like(x0); attr = torch.zeros_like(x0)
    n_total = x0.numel(); n_up = max(1, int(fraction * n_total))
    model.eval()
    for _ in range(steps):
        dx = x0 - x_curr
        if dx.abs().max() < 1e-6: break
        x_t = x_curr.unsqueeze(0).requires_grad_(True)
        grad = torch.autograd.grad(model(x_t)[0, target], x_t)[0][0].detach()
        gain = (grad * dx).abs().view(-1)
        thresh = gain.kthvalue(max(1, n_total - n_up + 1)).values
        mask = (gain >= thresh).view_as(x0).float()
        step = dx * max_dist_frac * mask
        attr += grad * step; x_curr += step
    return attr.detach()

# ── KLIG-family raw attributions (eval nb cells 8,26) — return (C,H,W) ────────
def raw_klig_adaptive(model, x, target, sigma_final=None):
    x1 = _x1(x)
    sig = clamp_sigma(sigma_final if sigma_final is not None
                      else find_sigma_stop(model, x1, int(target), tau=0.95, n_samples=32, n_iter=12))
    return KLIntegratedGradients(model, n_steps=N_STEPS, n_samples=N_SAMPLES,
        sigma_final=sig, path=LinearPath(),
        device=next(model.parameters()).device).attribute(x1, target=int(target)).attr

def raw_klig_linear(model, x, target):
    x1 = _x1(x)
    return KLIntegratedGradients(model, n_steps=N_STEPS, n_samples=N_SAMPLES,
        sigma_final=SIGMA_FINAL, path=LinearPath(),
        device=next(model.parameters()).device).attribute(x1, target=int(target)).attr

def _build_klig2(model, phi, x_cf, sigma_start, lv_floor):
    return KLIGSquared(model, phi, x_cf, T=T_DESC, lr_mu=LR_MU, lr_lv=LR_LV,
        n_mc_path=N_MC_DESC, n_mc_grad=N_MC_GRAD, sigma_start=sigma_start, loss_stop=LOSS_STOP,
        lv_floor=lv_floor, lv_ceil=LV_CEIL, mu_min=MU_MIN, mu_max=MU_MAX,
        clamp_samples=True, device=next(model.parameters()).device)

def _klig2_integrate(k2, x1, target):
    """Build the rep-descent path and integrate attr_mu for `target`."""
    tm, tl, _ = k2._build_gradpath(x1, x1.shape)
    k2.model.eval()
    _, obj = k2._resolve_target(x1, int(target))
    saved = [p.requires_grad for p in k2.model.parameters()]
    for p in k2.model.parameters(): p.requires_grad_(False)
    acc = torch.zeros_like(x1)
    try:
        for k in range(len(tm) - 1):
            g, _ = k2._eval_gradients(tm[k], tl[k], x1.shape, obj)
            with torch.no_grad(): acc.add_(g * (tm[k] - tm[k + 1]))
    finally:
        for p, s in zip(k2.model.parameters(), saved): p.requires_grad_(s)
    return acc

def raw_klig2(model, x, target, x_cf, phi=None):
    x1 = _x1(x); xcf = _x1(x_cf).to(x1.device)
    k2 = _build_klig2(model, _get_phi(model, phi), xcf, SIGMA_FINAL, LV_FLOOR)
    return _klig2_integrate(k2, x1, target)

def raw_klig2_adaptive(model, x, target, x_cf, phi=None, sigma_final=None):
    x1 = _x1(x); xcf = _x1(x_cf).to(x1.device)
    sig = clamp_sigma(sigma_final if sigma_final is not None
                      else find_sigma_stop(model, x1, int(target), tau=0.95, n_samples=32, n_iter=12))
    k2 = _build_klig2(model, _get_phi(model, phi), xcf, sig, 2 * math.log(sig))
    return _klig2_integrate(k2, x1, target)

# ── unified entry point ──────────────────────────────────────────────────────
def raw_attr(method, model, x, target, *, x_cf=None, phi=None, sigma_final=None):
    """Return the raw (C,H,W) attribution for `method` (no collapse)."""
    if method == M_VG:    return raw_vanilla(model, x, target)
    if method == M_SG:    return raw_smoothgrad(model, x, target)
    if method == M_IGZ:   return raw_ig_zero(model, x, target)
    if method == M_BIG:   return raw_blur_ig(model, x, target)
    if method == M_IDG:   return raw_idg(model, x, target)
    if method == M_GIG:   return _attr_guided_ig(model, x, target)
    if method == M_EG:    return raw_expgrad(model, x, target)
    if method == M_KLIG:  return raw_klig_adaptive(model, x, target, sigma_final=sigma_final)
    if method == M_LIN:   return raw_klig_linear(model, x, target)
    if method == M_KLIG2:
        if x_cf is None: raise ValueError(f"{method} requires x_cf")
        return raw_klig2(model, x, target, x_cf, phi=phi)
    if method == M_KLIG2A:
        if x_cf is None: raise ValueError(f"{method} requires x_cf")
        return raw_klig2_adaptive(model, x, target, x_cf, phi=phi, sigma_final=sigma_final)
    raise ValueError(f"unknown method: {method!r}")

def attr_map(method, model, x, target, *, x_cf=None, phi=None, sigma_final=None):
    """Signed (H,W) absmax-collapsed attribution map (CPU tensor)."""
    raw = raw_attr(method, model, x, target, x_cf=x_cf, phi=phi, sigma_final=sigma_final)
    return absmax_collapse(raw).detach().cpu()
