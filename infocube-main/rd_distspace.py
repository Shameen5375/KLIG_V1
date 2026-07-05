"""
R-D DISTRIBUTION-SPACE ATTRIBUTION  —  peer of KL-IG on identical geometry.

KL-IG and R-D share the SAME probe space (per-pixel (mu, logvar) Gaussian),
the SAME sampler (mu + exp(logvar/2)*eps), the SAME ruler (klig.core.kl.gaussian_kl
to the N(0,1) prior) and the SAME endpoint sigma (klig.image.stopping.find_sigma_stop).
They DIFFER in the one thing that matters:

    KL-IG  = PATH INTEGRAL.  Walk (mu,logvar) from prior to input; accumulate the
             gradient.  -> distance TRAVELED in KL.
    R-D    = RATE ALLOCATION. Fix mu at the explicand; SOLVE for the per-pixel probe
             spread logvar* that spends the least total KL-rate while keeping the
             prediction intact (distortion <= tau).  -> rate ALLOCATED in KL.

The returned object is a SOLVED logvar* map (an argmin under a rate budget), NOT an
accumulated gradient.  That is what keeps R-D a peer of KL-IG rather than a special case.

Two deliberate, documented choices (see brief):
  1. NORMALIZED space.  KL-IG's prior is N(0,1) and its input is ImageNet-normalized,
     so `mu` here is the normalized image and probe samples are fed to the model with
     NO [0,1] clamp.  This is what makes the geometry byte-identical to KL-IG.
  2. EXCESS-RATE attribution.  gaussian_kl(mu,logvar) carries a 0.5*mu^2 term that is
     CONSTANT in the allocation variable (mu is fixed) and large for noise pixels
     (|mu| big) -> returning it verbatim would hand high rate to noise, re-introducing
     the gameability R-D exists to kill (and failing validation check #2).  We return
     the allocation-decided part:  gaussian_kl(mu,logvar*) - gaussian_kl(mu, lv_ceil),
     in which mu^2 cancels.  This is "the rate a pixel is FORCED to keep beyond
     throwing it away" — exactly the quantity the brief describes.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F

# REUSE KL-IG's own primitives so the geometry is provably identical.
from klig.core.kl import gaussian_kl               # KL(N(mu,exp(lv)) || N(0,1)), per-element
from klig.image.stopping import find_sigma_stop    # adaptive endpoint sigma (binary search)


@dataclass
class RDConfig:
    tau: float = 0.10           # distortion budget: target fractional logit/margin drop tolerated
    n_mc: int = 16              # MC samples per probe evaluation (match KL-IG n_mc)
    n_iter: int = 150           # allocation optimizer steps
    lr: float = 0.05            # step size on logvar
    lam: float = 1.0            # initial dual variable (weight on distortion; adapts via dual ascent)
    lr_lam: float = 0.5         # dual-ascent step: drives distortion -> tau budget
    lv_floor: float | None = None   # tightest probe (most info); None -> from find_sigma_stop
    lv_ceil: float = 0.0        # loosest probe = prior N(0,1) (logvar=0) — the "throw away" end
    distortion: str = "logit"   # "logit" (single class) or "kl_out" (full-dist, KL-IG^2-style)
    contrastive_cf: bool = False    # if True: preserve target-vs-CF margin (class-sensitivity mode)
    adaptive_floor: bool = True     # use find_sigma_stop for lv_floor (shared endpoint w/ KL-IG)
    seed: int = 0


# ── PROBE — identical reparameterization to KL-IG (normalized space, NO clamp) ──────────
def sample_probe(mu, logvar, n_mc, gen=None):
    """x_samp = mu + exp(logvar/2)*eps.  mu is the fixed explicand center (1,C,H,W)."""
    eps = torch.randn((n_mc,) + mu.shape[1:], device=mu.device, generator=gen)
    return mu + torch.exp(0.5 * logvar) * eps          # normalized space: do NOT clamp to [0,1]


# ── DISTORTION — how much the prediction degrades under the current probe ───────────────
def distortion(model, mu, logvar, target, cfg, clean_ref, cf_ref=None, gen=None):
    x = sample_probe(mu, logvar, cfg.n_mc, gen)
    logits = model(x)
    if cfg.contrastive_cf:                              # class-sensitivity: preserve the margin
        margin = logits[:, target] - logits[:, cf_ref]
        return clean_ref - margin.mean()               # clean_ref = clean margin
    if cfg.distortion == "kl_out":                      # full output distribution (KL-IG^2-ish)
        logp = F.log_softmax(logits, -1)
        return F.kl_div(logp, clean_ref.expand_as(logp), reduction="batchmean")
    return clean_ref - logits[:, target].mean()        # clean_ref = clean target logit


# ── THE ESTIMATOR — RATE ALLOCATION (this is what makes it NOT KL-IG) ────────────────────
def allocate(model, mu, target, cfg, clean_ref, cf_ref=None, tau_abs=None):
    """
    minimize  total_rate(logvar) = sum_i gaussian_kl_i(mu, logvar)
    s.t.      distortion(logvar) <= tau      (solved via Lagrangian  L = D + lam*R)

    Direction (IBA-style, well-conditioned): start from lv_floor (the tight input probe,
    image intact, model gradients LIVE) and let the RATE term push logvar UP — loosen
    every pixel toward the throw-away end (lv_ceil = prior) — while the distortion term
    holds the important pixels tight.  At equilibrium a pixel loosens iff loosening it
    costs less distortion than it saves rate; the pixels that STAY tight are the ones the
    prediction cannot afford to throw away.  This is an ALLOCATION (argmin over the logvar
    map), NOT an integral along a path.

    [We deliberately start at the tight end rather than the loose end the raw skeleton
     suggests: from an all-loose probe in NORMALIZED space the image is pure noise, model
     gradients vanish, and nothing re-concentrates.  Starting tight keeps the signal live.]
    """
    gen = torch.Generator(device=mu.device).manual_seed(cfg.seed)
    lv_ceil = cfg.lv_ceil
    lv_floor = cfg.lv_floor if cfg.lv_floor is not None else -12.0

    logvar = torch.full_like(mu, lv_floor).requires_grad_(True)   # start: full info (tight probe)
    opt = torch.optim.Adam([logvar], lr=cfg.lr)

    # Constrained form:  minimize RATE  s.t.  distortion <= tau.
    #   L(logvar, lam) = R + lam * (D - tau),   min over logvar, max over lam >= 0.
    # Dual ascent raises lam when the prediction breaks (D > tau -> tighten) and lowers it
    # when there is slack (D < tau -> loosen, spend less rate).  Equilibrium sits at D = tau:
    # the MIN-RATE allocation that keeps the prediction within tolerance (a true R-D point).
    lam = cfg.lam                                        # dual variable (rate stays in objective)
    D0 = None
    for _ in range(cfg.n_iter):
        opt.zero_grad()
        D = distortion(model, mu, logvar, target, cfg, clean_ref, cf_ref, gen)
        if D0 is None:
            D0 = float(D.detach())
        R = gaussian_kl(mu, logvar).mean()              # SHARED KL ruler, per-pixel mean rate
        loss = R + lam * D                               # min rate, distortion as constraint
        loss.backward()
        opt.step()
        with torch.no_grad():
            logvar.clamp_(lv_floor, lv_ceil)
            lam = min(1e4, max(0.0, lam + cfg.lr_lam * (float(D.detach()) - tau_abs)))   # dual ascent

    with torch.no_grad():
        # EXCESS rate = rate forced beyond the throw-away baseline (mu^2 cancels -> noise-robust).
        base = gaussian_kl(mu, torch.full_like(mu, lv_ceil))
        rate_map = (gaussian_kl(mu, logvar) - base).clamp_min(0.0)
        D_final = float(distortion(model, mu, logvar, target, cfg, clean_ref, cf_ref, gen).detach())
    return logvar.detach(), rate_map, dict(D0=D0, D_final=D_final, tau_abs=tau_abs,
                                           lam_final=lam, total_rate=float(rate_map.sum()))


# ── ORCHESTRATOR ────────────────────────────────────────────────────────────────────────
def rd_attribution(model, image01_norm, target=None, cfg: RDConfig = RDConfig(), x_cf=None):
    """
    image01_norm : (1,C,H,W) ImageNet-NORMALIZED image = mu (explicand, fixed center).
    x_cf         : normalized counterfactual, only if cfg.contrastive_cf (class-sensitivity).
    Returns dict with per-pixel excess-rate attribution (H,W) + the solved logvar* map.
    """
    mu = image01_norm if image01_norm.dim() == 4 else image01_norm.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        logits = model(mu)
        if target is None:
            target = int(logits.argmax())
        if cfg.contrastive_cf:
            cf_ref = int(model(x_cf if x_cf.dim() == 4 else x_cf.unsqueeze(0)).argmax())
            clean_ref = (logits[:, target] - logits[:, cf_ref])           # clean margin (tensor)
        elif cfg.distortion == "kl_out":
            clean_ref = F.softmax(logits, -1); cf_ref = None
        else:
            clean_ref = logits[:, target]; cf_ref = None                  # clean target logit (tensor)

    # distortion budget (absolute): tau * clean scale
    if cfg.distortion == "kl_out":
        tau_abs = cfg.tau                                             # absolute output-KL budget
    else:
        scale = abs(float(clean_ref.mean()))                         # clean logit or margin scale
        tau_abs = cfg.tau * scale

    # shared endpoint with KL-IG: tightest probe = input distribution N(mu, sigma_stop^2)
    if cfg.adaptive_floor and cfg.lv_floor is None:
        sigma_stop = find_sigma_stop(model, mu.squeeze(0), target)
        cfg = _with(cfg, lv_floor=2.0 * math.log(max(sigma_stop, 1e-3)))

    logvar_star, rate_map, info = allocate(model, mu, target, cfg, clean_ref, cf_ref, tau_abs)
    attribution = rate_map.sum(1)[0]                    # collapse channels -> (H,W)
    return dict(attribution=attribution, logvar_star=logvar_star,
                rate_map=rate_map, target=target, info=info, cfg=cfg)


def _with(cfg: RDConfig, **kw) -> RDConfig:
    from dataclasses import replace
    return replace(cfg, **kw)


# ── R(D) SWEEP — trace the rate-distortion curve by sweeping the rate weight lambda ──────
def rd_curve(model, image01_norm, target=None, cfg: RDConfig = RDConfig(),
             lams=(0.02, 0.05, 0.1, 0.3, 1.0, 3.0), x_cf=None):
    """Sweep lam (budget knob, KL-IG's t played as a BUDGET not a path param).
    Returns list of (lam, total_rate, distortion) — the R(D) trace."""
    out = []
    for lam in lams:
        r = rd_attribution(model, image01_norm, target, _with(cfg, lam=lam), x_cf)
        out.append((lam, r["info"]["total_rate"], r["info"]["D_final"]))
    return out
