
"""
R-D Path Attribution — attribution as RATE ALLOCATION.

Score each region by how many bits it must retain for the model to keep predicting class c.
A region you can noise into oblivion without moving the logit is unimportant; a region that
demands precision (breaks under the slightest noise) is the explanation. Adding noise COSTS
rate rather than earning attribution credit -> resistant to the noise-gameability failure mode.

Module boundaries (per design brief §10):
  ModelWrapper · patch_regions · Noise/Blur operators · measure_rd_curves ·
  sufficiency/sensitivity heads · global_rd_curve (reverse-waterfilling) · validation · run()

Library only — visualization lives in the notebook. Demo: `python rd_attribution.py`.
"""
from __future__ import annotations
import sys, pickle, warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
import numpy as np, torch, torch.nn.functional as F
from scipy.ndimage import gaussian_filter
warnings.filterwarnings('ignore')
EPS = 1e-8


# ─────────────────────────────── config (§8) ───────────────────────────────
@dataclass
class RDConfig:
    target_class: int | None = None          # None -> model top-1 on the clean image
    grid: int = 14                            # coarse disjoint grid (global R(D) curve + validation)
    window: int = 48                          # fine map: sliding-window size (receptive field, px)
    stride: int = 12                          # fine map: window stride (smaller = finer + more forwards)
    smooth: float = 2.5                       # fine map: Gaussian smoothing σ (px) after upsample
    soft_window: bool = True                   # Gaussian-soft windows (smooth, non-patchy) vs hard squares
    operator: str = 'noise'                   # 'noise' (amplitude) or 'blur' (spatial, §7)
    levels: tuple = ()                        # degradation grid; filled by default per operator
    n_mc: int = 4                             # Monte-Carlo draws per (region, level)
    tau: float = 0.30                         # sufficiency tolerance (meaning depends on thr_mode)
    rate_model: str = 'linear'                # 'linear' (graded, bounded [0,1]) or 'log' (sparse skeleton)
    thr_mode: str = 'global'                  # 'global' (valid: bg floored) | 'per' (dense but bg lights up) | 'abs'
    batch: int = 64
    seed: int = 0
    def default_levels(self):
        if self.levels: return np.asarray(self.levels, float)
        return np.array([0.02, 0.05, 0.10, 0.20, 0.35, 0.60]) if self.operator == 'noise' \
               else np.array([0.5, 1.0, 2.0, 4.0, 8.0])       # blur sigma (px)


# ─────────────────────────────── model wrapper ─────────────────────────────
class ModelWrapper:
    """Takes [0,1] images (B,3,H,W), returns pre-softmax logits. Normalizes internally."""
    def __init__(self, model, device, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.model = model.eval(); self.device = device
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
    @torch.no_grad()
    def logits(self, img01):
        x = (img01.to(self.device) - self.mean) / self.std
        return self.model(x)
    @torch.no_grad()
    def target_logit(self, img01, cls):
        return self.logits(img01)[:, cls]


# ─────────────────────────────── region partition ──────────────────────────
def patch_regions(H, W, grid):
    """Coarse DISJOINT label map (H,W) int in [0, grid*grid). Used for the global R(D) curve."""
    ys = np.linspace(0, grid, H + 1)[:-1].astype(int); xs = np.linspace(0, grid, W + 1)[:-1].astype(int)
    lab = (ys[:, None] * grid + xs[None, :]).astype(np.int64)
    return lab, grid * grid

def _label_masks(lab, n_reg, device):
    H, W = lab.shape
    return [torch.from_numpy((lab == r).astype('float32')).to(device).view(1, 1, H, W) for r in range(n_reg)]

def window_masks(H, W, window, stride, device, soft=True):
    """OVERLAPPING windows on a regular center grid (fine map). Returns (masks, n_cy, n_cx).
    soft=True -> Gaussian bumps (contributions BLEND -> smooth, non-patchy map);
    soft=False -> hard squares."""
    cys = list(range(0, H, stride)); cxs = list(range(0, W, stride))
    masks = []
    if soft:
        yy, xx = np.mgrid[0:H, 0:W]; s2 = 2.0 * (window / 2.0) ** 2
        for cy in cys:
            for cx in cxs:
                m = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / s2).astype('float32')
                masks.append(torch.from_numpy(m).to(device).view(1, 1, H, W))
    else:
        h = window // 2
        for cy in cys:
            for cx in cxs:
                m = np.zeros((H, W), 'float32'); m[max(0, cy-h):min(H, cy+h+1), max(0, cx-h):min(W, cx+h+1)] = 1.0
                masks.append(torch.from_numpy(m).to(device).view(1, 1, H, W))
    return masks, len(cys), len(cxs)

def field_to_map(scores, n_cy, n_cx, H, W, smooth):
    """Sparse window-center scores -> dense H×W map via BILINEAR upsample + light Gaussian smoothing."""
    g = torch.tensor(np.asarray(scores, dtype='float32').reshape(n_cy, n_cx))[None, None]
    up = F.interpolate(g, size=(H, W), mode='bilinear', align_corners=True)[0, 0].cpu().numpy()
    return gaussian_filter(up, smooth) if smooth and smooth > 0 else up


# ─────────────────────── perturbation operators (swappable) ─────────────────
class NoiseOperator:
    """Amplitude rate: add Gaussian noise (level=σ), clip to [0,1]. §3"""
    name = 'noise'
    def degrade(self, img01, level, rng):        # full-image degraded version (composited later)
        n = torch.from_numpy(rng.standard_normal(img01.shape).astype('float32')).to(img01.device)
        return (img01 + n * level).clamp(0, 1)

class BlurOperator:
    """Spatial rate: Gaussian blur (level=σ_px). §7  rate becomes spatial resolution."""
    name = 'blur'
    def degrade(self, img01, level, rng=None):
        k = max(3, int(2 * round(3 * level) + 1)); c = torch.arange(k, device=img01.device) - k // 2
        g = torch.exp(-(c.float() ** 2) / (2 * level ** 2)); g = (g / g.sum())
        kh = g.view(1, 1, -1, 1).expand(3, 1, -1, 1); kw = g.view(1, 1, 1, -1).expand(3, 1, 1, -1)
        x = F.pad(img01, (0, 0, k // 2, k // 2), mode='reflect'); x = F.conv2d(x, kh, groups=3)
        x = F.pad(x, (k // 2, k // 2, 0, 0), mode='reflect'); return F.conv2d(x, kw, groups=3)

def get_operator(name): return NoiseOperator() if name == 'noise' else BlurOperator()

def rise_smooth_map(wrap, img01, operator, level, cls, n_masks=3000, s=8, p=0.5, batch=64, seed=0):
    """SMOOTH pixel-level importance via RISE-style random masks (no patch grid).
    Thousands of random smooth masks; noise where mask=1; weight each mask by its logit drop.
    importance[px] = mean logit-drop over masks that noised that pixel -> smooth heatmap."""
    dev = img01.device; H, W = img01.shape[-2:]
    L0 = float(wrap.target_logit(img01, cls).item()); rng = np.random.default_rng(seed)
    ch, cw = int(np.ceil(H / s)), int(np.ceil(W / s)); uh, uw = (s + 1) * ch, (s + 1) * cw
    acc = torch.zeros(1, H, W, device=dev); wsum = torch.zeros(1, H, W, device=dev); done = 0
    while done < n_masks:
        b = min(batch, n_masks - done)
        grid = torch.from_numpy((rng.random((b, 1, s, s)) < p).astype('float32')).to(dev)
        up = F.interpolate(grid, size=(uh, uw), mode='bilinear', align_corners=False)
        M = torch.empty(b, 1, H, W, device=dev)
        for i in range(b):
            oy, ox = int(rng.integers(0, ch)), int(rng.integers(0, cw)); M[i] = up[i, :, oy:oy+H, ox:ox+W]
        deg = operator.degrade(img01, level, rng)
        L = wrap.target_logit(img01 * (1 - M) + deg * M, cls)          # (b,)
        acc += ((L0 - L).view(b, 1, 1, 1) * M).sum(0); wsum += M.sum(0); done += b
    return (acc / (wsum + EPS))[0].cpu().numpy()

def rate_of(level, level_max, model='log'):      # bits/importance kept; monotone DECREASING in degradation. §3
    """'log': log(lmax/level) (unbounded, log-compressed, hard floor)  ·
    'linear': 1 - level/lmax (bounded [0,1], graded, gentler floor)."""
    if model == 'linear':
        return float(max(0.0, 1.0 - max(level, EPS) / level_max))
    return float(np.log(level_max / max(level, EPS)))


# ─────────────────────── R-D curve measurement (§4, the expensive step) ─────
def measure_rd_curves(wrap, img01, masks, operator, levels, n_mc, cls, batch, seed):
    """masks: list of (1,1,H,W) float region masks (disjoint OR overlapping windows).
    Return drops[n_reg, n_levels] = mean_MC (L0 - L(noise in region only at level))."""
    L0 = float(wrap.target_logit(img01, cls).item()); n_reg = len(masks)
    drops = np.zeros((n_reg, len(levels)))
    rng = np.random.default_rng(seed)
    buf_img, buf_idx = [], []                     # batch buffer of composited perturbed images
    def flush():
        if not buf_img: return
        L = wrap.target_logit(torch.cat(buf_img, 0), cls).cpu().numpy()
        for (r, li), lv in zip(buf_idx, (L0 - L)): drops[r, li] += lv
        buf_img.clear(); buf_idx.clear()
    for li, level in enumerate(levels):
        for _ in range(n_mc):
            degraded = operator.degrade(img01, float(level), rng)          # shared draw across regions
            for r in range(n_reg):
                m = masks[r]; buf_img.append(img01 * (1 - m) + degraded * m); buf_idx.append((r, li))
                if len(buf_img) >= batch: flush()
    flush()
    drops /= n_mc
    return drops, L0


# ─────────────────────── score heads (§5) ───────────────────────────────────
def sufficiency_score(drops, levels, tau, L0, rate_model='linear', thr_mode='per'):
    """Primary, noise-robust. σ* = largest noise a region tolerates before its logit drop reaches
    the tolerance; score = rate(σ*). High required-rate = important (breaks under little noise).

    thr_mode: 'per'    -> thr = τ · this region's OWN max drop  (decoupled, graded — default)
              'global' -> thr = τ · max drop over ALL regions   (couples every window to the hottest)
              'abs'    -> thr = τ  (absolute logit-drop tolerance)
    rate_model: 'linear' (bounded [0,1], graded)  or  'log' (unbounded, sparse skeleton).
    Monotone drop envelope + interpolated crossing for a smooth map."""
    levels = np.asarray(levels, float); lmax = float(levels[-1]); n_reg = len(drops)
    Dmax_g = max(float(np.max(drops)), EPS); sc = np.zeros(n_reg)
    for r in range(n_reg):
        c = np.maximum.accumulate(np.clip(drops[r], 0.0, None))  # monotone "damage so far" envelope
        thr = (tau * Dmax_g) if thr_mode == 'global' else tau if thr_mode == 'abs' \
              else tau * max(float(c[-1]), EPS)                   # 'per': fraction of region's own max
        if c[-1] < thr:                                          # never reaches thr -> tolerant -> unimportant
            star = lmax
        elif c[0] >= thr:                                        # breaks at the smallest noise -> very important
            star = levels[0] * 0.5
        else:
            j = int(np.argmax(c >= thr))                         # first crossing
            l0, l1, c0, c1 = levels[j-1], levels[j], c[j-1], c[j]
            star = l0 + (l1 - l0) * (thr - c0) / (c1 - c0 + EPS)  # interpolate σ*
        sc[r] = rate_of(star, lmax, rate_model)
    return sc

def sensitivity_score(drops, levels):
    """Secondary baseline. Area under the logit-drop curve. Faster, more gameable."""
    return np.trapezoid(np.clip(drops, 0, None), levels, axis=1)


# ─────────────────── global R(D) curve via reverse-waterfilling (§6) ────────
def global_rd_curve(wrap, img01, lab, n_reg, scores, operator, levels, cls, seed, steps=16, n_mc=3):
    """Protect high-score regions, dump max noise on the rest. Sweep #protected -> Δlogit.
    Also a RANDOM-order baseline: importance ordering should give LOWER distortion."""
    dev = img01.device; H, W = img01.shape[-2:]; lmax = float(levels[-1])
    L0 = float(wrap.target_logit(img01, cls).item())
    order_imp = np.argsort(-scores)                       # most important first (protect these first)
    rng = np.random.default_rng(seed); order_rnd = rng.permutation(n_reg)
    ks = np.unique(np.linspace(0, n_reg, steps).astype(int))
    def distortion_for(protect_set):
        keep = np.zeros(n_reg, bool); keep[list(protect_set)] = True
        noise_mask = torch.from_numpy(np.isin(lab, np.where(~keep)[0]).astype('float32')).to(dev).view(1, 1, H, W)
        acc = 0.0
        for _ in range(n_mc):
            deg = operator.degrade(img01, lmax, rng)
            L = float(wrap.target_logit(img01 * (1 - noise_mask) + deg * noise_mask, cls).item())
            acc += (L0 - L)
        return acc / n_mc
    dist_imp = np.array([distortion_for(order_imp[:k]) for k in ks])
    dist_rnd = np.array([distortion_for(order_rnd[:k]) for k in ks])
    budget = ks / n_reg                                  # fraction of regions protected (rate proxy)
    return dict(budget=budget, dist_importance=dist_imp, dist_random=dist_rnd, L0=L0,
                Dmax=float(dist_imp.max() + EPS))


# ─────────────────────── validation hooks (§9) ─────────────────────────────
def validate(wrap, img01, cfg, score_map, lab, n_reg, scores):
    """Sanity floor + noise-gameability. Returns dict of pass/fail + numbers."""
    dev = img01.device; H, W = img01.shape[-2:]; g = cfg.grid
    out = {}
    # (a) sanity floor: corner patch (background proxy) should score below the image median.
    corner = 0                                            # region (0,0)
    out['corner_score'] = float(scores[corner]); out['median_score'] = float(np.median(scores))
    out['sanity_floor_pass'] = bool(scores[corner] <= np.median(scores))
    # (b) noise-gameability: fill a corner region with pure noise, re-measure ITS sufficiency score.
    #     A gradient/differ metric would light up; the sufficiency head must keep it LOW.
    rng = np.random.default_rng(cfg.seed + 7)
    m = torch.from_numpy((lab == corner).astype('float32')).to(dev).view(1, 1, H, W)
    noisy_img = (img01 * (1 - m) + torch.from_numpy(rng.random(img01.shape).astype('float32')).to(dev) * m)
    cls = cfg.target_class
    op = get_operator(cfg.operator); levels = cfg.default_levels()
    d2, L0b = measure_rd_curves(wrap, noisy_img, [m, 1 - m], op, levels, cfg.n_mc, cls, cfg.batch, cfg.seed)
    s2 = sufficiency_score(d2, levels, cfg.tau, L0b, cfg.rate_model, cfg.thr_mode)   # region 0 = the noise patch
    out['noise_patch_score'] = float(s2[0]); out['noise_patch_low'] = bool(s2[0] <= np.median(scores) + EPS)
    return out


# ─────────────────────── orchestrator (§10) ─────────────────────────────────
def run_rd_attribution(model, img01, cfg: RDConfig, device, full=True):
    """img01: (1,3,H,W) or (3,H,W) in [0,1].
    Fine map: σ* scored on OVERLAPPING windows -> bilinear upsample -> Gaussian smooth (H×W, smooth).
    full=True also computes the coarse disjoint grid for the global R(D) curve + validation."""
    wrap = ModelWrapper(model, device)
    if img01.dim() == 3: img01 = img01.unsqueeze(0)
    img01 = img01.to(device)
    if cfg.target_class is None:                               # don't mutate the caller's cfg (reuse-safe)
        cfg = replace(cfg, target_class=int(wrap.logits(img01)[0].argmax()))
    H, W = img01.shape[-2:]
    op = get_operator(cfg.operator); levels = cfg.default_levels()
    # ── fine, smooth map from overlapping windows ──
    wmasks, ncy, ncx = window_masks(H, W, cfg.window, cfg.stride, device, cfg.soft_window)
    dW, L0 = measure_rd_curves(wrap, img01, wmasks, op, levels, cfg.n_mc, cfg.target_class, cfg.batch, cfg.seed)
    suffW = sufficiency_score(dW, levels, cfg.tau, L0, cfg.rate_model, cfg.thr_mode); sensW = sensitivity_score(dW, levels)
    out = dict(cfg=cfg, target_class=cfg.target_class, L0=L0, levels=levels, n_windows=len(wmasks),
               suff_map=field_to_map(suffW, ncy, ncx, H, W, cfg.smooth),
               sens_map=field_to_map(sensW, ncy, ncx, H, W, cfg.smooth))
    if full:
        lab, n_reg = patch_regions(H, W, cfg.grid)
        dC, _ = measure_rd_curves(wrap, img01, _label_masks(lab, n_reg, device), op, levels,
                                  cfg.n_mc, cfg.target_class, cfg.batch, cfg.seed)
        suffC = sufficiency_score(dC, levels, cfg.tau, L0, cfg.rate_model, cfg.thr_mode); sensC = sensitivity_score(dC, levels)
        out.update(drops=dC, suff=suffC, sens=sensC, labels=lab, n_reg=n_reg,
                   rd_curve=global_rd_curve(wrap, img01, lab, n_reg, suffC, op, levels, cfg.target_class, cfg.seed),
                   validation=validate(wrap, img01, cfg, suffC[lab], lab, n_reg, suffC))
    return out


# ─────────────────────── demo (library self-test) ──────────────────────────
def _load_demo_image(device):
    for p in ['cs_viz_outputs/segment_store_vit.pkl', 'cs_viz_cache/cands.pkl']:
        if Path(p).exists():
            d = pickle.load(open(p, 'rb'))[0]; x = d['x']; x = x.squeeze(0) if x.dim() == 4 else x
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            return (x * std + mean).clamp(0, 1)           # de-normalize store tensor -> [0,1]
    raise FileNotFoundError('no demo image cache found')

if __name__ == '__main__':
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from torchvision.models import resnet50, ResNet50_Weights
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(dev).eval()
    cats = ResNet50_Weights.IMAGENET1K_V2.meta['categories']
    img01 = _load_demo_image(dev)
    grid = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    cfg = RDConfig(grid=grid, operator='noise', n_mc=3)
    print(f'[demo] device={dev} grid={grid} operator=noise')
    res = run_rd_attribution(model, img01, cfg, dev)
    print(f'target class = {res["target_class"]} ({cats[res["target_class"]].split(",")[0]})  L0={res["L0"]:.2f}')
    print(f'sufficiency score: min={res["suff"].min():.2f} max={res["suff"].max():.2f}')
    rd = res['rd_curve']; d = rd['dist_importance']; b = rd['budget']
    d50 = b[np.argmax(d <= 0.5 * rd['Dmax'])] if (d <= 0.5 * rd['Dmax']).any() else 1.0
    print(f'global R(D): Dmax={rd["Dmax"]:.2f}  D_50%@budget≈{d50:.2f}  '
          f'(importance AUC {np.trapezoid(d, b):.2f} vs random {np.trapezoid(rd["dist_random"], b):.2f} — importance should be LOWER)')
    v = res['validation']
    print(f'[validate] sanity-floor {"PASS" if v["sanity_floor_pass"] else "FAIL"} '
          f'(corner {v["corner_score"]:.2f} vs median {v["median_score"]:.2f});  '
          f'noise-gameability {"PASS" if v["noise_patch_low"] else "FAIL"} (noise patch {v["noise_patch_score"]:.2f})')
    np.save('cs_viz_outputs/rd_suff_map.npy', res['suff_map'])
    print('saved cs_viz_outputs/rd_suff_map.npy')
