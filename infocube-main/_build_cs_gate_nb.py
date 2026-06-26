# Assembles cs_gate_eval.ipynb from the cell sources below.
import json, nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []
def md(s):   cells.append(new_markdown_cell(s.strip("\n")))
def code(s): cells.append(new_code_cell(s.strip("\n")))

# ════════════════════════════════════════════════════════════════════════════
md(r'''
# Class Sensitivity — **The Gate**  (pixel-CS vs CS_latent)

A fresh, self-contained evaluation of class sensitivity. We compare two metrics that
consume **identical attribution masks** and differ by a single line — *encode vs flatten*:

| metric | space | formula |
|---|---|---|
| `cs_pixel`  | pixel space      | `1 − cos( masked_image(x,m₁).flatten(), masked_image(x,m₂).flatten() )` |
| `cs_latent` | representation   | `1 − cos( φ(masked_image(x,m₁)), φ(masked_image(x,m₂)) )`  (φ = ViT-B/16) |

where `mₖ = |A_yk| / max|A_yk|` is the per-class attribution mask (same normalization for both).

**Machinery reused from the existing pipeline**
- `klig_methods.attr_map(...)` — one entry point for **all 11 methods**
  (Vanilla Grad, SmoothGrad, IG-zero, Blur-IG, IDG, Guided IG, ExpGrad,
   KLIG-Adaptive, KL-IG linear, KL-IG², KL-IG² adaptive) + a local **Random** branch.
- `enc_vit` — ViT-B/16 patch-token encoder (the same φ used by CS_latent in `kl_ig2__eval`).

**Experiments**
- **Task 1.1** — Noise-injection control (the gate): corrupt one y₁ mask, hold y₂ fixed,
  watch the two metrics diverge.
- **Task 1.2** — Decision checkpoint.
- **Task 1.4** — Crossover table (n=20–30, all methods, both metrics, same masks).
- **Task 1.5** — Record verdict; save `noise_curve.png`, `crossover_table.csv`.
''')

# ────────────────────────────────────────────────────────────────────────────
code(r'''
# ── imports ──────────────────────────────────────────────────────────────────
import importlib, os, math, pickle, warnings, random
from pathlib import Path

import numpy as np
if not hasattr(np, 'trapz'): np.trapz = np.trapezoid   # NumPy>=2.0 compat
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
warnings.filterwarnings('ignore')

# unified 11-method attribution dispatch (lives next to this notebook)
import klig_methods as KM
importlib.reload(KM)
from klig_methods import attr_map, METHODS, needs_cf, make_phi

from torchvision.models import resnet50, ResNet50_Weights
import timm

print('imports OK — methods:', METHODS)
''')

# ────────────────────────────────────────────────────────────────────────────
code(r'''
# ── config ───────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', DEVICE)
EPS = 1e-8

CACHE_DIR = Path('cs_gate_cache'); CACHE_DIR.mkdir(exist_ok=True)
FORCE_RECOMPUTE = False

# pool sizes
N_POOL_TARGET = 40          # multi-class images to collect
N_CROSSOVER   = 24          # images used in Task 1.4 (20–30)
CS_PROB_THRESH = 0.10       # a class "co-occurs" if softmax prob > this
CS_MAX_SCAN    = 3000       # val images to scan when building the pool

# noise gate
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.4, 0.8, 1.6]
N_GATE       = 50                    # images to average the gate over (Task 1.1)
GOOD_METHOD  = 'KL-IG² (adaptive)'   # the known-good map driving Task 1.1

# speed: lighter KLIG/IG settings so a multi-method sweep is tractable
KM.N_STEPS, KM.N_SAMPLES = 25, 3
KM.IG_STEPS = 25
KM.SG_SAMPLES, KM.EG_SAMPLES = 25, 25
KM.T_DESC, KM.N_MC_DESC, KM.N_MC_GRAD = 25, 8, 3

# roster for the crossover = every method + Random floor
ROSTER = list(METHODS) + ['Random']
print('roster (%d):' % len(ROSTER), ROSTER)
''')

# ────────────────────────────────────────────────────────────────────────────
code(r'''
# ── model + φ + latent encoder ───────────────────────────────────────────────
weights  = ResNet50_Weights.IMAGENET1K_V2
model    = resnet50(weights=weights).to(DEVICE).eval()
preprocess      = weights.transforms()
imagenet_labels = weights.meta['categories']
phi = make_phi(model)                       # layer4 φ for KL-IG² descent

# latent encoder φ_enc = ViT-B/16 mean-pooled patch tokens (768-d) — same as CS_latent
_vit = timm.create_model('vit_base_patch16_224', pretrained=True).to(DEVICE).eval()
def enc_vit(t):
    with torch.no_grad():
        f = _vit.forward_features(t.unsqueeze(0).to(DEVICE))   # (1, 1+N, 768)
    return f[0, 1:].mean(0)                                     # drop CLS → (768,)
ENC = enc_vit
print('model + ViT encoder ready')
''')

# ────────────────────────────────────────────────────────────────────────────
code(r'''
# ── image pool: val images where ≥2 classes co-occur (top-1, top-2 both confident) ──
_cache_pool = CACHE_DIR / 'pool.pkl'
if not FORCE_RECOMPUTE and _cache_pool.exists():
    _pool = pickle.load(open(_cache_pool, 'rb'))
    _pool = [{**d, 'x': d['x'].to(DEVICE)} for d in _pool]
    print(f'[cache] pool n={len(_pool)}')
else:
    from datasets import load_dataset as _hf
    _ds = _hf('evanarlian/imagenet_1k_resized_256', split='val', streaming=True)
    _ds = _ds.shuffle(seed=42, buffer_size=5000)
    _pool, scanned = [], 0
    for item in tqdm(_ds, total=CS_MAX_SCAN, desc='scanning multi-class'):
        if len(_pool) >= N_POOL_TARGET or scanned >= CS_MAX_SCAN: break
        scanned += 1
        img = item['image']
        if img.mode != 'RGB': img = img.convert('RGB')
        x = preprocess(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = model(x).softmax(-1)[0].cpu()
        high = (probs > CS_PROB_THRESH).nonzero(as_tuple=True)[0].tolist()
        if len(high) < 2: continue
        high = sorted(high, key=lambda c: probs[c].item(), reverse=True)
        _pool.append({'idx': len(_pool), 'x': x, 'high_cls': high,
                      'high_probs': [probs[c].item() for c in high]})
    pickle.dump([{**d, 'x': d['x'].cpu()} for d in _pool], open(_cache_pool, 'wb'))
    print(f'pool n={len(_pool)} (scanned {scanned})')

for d in _pool[:3]:
    y1, y2 = d['high_cls'][:2]
    print(f"  img {d['idx']}: y1={imagenet_labels[y1]!r:24s} ({d['high_probs'][0]:.2f})  "
          f"y2={imagenet_labels[y2]!r:24s} ({d['high_probs'][1]:.2f})")
''')

# ────────────────────────────────────────────────────────────────────────────
code(r'''
# ── counterfactual pool: a real image of each needed y2 class (KL-IG² baseline) ──
_cache_cf = CACHE_DIR / 'cf.pkl'
_need = {int(d['high_cls'][1]) for d in _pool}
print(f'CF needed for {len(_need)} distinct y2 classes')

if not FORCE_RECOMPUTE and _cache_cf.exists():
    _cf_cpu = pickle.load(open(_cache_cf, 'rb'))
    if len(set(_cf_cpu) & _need) < len(_need): _cache_cf.unlink()
if FORCE_RECOMPUTE or not _cache_cf.exists():
    _cf_cpu = {}
    # seed for free from pool images whose own top-1 IS a needed class
    for d in _pool:
        c0 = int(d['high_cls'][0])
        if c0 in _need and c0 not in _cf_cpu:
            xx = d['x']; _cf_cpu[c0] = (xx.squeeze(0) if xx.dim() == 4 else xx).cpu()
    _still = _need - set(_cf_cpu)
    print(f'  seeded {len(_cf_cpu)}/{len(_need)} from pool; streaming for {len(_still)}')
    if _still:
        from datasets import load_dataset as _hf
        _s = _hf('evanarlian/imagenet_1k_resized_256', split='val',
                 streaming=True).shuffle(seed=13, buffer_size=500)
        best, lock, sc = {c: (-1.0, None) for c in _still}, set(), 0
        pb = tqdm(total=len(_still), desc='CF pool')
        for item in _s:
            sc += 1
            if len(lock) >= len(_still) or sc >= 8000: break
            im = item['image']
            if im.mode != 'RGB': im = im.convert('RGB')
            xx = preprocess(im).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pr = model(xx).softmax(-1)[0].cpu()
            for c in _still:
                if c in lock: continue
                if float(pr[c]) > best[c][0]:
                    best[c] = (float(pr[c]), xx.squeeze(0).cpu())
                    if pr[c] >= 0.30: lock.add(c); pb.update(1)
        pb.close()
        for c in _still:
            if best[c][1] is not None: _cf_cpu[c] = best[c][1]
    pickle.dump(_cf_cpu, open(_cache_cf, 'wb'))
cf_csl = {c: v.to(DEVICE) for c, v in _cf_cpu.items()}
print(f'CF pool ready: {len(cf_csl)}/{len(_need)} y2 classes')
''')

# ════════════════════════════════════════════════════════════════════════════
md(r'''
## Shared — mask + encode helpers

`make_mask` is applied **identically** before both metrics. The only line that differs
between `cs_pixel` and `cs_latent` is encode-vs-flatten — so the gap between them cannot
be a preprocessing artifact.
''')

code(r'''
_rng = np.random.default_rng(0)

def make_mask(attr):
    """|A| / max|A|  →  (H,W) tensor in [0,1] on DEVICE.  Same norm for both metrics."""
    a = attr if torch.is_tensor(attr) else torch.as_tensor(attr)
    a = a.float().abs().to(DEVICE)
    return a / (a.max() + EPS)

def masked_image(x, mask):
    return x * mask.unsqueeze(0)                # broadcast (H,W) over the 3 channels

def cs_pixel(x, m1, m2):                        # pixel space — SAME masks, NO encoder
    v1 = masked_image(x, m1).flatten()
    v2 = masked_image(x, m2).flatten()
    return float(1.0 - F.cosine_similarity(v1[None], v2[None]).item())

def cs_latent(x, m1, m2):                       # representation space
    z1 = ENC(masked_image(x, m1))
    z2 = ENC(masked_image(x, m2))
    return float(1.0 - F.cosine_similarity(z1[None], z2[None]).item())

def attr_for(method, x1, cls, x_cf):
    """Signed (H,W) attribution for `method` & class `cls`. Adds the Random branch;
       everything else routes through the shared klig_methods dispatch."""
    H, W = x1.shape[-2], x1.shape[-1]
    if method == 'Random':
        return torch.from_numpy(_rng.standard_normal((H, W))).float()
    return attr_map(method, model, x1, int(cls), x_cf=x_cf, phi=phi)

print('helpers ready: make_mask, masked_image, cs_pixel, cs_latent, attr_for')
''')

# ════════════════════════════════════════════════════════════════════════════
md(r'''
## Task 1.1 — Noise-injection control  (the gate; method-free)

Take a known-good class-discriminative map. Hold the y₂ mask **fixed**. Progressively
inject Gaussian noise into the y₁ mask (re-normalizing after, so we test *decorrelation*,
not magnitude). Watch the two metrics diverge — **averaged over up to `N_GATE` (≈50) images**
so the gate verdict isn't a single-image fluke.

**Expected:** `cs_pixel` flat-or-**rising** (noise = free decorrelation) while `cs_latent`
**falls** (a noisy mask carves an incoherent input → the encoder reads it as garbage).
''')

code(r'''
# gate images = every pooled image with a real CF for its y2, up to N_GATE
gate_imgs = [d for d in _pool if int(d['high_cls'][1]) in cf_csl][:N_GATE]
assert gate_imgs, 'no pooled image has a CF for its y2 — rebuild cf_csl'
print(f'noise gate averaged over {len(gate_imgs)} images × {len(NOISE_LEVELS)} σ levels '
      f'(map = {GOOD_METHOD})')

# per-image clean masks (cached), then sweep σ; collect px/lat across images per σ
torch.manual_seed(0)
px_by_sigma  = {s: [] for s in NOISE_LEVELS}
lat_by_sigma = {s: [] for s in NOISE_LEVELS}
for d in tqdm(gate_imgs, desc='noise gate'):
    x      = d['x'].squeeze(0).to(DEVICE)
    y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
    x_cf   = cf_csl[y2]; x_cf = (x_cf.squeeze(0) if x_cf.dim() == 4 else x_cf).to(DEVICE)
    m1_clean = make_mask(attr_for(GOOD_METHOD, x, y1, x_cf))   # known-good y1 map
    m2       = make_mask(attr_for(GOOD_METHOD, x, y2, x_cf))   # FIXED reference
    for sigma in NOISE_LEVELS:
        m1_noisy = make_mask(m1_clean + torch.randn_like(m1_clean) * sigma)  # renorm AFTER noise
        px_by_sigma[sigma].append(cs_pixel(x, m1_noisy, m2))
        lat_by_sigma[sigma].append(cs_latent(x, m1_noisy, m2))

# aggregate: results_noise = (sigma, mean_px, mean_lat); keep std for the band
results_noise, px_sd, lat_sd = [], [], []
for sigma in NOISE_LEVELS:
    p, l = np.array(px_by_sigma[sigma]), np.array(lat_by_sigma[sigma])
    results_noise.append((sigma, float(p.mean()), float(l.mean())))
    px_sd.append(float(p.std())); lat_sd.append(float(l.std()))
    print(f'  σ={sigma:4.2f}   pixel-CS={p.mean():.4f}±{p.std():.3f}   '
          f'CS_latent={l.mean():.4f}±{l.std():.3f}')

# ── plot: mean ± 1 std band over the gate images ─────────────────────────────
sig = [r[0] for r in results_noise]
px  = np.array([r[1] for r in results_noise]); px_sd  = np.array(px_sd)
lat = np.array([r[2] for r in results_noise]); lat_sd = np.array(lat_sd)
fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
ax.plot(sig, px,  'o-', color='#1E90FF', lw=2, label='pixel-CS  (flatten)')
ax.fill_between(sig, px - px_sd, px + px_sd, color='#1E90FF', alpha=0.15)
ax.plot(sig, lat, 's-', color='#e41a1c', lw=2, label='CS_latent  (ViT φ)')
ax.fill_between(sig, lat - lat_sd, lat + lat_sd, color='#e41a1c', alpha=0.15)
ax.set_xlabel('noise σ injected into y₁ mask'); ax.set_ylabel('class-separation (1 − cos)')
ax.set_title(f'Noise-injection gate — {GOOD_METHOD}\n'
             f'mean ± 1σ over {len(gate_imgs)} images',
             fontsize=11, fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('noise_curve.png', dpi=160, bbox_inches='tight'); plt.show()
print('saved noise_curve.png')
''')

# ════════════════════════════════════════════════════════════════════════════
md(r'''
### Task 1.1b — Noise-injection control for **Vanilla Grad**

Same gate, same images, same σ levels — but the map driving it is the noisy primitive
**Vanilla Grad** instead of KL-IG². A clean attribution should still show the
pixel-up / latent-down split; a noisy primitive's y₁ map is already incoherent, so the
encoder has less coherent signal to lose and the latent drop is expected to be **shallower**.
''')

code(r'''
# identical gate, driven by Vanilla Grad (averaged over the same gate_imgs)
VG_METHOD = 'Vanilla Grad'
torch.manual_seed(0)
px_by_sigma_vg  = {s: [] for s in NOISE_LEVELS}
lat_by_sigma_vg = {s: [] for s in NOISE_LEVELS}
for d in tqdm(gate_imgs, desc='noise gate (Vanilla Grad)'):
    x      = d['x'].squeeze(0).to(DEVICE)
    y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
    x_cf   = cf_csl[y2]; x_cf = (x_cf.squeeze(0) if x_cf.dim() == 4 else x_cf).to(DEVICE)
    m1_clean = make_mask(attr_for(VG_METHOD, x, y1, x_cf))
    m2       = make_mask(attr_for(VG_METHOD, x, y2, x_cf))     # FIXED reference
    for sigma in NOISE_LEVELS:
        m1_noisy = make_mask(m1_clean + torch.randn_like(m1_clean) * sigma)
        px_by_sigma_vg[sigma].append(cs_pixel(x, m1_noisy, m2))
        lat_by_sigma_vg[sigma].append(cs_latent(x, m1_noisy, m2))

results_noise_vg, px_sd_vg, lat_sd_vg = [], [], []
for sigma in NOISE_LEVELS:
    p, l = np.array(px_by_sigma_vg[sigma]), np.array(lat_by_sigma_vg[sigma])
    results_noise_vg.append((sigma, float(p.mean()), float(l.mean())))
    px_sd_vg.append(float(p.std())); lat_sd_vg.append(float(l.std()))
    print(f'  σ={sigma:4.2f}   pixel-CS={p.mean():.4f}±{p.std():.3f}   '
          f'CS_latent={l.mean():.4f}±{l.std():.3f}')

# ── plot: mean ± 1 std band ──────────────────────────────────────────────────
sig_vg = [r[0] for r in results_noise_vg]
px_vg  = np.array([r[1] for r in results_noise_vg]); px_sd_vg  = np.array(px_sd_vg)
lat_vg = np.array([r[2] for r in results_noise_vg]); lat_sd_vg = np.array(lat_sd_vg)
fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
ax.plot(sig_vg, px_vg,  'o-', color='#1E90FF', lw=2, label='pixel-CS  (flatten)')
ax.fill_between(sig_vg, px_vg - px_sd_vg, px_vg + px_sd_vg, color='#1E90FF', alpha=0.15)
ax.plot(sig_vg, lat_vg, 's-', color='#e41a1c', lw=2, label='CS_latent  (ViT φ)')
ax.fill_between(sig_vg, lat_vg - lat_sd_vg, lat_vg + lat_sd_vg, color='#e41a1c', alpha=0.15)
ax.set_xlabel('noise σ injected into y₁ mask'); ax.set_ylabel('class-separation (1 − cos)')
ax.set_title(f'Noise-injection gate — {VG_METHOD}\n'
             f'mean ± 1σ over {len(gate_imgs)} images',
             fontsize=11, fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('noise_curve_vanilla.png', dpi=160, bbox_inches='tight'); plt.show()
print('saved noise_curve_vanilla.png')
''')

# ════════════════════════════════════════════════════════════════════════════
md(r'''
## Task 1.2 — Decision checkpoint

If `CS_latent` does **not** fall as σ rises, the premise (the encoder filters incoherent-mask
noise) is wrong — **stop and reframe** before going further. Otherwise proceed to the crossover.
''')

code(r'''
lat0, latN = results_noise[0][2], results_noise[-1][2]
px0,  pxN  = results_noise[0][1], results_noise[-1][1]
latent_falls = latN < lat0 - 1e-3
pixel_flat_or_rises = pxN >= px0 - 1e-3

print(f'CS_latent: {lat0:.4f} → {latN:.4f}   Δ={latN-lat0:+.4f}   '
      f'{"FALLS ✓" if latent_falls else "does NOT fall ✗"}')
print(f'pixel-CS : {px0:.4f} → {pxN:.4f}   Δ={pxN-px0:+.4f}   '
      f'{"flat/rising ✓" if pixel_flat_or_rises else "falls ✗"}')

if not latent_falls:
    print('\n⛔ CHECKPOINT FAILED — CS_latent did not fall. '
          'Premise (encoder filters noise) is wrong. Reframe before the crossover.')
else:
    print('\n✅ CHECKPOINT PASSED — proceed to Task 1.4 (crossover).')
''')

# ════════════════════════════════════════════════════════════════════════════
md(r'''
## Task 1.4 — Crossover table  (n=20–30, both metrics, identical masks)

Every method's y₁/y₂ masks are scored by **both** metrics inside the **same loop** — the
only difference is encode-vs-flatten. Noisy primitives (Vanilla Grad, IG-zero) are expected
to rank **higher on pixel-CS than on CS_latent** (decorrelation flatters them in pixel space);
Random should sit near the floor on **both**.
''')

code(r'''
scores = {m: {'pixel': [], 'latent': []} for m in ROSTER}
used = 0
for d in tqdm(_pool, desc='crossover'):
    if used >= N_CROSSOVER: break
    x1 = d['x'].squeeze(0).to(DEVICE)
    y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
    x_cf = cf_csl.get(y2)
    if x_cf is None: continue
    x_cf = (x_cf.squeeze(0) if x_cf.dim() == 4 else x_cf).to(DEVICE)
    for m in ROSTER:
        a1 = attr_for(m, x1, y1, x_cf)
        a2 = attr_for(m, x1, y2, x_cf)
        m1, m2 = make_mask(a1), make_mask(a2)
        scores[m]['pixel'].append(cs_pixel(x1, m1, m2))     # identical masks ↓
        scores[m]['latent'].append(cs_latent(x1, m1, m2))
    used += 1
print(f'scored {used} images × {len(ROSTER)} methods')

# ── aggregate → mean per method per metric, with ranks + rank-shift ──────────
table = {m: (float(np.mean(scores[m]['pixel'])), float(np.mean(scores[m]['latent'])))
         for m in ROSTER}
rank_px  = {m: i + 1 for i, m in enumerate(sorted(ROSTER, key=lambda m: -table[m][0]))}
rank_lat = {m: i + 1 for i, m in enumerate(sorted(ROSTER, key=lambda m: -table[m][1]))}

df = pd.DataFrame([{
    'method': m, 'pixel_CS': table[m][0], 'rank_px': rank_px[m],
    'CS_latent': table[m][1], 'rank_lat': rank_lat[m],
    'rank_shift': rank_px[m] - rank_lat[m],          # +ve ⇒ demoted by the latent metric
} for m in ROSTER]).sort_values('CS_latent', ascending=False).reset_index(drop=True)

df.to_csv('crossover_table.csv', index=False)
print('\n' + df.round(4).to_string(index=False))
print('\nsaved crossover_table.csv   (rank_shift = rank_px − rank_lat; +ve ⇒ demoted in latent space)')
''')

# ════════════════════════════════════════════════════════════════════════════
md(r'''
## Task 1.5 — Record verdict

Mark each prediction **held / partial / failed** and persist the artifacts
(`noise_curve.png`, `crossover_table.csv`).
''')

code(r'''
verdict = {}

# P1 — gate: CS_latent falls with noise
verdict['gate_latent_falls'] = 'held' if latent_falls else 'failed'
# P2 — gate: pixel-CS flat or rising
verdict['gate_pixel_flat_or_rises'] = 'held' if pixel_flat_or_rises else 'failed'

# P3 — crossover: noisy primitives rank higher on pixel than latent (positive rank_shift)
_prims = [m for m in ['Vanilla Grad', 'IG-zero'] if m in rank_px]
_shift = np.mean([rank_px[m] - rank_lat[m] for m in _prims]) if _prims else 0.0
verdict['crossover_primitives_demoted'] = (
    'held' if _shift > 0.5 else 'partial' if _shift > 0 else 'failed')

# P4 — Random near the floor on BOTH (bottom third of the roster)
_floor = max(1, len(ROSTER) // 3)
rand_floor = ('Random' in rank_px and rank_px['Random'] > len(ROSTER) - _floor
              and rank_lat['Random'] > len(ROSTER) - _floor)
verdict['random_near_floor_both'] = 'held' if rand_floor else 'partial'

print('=== VERDICT ===')
for k, v in verdict.items():
    print(f'  {k:32s}: {v}')
print(f'\n  primitive mean rank_shift (px−lat) = {_shift:+.2f}')
print(f'  Random ranks: pixel #{rank_px.get("Random","-")}  latent #{rank_lat.get("Random","-")}  of {len(ROSTER)}')
print('\nartifacts: noise_curve.png, crossover_table.csv')

pd.DataFrame([{'prediction': k, 'result': v} for k, v in verdict.items()]
            ).to_csv('gate_verdict.csv', index=False)
print('saved gate_verdict.csv')
''')

# ════════════════════════════════════════════════════════════════════════════
md(r'''
# Part 2 — CS_latent across the full roster & orthogonality vs faithfulness

A second, complementary read on class sensitivity. Same masks/encoder as Part 1, but here we
(1) rank **every** method by `CS_latent` (does it return *different* maps for y₁ vs y₂?), then
(2) ask whether class sensitivity is a **separate axis** from faithfulness (Insertion-AUC) —
if the two don't line up, CS is non-redundant. **Roster = all methods in the notebook + Random.**
''')

md(r'''
## Task 1.3 — Run CS_latent across the roster  (n=30–50, ViT)

Per image: attribute for y₁ and y₂, mask, encode, `CS_latent = 1 − cos(z_y1, z_y2)`.
We compute the **Insertion-AUC of the y₁ map in the same loop** (faithfulness axis for Task 1.4),
so attributions are computed once and reused — no `<ins_auc>` placeholders.

**Watch:** if **Vanilla Grad** ranks near the top of CS_latent, that's the pixel-style gaming
problem — flag it; it motivates the crossover.
''')

code(r'''
N_ROSTER     = 40          # images for the roster sweep (30–50)
N_INS_STEPS  = 50          # insertion/deletion quadrature steps

def insertion_deletion(model, xb, attr_hw, target, n_steps=N_INS_STEPS):
    """Ins/Del AUC for a (H,W) attribution. xb is (1,C,H,W). Blur baseline (matches pipeline)."""
    C, H, W = xb.shape[1], xb.shape[2], xb.shape[3]
    order = attr_hw.detach().reshape(-1).abs().argsort(descending=True)
    pps   = max(1, H * W // n_steps)
    blur  = F.avg_pool2d(xb, kernel_size=31, stride=1, padding=15)
    x_ins, x_del = blur.clone(), xb.clone()
    ins_s, del_s = [], []
    with torch.no_grad():
        for step in range(n_steps):
            pix = order[step * pps:(step + 1) * pps]
            for ch in range(C):
                x_ins[:, ch].reshape(-1)[pix] = xb[:, ch].reshape(-1)[pix]
                x_del[:, ch].reshape(-1)[pix] = blur[:, ch].reshape(-1)[pix]
            ins_s.append(model(x_ins).softmax(-1)[0, target].item())
            del_s.append(model(x_del).softmax(-1)[0, target].item())
    return float(np.trapz(ins_s) / n_steps), float(np.trapz(del_s) / n_steps)

cs_scores  = {m: [] for m in ROSTER}
ins_scores = {m: [] for m in ROSTER}     # Ins-AUC of the y1 map (faithfulness axis)
used = 0
for d in tqdm(_pool, desc='CS_latent roster'):
    if used >= N_ROSTER: break
    x1 = d['x'].squeeze(0).to(DEVICE)
    xb = x1.unsqueeze(0)
    y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
    x_cf = cf_csl.get(y2)
    if x_cf is None: continue
    x_cf = (x_cf.squeeze(0) if x_cf.dim() == 4 else x_cf).to(DEVICE)
    for m in ROSTER:
        a1 = attr_for(m, x1, y1, x_cf)
        a2 = attr_for(m, x1, y2, x_cf)
        m1, m2 = make_mask(a1), make_mask(a2)
        cs_scores[m].append(cs_latent(x1, m1, m2))
        ins_scores[m].append(insertion_deletion(model, xb, a1.to(DEVICE), y1)[0])
    used += 1
print(f'scored {used} images × {len(ROSTER)} methods')

cs_mean  = {m: float(np.mean(cs_scores[m]))  for m in ROSTER}
ins_mean = {m: float(np.mean(ins_scores[m])) for m in ROSTER}
print('\nCS_latent per method (ranked):')
for m, v in sorted(cs_mean.items(), key=lambda kv: -kv[1]):
    print(f'  {m:20s}  CS_latent={v:.4f}   Ins-AUC={ins_mean[m]:.4f}')

# Vanilla-Grad gaming flag
_vg_rank = sorted(ROSTER, key=lambda m: -cs_mean[m]).index('Vanilla Grad') + 1
vg_games_cs = _vg_rank <= max(2, len(ROSTER) // 4)
print(f'\n[flag] Vanilla Grad CS_latent rank = #{_vg_rank}/{len(ROSTER)}  '
      f'→ {"GAMING CS (pixel-style) — flag for crossover" if vg_games_cs else "not top-ranked (ok)"}')
''')

md(r'''
## Task 1.4 — Orthogonality: CS vs Faithfulness  (the finding)

Scatter each method by (Insertion-AUC →, CS_latent →). If the methods do **not** line up on a
diagonal, the two axes measure different things and class sensitivity is a genuinely new axis.
''')

code(r'''
points = [(ins_mean[m], cs_mean[m], m) for m in ROSTER]
fig, ax = plt.subplots(figsize=(7.5, 6), facecolor='white')
for fx, cy, m in points:
    col = KM.COLORS.get(m, '#999999')
    ax.scatter(fx, cy, s=90, color=col, edgecolor='black', lw=0.6, zorder=3)
    ax.annotate(m, (fx, cy), fontsize=8, xytext=(5, 4),
                textcoords='offset points', zorder=4)
ax.set_xlabel('Faithfulness (Insertion-AUC) →', fontsize=11)
ax.set_ylabel('Class Sensitivity (CS_latent) →', fontsize=11)
ax.set_title('CS vs Faithfulness across methods\n(off-diagonal spread ⇒ orthogonal axes)',
             fontsize=11, fontweight='bold')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('cs_vs_faithfulness_scatter.png', dpi=160, bbox_inches='tight'); plt.show()
print('saved cs_vs_faithfulness_scatter.png')
''')

md(r'''
## Task 1.5 — Quantify orthogonality  (one number)  + save

Spearman ρ between Insertion-AUC and CS_latent across methods. **ρ ≈ 0 / not significant ⇒
CS is non-redundant with faithfulness — the finding.** Strongly positive ρ ⇒ CS just re-measures
faithfulness (premise weakens; this is the gate).
''')

code(r'''
from scipy.stats import spearmanr
xs = [ins_mean[m] for m in ROSTER]
ys = [cs_mean[m]  for m in ROSTER]
rho, pval = spearmanr(xs, ys)
print(f'CS_latent vs faithfulness (Ins-AUC) across {len(ROSTER)} methods: '
      f'rho={rho:+.2f}  p={pval:.3f}')
if pval < 0.05 and rho > 0.5:
    print('  ⚠ strongly positive → CS may just re-measure faithfulness (premise weakens).')
else:
    print('  ✅ near-zero / not significant → CS is an ORTHOGONAL, non-redundant axis. (the finding)')

# ── save: per-method roster table ────────────────────────────────────────────
roster_df = pd.DataFrame([{
    'method': m, 'CS_latent': cs_mean[m], 'Ins_AUC': ins_mean[m],
    'cs_rank': sorted(ROSTER, key=lambda k: -cs_mean[k]).index(m) + 1,
} for m in ROSTER]).sort_values('CS_latent', ascending=False).reset_index(drop=True)
roster_df.to_csv('cs_latent_roster.csv', index=False)
print('\n' + roster_df.round(4).to_string(index=False))
print(f'\nrho={rho:+.3f} (p={pval:.3f}); Vanilla Grad CS rank #{_vg_rank}/{len(ROSTER)}')
print('saved: cs_latent_roster.csv, cs_vs_faithfulness_scatter.png')
''')

# ════════════════════════════════════════════════════════════════════════════
md(r'''
# Day 2 — Validity Battery

Two known-answer **oracles** (no attribution method in the loop) + the assembled validity
ladder + a cross-encoder robustness check. The **class-perfect oracle is the gate**: it must
score *above* Vanilla Grad — otherwise the metric saturates at noise level and can't tell
true class evidence from a noisy primitive.

GT per-class masks come from **CLIPSeg** (text-prompted segmentation), so we get an "ideal
evidence" mask for *both* co-occurring classes in the same image without any attribution method.
''')

code(r'''
# ── Day 2 setup: extra encoders (ResNet, CLIP) + CLIPSeg GT-proxy segmenter ──
from torchvision import transforms as _T
from transformers import CLIPModel, CLIPSegProcessor, CLIPSegForImageSegmentation

N_VALID = 30                          # images per Day-2 loop (oracles + cross-encoder)

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
def denorm(t): return (t.detach().cpu() * _STD + _MEAN).clamp(0, 1)
_to_pil = _T.ToPILImage()

def cs_latent_enc(x, m1, m2, enc):    # generic-encoder CS_latent
    z1, z2 = enc(masked_image(x, m1)), enc(masked_image(x, m2))
    return float(1.0 - F.cosine_similarity(z1[None], z2[None]).item())

# ResNet-50 avgpool feature (2048) — task-model oracle
_res_feats = {}
model.avgpool.register_forward_hook(lambda m, i, o: _res_feats.__setitem__('z', o.detach()))
def enc_resnet(t):
    with torch.no_grad(): model(t.unsqueeze(0).to(DEVICE))
    return _res_feats['z'].flatten()

# CLIP image encoder (512)
_clip_mdl = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(DEVICE).eval()
def enc_clip(t):
    with torch.no_grad():
        vo = _clip_mdl.vision_model(pixel_values=t.unsqueeze(0).to(DEVICE))
        z  = _clip_mdl.visual_projection(vo.pooler_output)
    return z[0]

ENCODERS = {'ViT': enc_vit, 'ResNet': enc_resnet, 'CLIP': enc_clip}

# CLIPSeg: text → per-class segmentation (the GT-proxy "ideal" mask)
_clipseg_proc = CLIPSegProcessor.from_pretrained('CIDAS/clipseg-rd64-refined')
_clipseg = CLIPSegForImageSegmentation.from_pretrained(
    'CIDAS/clipseg-rd64-refined').to(DEVICE).eval()
def gt_class_mask(x1, cls, H, W):
    """CLIPSeg segmentation for `cls` → (H,W) heatmap in [0,1]. No attribution method."""
    text = imagenet_labels[int(cls)].split(',')[0].strip()
    inp  = _clipseg_proc(text=[text], images=[_to_pil(denorm(x1))],
                         return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        pr = _clipseg(**inp).logits.sigmoid().float()
    while pr.dim() < 4: pr = pr.unsqueeze(0)
    return F.interpolate(pr, size=(H, W), mode='bilinear', align_corners=False)[0, 0]

print('Day-2 ready: ENCODERS =', list(ENCODERS), '| CLIPSeg loaded | N_VALID =', N_VALID)
''')

md(r'''
## Task 2.1 — Class-blind oracle  (must score ≈ 0)

Feed the **same** map for both classes. If the metric has a true zero, identical masks →
identical encodings → CS ≈ 0. A non-zero result means the encoder isn't deterministic or the
masking is buggy — fix before trusting anything else.
''')

code(r'''
blind, used = [], 0
for d in tqdm(_pool, desc='2.1 class-blind oracle'):
    if used >= N_VALID: break
    x = d['x'].squeeze(0).to(DEVICE); H, W = x.shape[1], x.shape[2]
    y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
    x_cf = cf_csl.get(y2)
    if x_cf is None: continue
    x_cf = (x_cf.squeeze(0) if x_cf.dim() == 4 else x_cf).to(DEVICE)
    m = make_mask(attr_for(GOOD_METHOD, x, y1, x_cf))      # ONE map
    blind.append(cs_latent_enc(x, m, m, enc_vit))          # SAME mask twice
    used += 1

blind_mean = float(np.mean(blind))
print(f'class-blind oracle CS = {blind_mean:.5f}  (n={len(blind)})')
print('  ' + ('✅ ≈ 0 — metric has a true zero' if abs(blind_mean) < 1e-3
              else '⚠ NOT ≈ 0 — check encoder determinism / masking before continuing'))
''')

md(r'''
## Task 2.2 — Class-perfect oracle  (THE GATE: must beat Vanilla Grad)

CLIPSeg per-class masks are the "ideal evidence" ceiling — the actual pixels of each class,
no attribution method. The metric should reward this *above* the noisy-primitive floor.

**Confound (not hidden):** GT-style masks are contiguous blobs; attribution masks are sparse.
If perfect scores high partly because it's *coherent* rather than *correct*, that's the same
coherence point Day 3 makes. The **shuffle control** (same density, scrambled location) tests
it: if it drops toward Random, the signal is *location*, not blob.
''')

code(r'''
perfect, perfect_shuf, used = [], [], 0
_perm_rng = torch.Generator(device='cpu').manual_seed(0)
for d in tqdm(_pool, desc='2.2 class-perfect oracle'):
    if used >= N_VALID: break
    x = d['x'].squeeze(0).to(DEVICE); H, W = x.shape[1], x.shape[2]
    y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
    seg1, seg2 = gt_class_mask(x, y1, H, W), gt_class_mask(x, y2, H, W)
    if seg1 is None or seg2 is None: continue
    m1, m2 = make_mask(seg1), make_mask(seg2)
    perfect.append(cs_latent_enc(x, m1, m2, enc_vit))
    # shuffle control: same mask densities, scrambled spatial location
    p = torch.randperm(H * W, generator=_perm_rng)
    s1 = make_mask(m1.flatten()[p.to(DEVICE)].reshape(H, W))
    s2 = make_mask(m2.flatten()[p.to(DEVICE)].reshape(H, W))
    perfect_shuf.append(cs_latent_enc(x, s1, s2, enc_vit))
    used += 1

perfect_mean = float(np.mean(perfect)); shuf_mean = float(np.mean(perfect_shuf))
VG_REF   = float(cs_mean.get('Vanilla Grad', 0.1589))   # measured Day-1 VG (fallback 0.1589)
RAND_REF = float(cs_mean.get('Random', float('nan')))
print(f'class-perfect oracle CS = {perfect_mean:.4f}  (n={len(perfect)})')
print(f'  gate ref (Vanilla Grad) = {VG_REF:.4f}')
print('  ' + ('✅ GATE PASS — headroom above noise (only true class evidence reaches the ceiling)'
              if perfect_mean > VG_REF else
              '⛔ GATE FAIL — metric SATURATES at noise level; STOP and rethink'))
print(f'  shuffle control CS = {shuf_mean:.4f}  (should drop toward Random {RAND_REF:.4f}; '
      f'{"location-driven ✓" if shuf_mean < perfect_mean - 0.01 else "blob/coherence-driven ⚠"})')
''')

md(r'''
## Task 2.3 — Assemble the validity ladder

Slot the oracles around the Day-1 numbers. The claim: **blind ≈ 0  <  Random  <  real methods
<  perfect.** If that ordering holds, CS_latent is validated against known-answer cases with no
attribution method in the loop — it isn't circular.
''')

code(r'''
_real = [m for m in ROSTER if m != 'Random']
real_lo = float(min(cs_mean[m] for m in _real))
real_hi = float(max(cs_mean[m] for m in _real))
ladder = {
    'class-blind oracle':   blind_mean,           # expect ≈ 0
    'Random':               cs_mean.get('Random', RAND_REF),
    'real methods (range)': (real_lo, real_hi),   # Day-1 spread
    'class-perfect oracle': perfect_mean,         # expect > real_hi ideally
}
monotone = (blind_mean < ladder['Random'] < real_lo) and (perfect_mean > VG_REF)
print('=== validity ladder ===')
for k, v in ladder.items():
    print(f'  {k:22s}: {("%.4f–%.4f" % v) if isinstance(v, tuple) else "%.4f" % v}')
print('  ' + ('✅ monotone ordering holds — CS_latent validated (non-circular)'
              if monotone else '⚠ ordering broken — inspect which rung is out of place'))

# ── plot ladder ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4), facecolor='white')
ax.barh(0, blind_mean, color='#bbbbbb', height=0.55, label='class-blind (≈0)')
ax.barh(1, ladder['Random'], color='#888888', height=0.55, label='Random floor')
ax.barh(2, real_hi - real_lo, left=real_lo, color='#1E90FF', height=0.55, label='real methods (range)')
ax.barh(3, perfect_mean, color='#8b0000', height=0.55, label='class-perfect (ceiling)')
ax.axvline(VG_REF, color='black', ls='--', lw=1.2, label=f'Vanilla Grad gate = {VG_REF:.3f}')
ax.set_yticks(range(4)); ax.set_yticklabels(list(ladder.keys()))
ax.set_xlabel('CS_latent (ViT)'); ax.set_title('Validity ladder', fontweight='bold')
ax.legend(fontsize=8, loc='lower right'); ax.grid(axis='x', alpha=0.3)
plt.tight_layout(); plt.savefig('validity_ladder.png', dpi=160, bbox_inches='tight'); plt.show()
print('saved validity_ladder.png')
''')

md(r'''
## Task 2.4 — Cross-encoder robustness

Rerun the roster on **ViT + ResNet + CLIP**. Absolute CS values shift between encoders, but the
**ranking** should be stable — that means we're measuring class structure, not a single encoder's
artifact. Watch where Vanilla Grad lands per encoder: lower on ResNet/CLIP than ViT = partial demotion.
''')

code(r'''
import itertools as _it
from scipy.stats import spearmanr

roster_scores = {e: {m: [] for m in ROSTER} for e in ENCODERS}
used = 0
for d in tqdm(_pool, desc='2.4 cross-encoder'):
    if used >= N_VALID: break
    x = d['x'].squeeze(0).to(DEVICE); H, W = x.shape[1], x.shape[2]
    y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
    x_cf = cf_csl.get(y2)
    if x_cf is None: continue
    x_cf = (x_cf.squeeze(0) if x_cf.dim() == 4 else x_cf).to(DEVICE)
    for m in ROSTER:
        a1, a2 = attr_for(m, x, y1, x_cf), attr_for(m, x, y2, x_cf)
        msk1, msk2 = make_mask(a1), make_mask(a2)            # attribute once, encode 3×
        for ename, enc in ENCODERS.items():
            roster_scores[ename][m].append(cs_latent_enc(x, msk1, msk2, enc))
    used += 1
print(f'scored {used} images × {len(ROSTER)} methods × {len(ENCODERS)} encoders')

means = {e: {m: float(np.mean(roster_scores[e][m])) for m in ROSTER} for e in ENCODERS}
def _rank(dd):
    order = sorted(dd, key=lambda k: -dd[k]); return {m: i + 1 for i, m in enumerate(order)}
ranks = {e: _rank(means[e]) for e in ENCODERS}

ce_df = pd.DataFrame([{'method': m,
                       **{f'{e}_CS': means[e][m] for e in ENCODERS},
                       **{f'{e}_rank': ranks[e][m] for e in ENCODERS}}
                      for m in ROSTER]).sort_values('ViT_rank').reset_index(drop=True)
ce_df.to_csv('cross_encoder_ranks.csv', index=False)

corrs = []
for e1, e2 in _it.combinations(ENCODERS, 2):
    rho = spearmanr([ranks[e1][m] for m in ROSTER], [ranks[e2][m] for m in ROSTER]).correlation
    corrs.append(rho); print(f'  rank Spearman {e1}-{e2}: {rho:+.3f}')
rank_corr = float(np.mean(corrs))
print(f'\nmean cross-encoder rank Spearman = {rank_corr:+.3f}  '
      f'{"✅ encoder-robust (>0.7)" if rank_corr > 0.7 else "⚠ encoder-sensitive"}')
print('\n' + ce_df.round(4).to_string(index=False))
vg_ranks = {e: ranks[e].get('Vanilla Grad') for e in ENCODERS}
print(f'\nVanilla Grad rank per encoder: {vg_ranks}  (lower on ResNet/CLIP than ViT ⇒ partial demotion)')
print('saved cross_encoder_ranks.csv  (also: validity_ladder.png)')
''')

# ════════════════════════════════════════════════════════════════════════════
md(r'''
# Day 3 — Three metrics, the crossover, and the CF-existence proof

Now we put `CS_latent` (the one under suspicion) next to two alternatives that need **no
encoder**: a pixel-correlation primitive and a **location-based top-k region overlap**. The
question Day 3 adjudicates: *does the method ranking change across measurement spaces?* If
`latent` and `overlap` agree on the top methods, latent's oracle problem doesn't bite the
ranking. If they disagree, **overlap is the trustworthy one** (it's purely spatial).

Roster = all methods in the notebook + Random (your `'IG'` ≡ this notebook's `'IG-zero'`).
''')

code(r'''
# ── the three metrics: all compare the y1-map vs the y2-map; they differ in HOW ──
from captum.attr import IntegratedGradients as _CaptumIG

def _flat(a): return a.detach().float().reshape(-1).cpu()

def pearson(u, v):
    u = u - u.mean(); v = v - v.mean()
    return float((u @ v) / (u.norm() * v.norm() + EPS))

# (A) latent — mask, encode, cosine (the metric under suspicion)
def cs_latent3(x, a1, a2, enc=enc_vit):
    return cs_latent_enc(x, make_mask(a1), make_mask(a2), enc)

# (B) overlap — top-k% region IoU, location-based, NO encoder
def topk_region(a, k):
    a = _flat(a).abs(); n = max(1, int(k * a.numel()))
    thr = torch.topk(a, n).values.min()
    return a >= thr                                  # binary (H*W,)
def cs_overlap(x, a1, a2, k=0.10):
    r1, r2 = topk_region(a1, k), topk_region(a2, k)
    iou = (r1 & r2).sum().item() / ((r1 | r2).sum().item() + EPS)
    return float(1.0 - iou)                          # high = different regions = sensitive

# (C) pixel-correlation — the noisy "primitive" baseline for the crossover
def cs_pixel_corr(x, a1, a2):
    return float(1.0 - abs(pearson(_flat(a1), _flat(a2))))

def spatial_shuffle(a):
    f = _flat(a); p = torch.randperm(f.numel())
    return f[p].reshape(tuple(a.shape))

print('Day-3 metrics ready: cs_latent3, cs_overlap (top-k IoU), cs_pixel_corr')
''')

md(r'''
## Task 3.1 — Crossover (3 metrics, full roster)

Same y1/y2 maps scored three ways. **Watch:** Vanilla Grad should rank *high* on pixel (noise
inflates correlation-distance); if it **drops on overlap**, the location metric is filtering noise
correctly. Random should floor on **all three** — if it doesn't floor on latent, that's latent
failing in plain sight.
''')

code(r'''
N_DAY3 = 50          # 50–200, your call (KL-IG² variants dominate the cost)

scores3 = {m: {'pixel': [], 'latent': [], 'overlap': []} for m in ROSTER}
used = 0
for d in tqdm(_pool, desc='3.1 crossover (3 metrics)'):
    if used >= N_DAY3: break
    x = d['x'].squeeze(0).to(DEVICE); H, W = x.shape[1], x.shape[2]
    y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
    x_cf = cf_csl.get(y2)
    if x_cf is None: continue
    x_cf = (x_cf.squeeze(0) if x_cf.dim() == 4 else x_cf).to(DEVICE)
    for m in ROSTER:
        a1, a2 = attr_for(m, x, y1, x_cf), attr_for(m, x, y2, x_cf)
        scores3[m]['pixel'].append(  cs_pixel_corr(x, a1, a2))
        scores3[m]['latent'].append( cs_latent3(x, a1, a2))
        scores3[m]['overlap'].append(cs_overlap(x, a1, a2))
    used += 1
print(f'scored {used} images × {len(ROSTER)} methods × 3 metrics')

means3 = {m: {k: float(np.mean(scores3[m][k])) for k in ('pixel', 'latent', 'overlap')}
          for m in ROSTER}
def rank_by(mm, key):
    order = sorted(mm, key=lambda m: -mm[m][key]); return {m: i + 1 for i, m in enumerate(order)}
rk = {k: rank_by(means3, k) for k in ('pixel', 'latent', 'overlap')}

df3 = pd.DataFrame([{
    'method': m,
    'pixel': means3[m]['pixel'],   'rank_px':  rk['pixel'][m],
    'latent': means3[m]['latent'], 'rank_lat': rk['latent'][m],
    'overlap': means3[m]['overlap'],'rank_ov':  rk['overlap'][m],
} for m in ROSTER]).sort_values('rank_lat').reset_index(drop=True)
df3.to_csv('crossover_3metric.csv', index=False)
print('\n' + df3.round(4).to_string(index=False))

from scipy.stats import spearmanr as _sr
_lat_ov = _sr([rk['latent'][m] for m in ROSTER], [rk['overlap'][m] for m in ROSTER]).correlation
print(f'\nlatent–overlap rank Spearman = {_lat_ov:+.3f}  '
      f'{"→ agree (latent oracle issue does not bite ranking)" if _lat_ov > 0.7 else "→ DISAGREE (trust overlap, location-based)"}')
for tag in ('Vanilla Grad', 'Random'):
    if tag in rk['pixel']:
        print(f'  {tag:13s}: pixel #{rk["pixel"][tag]:<2d} latent #{rk["latent"][tag]:<2d} overlap #{rk["overlap"][tag]:<2d}')
print('saved crossover_3metric.csv')
''')

md(r'''
## Task 3.2 — CF-existence proof (both metrics)

Toggle IG's baseline: class-agnostic (zero) → y2-counterfactual image. Does targeting the
counterfactual you want sensitivity *to* raise CS? **Expected:** CF lift > 0 on **both**
latent and overlap, significant (Wilcoxon) — a measurement-space-independent existence proof.
''')

code(r'''
from scipy.stats import wilcoxon
_IG_STEPS_CF = 25
_ig = _CaptumIG(model)
def ig_base(x1, cls, baseline):
    a = _ig.attribute(x1.unsqueeze(0), target=int(cls),
                      baselines=baseline.unsqueeze(0), n_steps=_IG_STEPS_CF)
    return KM.absmax_collapse(a.squeeze(0)).detach().cpu()

cf_lift = {'latent': [], 'overlap': []}
used = 0
for d in tqdm(_pool, desc='3.2 CF-existence (IG baseline toggle)'):
    if used >= N_DAY3: break
    x = d['x'].squeeze(0).to(DEVICE); H, W = x.shape[1], x.shape[2]
    y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
    x_cf = cf_csl.get(y2)
    if x_cf is None: continue
    x_cf = (x_cf.squeeze(0) if x_cf.dim() == 4 else x_cf).to(DEVICE)
    zero = torch.zeros_like(x)
    a1_ag, a2_ag = ig_base(x, y1, zero), ig_base(x, y2, zero)        # agnostic baseline
    a1_cf, a2_cf = ig_base(x, y1, x_cf), ig_base(x, y2, x_cf)        # y2-CF baseline
    cf_lift['latent'].append( cs_latent3(x, a1_cf, a2_cf) - cs_latent3(x, a1_ag, a2_ag))
    cf_lift['overlap'].append(cs_overlap(x, a1_cf, a2_cf) - cs_overlap(x, a1_ag, a2_ag))
    used += 1

rows = []
for metric in ('latent', 'overlap'):
    vals = np.array(cf_lift[metric]); mu = float(vals.mean())
    try:    p = float(wilcoxon(vals).pvalue)
    except Exception: p = float('nan')
    sig = (p < 0.05) and (mu > 0)
    rows.append({'metric': metric, 'cf_lift_mean': mu, 'wilcoxon_p': p,
                 'n': len(vals), 'pos_and_sig': sig})
    print(f'  {metric:8s} CF lift mean={mu:+.4f}  Wilcoxon p={p:.3g}  '
          f'{"✅ >0 & significant" if sig else "⚠ not a clean positive lift"}')
pd.DataFrame(rows).to_csv('cf_lift_dual.csv', index=False)
print('saved cf_lift_dual.csv')
''')

md(r'''
## Task 3.3 — Overlap-only oracle re-check  (settles it)

Quick known-answer check on the **overlap** metric: GT classes sit in different places → low
IoU → high CS. If the CLIPSeg "perfect" masks score well above the method spread (and the
spatial-shuffle control changes it), overlap has the headroom latent may have lacked.
''')

code(r'''
perfect_ov, shuffle_ov, random_ov = [], [], []
used = 0
for d in tqdm(_pool, desc='3.3 overlap oracle'):
    if used >= N_DAY3: break
    x = d['x'].squeeze(0).to(DEVICE); H, W = x.shape[1], x.shape[2]
    y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
    seg1, seg2 = gt_class_mask(x, y1, H, W), gt_class_mask(x, y2, H, W)
    if seg1 is None or seg2 is None: continue
    perfect_ov.append(cs_overlap(x, seg1, seg2))
    shuffle_ov.append(cs_overlap(x, seg1, spatial_shuffle(seg2)))     # same density, wrong place
    random_ov.append(cs_overlap(x, torch.randn(H, W), torch.randn(H, W)))
    used += 1

p_ov, s_ov, r_ov = float(np.mean(perfect_ov)), float(np.mean(shuffle_ov)), float(np.mean(random_ov))
method_hi = max(means3[m]['overlap'] for m in ROSTER if m != 'Random') if 'means3' in dir() else float('nan')
print(f'overlap oracle  perfect={p_ov:.4f}  shuffle={s_ov:.4f}  random={r_ov:.4f}  (n={len(perfect_ov)})')
print(f'  best real-method overlap CS = {method_hi:.4f}  → '
      f'{"perfect has headroom ✅" if p_ov > method_hi else "no headroom over methods ⚠"}')
print(f'  shuffle {"changes the score ✅ (location-driven)" if abs(p_ov - s_ov) > 0.01 else "≈ perfect ⚠ (blob/coherence-driven)"}')
with open('overlap_oracle.txt', 'w') as f:
    f.write(f'overlap oracle\nperfect={p_ov:.4f}\nshuffle={s_ov:.4f}\nrandom={r_ov:.4f}\n'
            f'best_real_method={method_hi:.4f}\nn={len(perfect_ov)}\n')
print('saved overlap_oracle.txt')
''')

# ════════════════════════════════════════════════════════════════════════════
md(r'''
# Attribution-map gallery — all methods, Top-1 vs Top-2

The class-sensitivity examples figure, generalized to the **full roster**. Rows = images;
columns = `Original` then each method's Top-1 / Top-2 attribution maps. `vmax` is computed
**per map** (99th-percentile of |attr|) so no method washes out another. This is the visual
companion to the metrics — you can eyeball whether a method actually moves its evidence between
y₁ and y₂. Self-contained: needs only Day-1 (`_pool`, `cf_csl`, `attr_for`, `model`).
''')

code(r'''
N_IMGS_VIZ  = 3                       # rows (images)
TOP_K_VIZ   = 2                       # Top-1, Top-2
VIZ_METHODS = list(ROSTER)            # all methods + Random; subset to taste

# self-contained denormalize (ImageNet stats) for display
_vMEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_vSTD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
def _denorm_disp(t): return (t.detach().cpu() * _vSTD + _vMEAN).clamp(0, 1)

# pick N ANIMAL images (top-1 ∈ ImageNet 0–399) with a real CF for y2, deduped by top-1 class
ANIMAL_CLS = set(range(0, 400))          # ImageNet 0–399 ≈ animals
def _gather(animal_only):
    out, seen = [], set()
    for d in _pool:
        y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
        if y2 not in cf_csl or y1 in seen: continue
        if animal_only and y1 not in ANIMAL_CLS: continue
        out.append(d); seen.add(y1)
        if len(out) >= N_IMGS_VIZ: break
    return out

viz_sel = _gather(animal_only=True)
if len(viz_sel) < N_IMGS_VIZ:            # fall back: pad with non-animals if pool is thin
    print(f'only {len(viz_sel)} animal images in pool — padding with non-animals')
    _have = {d['idx'] for d in viz_sel}
    for d in _gather(animal_only=False):
        if d['idx'] not in _have:
            viz_sel.append(d); _have.add(d['idx'])
        if len(viz_sel) >= N_IMGS_VIZ: break
print(f'gallery: {len(viz_sel)} images '
      f'({sum(int(d["high_cls"][0]) in ANIMAL_CLS for d in viz_sel)} animal) '
      f'× {len(VIZ_METHODS)} methods × Top-{TOP_K_VIZ}')

# compute maps
viz_data = []
for d in tqdm(viz_sel, desc='gallery attrs'):
    x1 = d['x'].squeeze(0).to(DEVICE); H, W = x1.shape[1], x1.shape[2]
    high_cls  = d['high_cls'][:TOP_K_VIZ]
    high_prob = d['high_probs'][:TOP_K_VIZ]
    x_cf = cf_csl[int(d['high_cls'][1])]
    x_cf = (x_cf.squeeze(0) if x_cf.dim() == 4 else x_cf).to(DEVICE)
    maps = {}
    for m in VIZ_METHODS:
        for ci, cls in enumerate(high_cls):
            a = attr_for(m, x1, cls, x_cf)
            maps[(m, ci)] = a.detach().cpu().numpy().reshape(H, W)
    viz_data.append({'x': x1, 'high_cls': high_cls, 'high_probs': high_prob, 'maps': maps})
''')

code(r'''
# ── plot: ONE METHOD PER ROW (columns grouped by image: Top-1 | Top-2) ───────
N_METH = len(VIZ_METHODS)
N_COLS = N_IMGS_VIZ * TOP_K_VIZ
N_ROWS = 1 + N_METH                      # row 0 = originals + class labels
fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(2.4 * N_COLS, 2.2 * N_ROWS),
                         facecolor='white', squeeze=False)

def _blank(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

# row 0 — originals (first sub-col of each image group) + per-column class titles
for i, item in enumerate(viz_data):
    img = _denorm_disp(item['x']).permute(1, 2, 0).numpy()
    for ci, cls in enumerate(item['high_cls']):
        ax = axes[0, i * TOP_K_VIZ + ci]; _blank(ax)
        if ci == 0: ax.imshow(img)
        ax.set_title(f"img{i+1} · T{ci+1}\n{imagenet_labels[cls].split(',')[0][:14]}"
                     f" (p={item['high_probs'][ci]:.2f})", fontsize=7)
axes[0, 0].set_ylabel('Original', fontsize=10, fontweight='bold',
                      color='#222', rotation=90, labelpad=8)

# one row per method
for m_i, m in enumerate(VIZ_METHODS):
    row = 1 + m_i
    for i, item in enumerate(viz_data):
        for ci, cls in enumerate(item['high_cls']):
            ax = axes[row, i * TOP_K_VIZ + ci]
            a = item['maps'][(m, ci)]
            vmax = max(np.percentile(np.abs(a), 99), 1e-9)
            ax.imshow(a, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
    # method name as the row label on the leftmost cell (colored)
    axes[row, 0].set_ylabel(m, fontsize=10, fontweight='bold',
                            color=KM.COLORS.get(m, '#777777'), rotation=90, labelpad=8)

# vertical separators between image groups
for i in range(1, N_IMGS_VIZ):
    bb_l = axes[0, i * TOP_K_VIZ - 1].get_position()
    bb_r = axes[0, i * TOP_K_VIZ].get_position()
    sep = (bb_l.x1 + bb_r.x0) / 2
    fig.add_artist(plt.Line2D([sep, sep], [0.02, 0.96], color='#bbb', lw=1.2,
                              transform=fig.transFigure))

plt.suptitle('Class-Sensitivity Attribution Gallery — one method per row '
             '(Top-1 vs Top-2, per-map vmax)',
             fontsize=13, fontweight='bold', y=1.005)
plt.tight_layout(); plt.subplots_adjust(hspace=0.3, wspace=0.1)
plt.savefig('cs_attr_gallery_all_methods_top2.png', dpi=140, bbox_inches='tight')
plt.show()
print('saved cs_attr_gallery_all_methods_top2.png')
''')

nb = new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
    "accelerator": "GPU",
}
with open('cs_gate_eval.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print('wrote cs_gate_eval.ipynb  with', len(cells), 'cells')
