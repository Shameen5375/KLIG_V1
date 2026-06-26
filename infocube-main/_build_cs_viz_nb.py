# Assembles cs_viz_playground.ipynb — a standalone visualization notebook.
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []
def md(s):   cells.append(new_markdown_cell(s.strip("\n")))
def code(s): cells.append(new_code_cell(s.strip("\n")))

# ════════════════════════════════════════════════════════════════════════════
md(r'''
# Class-Sensitivity Visualization Playground

A standalone notebook for **browsing the whole ImageNet-val stream** and rendering attribution
galleries — animals or any class, any method. Same Top-1/Top-2 logic as the eval notebooks
(a class "co-occurs" if its softmax prob > `PROB_THRESH`; images need ≥2 such classes), same
per-map `vmax` one-method-per-row gallery.

Workflow:
1. **scan** a large slice of val → cached candidate pool (`cands`)
2. **browse / pick** images (filter by animal, class name, index)
3. **gallery(sel)** → attribution maps for every method, Top-1 vs Top-2

Reuses `klig_methods.attr_map` for all 11 methods + a local Random branch.
''')

code(r'''
import importlib, math, pickle, warnings, random
from pathlib import Path
import numpy as np
if not hasattr(np, 'trapz'): np.trapz = np.trapezoid
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
warnings.filterwarnings('ignore')

import klig_methods as KM
importlib.reload(KM)
from klig_methods import attr_map, METHODS, make_phi
from torchvision.models import resnet50, ResNet50_Weights

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# all figures/tables this notebook saves land in ONE folder
OUT = Path('cs_viz_outputs'); OUT.mkdir(exist_ok=True)
def out(name): return str(OUT / name)            # use out('foo.png') in every save call
print('device:', DEVICE, '| outputs →', OUT, '| methods:', METHODS)
''')

code(r'''
# ── model + attribution dispatch ─────────────────────────────────────────────
weights  = ResNet50_Weights.IMAGENET1K_V2
model    = resnet50(weights=weights).to(DEVICE).eval()
preprocess      = weights.transforms()
imagenet_labels = weights.meta['categories']
phi = make_phi(model)

# lighter KLIG/IG settings so a gallery renders quickly
KM.N_STEPS, KM.N_SAMPLES = 25, 3
KM.IG_STEPS = 25; KM.SG_SAMPLES = 25; KM.EG_SAMPLES = 25
KM.T_DESC, KM.N_MC_DESC, KM.N_MC_GRAD = 25, 8, 3

ROSTER = list(METHODS) + ['Random']
ANIMAL_CLS = set(range(0, 400))          # ImageNet 0–399 ≈ animals
_rng = np.random.default_rng(0)

def attr_for(method, x1, cls, x_cf):
    H, W = x1.shape[-2], x1.shape[-1]
    if method == 'Random':
        return torch.from_numpy(_rng.standard_normal((H, W))).float()
    return attr_map(method, model, x1, int(cls), x_cf=x_cf, phi=phi)

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
def denorm(t): return (t.detach().cpu() * _STD + _MEAN).clamp(0, 1)
print('ready. roster:', ROSTER)
''')

md(r'''
## 1 — Scan the dataset → candidate pool

Streams `evanarlian/imagenet_1k_resized_256` (val) and keeps every **multi-class** image
**Every image qualifies** — `(y1, y2)` are just its top-2 predicted classes (`multiclass` flags
the genuinely confusable ones where both clear `PROB_THRESH`). Cap is `POOL_TARGET`. Cached to
`cs_viz_cache/cands.pkl` (checkpointed every 200 so a network drop doesn't lose progress); set
`RESCAN=True` to rebuild. Bigger `POOL_TARGET` → more variety.
''')

code(r'''
PROB_THRESH = 0.10      # only used to FLAG multi-class images; no longer required
POOL_TARGET = 600       # cap (bigger pool → more variety). every image qualifies now
MAX_SCAN    = 20000     # val images to stream through
RESCAN      = False     # set True (network up) to rebuild the pool with these settings

CACHE = Path('cs_viz_cache'); CACHE.mkdir(exist_ok=True)
_cache_cands = CACHE / 'cands.pkl'

if not RESCAN:
    # OFFLINE: merge every local cache that holds multi-class image lists (deduped) → big pool
    _srcs = ['cs_viz_cache/cands.pkl', 'klig2_dist_cache/klig2_dist_multiprob.pkl',
             'klig2_val_cache/klig2_dist_multiprob.pkl', 'cs_gate_cache/pool.pkl']
    merged, _seen = [], set()
    for _s in _srcs:
        if not Path(_s).exists(): continue
        for d in pickle.load(open(_s, 'rb')):
            if len(d.get('high_cls', [])) < 2: continue
            x = d['x']; x = x.squeeze(0) if x.dim() == 4 else x
            fp = round(float(x.float().sum()), 1)              # cheap dedup fingerprint
            if fp in _seen: continue
            _seen.add(fp)
            merged.append({'idx': len(merged), 'x': x.cpu(),
                           'high_cls': [int(c) for c in d['high_cls'][:2]],
                           'high_probs': [float(p) for p in d.get('high_probs', [1.0, 0.5])[:2]],
                           'multiclass': True})
    if merged:
        cands = [{**d, 'x': d['x'].to(DEVICE)} for d in merged]
        pickle.dump([{**d, 'x': d['x'].cpu()} for d in cands], open(_cache_cands, 'wb'))
        print(f'[offline] merged {len(cands)} unique images from {len(_srcs)} local caches')
    else:
        print('[offline] no local caches found → set RESCAN=True to stream'); cands = []
else:
    from datasets import load_dataset as _hf
    _ds = _hf('evanarlian/imagenet_1k_resized_256', split='val', streaming=True)
    _ds = _ds.shuffle(seed=42, buffer_size=5000)
    cands, scanned = [], 0
    try:
        for item in tqdm(_ds, total=MAX_SCAN, desc='scanning val'):
            if len(cands) >= POOL_TARGET or scanned >= MAX_SCAN: break
            scanned += 1
            img = item['image']
            if img.mode != 'RGB': img = img.convert('RGB')
            x = preprocess(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probs = model(x).softmax(-1)[0].cpu()
            # ALL images qualify now: always take the top-2 classes as (y1, y2)
            top2 = probs.topk(2).indices.tolist()
            n_conf = int((probs > PROB_THRESH).sum())           # how many classes are confident
            cands.append({'idx': len(cands), 'x': x, 'high_cls': top2,
                          'high_probs': [float(probs[c]) for c in top2],
                          'multiclass': n_conf >= 2})           # flag the genuinely confusable ones
            if scanned % 200 == 0:                              # checkpoint so a drop doesn't lose all
                pickle.dump([{**d, 'x': d['x'].cpu()} for d in cands], open(_cache_cands, 'wb'))
    except Exception as e:
        print(f'  [warn] scan interrupted ({type(e).__name__}); keeping {len(cands)} collected so far')
    pickle.dump([{**d, 'x': d['x'].cpu()} for d in cands], open(_cache_cands, 'wb'))
    print(f'kept {len(cands)} images (scanned {scanned}; '
          f'{sum(d["multiclass"] for d in cands)} multi-class)')

_n_animal = sum(int(d['high_cls'][0]) in ANIMAL_CLS for d in cands)
print(f'animals (top-1 ∈ 0–399): {_n_animal}/{len(cands)}')
''')

md(r'''
## 2 — Browse / pick images

`browse(...)` lists candidates; `pick_images(...)` selects a deduped set (one per top-1 class);
`by_idx([...])` grabs specific rows. All operate in-memory on `cands` — no re-streaming.
''')

code(r'''
def browse(animal_only=False, contains=None, n=40):
    shown = 0
    for d in cands:
        y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
        if animal_only and y1 not in ANIMAL_CLS: continue
        if contains and contains.lower() not in imagenet_labels[y1].lower(): continue
        print(f"  idx={d['idx']:4d}  top1={imagenet_labels[y1].split(',')[0][:22]:22s}"
              f"(p={d['high_probs'][0]:.2f}) | top2={imagenet_labels[y2].split(',')[0][:22]:22s}"
              f"(p={d['high_probs'][1]:.2f})  {'🐾' if y1 in ANIMAL_CLS else ''}")
        shown += 1
        if shown >= n: break
    print(f'{shown} shown')

def pick_images(n=3, animal_only=False, top1_class=None, contains=None, dedup=True, seed=0):
    # dedup=True → one image per top-1 class (variety); dedup=False → just take n (for big averages)
    pool = list(cands); random.Random(seed).shuffle(pool)
    sel, seen = [], set()
    for d in pool:
        y1 = int(d['high_cls'][0])
        if animal_only and y1 not in ANIMAL_CLS: continue
        if top1_class is not None and y1 != int(top1_class): continue
        if contains and contains.lower() not in imagenet_labels[y1].lower(): continue
        if dedup and y1 in seen: continue
        sel.append(d); seen.add(y1)
        if len(sel) >= n: break
    print(f'picked {len(sel)} images ({len(seen)} distinct top-1 classes)')
    return sel

def by_idx(ids):
    m = {d['idx']: d for d in cands}
    return [m[i] for i in ids if i in m]

# example: list candidates (set animal_only=True if you want only animals)
browse(n=25)
''')

md(r'''
## 3 — Counterfactuals (for KL-IG² methods only)

KL-IG² needs a real image of each selected image's y₂ class. `build_cf_for(sel)` is **offline by
default** — it seeds CFs from the cached `cands` pool (any image whose top-1 or top-2 is the
needed class) and uses a cached fallback for anything missing, so it never touches the network.
Pass `allow_stream=True` only if you want it to stream the val set for exact CFs (needs HF). Other
methods ignore the CF.
''')

code(r'''
def build_cf_for(sel, allow_stream=False):           # default OFFLINE: cache only, never streams
    need = {int(d['high_cls'][1]) for d in sel}
    cf = {}
    # free seeds from the already-downloaded pool: any cand whose top-1 OR top-2 is a needed class
    for d in cands:
        for c in (int(d['high_cls'][0]),
                  int(d['high_cls'][1]) if len(d['high_cls']) > 1 else -1):
            if c in need and c not in cf:
                xx = d['x']; cf[c] = (xx.squeeze(0) if xx.dim() == 4 else xx).to(DEVICE)
    still = need - set(cf)
    if still and allow_stream:                       # stream the rest — but survive network drops
        try:
            from datasets import load_dataset as _hf
            _s = _hf('evanarlian/imagenet_1k_resized_256', split='val',
                     streaming=True).shuffle(seed=13, buffer_size=500)
            best, lock, sc = {c: (-1.0, None) for c in still}, set(), 0
            for item in tqdm(_s, desc='CF stream', total=6000):
                sc += 1
                if len(lock) >= len(still) or sc >= 6000: break
                im = item['image']
                if im.mode != 'RGB': im = im.convert('RGB')
                xx = preprocess(im).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    pr = model(xx).softmax(-1)[0].cpu()
                for c in still:
                    if c in lock: continue
                    if float(pr[c]) > best[c][0]:
                        best[c] = (float(pr[c]), xx.squeeze(0).to(DEVICE))
                        if pr[c] >= 0.30: lock.add(c)
            for c in still:
                if best[c][1] is not None: cf[c] = best[c][1]
        except Exception as e:
            print(f'  [warn] CF streaming failed ({type(e).__name__}: {e}); using cached fallback')
    # final fallback: any cached image, so KL-IG² still runs (approximate CF) instead of crashing
    missing = need - set(cf)
    if missing and cands:
        fb = cands[0]['x']; fb = (fb.squeeze(0) if fb.dim() == 4 else fb).to(DEVICE)
        for c in missing: cf[c] = fb
        print(f'  [warn] {len(missing)} y2 classes had no real CF → cached-fallback image')
    print(f'CF ready for {len(cf)}/{len(need)} y2 classes')
    return cf
''')

md(r'''
## 4 — Gallery  (one method per row, Top-1 vs Top-2, per-map vmax)

`gallery(sel)` computes attribution maps for every method and plots them. Columns are grouped
by image (Top-1 | Top-2); rows are methods. Each map gets its own `vmax` (99th-pct |attr|).
''')

code(r'''
def gallery(sel, cf=None, methods=ROSTER, top_k=2, save=out('cs_viz_gallery.png')):
    if cf is None: cf = build_cf_for(sel)
    # compute maps
    viz = []
    for d in tqdm(sel, desc='gallery attrs'):
        x1 = d['x'].squeeze(0).to(DEVICE); H, W = x1.shape[1], x1.shape[2]
        hc, hp = d['high_cls'][:top_k], d['high_probs'][:top_k]
        y2 = int(d['high_cls'][1]); x_cf = cf.get(y2)
        if x_cf is not None:
            x_cf = (x_cf.squeeze(0) if x_cf.dim() == 4 else x_cf).to(DEVICE)
        maps = {}
        for m in methods:
            for ci, cls in enumerate(hc):
                maps[(m, ci)] = attr_for(m, x1, cls, x_cf).detach().cpu().numpy().reshape(H, W)
        viz.append({'x': x1, 'high_cls': hc, 'high_probs': hp, 'maps': maps})

    # plot
    N_M, N_COLS, N_ROWS = len(methods), len(sel) * top_k, 1 + len(methods)
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(2.4 * N_COLS, 2.2 * N_ROWS),
                             facecolor='white', squeeze=False)
    def _blank(ax):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
    for i, item in enumerate(viz):                       # row 0 — originals + class titles
        img = denorm(item['x']).permute(1, 2, 0).numpy()
        for ci, cls in enumerate(item['high_cls']):
            ax = axes[0, i * top_k + ci]; _blank(ax)
            if ci == 0: ax.imshow(img)
            ax.set_title(f"img{i+1}·T{ci+1}\n{imagenet_labels[cls].split(',')[0][:14]}"
                         f" (p={item['high_probs'][ci]:.2f})", fontsize=7)
    axes[0, 0].set_ylabel('Original', fontsize=10, fontweight='bold', rotation=90, labelpad=8)
    for m_i, m in enumerate(methods):                    # one row per method
        for i, item in enumerate(viz):
            for ci, cls in enumerate(item['high_cls']):
                ax = axes[1 + m_i, i * top_k + ci]
                a = item['maps'][(m, ci)]
                vmax = max(np.percentile(np.abs(a), 99), 1e-9)
                ax.imshow(a, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values(): sp.set_visible(False)
        axes[1 + m_i, 0].set_ylabel(m, fontsize=10, fontweight='bold',
                                    color=KM.COLORS.get(m, '#777777'), rotation=90, labelpad=8)
    for i in range(1, len(sel)):                         # separators between images
        bb_l = axes[0, i * top_k - 1].get_position(); bb_r = axes[0, i * top_k].get_position()
        sep = (bb_l.x1 + bb_r.x0) / 2
        fig.add_artist(plt.Line2D([sep, sep], [0.02, 0.96], color='#bbb', lw=1.2,
                                  transform=fig.transFigure))
    plt.suptitle('Attribution Gallery — one method per row (Top-1 vs Top-2, per-map vmax)',
                 fontsize=13, fontweight='bold', y=1.005)
    plt.tight_layout(); plt.subplots_adjust(hspace=0.3, wspace=0.1)
    if save: plt.savefig(save, dpi=140, bbox_inches='tight'); print('saved', save)
    plt.show()
''')

md(r'''
## 5 — Run it

Picks from the whole pool by default. Swap the call to play around:
- `pick_images(n=4, animal_only=True)` — animals only
- `pick_images(n=3, contains='dog')` — class name contains "dog"
- `gallery(by_idx([5, 12, 30]))` — specific images by index
- `gallery(sel, methods=['Vanilla Grad','IG-zero','KL-IG² (adaptive)','Random'])` — fewer methods
''')

code(r'''
sel = pick_images(n=3)                 # all images (add animal_only=True to restrict)
gallery(sel)
''')

# ════════════════════════════════════════════════════════════════════════════
md(r'''
# 6 — Blur-masking probe: does blur kill Random's noise-difference?

**Question:** if we blur the attribution map *before* masking, do we wash out Random's
fine speckle "difference" while keeping a real method's coarse, object-level difference?
Look at the maps, not just the number. **CS is measured in pixel space** here — cosine between
the masked images directly (no encoder), so we're testing the masks, not a representation.

- `mask_soft` — current: `|a| / max|a|`
- `mask_blur` — `gaussian_blur(|a|, σ)` then normalize (collapses fine noise, keeps coarse shape)

**Win condition:** `Random` → `cs_blur ≪ cs_soft` (speckle blurred away → floors);
`KL-IG²` → `cs_blur ≈ cs_soft` (coarse object diff survives). σ is the knob: too high kills real
methods too; too low and Random stays high. *Blur fixes noise-gaming, not location washout —
a blurred wrong-location blob is still a coherent blob.*
''')

code(r'''
# ── mask modes (CS measured in PIXEL space — no encoder) ─────────────────────
import torch.nn.functional as F

def _abs_dev(attr):
    a = attr if torch.is_tensor(attr) else torch.as_tensor(attr)
    return a.float().abs().to(DEVICE)

def _gauss1d(sigma, device):
    r = max(1, int(math.ceil(3 * sigma)))
    xs = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    k = torch.exp(-(xs ** 2) / (2 * sigma * sigma)); return k / k.sum()

def gaussian_blur2d(t, sigma):                      # t: (H,W)
    k = _gauss1d(sigma, t.device); ks = k.numel(); pad = ks // 2
    x = t[None, None]
    x = F.conv2d(x, k.view(1, 1, ks, 1), padding=(pad, 0))
    x = F.conv2d(x, k.view(1, 1, 1, ks), padding=(0, pad))
    return x[0, 0]

def mask_soft(attr):
    a = _abs_dev(attr); return a / (a.max() + 1e-8)
def mask_blur(attr, sigma):
    b = gaussian_blur2d(_abs_dev(attr), sigma); return b / (b.max() + 1e-8)

def cs(x1, a1, a2, mask_fn):                        # PIXEL-space CS: cosine of masked images
    m1, m2 = mask_fn(a1), mask_fn(a2)
    v1 = (x1 * m1.unsqueeze(0)).flatten()
    v2 = (x1 * m2.unsqueeze(0)).flatten()
    return float(1.0 - F.cosine_similarity(v1[None], v2[None]).item())
print('mask modes ready (pixel-space cs, mask_soft, mask_blur)')
''')

code(r'''
# ── the probe: soft vs blur, with per-(image,method) two-row visualization ───
def blur_probe(sel, methods=ROSTER, sigma=5.0,
               show_methods=('KL-IG² (adaptive)', 'Vanilla Grad', 'Random')):
    # methods       → quantified in the summary (default: ALL)
    # show_methods  → get the detailed 2-row viz (default: 3 diagnostics; None = all; [] = none)
    cf = build_cf_for(sel)
    rec = {m: {'soft': [], 'blur': []} for m in methods}
    soft_fn = mask_soft
    blur_fn = lambda a: mask_blur(a, sigma)

    for d in sel:
        x1 = d['x'].squeeze(0).to(DEVICE); H, W = x1.shape[1], x1.shape[2]
        y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
        x_cf = cf.get(y2)
        if x_cf is not None: x_cf = (x_cf.squeeze(0) if x_cf.dim() == 4 else x_cf).to(DEVICE)
        orig = denorm(x1).permute(1, 2, 0).numpy()
        for m in methods:
            a1, a2 = attr_for(m, x1, y1, x_cf), attr_for(m, x1, y2, x_cf)
            cs_soft, cs_blur = cs(x1, a1, a2, soft_fn), cs(x1, a1, a2, blur_fn)
            rec[m]['soft'].append(cs_soft); rec[m]['blur'].append(cs_blur)
            if show_methods is not None and m not in show_methods: continue
            fig, ax = plt.subplots(2, 6, figsize=(15, 5.2), facecolor='white')
            for r, (tag, fn, csv) in enumerate([('soft', soft_fn, cs_soft),
                                                ('blur σ=%g' % sigma, blur_fn, cs_blur)]):
                m1, m2 = fn(a1).cpu().numpy(), fn(a2).cpu().numpy()
                t1 = denorm(x1 * fn(a1).unsqueeze(0)).permute(1, 2, 0).numpy()
                t2 = denorm(x1 * fn(a2).unsqueeze(0)).permute(1, 2, 0).numpy()
                dv = max(np.abs(m1 - m2).max(), 1e-9)
                panels = [(m1, 'magma', '|a1|'), (m2, 'magma', '|a2|'),
                          (m1 - m2, 'RdBu_r', 'a1−a2'), (t1, None, 'masked T1'),
                          (t2, None, 'masked T2'), (orig, None, 'orig')]
                for c, (im, cmap, lab) in enumerate(panels):
                    a = ax[r, c]
                    if cmap == 'RdBu_r': a.imshow(im, cmap=cmap, vmin=-dv, vmax=dv)
                    elif cmap:           a.imshow(im, cmap=cmap)
                    else:                a.imshow(np.clip(im, 0, 1))
                    a.axis('off')
                    if r == 0: a.set_title(lab, fontsize=9)
                ax[r, 0].set_ylabel(f'{tag}\nCS={csv:.3f}', fontsize=10, fontweight='bold',
                                    rotation=90, labelpad=10); ax[r, 0].axis('on')
                ax[r, 0].set_xticks([]); ax[r, 0].set_yticks([])
                for sp in ax[r, 0].spines.values(): sp.set_visible(False)
            plt.suptitle(f'{m}   |   {imagenet_labels[y1].split(",")[0]}  vs  '
                         f'{imagenet_labels[y2].split(",")[0]}', fontsize=12, fontweight='bold')
            plt.tight_layout(); plt.show()

    # summary
    print('\nmean CS per method (soft vs blur σ=%g):' % sigma)
    for m in methods:
        s, b = np.mean(rec[m]['soft']), np.mean(rec[m]['blur'])
        print(f'  {m:20s} soft={s:.3f}  blur={b:.3f}  Δ={b-s:+.3f}')
    fig, ax = plt.subplots(figsize=(7, 4), facecolor='white')
    xs = np.arange(len(methods)); w = 0.38
    ax.bar(xs - w/2, [np.mean(rec[m]['soft']) for m in methods], w, label='soft', color='#1E90FF')
    ax.bar(xs + w/2, [np.mean(rec[m]['blur']) for m in methods], w, label=f'blur σ={sigma:g}', color='#e41a1c')
    ax.set_xticks(xs); ax.set_xticklabels(methods, rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('mean CS_pixel'); ax.set_title('Soft vs blur masking (pixel space)', fontweight='bold')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(out('blur_probe_summary.png'), dpi=140, bbox_inches='tight'); plt.show()
    print('saved blur_probe_summary.png')
    return rec
''')

code(r'''
# ALL methods, averaged over n=50 images, blur σ=2 (fixed — no sweep).
# show_methods=[] → numbers + summary bar only (per-image figures would be thousands of plots).
sel50 = pick_images(n=50, animal_only=False, dedup=False)
rec = blur_probe(sel50, methods=ROSTER, sigma=2.0, show_methods=[])
# to eyeball maps for a few methods on one image:
#   blur_probe(pick_images(n=1), methods=['Vanilla Grad','KL-IG² (adaptive)','Random'], show_methods=None)
''')

# ════════════════════════════════════════════════════════════════════════════
md(r'''
# Family 2 — Coherence-based comparison (connected-component thresholding)

Kill noise by **spatial structure**, not frequency. Threshold each map to its strong pixels,
then keep only pixels in a connected component above a size floor: isolated specks (noise) get
dropped, coherent blobs (signal) survive — **fine or coarse**, so your method shouldn't get
buried the way it did under blur. Then compare the two cleaned region-maps (IoU by default).

The checks that decide if this works (run these, not just the ranking):
1. **Random must FLOOR** — scattered top-k pixels → no component clears `min_size`. If Random
   stays high, raise `min_size`.
2. **KL-IG² (adaptive) must NOT be buried** — fine-but-connected regions survive (the blur failure).
3. **Oracle** — GT segmentation y₁ vs y₂ → different objects → low IoU → HIGH CS; shuffling y₂'s
   location should change it. Want oracle > real methods, shuffle ≠ oracle, Random low.

Knobs: `min_size` (too small → noise survives; too big → fine real regions killed; sweep {10,20,50})
and `keep_frac` (how much counts as "strong"; 0.15–0.25 typical).
''')

code(r'''
from scipy import ndimage
import torch.nn.functional as F

def _np_abs(attr):
    a = attr.detach().cpu().numpy() if torch.is_tensor(attr) else np.asarray(attr)
    return np.abs(a).astype(np.float64)

def coherent_mask(attr, keep_frac=0.10, min_size=50):
    """|attr| → top-keep_frac binary → drop connected components smaller than min_size."""
    m = _np_abs(attr)
    binary = m >= np.quantile(m.ravel(), 1 - keep_frac)          # top keep_frac% pixels
    labels, n = ndimage.label(binary, structure=np.ones((3, 3)))  # 8-connectivity
    if n == 0: return np.zeros_like(m)
    sizes = ndimage.sum(binary, labels, range(1, n + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if s >= min_size]
    return np.isin(labels, keep).astype(float)                   # only coherent regions = 1

def cs_coherent(a1, a2, keep_frac=0.10, min_size=50, mode='iou'):
    r1 = coherent_mask(a1, keep_frac, min_size)
    r2 = coherent_mask(a2, keep_frac, min_size)
    if mode == 'iou':
        inter = (r1 * r2).sum(); union = ((r1 + r2) > 0).sum()
        if union == 0:                                           # both empty (e.g. Random) →
            return 0.0                                           # no regions = no difference = floor
        return float(1.0 - inter / union)                        # high = regions differ
    v1, v2 = r1.ravel(), r2.ravel()
    return float(1.0 - (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))

def spatial_shuffle(attr):                                       # same density, scrambled location
    m = _np_abs(attr); f = m.ravel().copy()
    np.random.default_rng(0).shuffle(f); return f.reshape(m.shape)
print('coherence metric ready: coherent_mask, cs_coherent, spatial_shuffle')
''')

code(r'''
# ── GT segmentation for the oracle (CLIPSeg, text-prompted; no attribution method) ──
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from torchvision import transforms as _T
_to_pil  = _T.ToPILImage()
_csproc  = CLIPSegProcessor.from_pretrained('CIDAS/clipseg-rd64-refined')
_csmdl   = CLIPSegForImageSegmentation.from_pretrained('CIDAS/clipseg-rd64-refined').to(DEVICE).eval()
def gt_class_mask(x1, cls, H, W):
    text = imagenet_labels[int(cls)].split(',')[0].strip()
    inp  = _csproc(text=[text], images=[_to_pil(denorm(x1))], return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        pr = _csmdl(**inp).logits.sigmoid().float()
    while pr.dim() < 4: pr = pr.unsqueeze(0)
    return F.interpolate(pr, size=(H, W), mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
print('CLIPSeg GT segmenter ready (gt_class_mask)')
''')

code(r'''
def run_coherent(sel, methods=ROSTER, keep_frac=0.10, min_size=50):
    cf = build_cf_for(sel)
    scores = {m: [] for m in methods}
    for d in tqdm(sel, desc='coherent CS'):
        x1 = d['x'].squeeze(0).to(DEVICE)
        y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
        x_cf = cf.get(y2)
        if x_cf is not None: x_cf = (x_cf.squeeze(0) if x_cf.dim() == 4 else x_cf).to(DEVICE)
        for m in methods:
            a1, a2 = attr_for(m, x1, y1, x_cf), attr_for(m, x1, y2, x_cf)
            scores[m].append(cs_coherent(a1, a2, keep_frac, min_size))
    means = {m: float(np.mean(scores[m])) for m in methods}
    order = sorted(means, key=lambda k: -means[k])
    print(f'\ncoherent CS (keep_frac={keep_frac}, min_size={min_size}) — ranked:')
    for r, m in enumerate(order, 1):
        print(f'  {r:2d}. {m:20s} {means[m]:.4f}')
    rank = lambda mm: order.index(mm) + 1
    print(f"\n[check 1] Random rank: {rank('Random')}/{len(methods)}  (want LAST)")
    print(f"[check 2] KL-IG² (adaptive) rank: {rank('KL-IG² (adaptive)')}/{len(methods)}  (want NOT buried)")
    return means, scores, order

def coherent_oracle(sel, keep_frac=0.10, min_size=50):
    oracle, shuffle = [], []
    for d in tqdm(sel, desc='coherent oracle'):
        x1 = d['x'].squeeze(0).to(DEVICE); H, W = x1.shape[1], x1.shape[2]
        y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
        seg1, seg2 = gt_class_mask(x1, y1, H, W), gt_class_mask(x1, y2, H, W)
        oracle.append( cs_coherent(seg1, seg2, keep_frac, min_size))
        shuffle.append(cs_coherent(seg1, spatial_shuffle(seg2), keep_frac, min_size))
    return float(np.mean(oracle)), float(np.mean(shuffle))
''')

code(r'''
# RUN — roster + the validity-ladder checks (n=50; CLIPSeg oracle on a 30-image subset)
# Calibrated defaults: on synthetic blobs vs noise, keep_frac=0.10/min_size=50 gives
# real-diff≈1.0, real-same≈0.03, Random=0.0 (the pseudocode's 0.20/20 lets noise percolate).
KEEP_FRAC, MIN_SIZE = 0.10, 50
sel_coh = pick_images(n=50, animal_only=False, dedup=False)
means_coh, scores_coh, order_coh = run_coherent(sel_coh, methods=ROSTER,
                                                 keep_frac=KEEP_FRAC, min_size=MIN_SIZE)

o, s = coherent_oracle(sel_coh[:30], keep_frac=KEEP_FRAC, min_size=MIN_SIZE)
best_real = max(means_coh[m] for m in ROSTER if m != 'Random')
print(f"\n[check 3] oracle={o:.3f}  shuffle={s:.3f}  best-real={best_real:.3f}  Random={means_coh['Random']:.3f}")
print('  want: oracle > best-real,  shuffle ≠ oracle,  Random low')

# bar chart
fig, ax = plt.subplots(figsize=(9, 4), facecolor='white')
ax.bar(range(len(order_coh)), [means_coh[m] for m in order_coh],
       color=[KM.COLORS.get(m, '#777777') for m in order_coh])
ax.axhline(o, color='green', ls='--', lw=1.5, label=f'GT oracle = {o:.3f}')
ax.axhline(means_coh['Random'], color='red', ls=':', lw=1.5, label=f"Random = {means_coh['Random']:.3f}")
ax.set_xticks(range(len(order_coh))); ax.set_xticklabels(order_coh, rotation=40, ha='right', fontsize=8)
ax.set_ylabel('coherent CS (1−IoU)'); ax.set_title(
    f'Coherence-based class sensitivity (keep_frac={KEEP_FRAC}, min_size={MIN_SIZE})', fontweight='bold')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig(out('coherent_cs_roster.png'), dpi=140, bbox_inches='tight'); plt.show()
print('saved coherent_cs_roster.png')
# min_size knob: if Random not last → raise; if KL-IG² buried → lower. Try {20,50,100}.
# (smaller keep_frac also helps Random floor: noise percolates less when fewer pixels pass.)
''')

# ════════════════════════════════════════════════════════════════════════════
md(r'''
## Map multiply / diff table  (cache the attributions, then derive metrics for free)

Cache the per-(image, method) attribution maps **once**; afterward any map-level metric is just
arithmetic on the cache — no attribution rerun. We report two: the **multiply** of the two
normalized maps (`mean m₁·m₂`, overlap-like) and the **diff** (`mean |m₁−m₂|`, how different the
class maps are). Maps are normalized `|a|/max` so methods with different gradient scales compare
fairly. Ranked by `diff` (most class-different on top).
''')

code(r'''
# 1) cache the attribution maps ONCE (the only cell that runs attr_for)
cf = build_cf_for(sel_coh)
ATTR_CACHE = []                                   # [{method: (a1, a2)} per image]
for d in tqdm(sel_coh, desc='cache attr'):
    x1 = d['x'].squeeze(0).to(DEVICE)
    y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
    x_cf = cf.get(y2)
    if x_cf is not None: x_cf = (x_cf.squeeze(0) if x_cf.dim() == 4 else x_cf).to(DEVICE)
    ATTR_CACHE.append({m: (attr_for(m, x1, y1, x_cf).cpu(),
                           attr_for(m, x1, y2, x_cf).cpu()) for m in ROSTER})
print(f'cached {len(ATTR_CACHE)} images × {len(ROSTER)} methods — re-derive any metric for free')
''')

code(r'''
# 2) table (method | multiply | diff | rank) + bar plot, all from the cache (no rerun)
import pandas as pd
def _nm(a): a = a.abs().float(); return a / (a.max() + 1e-8)     # |a|/max → [0,1]
mult, diff = {m: [] for m in ROSTER}, {m: [] for m in ROSTER}
for maps in ATTR_CACHE:
    for m in ROSTER:
        m1, m2 = _nm(maps[m][0]), _nm(maps[m][1])
        mult[m].append(float((m1 * m2).mean()))                  # multiply of the maps
        diff[m].append(float((m1 - m2).abs().mean()))            # diff of the maps

tbl = (pd.DataFrame([{'method': m, 'multiply': float(np.mean(mult[m])),
                      'diff': float(np.mean(diff[m]))} for m in ROSTER])
       .sort_values('diff', ascending=False).reset_index(drop=True))
tbl['rank'] = tbl.index + 1
print(tbl.round(4).to_string(index=False))
tbl.to_csv(out('map_mult_diff_table.csv'), index=False)

fig, ax = plt.subplots(figsize=(9, 4), facecolor='white'); xs = np.arange(len(tbl)); w = 0.4
ax.bar(xs - w/2, tbl['multiply'], w, label='multiply (mean m1·m2)')
ax.bar(xs + w/2, tbl['diff'],     w, label='diff (mean |m1−m2|)')
ax.set_xticks(xs); ax.set_xticklabels(tbl['method'], rotation=40, ha='right', fontsize=8)
ax.set_ylabel('score'); ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig(out('map_mult_diff.png'), dpi=140, bbox_inches='tight'); plt.show()
print('saved →', OUT, ': map_mult_diff_table.csv, map_mult_diff.png')
''')

# ════════════════════════════════════════════════════════════════════════════
md(r'''
# Attribution vs GT — per-image panel  (random image every run)

One row per method, columns: orig | GT y1 outline | A y1 heatmap | A∩GT y1 | (same for y2).
GT = green CLIPSeg contour; attribution = magma heatmap; `hit = IoU(attr region, GT region)`
under the overlap columns (green if ≥0.3). Random image each run (no repeat until pool used up).
''')

code(r'''
import matplotlib.cm as cm, random
def _ioup(a, b):
    u = ((a + b) > 0).sum(); return 0.0 if u == 0 else float((a * b).sum() / u)

def gt_attr_panel(d, methods=('Vanilla Grad', 'IG-zero', 'KLIG-Adaptive', 'KL-IG² (adaptive)'),
                  keep_frac=0.10, min_size=50, save=None):
    x1 = d['x'].squeeze(0).to(DEVICE); H, W = x1.shape[1], x1.shape[2]
    y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
    cf = build_cf_for([d]); x_cf = cf.get(y2)
    if x_cf is not None: x_cf = (x_cf.squeeze(0) if x_cf.dim() == 4 else x_cf).to(DEVICE)
    orig = denorm(x1).permute(1, 2, 0).numpy()
    G1 = coherent_mask(gt_class_mask(x1, y1, H, W), keep_frac, min_size)
    G2 = coherent_mask(gt_class_mask(x1, y2, H, W), keep_frac, min_size)
    def absn(A): a = np.abs(A).astype(float); return a / (a.max() + 1e-8)
    GREEN = '#39FF14'; cols = ['orig', 'GT y1', 'A y1', 'A∩GT y1', 'GT y2', 'A y2', 'A∩GT y2']
    fig, ax = plt.subplots(len(methods), 7, figsize=(2.0 * 7, 2.3 * len(methods)),
                           facecolor='white', squeeze=False)
    def ctr(a, G):
        if G.any(): a.contour(G, levels=[0.5], colors=[GREEN], linewidths=1.6)
    for r, m in enumerate(methods):
        A1 = attr_for(m, x1, y1, x_cf).detach().cpu().numpy()
        A2 = attr_for(m, x1, y2, x_cf).detach().cpu().numpy()
        R1, R2 = coherent_mask(A1, keep_frac, min_size), coherent_mask(A2, keep_frac, min_size)
        h1, h2 = _ioup(R1, G1), _ioup(R2, G2)
        for c in range(7): ax[r, c].axis('off')
        ax[r, 0].imshow(orig)
        ax[r, 1].imshow(orig); ctr(ax[r, 1], G1)
        ax[r, 2].imshow(absn(A1), cmap='magma', vmin=0, vmax=1)
        ax[r, 3].imshow(absn(A1), cmap='magma', vmin=0, vmax=1); ctr(ax[r, 3], G1)
        ax[r, 4].imshow(orig); ctr(ax[r, 4], G2)
        ax[r, 5].imshow(absn(A2), cmap='magma', vmin=0, vmax=1)
        ax[r, 6].imshow(absn(A2), cmap='magma', vmin=0, vmax=1); ctr(ax[r, 6], G2)
        for c, h in ((3, h1), (6, h2)):
            ax[r, c].text(0.5, -0.07, f'hit={h:.2f}', transform=ax[r, c].transAxes,
                          ha='center', va='top', fontsize=11, fontweight='bold',
                          color='#1a9850' if h >= 0.3 else '#d73027')
        ax[r, 0].axis('on'); ax[r, 0].set_xticks([]); ax[r, 0].set_yticks([])
        for sp in ax[r, 0].spines.values(): sp.set_visible(False)
        ax[r, 0].set_ylabel(m, fontsize=10, fontweight='bold', rotation=90, labelpad=8)
    for c, t in enumerate(cols): ax[0, c].set_title(t, fontsize=9)
    plt.suptitle(f'GT (green) vs attribution (magma) — '
                 f'y1={imagenet_labels[y1].split(",")[0]} | y2={imagenet_labels[y2].split(",")[0]}',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save: plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()

try: _shown_panel
except NameError: _shown_panel = set()
_rem = [x for x in cands if x['idx'] not in _shown_panel]
if not _rem: _shown_panel.clear(); _rem = list(cands)
d = random.choice(_rem); _shown_panel.add(d['idx'])
print(f"image idx {d['idx']}  ({len(_shown_panel)}/{len(cands)} shown)  "
      f"y1={imagenet_labels[int(d['high_cls'][0])].split(',')[0]} | "
      f"y2={imagenet_labels[int(d['high_cls'][1])].split(',')[0]}")
gt_attr_panel(d, save=out(f"gt_attr_panel_{d['idx']}.png"))
''')

# ════════════════════════════════════════════════════════════════════════════
md(r'''
# CS_loc — localization correctness (soft CLIPSeg GT)

`hit(A, GT) = Σ|A|·GT / Σ|A|` = fraction of attribution mass on the object (no thresholding).
`CS_loc_mult = hit_y1·hit_y2` (both maps must land on their own object — correctness, not
difference, so noise/shuffle can't game it). `CS_loc_sub = hit_y1 − hit_y2` (asymmetry).
''')

code(r'''
import pandas as pd
def hit(A, GT):
    a = np.abs(A).astype(float); return float((a * GT).sum() / (a.sum() + 1e-8))

def run_cs_loc(N=50, methods=None):
    methods = methods or [m for m in METHODS]                 # 11 methods, no Random
    sel = pick_images(n=N, animal_only=False, dedup=False); cf = build_cf_for(sel)
    sx, ss, hh = ({m: [] for m in methods} for _ in range(3))
    for d in tqdm(sel, desc='CS_loc'):
        x = d['x'].squeeze(0).to(DEVICE); H, W = x.shape[1], x.shape[2]
        y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
        g1, g2 = gt_class_mask(x, y1, H, W), gt_class_mask(x, y2, H, W)
        if g1.max() < 1e-6 or g2.max() < 1e-6: continue
        GT1, GT2 = g1 / g1.max(), g2 / g2.max()
        x_cf = cf.get(y2)
        if x_cf is None: continue
        x_cf = (x_cf.squeeze(0) if x_cf.dim() == 4 else x_cf).to(DEVICE)
        for m in methods:
            A1 = attr_for(m, x, y1, x_cf).detach().cpu().numpy()
            A2 = attr_for(m, x, y2, x_cf).detach().cpu().numpy()
            h1, h2 = hit(A1, GT1), hit(A2, GT2)
            sx[m].append(h1 * h2); ss[m].append(h1 - h2); hh[m].append((h1, h2))
    tbl = (pd.DataFrame([{'method': m,
            'hit_y1': float(np.mean([h[0] for h in hh[m]])),
            'hit_y2': float(np.mean([h[1] for h in hh[m]])),
            'CS_loc_mult': float(np.mean(sx[m])), 'CS_loc_sub': float(np.mean(ss[m]))}
           for m in methods]).sort_values('CS_loc_mult', ascending=False).reset_index(drop=True))
    tbl['rank'] = tbl.index + 1
    print(tbl.round(4).to_string(index=False)); tbl.to_csv(out('cs_loc_roster.csv'), index=False)
    fig, ax = plt.subplots(figsize=(9, 4), facecolor='white'); xs = np.arange(len(tbl)); w = 0.4
    ax.bar(xs - w/2, tbl['CS_loc_mult'], w, label='multiply (h1·h2)')
    ax.bar(xs + w/2, tbl['CS_loc_sub'],  w, label='subtract (h1−h2)')
    ax.set_xticks(xs); ax.set_xticklabels(tbl['method'], rotation=40, ha='right', fontsize=8)
    ax.axhline(0, color='k', lw=0.6); ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(out('cs_loc_roster.png'), dpi=140, bbox_inches='tight'); plt.show()
    return tbl

cs_loc_tbl = run_cs_loc(N=50)
''')

# ════════════════════════════════════════════════════════════════════════════
md(r'''
# Occlusion causal GT  (+ CS_loc on it, + same-region diagnostic)

Method-independent GT: slide an occluder; where hiding pixels drops the class logit most = the
region the *model* causally uses. `same_region` cosine(GT_y1, GT_y2) is the key diagnostic —
high ⇒ model uses the SAME region for both classes (featural, not spatial → location metrics
can't separate). Slow, so start small; GTs are cached per (image, class).
''')

code(r'''
import matplotlib.cm as _cm
try: _occ_cache
except NameError: _occ_cache = {}
def occlusion_gt(x, c, idx, patch=32, stride=16, fill=0.0, chunk=64):
    key = (idx, int(c))
    if key in _occ_cache: return _occ_cache[key]
    H, W = x.shape[1], x.shape[2]
    with torch.no_grad(): base = model(x.unsqueeze(0))[0, c].item()
    coords = [(i, j) for i in range(0, H - patch + 1, stride) for j in range(0, W - patch + 1, stride)]
    imp, cnt = np.zeros((H, W)), np.zeros((H, W))
    for k in range(0, len(coords), chunk):
        bc = coords[k:k + chunk]; xb = x.unsqueeze(0).repeat(len(bc), 1, 1, 1).clone()
        for b, (i, j) in enumerate(bc): xb[b, :, i:i+patch, j:j+patch] = fill
        with torch.no_grad(): lg = model(xb)[:, c].cpu().numpy()
        for b, (i, j) in enumerate(bc):
            imp[i:i+patch, j:j+patch] += base - lg[b]; cnt[i:i+patch, j:j+patch] += 1
    imp = np.clip(imp / (cnt + 1e-8), 0, None); imp = imp / (imp.max() + 1e-8)
    _occ_cache[key] = imp; return imp
def _cos(u, v):
    u, v = u.ravel(), v.ravel(); return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-8))

def run_occlusion(N=12, methods=None):
    methods = methods or [m for m in METHODS]
    sel = pick_images(n=N, animal_only=False, dedup=False); cf = build_cf_for(sel)
    score = {m: [] for m in methods}; same = []
    for d in tqdm(sel, desc='occlusion CS_loc'):
        x = d['x'].squeeze(0).to(DEVICE); y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
        G1, G2 = occlusion_gt(x, y1, d['idx']), occlusion_gt(x, y2, d['idx'])
        same.append(_cos(G1, G2))
        x_cf = cf.get(y2)
        if x_cf is None: continue
        x_cf = (x_cf.squeeze(0) if x_cf.dim() == 4 else x_cf).to(DEVICE)
        for m in methods:
            A1 = attr_for(m, x, y1, x_cf).detach().cpu().numpy()
            A2 = attr_for(m, x, y2, x_cf).detach().cpu().numpy()
            score[m].append(hit(A1, G1) * hit(A2, G2))
    tbl = (pd.DataFrame([{'method': m, 'CS_loc': float(np.mean(score[m]))} for m in methods])
           .sort_values('CS_loc', ascending=False).reset_index(drop=True)); tbl['rank'] = tbl.index + 1
    print(tbl.round(4).to_string(index=False)); tbl.to_csv(out('cs_loc_occlusion.csv'), index=False)
    print(f'\nmean y1-vs-y2 causal-region similarity = {np.mean(same):.3f}  '
          f'(HIGH→same region=featural; LOW→spatial CS meaningful)')
    return tbl, same

def show_occlusion_examples(n_ex=4):
    exs = random.sample(cands, min(n_ex, len(cands)))
    fig, ax = plt.subplots(n_ex, 3, figsize=(9, 3 * n_ex), facecolor='white', squeeze=False)
    def ov(a, orig, G):
        a.imshow(orig); rgba = _cm.jet(np.clip(G, 0, 1)); rgba[..., 3] = np.clip(G, 0, 1) * 0.6
        a.imshow(rgba); a.axis('off')
    for r, d in enumerate(exs):
        x = d['x'].squeeze(0).to(DEVICE); y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
        orig = denorm(x).permute(1, 2, 0).numpy()
        G1, G2 = occlusion_gt(x, y1, d['idx']), occlusion_gt(x, y2, d['idx']); sim = _cos(G1, G2)
        ax[r, 0].imshow(orig); ov(ax[r, 1], orig, G1); ov(ax[r, 2], orig, G2)
        ax[r, 0].set_title(f"img {d['idx']}", fontsize=9)
        ax[r, 1].set_title(f"causal y1: {imagenet_labels[y1].split(',')[0]}", fontsize=9)
        ax[r, 2].set_title(f"causal y2: {imagenet_labels[y2].split(',')[0]}", fontsize=9)
        ax[r, 0].set_xticks([]); ax[r, 0].set_yticks([])
        for sp in ax[r, 0].spines.values(): sp.set_visible(False)
        ax[r, 0].set_ylabel(f"sim={sim:.2f}", fontsize=11, fontweight='bold', rotation=90,
                            color='#d73027' if sim > 0.7 else '#1a9850')
    plt.suptitle('Occlusion causal regions y1 vs y2 (sim high/red = same region)',
                 fontsize=12, fontweight='bold', y=1.0)
    plt.tight_layout(); plt.savefig(out('occlusion_examples.png'), dpi=140, bbox_inches='tight'); plt.show()

occ_tbl, occ_same = run_occlusion(N=12)
show_occlusion_examples(4)
''')

# ════════════════════════════════════════════════════════════════════════════
md(r'''
# Family D — Causal class sensitivity (prediction-steering)

Use each class's attribution as a soft selector; keep only those pixels and re-predict. A
class-sensitive method's y1-map steers the model to y1 and its y2-map to y2.
`CS = ½[(p_y1[y1]−p_y2[y1]) + (p_y2[y2]−p_y1[y2])]`. Grounded in prediction → Random and the
class-blind control (same map for both classes) floor by construction.
''')

code(r'''
from scipy import ndimage as _ndi
from scipy.stats import wilcoxon as _wil
import torch.nn.functional as _F

def soft_mask(A, keep_frac=0.25, blur=2):
    a = np.abs(A).astype(float); a = a / (a.max() + 1e-8)
    thr = np.quantile(a, 1 - keep_frac)
    m = np.clip((a - thr) / (a.max() - thr + 1e-8), 0, 1)
    if blur: m = _ndi.gaussian_filter(m, sigma=blur)
    return torch.from_numpy(m).float()
@torch.no_grad()
def _class_probs(x, m):
    return _F.softmax(model((x * m.unsqueeze(0).to(x.device)).unsqueeze(0))[0], dim=0).cpu().numpy()
def cs_causal(x, A1, A2, y1, y2):
    p1, p2 = _class_probs(x, soft_mask(A1)), _class_probs(x, soft_mask(A2))
    return 0.5 * ((p1[y1] - p2[y1]) + (p2[y2] - p1[y2]))

def run_causal(N=100):
    methods = [m for m in METHODS] + ['Random']
    sel = pick_images(n=N, animal_only=False, dedup=False); print('sel:', len(sel))
    cf = build_cf_for(sel); score = {m: [] for m in methods}; blind = []
    for d in tqdm(sel, desc='causal CS'):
        x = d['x'].squeeze(0).to(DEVICE); y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
        x_cf = cf.get(y2)
        if x_cf is None: continue
        x_cf = (x_cf.squeeze(0) if x_cf.dim() == 4 else x_cf).to(DEVICE); bmap = None
        for m in methods:
            A1 = attr_for(m, x, y1, x_cf).detach().cpu().numpy()
            A2 = attr_for(m, x, y2, x_cf).detach().cpu().numpy()
            score[m].append(cs_causal(x, A1, A2, y1, y2))
            if m == 'KL-IG² (adaptive)': bmap = A1
        if bmap is not None: blind.append(cs_causal(x, bmap, bmap, y1, y2))
    tbl = (pd.DataFrame([{'method': m, 'CS_causal': float(np.mean(score[m])),
            'se': float(np.std(score[m]) / np.sqrt(max(1, len(score[m]))))} for m in methods])
           .sort_values('CS_causal', ascending=False).reset_index(drop=True)); tbl['rank'] = tbl.index + 1
    print(tbl.round(4).to_string(index=False)); tbl.to_csv(out('cs_causal_roster.csv'), index=False)
    print(f"\nRandom rank: {tbl.set_index('method')['rank']['Random']}/{len(methods)} (want LAST)")
    print(f"class-blind oracle: {np.mean(blind):.4f} (want ~0)")
    fig, ax = plt.subplots(figsize=(9, 4), facecolor='white')
    ax.bar(range(len(tbl)), tbl['CS_causal'], yerr=tbl['se'], capsize=3,
           color=[KM.COLORS.get(m, '#777') for m in tbl['method']])
    ax.set_xticks(range(len(tbl))); ax.set_xticklabels(tbl['method'], rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('CS_causal'); ax.axhline(0, color='k', lw=0.6); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(out('cs_causal_roster.png'), dpi=140, bbox_inches='tight'); plt.show()
    return tbl

causal_tbl = run_causal(N=50)
''')

# ════════════════════════════════════════════════════════════════════════════
md(r'''
# Top-1 vs Top-2 attribution overlay (per method)

One image, rows = [Model (occlusion GT)] + methods, columns = image | Top-1 map | Top-2 map,
each attribution **smoothed for display** and overlaid (inferno). Shows *where* each method looks
for y1 vs y2 — magnitude only, so it's **sign-agnostic** (the right view given KL-IG²'s sign flip:
you see it lands on the correct regions without the sign confusing the picture).
''')

code(r'''
from scipy import ndimage as _ndi2
def denorm_disp(x):
    a = x.cpu().numpy().transpose(1, 2, 0); return (a - a.min()) / (np.ptp(a) + 1e-8)

def overlay(ax, img, amap, disp_blur=3, alpha=0.55):
    ax.imshow(img)
    a = np.abs(amap).astype(float); a = _ndi2.gaussian_filter(a, disp_blur)
    a = (a - a.min()) / (np.ptp(a) + 1e-8)
    ax.imshow(a, cmap='inferno', alpha=alpha); ax.set_xticks([]); ax.set_yticks([])

@torch.no_grad()
def occ_drops(x, y1, y2, patch=32, stride=16):
    H, W = x.shape[1], x.shape[2]
    base = F.softmax(model(x.unsqueeze(0))[0], 0); b1, b2 = base[y1].item(), base[y2].item()
    D1 = np.zeros((H, W)); D2 = np.zeros((H, W)); c = np.zeros((H, W))
    for i in range(0, H-patch+1, stride):
        for j in range(0, W-patch+1, stride):
            xo = x.clone(); xo[:, i:i+patch, j:j+patch] = 0
            p = F.softmax(model(xo.unsqueeze(0))[0], 0)
            D1[i:i+patch, j:j+patch] += b1 - p[y1].item()
            D2[i:i+patch, j:j+patch] += b2 - p[y2].item(); c[i:i+patch, j:j+patch] += 1
    return D1/(c+1e-8), D2/(c+1e-8)

def overlay_panel(d, methods=('KL-IG² (adaptive)', 'KL-IG²', 'KL-IG (linear)', 'Vanilla Grad'),
                  save=None):
    x = d['x'].squeeze(0).to(DEVICE); y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
    with torch.no_grad():
        probs = F.softmax(model(x.unsqueeze(0))[0], 0)
    p1, p2 = float(probs[y1]), float(probs[y2])            # model's predicted confidence
    x_cf = build_cf_for([d]).get(y2)
    if x_cf is not None: x_cf = (x_cf.squeeze(0) if x_cf.dim() == 4 else x_cf).to(DEVICE)
    img = denorm_disp(x)
    rows = [('Model (occlusion)', None)] + [(m, m) for m in methods]
    fig, ax = plt.subplots(len(rows), 3, figsize=(3.4*3, 3.4*len(rows)), facecolor='white', squeeze=False)
    D1, D2 = occ_drops(x, y1, y2)
    for r, (label, m) in enumerate(rows):
        ax[r,0].imshow(img); ax[r,0].set_xticks([]); ax[r,0].set_yticks([])
        ax[r,0].set_ylabel(label, fontsize=10, fontweight='bold', rotation=90, labelpad=8)
        if m is None:
            overlay(ax[r,1], img, D1); overlay(ax[r,2], img, D2)
        else:
            A1 = attr_for(m, x, y1, x_cf).detach().cpu().numpy()
            A2 = attr_for(m, x, y2, x_cf).detach().cpu().numpy()
            overlay(ax[r,1], img, A1); overlay(ax[r,2], img, A2)
    ax[0,0].set_title('image', fontsize=11)
    ax[0,1].set_title(f'TOP-1: {imagenet_labels[y1].split(",")[0]}  (p={p1:.2f})', fontsize=11)
    ax[0,2].set_title(f'TOP-2: {imagenet_labels[y2].split(",")[0]}  (p={p2:.2f})', fontsize=11)
    plt.suptitle(f'Top-1 vs Top-2 attribution per method  —  model: '
                 f'{imagenet_labels[y1].split(",")[0]} {p1:.2f} / {imagenet_labels[y2].split(",")[0]} {p2:.2f}',
                 fontsize=13, fontweight='bold', y=1.0)
    plt.tight_layout()
    if save: plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()

# random image each run (re-run for a new one); fixed method roster
OV_METHODS = ['KL-IG² (adaptive)', 'KLIG-Adaptive', 'Vanilla Grad', 'IG-zero', 'Blur-IG']
d_ov = random.choice(cands)
print(f"image idx {d_ov['idx']}  y1={imagenet_labels[int(d_ov['high_cls'][0])].split(',')[0]} | "
      f"y2={imagenet_labels[int(d_ov['high_cls'][1])].split(',')[0]}")
overlay_panel(d_ov, methods=OV_METHODS, save=out('top1_vs_top2_overlay.png'))
print('saved → cs_viz_outputs/top1_vs_top2_overlay.png')
''')

nb = new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
    "accelerator": "GPU",
}
with open('cs_viz_playground.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print('wrote cs_viz_playground.ipynb with', len(cells), 'cells')
