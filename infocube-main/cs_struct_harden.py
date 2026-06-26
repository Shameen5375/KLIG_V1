"""
Harden CS_struct — class-sensitivity metric validation.

CS_struct(A_y1, A_y2, sigma) = energy(gaussian_blur(D, sigma)) / energy(D),  D = (A_y1-A_y2)/max|A_y1-A_y2|
  i.e. fraction of the difference-map energy that survives a low-pass blur = structural coherence
  of the y1-vs-y2 difference.  (Implemented per spec; NOT redesigned.)

Reuses the cs_viz machinery: klig_methods.attr_map, offline pool merge, CLIPSeg GT.
Run:  .venv/Scripts/python cs_struct_harden.py [N]      (N = real-run size, default 100)
Outputs tables to stdout + figures/CSVs to cs_viz_outputs/.
"""
import os, sys, math, pickle, warnings
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import wilcoxon, spearmanr
warnings.filterwarnings('ignore')

N      = int(sys.argv[1]) if len(sys.argv) > 1 else 100      # real-run size
SEED   = 0
EPS    = 1e-8
OUT    = Path('cs_viz_outputs'); OUT.mkdir(exist_ok=True)
def out(name): return str(OUT / name)
np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[setup] device={DEVICE}  N={N}')

# ── model + attribution dispatch (same as the notebook) ──────────────────────
import klig_methods as KM
from klig_methods import attr_map, METHODS, make_phi
from torchvision.models import resnet50, ResNet50_Weights
weights = ResNet50_Weights.IMAGENET1K_V2
model   = resnet50(weights=weights).to(DEVICE).eval()
preprocess      = weights.transforms()
imagenet_labels = weights.meta['categories']
phi = make_phi(model)
KM.N_STEPS, KM.N_SAMPLES = 25, 3
KM.IG_STEPS = 25; KM.SG_SAMPLES = 25; KM.EG_SAMPLES = 25
KM.T_DESC, KM.N_MC_DESC, KM.N_MC_GRAD = 25, 8, 3
ROSTER = list(METHODS) + ['Random']
KLIG2A = 'KL-IG² (adaptive)'
_rng = np.random.default_rng(SEED)

def attr_for(method, x1, cls, x_cf):
    H, W = x1.shape[-2], x1.shape[-1]
    if method == 'Random':
        return torch.from_numpy(_rng.standard_normal((H, W))).float()
    return attr_map(method, model, x1, int(cls), x_cf=x_cf, phi=phi)

def npmap(A):
    return (A.detach().cpu().numpy() if torch.is_tensor(A) else np.asarray(A)).astype(float)

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
def denorm(t): return (t.detach().cpu() * _STD + _MEAN).clamp(0, 1)

# ── offline pool (merge local caches, deduped) ───────────────────────────────
def load_pool():
    srcs = ['cs_viz_cache/cands.pkl', 'klig2_dist_cache/klig2_dist_multiprob.pkl',
            'klig2_val_cache/klig2_dist_multiprob.pkl', 'cs_gate_cache/pool.pkl']
    merged, seen = [], set()
    for s in srcs:
        if not Path(s).exists(): continue
        for d in pickle.load(open(s, 'rb')):
            if len(d.get('high_cls', [])) < 2: continue
            x = d['x']; x = x.squeeze(0) if x.dim() == 4 else x
            fp = round(float(x.float().sum()), 1)
            if fp in seen: continue
            seen.add(fp)
            merged.append({'idx': len(merged), 'x': x.cpu(),
                           'high_cls': [int(c) for c in d['high_cls'][:2]]})
    print(f'[pool] {len(merged)} unique images merged offline')
    return merged

CANDS = load_pool()
def pick_images(n, dedup=True, seed=SEED):
    pool = list(CANDS); import random as _r; _r.Random(seed).shuffle(pool)
    if not dedup: return pool[:n]
    sel, used = [], set()
    for d in pool:
        c = int(d['high_cls'][0])
        if c in used: continue
        used.add(c); sel.append(d)
        if len(sel) >= n: break
    return sel

def build_cf_for(sel):                                # cache-only (offline)
    need = {int(d['high_cls'][1]) for d in sel}
    cf = {}
    for d in CANDS:
        for c in (int(d['high_cls'][0]), int(d['high_cls'][1])):
            if c in need and c not in cf:
                cf[c] = d['x'].to(DEVICE)
    if CANDS:
        fb = CANDS[0]['x'].to(DEVICE)
        for c in need - set(cf): cf[c] = fb
    return cf

# ── CLIPSeg GT (for Step-1 correctness control) ──────────────────────────────
try:
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    from torchvision import transforms as _T
    _to_pil = _T.ToPILImage()
    _csproc = CLIPSegProcessor.from_pretrained('CIDAS/clipseg-rd64-refined')
    _csmdl  = CLIPSegForImageSegmentation.from_pretrained('CIDAS/clipseg-rd64-refined').to(DEVICE).eval()
    def gt_class_mask(x1, cls, H, W):
        inp = _csproc(text=[imagenet_labels[int(cls)].split(',')[0].strip()],
                      images=[_to_pil(denorm(x1))], return_tensors='pt').to(DEVICE)
        with torch.no_grad(): pr = _csmdl(**inp).logits.sigmoid().float()
        while pr.dim() < 4: pr = pr.unsqueeze(0)
        return F.interpolate(pr, size=(H, W), mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
    HAVE_GT = True
except Exception as e:
    print(f'[warn] CLIPSeg unavailable ({e}); Step-1 will use synthetic-blob proxy')
    HAVE_GT = False

# ── the metric (exactly as specified) ────────────────────────────────────────
def _energy(z): return float((z.astype(float) ** 2).sum())
def cs_struct(A_y1, A_y2, sigma=4):
    D = npmap(A_y1) - npmap(A_y2)
    D = D / (np.abs(D).max() + EPS)
    coherent = gaussian_filter(D, sigma)
    return _energy(coherent) / (_energy(D) + EPS)

def mean_se(v):
    v = np.asarray(v, float); return float(v.mean()), float(v.std() / np.sqrt(max(1, len(v))))

# ═════════════════════════════════════════════════════════════════════════════
# Precompute attribution maps ONCE per (image, method) — reused by Steps 2,3,4
# ═════════════════════════════════════════════════════════════════════════════
def precompute_maps(n):
    sel = pick_images(n, dedup=True)
    cf = build_cf_for(sel)
    recs = []
    from tqdm import tqdm
    for d in tqdm(sel, desc='attr maps'):
        x = d['x'].squeeze(0).to(DEVICE)
        y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
        x_cf = cf.get(y2); x_cf = (x_cf.squeeze(0) if x_cf.dim() == 4 else x_cf).to(DEVICE)
        mp = {m: (npmap(attr_for(m, x, y1, x_cf)), npmap(attr_for(m, x, y2, x_cf))) for m in ROSTER}
        recs.append({'idx': d['idx'], 'y1': y1, 'y2': y2, 'maps': mp})
    return sel, recs

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — correctness control (synthetic difference scenarios)
# ═════════════════════════════════════════════════════════════════════════════
def _blob(center, H, W, rad=22):
    yy, xx = np.mgrid[0:H, 0:W]
    return np.exp(-(((xx - center[1]) ** 2 + (yy - center[0]) ** 2) / (2.0 * rad ** 2))).astype(float)
def _centroid(mask):
    m = mask > 0.5 * (mask.max() + 1e-9); ys, xs = np.nonzero(m)
    if len(ys) == 0: return (mask.shape[0] // 2, mask.shape[1] // 2)
    return (int(ys.mean()), int(xs.mean()))

def step1_correctness(sel, n=50):
    rng = np.random.default_rng(SEED)
    H = W = 224
    cc, cw, noi = [], [], []
    note = 'real CLIPSeg object regions' if HAVE_GT else 'SYNTHETIC separated blobs (CLIPSeg unavailable)'
    from tqdm import tqdm
    for d in tqdm(sel[:n], desc='step1'):
        x = d['x'].squeeze(0).to(DEVICE); y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
        if HAVE_GT:
            c1 = _centroid(gt_class_mask(x, y1, H, W)); c2 = _centroid(gt_class_mask(x, y2, H, W))
        else:
            c1 = (rng.integers(40, 184), rng.integers(40, 184))
            c2 = (rng.integers(40, 184), rng.integers(40, 184))
        crand = (int(rng.integers(20, 204)), int(rng.integers(20, 204)))
        A1 = _blob(c1, H, W); A2c = _blob(c2, H, W); A2w = _blob(crand, H, W)
        cc.append(cs_struct(A1, A2c))                       # coherent-correct
        cw.append(cs_struct(A1, A2w))                       # coherent-wrong (random location)
        noi.append(cs_struct(rng.standard_normal((H, W)), rng.standard_normal((H, W))))  # noise
    rows = []
    for name, v in [('coherent-correct', cc), ('coherent-wrong', cw), ('incoherent (noise)', noi)]:
        mu, se = mean_se(v); rows.append({'scenario': name, 'mean_CS_struct': mu, 'se': se})
    tbl = pd.DataFrame(rows); tbl.to_csv(out('cs_struct_step1_correctness.csv'), index=False)
    mc, mw, mn = tbl['mean_CS_struct'].values
    if mc > mw * 1.10 and mw > mn: verdict = 'TRACKS CORRECTNESS (correct > wrong > noise)'
    elif abs(mc - mw) <= 0.10 * max(mc, 1e-9) and mc > mn: verdict = 'TRACKS COHERENCE ONLY (correct ≈ wrong > noise) → "structured distinctness", not correctness'
    else: verdict = 'inconclusive / unexpected ordering'
    print('\n=== STEP 1 — correctness control (' + note + ') ===')
    print(tbl.round(4).to_string(index=False)); print('VERDICT:', verdict)
    return tbl, verdict, note

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — sigma robustness
# ═════════════════════════════════════════════════════════════════════════════
def step2_sigma(recs, sigmas=(2, 4, 6, 8)):
    per = {s: {m: [] for m in ROSTER} for s in sigmas}
    for r in recs:
        for m in ROSTER:
            A1, A2 = r['maps'][m]
            for s in sigmas: per[s][m].append(cs_struct(A1, A2, sigma=s))
    means = {s: {m: float(np.mean(per[s][m])) for m in ROSTER} for s in sigmas}
    order = {s: sorted(ROSTER, key=lambda m: -means[s][m]) for s in sigmas}
    print('\n=== STEP 2 — sigma robustness ===')
    for s in sigmas:
        print(f'  σ={s}: ' + ' > '.join(f'{m}({means[s][m]:.3f})' for m in order[s][:4]) + ' ...')
    rankmaps = {s: {m: order[s].index(m) for m in ROSTER} for s in sigmas}
    sl = list(sigmas); rho = []
    for i in range(len(sl) - 1):
        rr = spearmanr([rankmaps[sl[i]][m] for m in ROSTER],
                       [rankmaps[sl[i+1]][m] for m in ROSTER]).correlation
        rho.append((sl[i], sl[i+1], rr))
        print(f'  rank Spearman σ{sl[i]}↔σ{sl[i+1]} = {rr:+.3f}')
    top_stable = all(order[s][0] == KLIG2A for s in sigmas)
    rand_stable = all(order[s][-1] == 'Random' for s in sigmas)
    rank_robust = top_stable and all(r[2] > 0.8 for r in rho)        # ranking stability vs σ
    rand_rank4 = order[4].index('Random') + 1
    print(f'  KLIG2A top across all σ: {top_stable} | rank-Spearman all > 0.8: {all(r[2] > 0.8 for r in rho)}')
    print(f'  Random rank @σ=4: {rand_rank4}/{len(ROSTER)} (bottom across all σ: {rand_stable})')
    print('VERDICT(rank-robustness):', 'σ-ROBUST — ranking stable, KLIG2A top at every σ' if rank_robust
          else 'σ-SENSITIVE — ranking shifts with σ')
    print('NOTE: Random is NOT the floor; class-insensitive methods (near-identical y1/y2 maps) score below it.'
          ' The true floor is the class-blind oracle (=0, Step 4).' if not rand_stable else
          'Random floors at the bottom across all σ.')
    robust = rank_robust                                              # robustness = rank stability
    pd.DataFrame([{'sigma': s, **means[s]} for s in sigmas]).to_csv(out('cs_struct_step2_sigma.csv'), index=False)
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    for m in ROSTER:
        ax.plot(list(sigmas), [means[s][m] for s in sigmas], 'o-',
                color=KM.COLORS.get(m, '#777'), lw=2 if m in (KLIG2A, 'Random') else 1,
                alpha=1 if m in (KLIG2A, 'Random') else 0.5, label=m)
    ax.set_xlabel('blur σ'); ax.set_ylabel('CS_struct'); ax.set_title('CS_struct vs σ (robustness)')
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out('cs_struct_step2_sigma.png'), dpi=140, bbox_inches='tight'); plt.close()
    return means, per, robust

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — significance & consistency  (σ=4)
# ═════════════════════════════════════════════════════════════════════════════
def step3_significance(per_sigma4):
    per = per_sigma4
    means = {m: np.mean(per[m]) for m in ROSTER}
    order = sorted(ROSTER, key=lambda m: -means[m])
    nonkl = next((m for m in order if not m.startswith('KL')), None)
    comps = [m for m in ['KL-IG²', 'KL-IG (linear)', nonkl] if m and m != KLIG2A]
    a = np.array(per[KLIG2A])
    rows = []
    for m in ROSTER:
        b = np.array(per[m]); mu, se = mean_se(b)
        win = float(np.mean(a > b)) if m != KLIG2A else np.nan
        try: p = float(wilcoxon(a, b).pvalue) if m in comps else np.nan
        except Exception: p = np.nan
        rows.append({'method': m, 'mean': mu, 'se': se,
                     'winrate_vs_KLIG2A': win, 'wilcoxon_p': p})
    tbl = pd.DataFrame(rows).sort_values('mean', ascending=False).reset_index(drop=True)
    tbl.to_csv(out('cs_struct_step3_significance.csv'), index=False)
    print('\n=== STEP 3 — significance & consistency (σ=4) ===')
    print(tbl.round(4).to_string(index=False))
    nearest = comps[0] if comps else None
    if nearest:
        p0 = tbl.set_index('method').loc[nearest, 'wilcoxon_p']
        w0 = tbl.set_index('method').loc[nearest, 'winrate_vs_KLIG2A']
        ok = (p0 < 0.05) and (w0 is not np.nan)
        print(f'VERDICT: vs nearest ({nearest}) p={p0:.4g}, win-rate={w0:.2f} → '
              + ('SOLID win' if (p0 < 0.05 and w0 > 0.65) else 'NOT clearly solid'))
    return tbl

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — full validity ladder  (σ=4)
# ═════════════════════════════════════════════════════════════════════════════
def step4_ladder(recs, per_sigma4, step1_tbl):
    blind = [cs_struct(r['maps'][KLIG2A][0], r['maps'][KLIG2A][0]) for r in recs]   # same map both
    rand  = per_sigma4['Random']
    real_means = {m: np.mean(per_sigma4[m]) for m in ROSTER if m != 'Random'}
    cw = float(step1_tbl.set_index('scenario').loc['coherent-wrong', 'mean_CS_struct'])
    cc = float(step1_tbl.set_index('scenario').loc['coherent-correct', 'mean_CS_struct'])
    rows = [
        ('class-blind oracle', float(np.mean(blind)),  '≈ 0'),
        ('Random (noise)',     float(np.mean(rand)),   'floor'),
        ('coherent-wrong',     cw,                     'low'),
        ('real methods (min)', float(min(real_means.values())), 'mid'),
        ('real methods (max)', float(max(real_means.values())), 'mid'),
        ('coherent-correct',   cc,                     'high'),
    ]
    tbl = pd.DataFrame([{'rung': r, 'observed': v, 'expected': e} for r, v, e in rows])
    tbl.to_csv(out('cs_struct_step4_ladder.csv'), index=False)
    seq = [np.mean(blind), np.mean(rand), cw, max(real_means.values()), cc]
    monotone = all(seq[i] <= seq[i+1] + 1e-6 for i in range(len(seq) - 1))
    print('\n=== STEP 4 — validity ladder (σ=4) ===')
    print(tbl.round(4).to_string(index=False))
    print('VERDICT:', 'LADDER HOLDS (monotone)' if monotone else 'LADDER BROKEN (non-monotone)')
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='white')
    ax.bar(tbl['rung'], tbl['observed'], color='#4477aa')
    ax.set_ylabel('CS_struct'); ax.set_title('Validity ladder'); plt.xticks(rotation=30, ha='right', fontsize=8)
    plt.tight_layout(); plt.savefig(out('cs_struct_step4_ladder.png'), dpi=140, bbox_inches='tight'); plt.close()
    return tbl, monotone

# ═════════════════════════════════════════════════════════════════════════════
def main():
    sel, recs = precompute_maps(N)
    s1, v1, note1 = step1_correctness(sel, n=min(50, N))
    means2, per2, robust = step2_sigma(recs)
    s3 = step3_significance(per2[4])
    s4, monotone = step4_ladder(recs, per2[4], s1)
    print('\n' + '=' * 64 + '\nSUMMARY\n' + '=' * 64)
    print(f'  Step 1 correctness : {v1}')
    print(f'                       (control = {note1})')
    print(f'  Step 2 σ-robust    : {"YES" if robust else "NO (σ-sensitive)"}')
    nearest = next((m for m in s3["method"] if m not in (KLIG2A,) and m.startswith("KL")), None)
    print(f'  Step 3 significance : see table (KLIG2A vs nearest competitors)')
    print(f'  Step 4 ladder       : {"HOLDS" if monotone else "BROKEN"}')
    print(f'\n  figures/CSVs → {OUT}/')

if __name__ == '__main__':
    main()
