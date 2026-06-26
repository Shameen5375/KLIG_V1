"""
Spatial vs Featural — contrastive occlusion.
Per patch: drop_y1 - drop_y2  (the shared object cancels → bias-free, fixes RISE's flaw).
  Structured Δ-map → SPATIAL ;  flat Δ-map → FEATURAL.

Two regime scores per pair:
  raw   = mean(Δ²)                 (spec; but conflates class-presence magnitude with structure)
  ratio = energy(Δ)/energy(Δshared)  (presence-invariant: fraction of causal signal that is differential)
Validity control: same image, two RANDOM unrelated classes. Confusable Δ-maps must differ from
random-pair Δ-maps; if they don't, the test can't separate (like RISE) → stop.

Run:  .venv/Scripts/python contrastive_regime.py [n_imgs]   (default 100)
"""
import sys, pickle, warnings
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
warnings.filterwarnings('ignore')

N_IMGS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
PATCH, STRIDE, CHUNK, SEED, EPS = 32, 16, 64, 0, 1e-8
OUT = Path('cs_viz_outputs'); OUT.mkdir(exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED); np.random.seed(SEED)
print(f'[setup] device={DEVICE}  n_imgs={N_IMGS}  patch={PATCH} stride={STRIDE}')

from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(DEVICE).eval()

srcs = ['cs_viz_cache/cands.pkl', 'klig2_dist_cache/klig2_dist_multiprob.pkl',
        'klig2_val_cache/klig2_dist_multiprob.pkl', 'cs_gate_cache/pool.pkl']
CANDS, seen = [], set()
for s in srcs:
    if not Path(s).exists(): continue
    for d in pickle.load(open(s, 'rb')):
        if len(d.get('high_cls', [])) < 2: continue
        x = d['x']; x = x.squeeze(0) if x.dim() == 4 else x
        fp = round(float(x.float().sum()), 1)
        if fp in seen: continue
        seen.add(fp)
        CANDS.append({'idx': len(CANDS), 'x': x.cpu(), 'high_cls': [int(c) for c in d['high_cls'][:2]]})
print(f'[pool] {len(CANDS)} unique images')

def pick_images(n, dedup=True, seed=SEED):
    import random as _r; pool = list(CANDS); _r.Random(seed).shuffle(pool)
    if not dedup: return pool[:n]
    sel, used = [], set()
    for d in pool:
        c = int(d['high_cls'][0])
        if c in used: continue
        used.add(c); sel.append(d)
        if len(sel) >= n: break
    return sel

@torch.no_grad()
def contrastive_occlusion(x, y1, y2, patch=PATCH, stride=STRIDE, chunk=CHUNK):
    H, W = x.shape[1], x.shape[2]
    base = F.softmax(model(x.unsqueeze(0))[0], dim=-1)
    b1, b2 = float(base[y1]), float(base[y2])
    coords = [(i, j) for i in range(0, H - patch + 1, stride) for j in range(0, W - patch + 1, stride)]
    dl, sh, cnt = np.zeros((H, W)), np.zeros((H, W)), np.zeros((H, W))
    for k in range(0, len(coords), chunk):
        bc = coords[k:k + chunk]; xb = x.unsqueeze(0).repeat(len(bc), 1, 1, 1).clone()
        for b, (i, j) in enumerate(bc): xb[b, :, i:i+patch, j:j+patch] = 0.0
        p = F.softmax(model(xb), dim=-1)
        d1 = (b1 - p[:, y1]).cpu().numpy(); d2 = (b2 - p[:, y2]).cpu().numpy()
        for b, (i, j) in enumerate(bc):
            dl[i:i+patch, j:j+patch] += (d1[b] - d2[b])      # >0 favors y1, <0 favors y2
            sh[i:i+patch, j:j+patch] += (d1[b] + d2[b])      # shared causal importance
            cnt[i:i+patch, j:j+patch] += 1
    return dl / (cnt + EPS), sh / (cnt + EPS)

def regime_raw(dl):      return float((dl ** 2).mean())
def regime_ratio(dl, sh): return float((dl ** 2).sum() / ((sh ** 2).sum() + EPS))

_rng = np.random.default_rng(SEED)
from tqdm import tqdm
rows = []
for d in tqdm(pick_images(N_IMGS, dedup=True), desc='contrastive occlusion'):
    x = d['x'].squeeze(0).to(DEVICE); y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
    dlc, shc = contrastive_occlusion(x, y1, y2)
    yr1, yr2 = _rng.choice([c for c in range(1000) if c not in (y1, y2)], size=2, replace=False)
    dlr, shr = contrastive_occlusion(x, int(yr1), int(yr2))
    rows.append({'idx': d['idx'],
                 'raw_conf': regime_raw(dlc), 'raw_rand': regime_raw(dlr),
                 'ratio_conf': regime_ratio(dlc, shc), 'ratio_rand': regime_ratio(dlr, shr)})
df = pd.DataFrame(rows); df.to_csv(OUT / 'contrastive_regime.csv', index=False)

def report(tag, conf, rand):
    conf, rand = np.array(conf), np.array(rand)
    thr = rand.mean() + rand.std()
    spatial = int((conf > thr).sum()); featural = int((conf <= thr).sum())
    try: _, p = wilcoxon(conf, rand)
    except Exception: p = float('nan')
    print(f'\n=== {tag} ===')
    print(f'  confusable: mean={conf.mean():.4g}  median={np.median(conf):.4g}')
    print(f'  random    : mean={rand.mean():.4g}  median={np.median(rand):.4g}')
    print(f'  threshold (rand mean+std) = {thr:.4g}')
    print(f'  SPATIAL (above control): {spatial}/{len(conf)}   FEATURAL (at/below): {featural}/{len(conf)}')
    print(f'  Wilcoxon confusable vs random: p={p:.2e}  → '
          + ('SEPARATES (confusable Δ-maps differ from random) ✓' if p < 0.05
             else 'CANNOT separate (like RISE) — stop'))
    return spatial, featural, p

print('\n' + '#' * 60)
report('RAW score  mean(Δ²)  [conflates presence]', df['raw_conf'], df['raw_rand'])
report('RATIO score  energy(Δ)/energy(shared)  [presence-invariant]', df['ratio_conf'], df['ratio_rand'])

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), facecolor='white')
for a, (cc, rr, t) in zip(ax, [(df['raw_conf'], df['raw_rand'], 'raw  mean(Δ²)'),
                               (df['ratio_conf'], df['ratio_rand'], 'ratio  energy(Δ)/energy(shared)')]):
    lo, hi = float(min(cc.min(), rr.min())), float(max(cc.max(), rr.max()))
    bins = np.linspace(lo, hi + 1e-9, 25)
    a.hist(cc, bins=bins, alpha=0.6, color='#4477aa', label='confusable y1/y2')
    a.hist(rr, bins=bins, alpha=0.6, color='#cc6677', label='random class pair')
    a.set_title(t); a.set_xlabel('regime score'); a.legend()
fig.suptitle(f'Contrastive-occlusion regime (n={N_IMGS}): high=spatial, low=featural', fontweight='bold')
plt.tight_layout(); plt.savefig(OUT / 'contrastive_regime.png', dpi=150, bbox_inches='tight'); plt.close()
print(f'\nsaved: {OUT}/contrastive_regime.png, contrastive_regime.csv')
