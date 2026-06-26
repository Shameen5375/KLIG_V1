"""
Visualize the contrastive-occlusion pipeline, 5 images, one row each.
Columns: orig | drop_y1 | drop_y2 | Δ=(y1-y2, bipolar) | shared=(y1+y2) | ratio label.
Δ bipolar/structured -> SPATIAL ;  Δ flat -> FEATURAL.
Run:  .venv/Scripts/python contrastive_viz.py [n_rows]   (default 5)
"""
import sys, pickle, warnings
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

NROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
PATCH, STRIDE, CHUNK, SEED, EPS = 32, 16, 64, 0, 1e-8
OUT = Path('cs_viz_outputs'); OUT.mkdir(exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED); np.random.seed(SEED)

from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(DEVICE).eval()
labels = ResNet50_Weights.IMAGENET1K_V2.meta['categories']
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
def denorm(t): return (t.detach().cpu() * _STD + _MEAN).clamp(0, 1)

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

def pick_images(n):
    import random as _r; pool = list(CANDS); _r.Random(SEED).shuffle(pool)
    sel, used = [], set()
    for d in pool:
        c = int(d['high_cls'][0])
        if c in used: continue
        used.add(c); sel.append(d)
        if len(sel) >= n: break
    return sel

@torch.no_grad()
def contrastive_maps(x, y1, y2, patch=PATCH, stride=STRIDE, chunk=CHUNK):
    H, W = x.shape[1], x.shape[2]
    base = F.softmax(model(x.unsqueeze(0))[0], dim=-1); b1, b2 = float(base[y1]), float(base[y2])
    coords = [(i, j) for i in range(0, H - patch + 1, stride) for j in range(0, W - patch + 1, stride)]
    D1, D2, cnt = np.zeros((H, W)), np.zeros((H, W)), np.zeros((H, W))
    for k in range(0, len(coords), chunk):
        bc = coords[k:k + chunk]; xb = x.unsqueeze(0).repeat(len(bc), 1, 1, 1).clone()
        for b, (i, j) in enumerate(bc): xb[b, :, i:i+patch, j:j+patch] = 0.0
        p = F.softmax(model(xb), dim=-1)
        d1 = (b1 - p[:, y1]).cpu().numpy(); d2 = (b2 - p[:, y2]).cpu().numpy()
        for b, (i, j) in enumerate(bc):
            D1[i:i+patch, j:j+patch] += d1[b]; D2[i:i+patch, j:j+patch] += d2[b]; cnt[i:i+patch, j:j+patch] += 1
    D1 /= (cnt + EPS); D2 /= (cnt + EPS)
    delta, shared = D1 - D2, D1 + D2
    ratio = float((delta ** 2).sum() / ((shared ** 2).sum() + EPS))
    return D1, D2, delta, shared, ratio

sel = pick_images(NROWS)
cols = ['orig', 'drop y1', 'drop y2', 'Δ = y1−y2', 'shared = y1+y2']
fig, ax = plt.subplots(NROWS, 5, figsize=(2.5 * 5, 2.7 * NROWS), facecolor='white', squeeze=False)
for r, d in enumerate(sel):
    x = d['x'].squeeze(0).to(DEVICE); y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
    D1, D2, delta, shared, ratio = contrastive_maps(x, y1, y2)
    orig = denorm(x).permute(1, 2, 0).numpy()
    vpos = max(D1.max(), D2.max(), shared.max(), 1e-9)          # shared positive scale
    vd = max(abs(delta).max(), 1e-9)                            # symmetric for bipolar Δ
    panels = [(orig, None, None), (D1, 'magma', (0, vpos)), (D2, 'magma', (0, vpos)),
              (delta, 'RdBu_r', (-vd, vd)), (shared, 'magma', (0, vpos))]
    for c, (im, cmap, vr) in enumerate(panels):
        a = ax[r, c]
        if cmap is None: a.imshow(im)
        else: a.imshow(im, cmap=cmap, vmin=vr[0], vmax=vr[1])
        a.set_xticks([]); a.set_yticks([])
        if r == 0: a.set_title(cols[c], fontsize=10)
        for sp in a.spines.values(): sp.set_visible(False)
    reg = 'SPATIAL' if ratio > 0.63 else 'FEATURAL'            # 0.63 = random-control threshold (n=100)
    ax[r, 0].set_ylabel(f"{labels[y1].split(',')[0][:12]} / {labels[y2].split(',')[0][:12]}\n"
                        f"ratio={ratio:.2f}  {reg}", fontsize=9, fontweight='bold',
                        color='#1a5fb4' if reg == 'SPATIAL' else '#a51d2d')
    ax[r, 0].axis('on')
fig.suptitle('Contrastive-occlusion regime, step by step  '
             '(Δ bipolar/structured → SPATIAL;  Δ flat → FEATURAL)', fontsize=13, fontweight='bold', y=1.005)
plt.tight_layout(); plt.savefig(OUT / 'contrastive_regime_steps.png', dpi=150, bbox_inches='tight'); plt.close()
print('saved', OUT / 'contrastive_regime_steps.png')
for r, d in enumerate(sel):
    print(f"  row {r+1}: idx={d['idx']}  {labels[d['high_cls'][0]].split(',')[0]} / {labels[d['high_cls'][1]].split(',')[0]}")
