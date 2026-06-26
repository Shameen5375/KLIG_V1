"""
Contrastive-alignment correctness metric.
GT  = drop_y1 - drop_y2  (contrastive occlusion, at the patch grid) = model's real y1-vs-y2 regions.
M   = A_y1 - A_y2        (method's class-difference), average-pooled to the SAME patch grid.
alignment = Pearson(M_grid, GT_grid).  High = method's distinction lands on the model's real regions.

Ladder (must hold): GT-as-method ≈ 1 ; Random ≈ 0 ; class-blind ≈ 0 ; Shuffle(GT) ≈ 0 ; real methods between.
Avoids: noise-gaming (both sides contrastive → object cancels, noise → random dir → 0),
        coarse-bias (downsample method to patch grid), CLIPSeg/y2-mask dependence (GT is causal, always exists).
Run:  .venv/Scripts/python contrastive_alignment.py [N]   (default 100)
"""
import sys, pickle, warnings
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
warnings.filterwarnings('ignore')

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
PATCH, STRIDE, CHUNK, SEED, EPS = 32, 16, 64, 0, 1e-8
OUT = Path('cs_viz_outputs'); OUT.mkdir(exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED); np.random.seed(SEED)
print(f'[setup] device={DEVICE}  N={N}')

import klig_methods as KM
from klig_methods import attr_map, METHODS, make_phi
from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(DEVICE).eval()
imagenet_labels = ResNet50_Weights.IMAGENET1K_V2.meta['categories']
phi = make_phi(model)
KM.N_STEPS, KM.N_SAMPLES = 25, 3; KM.IG_STEPS = 25; KM.SG_SAMPLES = 25; KM.EG_SAMPLES = 25
KM.T_DESC, KM.N_MC_DESC, KM.N_MC_GRAD = 25, 8, 3
ROSTER = list(METHODS) + ['Random']; KLIG2A = 'KL-IG² (adaptive)'
_rng = np.random.default_rng(SEED)
def attr_for(m, x1, cls, x_cf):
    H, W = x1.shape[-2], x1.shape[-1]
    if m == 'Random': return torch.from_numpy(_rng.standard_normal((H, W))).float()
    return attr_map(m, model, x1, int(cls), x_cf=x_cf, phi=phi)
def npm(A): return (A.detach().cpu().numpy() if torch.is_tensor(A) else np.asarray(A)).astype(float)

# offline pool + cf
srcs = ['cs_viz_cache/cands.pkl','klig2_dist_cache/klig2_dist_multiprob.pkl',
        'klig2_val_cache/klig2_dist_multiprob.pkl','cs_gate_cache/pool.pkl']
CANDS, seen = [], set()
for s in srcs:
    if not Path(s).exists(): continue
    for d in pickle.load(open(s,'rb')):
        if len(d.get('high_cls',[]))<2: continue
        x=d['x']; x=x.squeeze(0) if x.dim()==4 else x
        fp=round(float(x.float().sum()),1)
        if fp in seen: continue
        seen.add(fp); CANDS.append({'idx':len(CANDS),'x':x.cpu(),'high_cls':[int(c) for c in d['high_cls'][:2]]})
def pick(n):
    import random as _r; pool=list(CANDS); _r.Random(SEED).shuffle(pool)
    sel,used=[],set()
    for d in pool:
        c=int(d['high_cls'][0])
        if c in used: continue
        used.add(c); sel.append(d)
        if len(sel)>=n: break
    return sel
def cf_for(sel):
    need={int(d['high_cls'][1]) for d in sel}; cf={}
    for d in CANDS:
        for c in (int(d['high_cls'][0]),int(d['high_cls'][1])):
            if c in need and c not in cf: cf[c]=d['x'].to(DEVICE)
    fb=CANDS[0]['x'].to(DEVICE)
    for c in need-set(cf): cf[c]=fb
    return cf

COORDS = [(i, j) for i in range(0, 224-PATCH+1, STRIDE) for j in range(0, 224-PATCH+1, STRIDE)]
@torch.no_grad()
def gt_grid(x, y1, y2):                              # GT contrastive at patch resolution
    base = F.softmax(model(x.unsqueeze(0))[0], -1); b1, b2 = float(base[y1]), float(base[y2])
    g = np.zeros(len(COORDS))
    for k in range(0, len(COORDS), CHUNK):
        bc = COORDS[k:k+CHUNK]; xb = x.unsqueeze(0).repeat(len(bc),1,1,1).clone()
        for b,(i,j) in enumerate(bc): xb[b,:,i:i+PATCH,j:j+PATCH]=0
        p = F.softmax(model(xb), -1)
        d1 = (b1-p[:,y1]).cpu().numpy(); d2 = (b2-p[:,y2]).cpu().numpy()
        for b in range(len(bc)): g[k+b] = d1[b]-d2[b]
    return g
def to_grid(M):                                     # avg-pool method map to the same patch grid
    return np.array([M[i:i+PATCH, j:j+PATCH].mean() for (i,j) in COORDS])
def pear(a, b):
    a = a-a.mean(); b = b-b.mean(); d = np.linalg.norm(a)*np.linalg.norm(b)
    return 0.0 if d < 1e-12 else float(a@b/d)

sel = pick(N); cf = cf_for(sel)
ROWS = ROSTER + ['GT-oracle', 'class-blind', 'Shuffle']
align = {m: [] for m in ROWS}
from tqdm import tqdm
for d in tqdm(sel, desc='alignment'):
    x = d['x'].squeeze(0).to(DEVICE); y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
    x_cf = cf.get(y2); x_cf = (x_cf.squeeze(0) if x_cf.dim()==4 else x_cf).to(DEVICE)
    G = gt_grid(x, y1, y2)
    for m in ROSTER:
        M = npm(attr_for(m, x, y1, x_cf)) - npm(attr_for(m, x, y2, x_cf))
        align[m].append(pear(to_grid(M), G))
    align['GT-oracle'].append(pear(G, G))                                  # = 1
    align['class-blind'].append(pear(np.zeros_like(G), G))                 # M=0 → 0
    align['Shuffle'].append(pear(_rng.permutation(G), G))                  # location-shuffled → ~0

def mse(v): v=np.array(v); return float(v.mean()), float(v.std()/np.sqrt(max(1,len(v))))
tbl = pd.DataFrame([{'row': m, 'alignment': mse(align[m])[0], 'se': mse(align[m])[1]} for m in ROWS]
                   ).sort_values('alignment', ascending=False).reset_index(drop=True)
tbl['rank'] = tbl.index + 1
tbl.to_csv(OUT/'contrastive_alignment.csv', index=False)
print('\n' + tbl.round(4).to_string(index=False))

# ladder check + significance
top_real = next(m for m in tbl['row'] if m in ROSTER and m != 'Random')
a = np.array(align[KLIG2A])
for comp in ['KL-IG²', 'KL-IG (linear)']:
    if comp in align:
        try: p = wilcoxon(a, np.array(align[comp])).pvalue
        except Exception: p = float('nan')
        print(f'  Wilcoxon {KLIG2A} vs {comp}: p={p:.2e}')
M = tbl.set_index('row')['alignment']
print(f"\nLADDER: GT-oracle={M['GT-oracle']:.2f} (want ~1) | Random={M['Random']:.3f}, "
      f"class-blind={M['class-blind']:.3f}, Shuffle={M['Shuffle']:.3f} (want ~0) | top real={top_real} ({M[top_real]:.3f})")
ladder_ok = M['GT-oracle'] > 0.95 and abs(M['Random']) < 0.1 and abs(M['Shuffle']) < 0.1 and M[top_real] > 0.1
print('VERDICT:', 'LADDER HOLDS — alignment is a valid correctness metric' if ladder_ok
      else 'LADDER ISSUE — inspect (see table)')

fig, ax = plt.subplots(figsize=(10, 4.5), facecolor='white')
colors = ['#FFD700' if m == KLIG2A else '#888' if m in ('GT-oracle','Random','class-blind','Shuffle')
          else KM.COLORS.get(m, '#4477aa') for m in tbl['row']]
ax.bar(range(len(tbl)), tbl['alignment'], yerr=tbl['se'], capsize=3, color=colors)
ax.axhline(0, color='k', lw=0.6)
ax.set_xticks(range(len(tbl))); ax.set_xticklabels(tbl['row'], rotation=40, ha='right', fontsize=8)
ax.set_ylabel('contrastive alignment (Pearson with model GT)')
ax.set_title(f'Contrastive-alignment correctness (N={N})\ngold=KL-IG²-adaptive; gray=controls/oracle',
             fontweight='bold', fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig(OUT/'contrastive_alignment.png', dpi=150, bbox_inches='tight'); plt.close()
print('\nsaved: contrastive_alignment.csv, contrastive_alignment.png')
