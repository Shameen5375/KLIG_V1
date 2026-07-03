"""
Faithfulness comparison: insertion / deletion AUC for all 12 methods (11 roster + R-D) on N images.
Deletion: replace patches with a blurred baseline in importance order -> prob should DROP fast (low AUC good).
Insertion: start blurred, insert original patches in importance order -> prob should RISE fast (high AUC good).
Composite = Insertion − Deletion (higher = more faithful). ResNet50, patch grid ranking.
Run:  .venv/Scripts/python rd_faithfulness.py [n_images]   (default 5)
"""
import sys, pickle, warnings
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G = 14                                        # ranking patch grid

import klig_methods as KM
from klig_methods import attr_map, METHODS as _ALL, make_phi
import rd_attribution as RD
from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(DEVICE).eval()
phi = make_phi(model); wrap = RD.ModelWrapper(model, DEVICE)
KM.N_STEPS, KM.N_SAMPLES = 25, 3; KM.IG_STEPS = 25; KM.SG_SAMPLES = 25; KM.EG_SAMPLES = 25
KM.T_DESC, KM.N_MC_DESC, KM.N_MC_GRAD = 25, 8, 3
_mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1); _std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

pool, seen = [], set()
for p in ['cs_viz_cache/cands.pkl','klig2_dist_cache/klig2_dist_multiprob.pkl','cs_gate_cache/pool.pkl']:
    if not Path(p).exists(): continue
    for d in pickle.load(open(p,'rb')):
        if len(d.get('high_cls',[]))<2: continue
        x = d['x']; x = x.squeeze(0) if x.dim()==4 else x; fp = round(float(x.float().sum()),1)
        if fp in seen: continue
        seen.add(fp); pool.append({'x':x.cpu(),'y1':int(d['high_cls'][0]),'y2':int(d['high_cls'][1])})
cf_by = {}
for d in pool: cf_by.setdefault(d['y1'], d['x'])
sel = pool[:K]
PO = ['Vanilla Grad','SmoothGrad','IG-zero','Blur-IG','IDG','Guided IG','ExpGrad',
      'KLIG-Adaptive','KL-IG (linear)','KL-IG²','KL-IG² (adaptive)']
METHODS = [m for m in PO if m in _ALL] + [m for m in _ALL if m not in PO] + ['R-D (sufficiency)']
print(f'[setup] {len(sel)} images × {len(METHODS)} methods, grid={G}')

lab, _ = RD.patch_regions(224, 224, G); NP = G*G
pmask = torch.stack([torch.from_numpy((lab==r).astype('float32')) for r in range(NP)]).to(DEVICE)  # (NP,H,W)
def blur01(x): return RD.BlurOperator().degrade(x.unsqueeze(0), 8.0)[0]
def patch_order(amap):                         # rank patches by mean |attr| desc
    a = np.abs(amap); ps = np.array([a[lab==r].mean() for r in range(NP)]); return np.argsort(-ps)

@torch.no_grad()
def curve(start, other, order, cls, steps=20):
    chunk = max(1, NP//steps); cur = start.clone(); states=[cur.clone().unsqueeze(0)]; fr=[0.0]
    for i in range(0, NP, chunk):
        m = pmask[order[i:i+chunk]].sum(0).clamp(0,1)[None]     # (1,H,W)
        cur = cur*(1-m) + other*m; states.append(cur.clone().unsqueeze(0)); fr.append(min(i+chunk,NP)/NP)
    imgs = torch.cat(states,0)
    probs = F.softmax(wrap.logits(imgs), -1)[:, cls].cpu().numpy()
    return np.trapezoid(probs, fr)

res = {m: {'ins':[], 'del':[]} for m in METHODS}
from tqdm import tqdm
for d in tqdm(sel, desc='images'):
    x01 = (d['x']*_std+_mean).clamp(0,1).to(DEVICE); xn = d['x'].to(DEVICE)
    cls = int(model(xn.unsqueeze(0))[0].argmax()); xcf = (cf_by.get(d['y2'], pool[0]['x'])).to(DEVICE)
    blur = blur01(x01)
    for m in METHODS:
        if m.startswith('R-D'):
            amap = RD.run_rd_attribution(model, x01, RD.RDConfig(window=48, stride=12, n_mc=3), DEVICE, full=False)['suff_map']
        else:
            try: amap = attr_map(m, model, xn, cls, x_cf=xcf, phi=phi).detach().cpu().numpy()
            except Exception: continue
        o = patch_order(amap)
        res[m]['del'].append(curve(x01, blur, o, cls))      # delete important-first: prob drops
        res[m]['ins'].append(curve(blur, x01, o, cls))      # insert important-first: prob rises

rows = []
for m in METHODS:
    di = np.array(res[m]['del']); ii = np.array(res[m]['ins'])
    if not len(di): continue
    rows.append((m, ii.mean(), di.mean(), ii.mean()-di.mean(),
                 ii.std()/np.sqrt(len(ii)), di.std()/np.sqrt(len(di))))
rows.sort(key=lambda r: -r[3])
print(f'\n=== faithfulness (n={len(sel)})  higher Insertion, lower Deletion, higher Composite = better ===')
print(f'{"method":20s} {"Insertion↑":>11s} {"Deletion↓":>10s} {"Composite↑":>11s}')
for m, ins, dele, comp, ise, dse in rows:
    print(f'{m:20s} {ins:>11.3f} {dele:>10.3f} {comp:>+11.3f}')

fig, ax = plt.subplots(figsize=(9, 6), facecolor='white')
names = [r[0] for r in rows]; comp = [r[3] for r in rows]
cols = ['#b00020' if n.startswith('R-D') else '#2c7fb8' for n in names]
ax.barh(range(len(names)), comp, color=cols); ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
ax.invert_yaxis(); ax.set_xlabel('Composite faithfulness  (Insertion − Deletion AUC)')
ax.set_title(f'Attribution faithfulness — R-D vs 11 methods (ResNet50, n={len(sel)})\ninsertion/deletion, higher = better',
             fontweight='bold', fontsize=11)
for i, c in enumerate(comp): ax.text(c, i, f' {c:+.3f}', va='center', fontsize=8)
plt.tight_layout(); plt.savefig('cs_viz_outputs/rd_faithfulness.png', dpi=170, bbox_inches='tight'); plt.close()
print('\nsaved cs_viz_outputs/rd_faithfulness.png')
