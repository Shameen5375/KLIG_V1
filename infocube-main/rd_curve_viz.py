"""Rate-Distortion curve for R-D distribution-space attribution.
For each image, sweep the distortion budget tau and record the MIN total rate the allocation
spends.  Rate falls as you allow more distortion — the signature R(D) tradeoff.
Run:  .venv/Scripts/python rd_curve_viz.py [n]   (default 4)
"""
import sys, pickle, warnings
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
from pathlib import Path
import numpy as np, torch, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
K = int(sys.argv[1]) if len(sys.argv) > 1 else 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import rd_distspace as RD
from torchvision.models import resnet50, ResNet50_Weights
_w = ResNet50_Weights.IMAGENET1K_V2; model = resnet50(weights=_w).to(DEVICE).eval(); cats = _w.meta['categories']

pool, seen = [], set()
for p in ['cs_viz_cache/cands.pkl','klig2_dist_cache/klig2_dist_multiprob.pkl','cs_gate_cache/pool.pkl']:
    if not Path(p).exists(): continue
    for d in pickle.load(open(p,'rb')):
        x = d['x']; x = x.squeeze(0) if x.dim()==4 else x; fp = round(float(x.float().sum()),1)
        if fp in seen: continue
        seen.add(fp); pool.append(x.cpu())
sel = pool[:K]
TAUS = (0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70)
cfg = RD.RDConfig(n_mc=6, n_iter=100)

fig, ax = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')
allR = []
from tqdm import tqdm
for i, x in enumerate(tqdm(sel, desc='R(D)')):
    mu = x.to(DEVICE).unsqueeze(0); tgt = int(model(mu).argmax())
    curve = RD.rd_curve(model, mu, tgt, cfg, taus=TAUS)
    taus = np.array([c[0] for c in curve]); rate = np.array([c[1] for c in curve]); dfin = np.array([c[2] for c in curve])
    lbl = cats[tgt].split(',')[0]
    ax[0].plot(taus, rate, '-o', ms=4, lw=1.5, label=lbl, alpha=0.85)
    ax[1].plot(dfin, rate, '-o', ms=4, lw=1.5, label=lbl, alpha=0.85)
    allR.append(rate)
    print(f'  {lbl:16s} rate@tau: ' + '  '.join(f'{t:.2f}:{r/1e3:.0f}k' for t, r in zip(taus, rate)))

mR = np.mean(allR, 0)
ax[0].plot(TAUS, mR, 'k--', lw=2.5, label='mean', zorder=5)
ax[0].set_xlabel('distortion budget  tau  (fractional logit drop allowed)', fontsize=11)
ax[0].set_ylabel('total allocated rate  R  (bits, sum of per-pixel KL)', fontsize=11)
ax[0].set_yscale('log'); ax[0].set_title('R(D) curve — rate vs distortion budget', fontsize=12, fontweight='bold')
ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
ax[1].set_xlabel('achieved distortion  D  (logit drop)', fontsize=11)
ax[1].set_ylabel('total allocated rate  R  (bits)', fontsize=11)
ax[1].set_yscale('log'); ax[1].set_title('R(D) curve — rate vs achieved distortion', fontsize=12, fontweight='bold')
ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
plt.suptitle('R-D distribution-space attribution — rate-distortion tradeoff (allow more distortion -> spend less rate)',
             fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.96]); out='cs_viz_outputs/rd_curve.png'
plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close(); print('saved', out)
