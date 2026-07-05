"""Aggregate Rate-Distortion curve over N images (default 50) for R-D distribution-space
attribution.  One graph (mean +/- SE across images), NOT individual per-image lines.

Rates vary 1000x across images, so we normalize each image's curve to its own max-rate point
(the R(D) SHAPE), average those, and also show the absolute median rate for magnitude context.
Run:  .venv/Scripts/python rd_curve50.py [n]   (default 50)
"""
import sys, pickle, warnings, math
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
from pathlib import Path
import numpy as np, torch, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
K = int(sys.argv[1]) if len(sys.argv) > 1 else 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import rd_distspace as RD
from klig.image.stopping import find_sigma_stop
from torchvision.models import resnet50, ResNet50_Weights
_w = ResNet50_Weights.IMAGENET1K_V2; model = resnet50(weights=_w).to(DEVICE).eval()

pool, seen = [], set()
for p in ['cs_viz_cache/cands.pkl','klig2_dist_cache/klig2_dist_multiprob.pkl','cs_gate_cache/pool.pkl']:
    if not Path(p).exists(): continue
    for d in pickle.load(open(p,'rb')):
        x = d['x']; x = x.squeeze(0) if x.dim()==4 else x; fp = round(float(x.float().sum()),1)
        if fp in seen: continue
        seen.add(fp); pool.append(x.cpu())
sel = pool[:K]
TAUS = np.array([0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70])
print(f'[setup] {len(sel)} images | {DEVICE} | taus={list(TAUS)}')

CKPT = Path('cs_viz_cache/rd_curve.pkl'); rates = []
if CKPT.exists():
    rates = pickle.load(open(CKPT, 'rb')); print(f'[resume] {len(rates)} cached curves')

from tqdm import tqdm
for i in tqdm(range(len(rates), len(sel)), desc='R(D) 50'):
    x = sel[i]; mu = x.to(DEVICE).unsqueeze(0)
    with torch.no_grad(): tgt = int(model(mu).argmax())
    sig = find_sigma_stop(model, x.to(DEVICE), tgt)                 # once per image, not per-tau
    cfg = RD.RDConfig(n_mc=5, n_iter=80, lv_floor=2.0*math.log(max(sig,1e-3)), adaptive_floor=False)
    row = [RD.rd_attribution(model, mu, tgt, RD._with(cfg, tau=float(t)))['info']['total_rate'] for t in TAUS]
    rates.append(row)
    if (i+1) % 10 == 0: pickle.dump(rates, open(CKPT, 'wb'))
pickle.dump(rates, open(CKPT, 'wb'))

R = np.array(rates)                                                  # (N, len(TAUS))
n = R.shape[0]                                                        # NO exclusion — all images
# normalized shape: each image / its OWN max rate (bounded in [0,1] even for tiny-rate images;
# robust to MC-noise non-monotonicity).  Near-zero-rate images are INCLUDED, not dropped.
Rn = R / np.maximum(R.max(1, keepdims=True), 1.0)
mean_n, se_n = Rn.mean(0), Rn.std(0)/math.sqrt(n)
med, q1, q3 = np.median(R, 0), np.percentile(R, 25, 0), np.percentile(R, 75, 0)
n_low = int((R.max(1) < 500).sum())
print(f'[all] {n} images, NO exclusion ({n_low} are near-0-rate / prior-robust, still included)')

fig, ax = plt.subplots(1, 2, figsize=(13, 5), facecolor='white')
ax[0].plot(TAUS, mean_n, '-o', color='#b00020', lw=2.2, ms=6, label=f'mean (n={n})')
ax[0].fill_between(TAUS, mean_n-se_n, mean_n+se_n, color='#b00020', alpha=0.25, label='±1 SE')
ax[0].set_xlabel('distortion budget  tau  (fractional logit drop allowed)', fontsize=11)
ax[0].set_ylabel('normalized rate  R(tau) / max_tau R', fontsize=11)
ax[0].set_title(f'R(D) shape averaged over {n} images (no exclusion)',
                fontsize=12, fontweight='bold')
ax[0].grid(alpha=0.3); ax[0].legend(fontsize=10)

ax[1].plot(TAUS, med, '-o', color='#1f4e79', lw=2.2, ms=6, label='median')
ax[1].fill_between(TAUS, q1, q3, color='#1f4e79', alpha=0.22, label='IQR (25-75%)')
ax[1].set_yscale('log'); ax[1].set_xlabel('distortion budget  tau', fontsize=11)
ax[1].set_ylabel('total allocated rate  R  (bits)', fontsize=11)
ax[1].set_title(f'Absolute rate — median over {n} images', fontsize=12, fontweight='bold')
ax[1].grid(alpha=0.3, which='both'); ax[1].legend(fontsize=10)
plt.suptitle('R-D distribution-space attribution — aggregate rate-distortion curve '
             '(allow more distortion -> spend less rate)', fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.96]); out=f'cs_viz_outputs/rd_curve{n}.png'
plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
drop = 100*(mean_n[0] - mean_n[-1])/mean_n[0]
print(f'saved {out}  | mean rate falls {drop:.0f}% from tau=0.02 to 0.70 (n={n}, no exclusion)')
