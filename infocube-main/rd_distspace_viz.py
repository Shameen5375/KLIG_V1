"""Visualize R-D distribution-space attribution as a PEER of KL-IG (same geometry, different
estimator).  Rows = images; columns = [input, KL-IG^2 map, R-D (dist-space) map, R-D overlay].
Run:  .venv/Scripts/python rd_distspace_viz.py [n]   (default 5)
"""
import sys, pickle, warnings
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
from pathlib import Path
import numpy as np, torch, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.cm as cm
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import rd_distspace as RD
from klig.image.attribution import ImageAttributor
from torchvision.models import resnet50, ResNet50_Weights
_w = ResNet50_Weights.IMAGENET1K_V2; model = resnet50(weights=_w).to(DEVICE).eval(); cats = _w.meta['categories']
_mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1); _std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

pool, seen = [], set()
for p in ['cs_viz_cache/cands.pkl','klig2_dist_cache/klig2_dist_multiprob.pkl','cs_gate_cache/pool.pkl']:
    if not Path(p).exists(): continue
    for d in pickle.load(open(p,'rb')):
        x = d['x']; x = x.squeeze(0) if x.dim()==4 else x; fp = round(float(x.float().sum()),1)
        if fp in seen: continue
        seen.add(fp); pool.append(x.cpu())
sel = pool[:K]
cfg = RD.RDConfig(n_mc=8, n_iter=150)

def norm(a):
    a = np.abs(a); v = np.percentile(a, 99) + 1e-8; return np.clip(a/v, 0, 1)
def denorm(x): return (x*_std+_mean).clamp(0,1).permute(1,2,0).cpu().numpy()
def overlay(img, a):
    a = np.clip(a,0,1)**0.85; heat = cm.get_cmap('inferno')(a)[...,:3]; al=(0.25+0.55*a)[...,None]
    return np.clip(img*(1-al)+heat*al,0,1)

fig, ax = plt.subplots(K, 4, figsize=(11, 2.7*K), facecolor='white')
if K == 1: ax = ax[None, :]
from tqdm import tqdm
for r, x in enumerate(tqdm(sel, desc='distspace-viz')):
    mu = x.to(DEVICE).unsqueeze(0); tgt = int(model(mu).argmax())
    kl = ImageAttributor(model, n_steps=25, n_samples=8).attribute(x.to(DEVICE), target=tgt)
    a_kl = np.abs(kl.attr_map('sumabs').detach().cpu().numpy())
    rd = RD.rd_attribution(model, mu, tgt, cfg)
    a_rd = gaussian_filter(rd['attribution'].cpu().numpy(), 2)
    rho = spearmanr(a_rd.ravel(), a_kl.ravel()).correlation
    img = denorm(x)
    ax[r,0].imshow(img); ax[r,0].axis('off')
    ax[r,0].text(-0.08,0.5,cats[tgt].split(',')[0], transform=ax[r,0].transAxes, rotation=90,
                 va='center', ha='center', fontsize=9, fontweight='bold')
    ax[r,1].imshow(norm(a_kl), cmap='inferno'); ax[r,1].axis('off')
    ax[r,2].imshow(norm(a_rd), cmap='inferno'); ax[r,2].axis('off')
    ax[r,3].imshow(overlay(img, norm(a_rd))); ax[r,3].axis('off')
    ax[r,2].set_title(f'rho={rho:+.2f}', fontsize=8, color='#b00020')
for j,t in enumerate(['input','KL-IG (path integral)','R-D (rate allocation)','R-D overlay']):
    ax[0,j].set_title((ax[0,j].get_title()+'\n' if j==2 else '')+t, fontsize=10, fontweight='bold',
                      color=('#b00020' if j>=2 else 'black'))
plt.suptitle('R-D distribution-space attribution — PEER of KL-IG (same probe/ruler/endpoint; '
             'allocation vs path-integral). rho<1 => correlated, not identical.',
             fontsize=11, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.97]); out='cs_viz_outputs/rd_distspace_viz.png'
plt.savefig(out, dpi=140, bbox_inches='tight'); plt.close(); print('saved', out)
