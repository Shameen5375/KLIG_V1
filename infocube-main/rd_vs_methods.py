"""
Compare R-D Path Attribution against ALL existing attribution methods in the codebase, on 5 images.
Rows = methods (11 roster + R-D), columns = images. Each cell = per-map normalized |attribution|.
Run:  .venv/Scripts/python rd_vs_methods.py [n_images]   (default 5)
"""
import sys, pickle, warnings
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
from pathlib import Path
import numpy as np, torch, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import klig_methods as KM
from klig_methods import attr_map, METHODS as _ALL, make_phi
import rd_attribution as RD
from scipy.ndimage import gaussian_filter as _gf
from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(DEVICE).eval()
cats = ResNet50_Weights.IMAGENET1K_V2.meta['categories']; phi = make_phi(model); _wrapRD = RD.ModelWrapper(model, DEVICE)
KM.N_STEPS, KM.N_SAMPLES = 25, 3; KM.IG_STEPS = 25; KM.SG_SAMPLES = 25; KM.EG_SAMPLES = 25
KM.T_DESC, KM.N_MC_DESC, KM.N_MC_GRAD = 25, 8, 3
_mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1); _std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

# pool (normalized tensors) + counterfactual-by-class for KL methods
pool, seen = [], set()
for p in ['cs_viz_cache/cands.pkl','klig2_dist_cache/klig2_dist_multiprob.pkl','cs_gate_cache/pool.pkl']:
    if not Path(p).exists(): continue
    for d in pickle.load(open(p,'rb')):
        if len(d.get('high_cls',[]))<2: continue
        x = d['x']; x = x.squeeze(0) if x.dim()==4 else x
        fp = round(float(x.float().sum()),1)
        if fp in seen: continue
        seen.add(fp); pool.append({'x':x.cpu(),'y1':int(d['high_cls'][0]),'y2':int(d['high_cls'][1])})
cf_by = {}
for d in pool: cf_by.setdefault(d['y1'], d['x'])
def cf_for(y2): return (cf_by.get(y2, pool[0]['x'])).to(DEVICE)

sel = pool[:K]
print(f'[setup] {len(sel)} images × {len(_ALL)+1} methods (11 roster + R-D)')

def norm_map(a):
    a = np.abs(a); v = np.percentile(a, 99) + 1e-8; return np.clip(a/v, 0, 1)
def denorm(x): return (x*_std+_mean).clamp(0,1).permute(1,2,0).cpu().numpy()

# order methods (nice roster order) + R-D last
PO = ['Vanilla Grad','SmoothGrad','IG-zero','Blur-IG','IDG','Guided IG','ExpGrad',
      'KLIG-Adaptive','KL-IG (linear)','KL-IG²','KL-IG² (adaptive)']
METHODS = [m for m in PO if m in _ALL] + [m for m in _ALL if m not in PO]
ROWS = ['image'] + METHODS + ['R-D (sufficiency)']

fig, ax = plt.subplots(len(ROWS), len(sel), figsize=(2.6*len(sel), 2.5*len(ROWS)), facecolor='white')
if len(sel) == 1: ax = ax[:, None]
from tqdm import tqdm
for col, d in enumerate(tqdm(sel, desc='images')):
    xn = d['x'].to(DEVICE); tgt = int(model(xn.unsqueeze(0))[0].argmax()); xcf = cf_for(d['y2'])
    img = denorm(d['x'])
    ax[0, col].imshow(img); ax[0, col].set_title(cats[tgt].split(',')[0], fontsize=9); ax[0, col].axis('off')
    for r, m in enumerate(METHODS, start=1):
        try:
            A = attr_map(m, model, xn, tgt, x_cf=xcf, phi=phi).detach().cpu().numpy()
            ax[r, col].imshow(norm_map(A), cmap='inferno')
        except Exception as e:
            ax[r, col].text(0.5, 0.5, 'err', ha='center');
        ax[r, col].axis('off')
    # R-D map (RISE-style smooth pixel map) — SAME post-processing (norm_map + inferno) as every other row
    x01 = (d['x']*_std+_mean).clamp(0,1).unsqueeze(0).to(DEVICE)
    amap = RD.rise_smooth_map(_wrapRD, x01, RD.NoiseOperator(), 0.5, tgt, n_masks=3000, s=8, p=0.5, batch=64, seed=0)
    amap = _gf(np.clip(amap, 0, None), 4)
    ax[len(ROWS)-1, col].imshow(norm_map(amap), cmap='inferno'); ax[len(ROWS)-1, col].axis('off')

for r, name in enumerate(ROWS):
    ax[r, 0].text(-0.15, 0.5, name, transform=ax[r, 0].transAxes, rotation=90, va='center', ha='center',
                  fontsize=10, fontweight='bold', color=('#b00020' if name.startswith('R-D') else 'black'))
plt.suptitle('Attribution maps — 11 existing methods vs R-D Path Attribution (ResNet50, |attr| per-map normalized)',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.98]); plt.savefig('cs_viz_outputs/rd_vs_methods.png', dpi=130, bbox_inches='tight'); plt.close()
print('saved cs_viz_outputs/rd_vs_methods.png')
