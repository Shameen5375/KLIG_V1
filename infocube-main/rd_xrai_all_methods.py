"""XRAI-format region reveal (top-k% salient, rest solid gray) for ALL 12 methods (11 roster + R-D).
Rows = methods, columns = images.  arXiv:1906.02825 (Kapishnikov et al., ICCV 2019) visualization style.
Run:  .venv/Scripts/python rd_xrai_all_methods.py [n_images] [arch] [frac]   (default 5 resnet 0.25)
"""
import sys, pickle, warnings
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
from pathlib import Path
import numpy as np, torch, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
warnings.filterwarnings('ignore')
K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
ARCH = sys.argv[2] if len(sys.argv) > 2 else 'resnet'
FRAC = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import klig_methods as KM
from klig_methods import attr_map, METHODS as _ALL, make_phi
import rd_attribution as RD
import rd_distspace as RDS
if ARCH == 'vit':
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    _w = ViT_B_16_Weights.IMAGENET1K_V1; model = vit_b_16(weights=_w).to(DEVICE).eval()
    from klig import make_phi_from_layer; phi = make_phi_from_layer(model, model.encoder.ln)
else:
    from torchvision.models import resnet50, ResNet50_Weights
    _w = ResNet50_Weights.IMAGENET1K_V2; model = resnet50(weights=_w).to(DEVICE).eval(); phi = make_phi(model)
cats = _w.meta['categories']; wrap = RD.ModelWrapper(model, DEVICE)
KM.N_STEPS, KM.N_SAMPLES = 25, 3; KM.IG_STEPS = 25; KM.SG_SAMPLES = 25; KM.EG_SAMPLES = 25
KM.T_DESC, KM.N_MC_DESC, KM.N_MC_GRAD = 25, 8, 3
_mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1); _std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

pool, seen = [], set()
for p in ['cs_viz_cache/cands.pkl','klig2_dist_cache/klig2_dist_multiprob.pkl','cs_gate_cache/pool.pkl']:
    if not Path(p).exists(): continue
    for d in pickle.load(open(p,'rb')):
        if len(d.get('high_cls',[]))<2: continue
        x=d['x']; x=x.squeeze(0) if x.dim()==4 else x; fp=round(float(x.float().sum()),1)
        if fp in seen: continue
        seen.add(fp); pool.append({'x':x.cpu(),'y1':int(d['high_cls'][0]),'y2':int(d['high_cls'][1])})
cf_by={}
for d in pool: cf_by.setdefault(d['y1'], d['x'])
sel = pool[:K]
PO = ['Vanilla Grad','SmoothGrad','IG-zero','Blur-IG','IDG','Guided IG','ExpGrad',
      'KLIG-Adaptive','KL-IG (linear)','KL-IG²','KL-IG² (adaptive)']
METHODS = [m for m in PO if m in _ALL] + [m for m in _ALL if m not in PO] + ['R-D (noise)', 'R-D (dist-space)']
_rds_cfg = RDS.RDConfig(n_mc=8, n_iter=150)
print(f'[setup] arch={ARCH} {len(sel)} imgs × {len(METHODS)} methods, top-{int(FRAC*100)}% reveal')

def denorm(x): return (x*_std+_mean).clamp(0,1).permute(1,2,0).cpu().numpy()
def reveal(img, a, frac):
    a = gaussian_filter(np.abs(a).astype(float), 3); thr = np.quantile(a, 1-frac)
    m = (a>=thr)[...,None]; return np.where(m, img, np.full_like(img, 0.5))

# precompute maps[method][image]
maps = {m: [] for m in METHODS}
from tqdm import tqdm
for d in tqdm(sel, desc='maps'):
    xn = d['x'].to(DEVICE); tgt = int(model(xn.unsqueeze(0))[0].argmax()); xcf = (cf_by.get(d['y2'], pool[0]['x'])).to(DEVICE)
    x01 = (d['x']*_std+_mean).clamp(0,1).unsqueeze(0).to(DEVICE)
    for m in METHODS:
        if m == 'R-D (dist-space)':
            maps[m].append(RDS.rd_attribution(model, xn.unsqueeze(0), tgt, _rds_cfg)['attribution'].cpu().numpy())
        elif m.startswith('R-D'):
            maps[m].append(RD.rise_smooth_map(wrap, x01, RD.NoiseOperator(), 0.5, tgt, n_masks=3000, s=8, seed=0))
        else:
            try: maps[m].append(attr_map(m, model, xn, tgt, x_cf=xcf, phi=phi).detach().cpu().numpy())
            except Exception: maps[m].append(np.zeros((224,224)))

# rows = images, columns = [input image] + methods  (actual image in the leftmost panel)
ncol = 1 + len(METHODS)
fig, ax = plt.subplots(len(sel), ncol, figsize=(2.3*ncol, 2.5*len(sel)), facecolor='white')
if len(sel)==1: ax = ax[None, :]
for r, d in enumerate(sel):
    img = denorm(d['x']); tgt = int(model(d['x'].to(DEVICE).unsqueeze(0))[0].argmax())
    ax[r,0].imshow(img); ax[r,0].axis('off')
    ax[r,0].text(-0.10,0.5,cats[tgt].split(',')[0], transform=ax[r,0].transAxes, rotation=90,
                 va='center', ha='center', fontsize=9, fontweight='bold')
    for c, m in enumerate(METHODS, start=1):
        ax[r,c].imshow(reveal(img, maps[m][r], FRAC)); ax[r,c].axis('off')
ax[0,0].set_title('input', fontsize=9, fontweight='bold')
for c, m in enumerate(METHODS, start=1):
    ax[0,c].set_title(m, fontsize=7.5, fontweight='bold', rotation=35, ha='left',
                      color=('#b00020' if m.startswith('R-D') else 'black'))
plt.suptitle(f'Top-{int(FRAC*100)}% salient region reveal ({ARCH.upper()}) — XRAI format (arXiv:1906.02825), all methods',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.98]); out=f'cs_viz_outputs/rd_xrai_all_methods_{ARCH}.png'
plt.savefig(out, dpi=130, bbox_inches='tight'); plt.close(); print('saved', out)
