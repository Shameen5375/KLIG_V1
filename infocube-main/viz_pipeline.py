"""Visualize every step of the gated-CS_struct pipeline for one image.
Run:  .venv/Scripts/python viz_pipeline.py [store_index]   (default = a clear spatial example)
Saves: cs_viz_outputs/pipeline_viz.png
"""
import sys, pickle, warnings
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
import numpy as np, torch, matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
warnings.filterwarnings('ignore')
EPS = 1e-8; DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import klig_methods as KM
from klig_methods import attr_map, make_phi
from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(DEVICE).eval(); phi = make_phi(model)
KM.N_STEPS, KM.N_SAMPLES = 25, 3; KM.IG_STEPS = 25; KM.T_DESC, KM.N_MC_DESC, KM.N_MC_GRAD = 25, 8, 3
_rng = np.random.default_rng(0)
def attr_for(m, x1, cls, xcf):
    if m == 'Random':
        return torch.from_numpy(_rng.standard_normal(x1.shape[-2:])).float()
    return attr_map(m, model, x1, int(cls), x_cf=xcf, phi=phi)

CANDS = {}
for p in ['cs_viz_cache/cands.pkl', 'klig2_dist_cache/klig2_dist_multiprob.pkl',
          'klig2_val_cache/klig2_dist_multiprob.pkl', 'cs_gate_cache/pool.pkl']:
    try:
        for d in pickle.load(open(p, 'rb')):
            c = int(d['high_cls'][0]); x = d['x']; x = x.squeeze(0) if x.dim() == 4 else x
            if c not in CANDS: CANDS[c] = x.to(DEVICE)
    except Exception: pass

store = pickle.load(open('cs_viz_outputs/segment_store.pkl', 'rb'))

# ---- helpers ----
def paint(vals, seg, labs):                       # per-segment value → pixel map
    img = np.zeros(seg.shape, float)
    for k, lab in enumerate(labs): img[seg == lab] = vals[k]
    return img
def boundaries(seg):                              # segment edges (no skimage)
    b = np.zeros(seg.shape, bool)
    b[:-1, :] |= seg[:-1, :] != seg[1:, :]; b[:, :-1] |= seg[:, :-1] != seg[:, 1:]
    return b
def topseg(v, f=0.25):
    v = np.asarray(v, float)
    return v >= np.quantile(v, 1 - f) if np.ptp(v) > EPS else np.zeros(len(v), bool)
def cs_struct_gated(A1, A2, mask, sigma=4):
    D = (A1 - A2).astype(float) * mask; D = D / (np.abs(D).max() + EPS)
    coh = gaussian_filter(D, sigma); return float((coh ** 2).sum() / ((D ** 2).sum() + EPS)), D, coh
def denorm(x):
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    return np.clip(x.permute(1, 2, 0).cpu().numpy() * std + mean, 0, 1)
def pct(a, q=99):                                 # symmetric clip for diverging maps
    v = np.percentile(np.abs(a), q) + EPS; return -v, v

# ---- choose example: largest discriminative-area energy (clear spatial case) ----
if len(sys.argv) > 1:
    IDX = int(sys.argv[1])
else:
    score = [np.linalg.norm(np.abs(np.asarray(R['model_d1']) - np.asarray(R['model_d2']))) for R in store]
    IDX = int(np.argsort(score)[-3])              # a strong (not the absolute max) example
R = store[IDX]
x = R['x'].squeeze(0).to(DEVICE); y1, y2 = R['y1'], R['y2']; seg, labs = R['seg'], np.asarray(R['labels'])
d1, d2 = np.asarray(R['model_d1']), np.asarray(R['model_d2'])
disc = topseg(np.abs(d1 - d2)); region = np.isin(seg, labs[disc]).astype(float)
xcf = CANDS.get(y2, next(iter(CANDS.values())))
print(f'image idx={IDX}  y1={y1} y2={y2}  segments={len(labs)}  region pixels={int(region.sum())}')

img = denorm(x)
METHODS = ['KL-IG² (adaptive)', 'Vanilla Grad']
attrs = {}
for m in METHODS:
    A1 = attr_for(m, x, y1, xcf).detach().cpu().numpy(); A2 = attr_for(m, x, y2, xcf).detach().cpu().numpy()
    gcs, D, coh = cs_struct_gated(A1, A2, region)
    attrs[m] = dict(A1=A1, A2=A2, D=D, coh=coh, gcs=gcs)
    print(f'{m:20s} gated CS_struct = {gcs:.3f}')

# ---- figure: row0 = model pipeline, row1/2 = each method ----
fig, ax = plt.subplots(3, 6, figsize=(20, 10), facecolor='white')
def show(a, im, title, cmap='viridis', vmm=None, over=None):
    a.imshow(im, cmap=cmap, vmin=None if vmm is None else vmm[0], vmax=None if vmm is None else vmm[1])
    if over is not None: a.imshow(np.dstack([over, np.zeros_like(over), np.zeros_like(over), over*0.45]))
    a.set_title(title, fontsize=11); a.axis('off')

# Row 0 — model / region pipeline
bnd = boundaries(seg)
show(ax[0, 0], img, '1. image', cmap=None)
ov = img.copy(); ov[bnd] = [1, 1, 0]
show(ax[0, 1], ov, f'2. segments ({len(labs)})', cmap=None)
show(ax[0, 2], paint(d1, seg, labs), f'3. occlusion drop  y1={y1}', cmap='magma')
show(ax[0, 3], paint(d2, seg, labs), f'   occlusion drop  y2={y2}', cmap='magma')
show(ax[0, 4], paint(np.abs(d1 - d2), seg, labs), '4. differential area |d1−d2|', cmap='inferno')
show(ax[0, 5], img, '5. discriminative region', cmap=None, over=region)

# Rows 1–2 — methods
for ri, m in enumerate(METHODS, start=1):
    A = attrs[m]; vmA1 = pct(A['A1']); vmA2 = pct(A['A2']); vmD = pct(A['D']); vmC = pct(A['coh'])
    show(ax[ri, 0], np.abs(A['A1']), f'{m}\nA_y1', cmap='viridis', vmm=(0, np.percentile(np.abs(A['A1']), 99)+EPS))
    show(ax[ri, 1], np.abs(A['A2']), 'A_y2', cmap='viridis', vmm=(0, np.percentile(np.abs(A['A2']), 99)+EPS))
    show(ax[ri, 2], A['A1'] - A['A2'], '6. difference  A_y1 − A_y2', cmap='seismic', vmm=pct(A['A1'] - A['A2']))
    show(ax[ri, 3], A['D'], '7. difference GATED to region', cmap='seismic', vmm=vmD)
    show(ax[ri, 4], A['coh'], '8. coherence (blurred)', cmap='seismic', vmm=vmC)
    ax[ri, 5].axis('off')
    ax[ri, 5].text(0.5, 0.5, f'gated\nCS_struct\n\n{A["gcs"]:.3f}', ha='center', va='center',
                   fontsize=20, fontweight='bold',
                   color=('#1a7f37' if A['gcs'] > 0.05 else '#b00020'),
                   transform=ax[ri, 5].transAxes,
                   bbox=dict(boxstyle='round', fc='#eef6ee' if A['gcs'] > 0.05 else '#fdeeee', ec='gray'))

plt.suptitle(f'Gated CS_struct pipeline — image #{IDX}  (y1={y1} vs y2={y2})\n'
             'segments locate the model region → pixel coherence of the class-difference there.  '
             'Clean method scores high; scatter floors.', fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('cs_viz_outputs/pipeline_viz.png', dpi=160, bbox_inches='tight'); plt.close()
print('saved cs_viz_outputs/pipeline_viz.png')
