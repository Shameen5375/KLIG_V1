"""Smoke: pixel CS_struct gated to the model's segment-defined discriminative region."""
import sys, pickle, warnings
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
import numpy as np, torch, pandas as pd
from scipy.ndimage import gaussian_filter
warnings.filterwarnings('ignore')
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
eps = 1e-8; DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import klig_methods as KM
from klig_methods import attr_map, make_phi
from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(DEVICE).eval(); phi = make_phi(model)
KM.N_STEPS, KM.N_SAMPLES = 25, 3; KM.IG_STEPS = 25; KM.T_DESC, KM.N_MC_DESC, KM.N_MC_GRAD = 25, 8, 3
_rng = np.random.default_rng(0)
def attr_for(m, x1, cls, xcf):
    H, W = x1.shape[-2], x1.shape[-1]
    if m == 'Random': return torch.from_numpy(_rng.standard_normal((H, W))).float()
    return attr_map(m, model, x1, int(cls), x_cf=xcf, phi=phi)
CANDS = {}
for p in ['cs_viz_cache/cands.pkl', 'klig2_dist_cache/klig2_dist_multiprob.pkl',
          'klig2_val_cache/klig2_dist_multiprob.pkl', 'cs_gate_cache/pool.pkl']:
    try:
        for d in pickle.load(open(p, 'rb')):
            c = int(d['high_cls'][0]); x = d['x']; x = x.squeeze(0) if x.dim() == 4 else x
            if c not in CANDS: CANDS[c] = x.to(DEVICE)
    except Exception: pass
store = pickle.load(open('cs_viz_outputs/segment_store.pkl', 'rb'))[:N]
METHODS = ['KL-IG² (adaptive)', 'KL-IG²', 'KL-IG (linear)', 'KLIG-Adaptive', 'IG-zero', 'Blur-IG',
           'ExpGrad', 'Guided IG', 'IDG', 'SmoothGrad', 'Vanilla Grad', 'Random']
def topseg(v, f=0.25):
    v = np.asarray(v, float); return v >= np.quantile(v, 1 - f) if np.ptp(v) > 1e-12 else np.zeros(len(v), bool)
def cs_struct_gated(A1, A2, mask, sigma=4):
    D = (A1 - A2).astype(float) * mask; D = D / (np.abs(D).max() + eps)
    coh = gaussian_filter(D, sigma); return float((coh ** 2).sum() / ((D ** 2).sum() + eps))
rows = {m: [] for m in METHODS}
from tqdm import tqdm
for R in tqdm(store, desc='gated pixel CS_struct'):
    x = R['x'].squeeze(0).to(DEVICE); y1, y2 = R['y1'], R['y2']; seg, labs = R['seg'], R['labels']
    xcf = CANDS.get(y2, next(iter(CANDS.values())))
    disc = topseg(np.abs(R['model_d1'] - R['model_d2']))
    mask = np.isin(seg, np.asarray(labs)[disc]).astype(float)
    for m in METHODS:
        A1 = attr_for(m, x, y1, xcf).detach().cpu().numpy(); A2 = attr_for(m, x, y2, xcf).detach().cpu().numpy()
        rows[m].append(cs_struct_gated(A1, A2, mask))
df = pd.DataFrame([dict(method=m, gated_CSstruct=np.mean(v), se=np.std(v)/np.sqrt(len(v))) for m, v in rows.items()]
                  ).sort_values('gated_CSstruct', ascending=False)
print(f"\nn={len(store)}  pixel CS_struct WITHIN model's discriminative region:\n")
print(df.round(3).to_string(index=False))
PY = None
