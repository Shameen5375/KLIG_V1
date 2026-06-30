"""Isolate which attribution method segfaults on ViT. Prints+flushes before each call."""
import sys, pickle, warnings
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
import numpy as np, torch, torch.nn.functional as F
warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import klig_methods as KM
from klig_methods import attr_map, METHODS as _ALL
from klig import make_phi_from_layer
from torchvision.models import vit_b_16, ViT_B_16_Weights
model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to(DEVICE).eval()
phi = make_phi_from_layer(model, model.encoder.ln)
KM.N_STEPS, KM.N_SAMPLES = 25, 3; KM.IG_STEPS = 25; KM.SG_SAMPLES = 25; KM.EG_SAMPLES = 25
KM.T_DESC, KM.N_MC_DESC, KM.N_MC_GRAD = 25, 8, 3

POOL = [d['x'].squeeze(0) if d['x'].dim()==4 else d['x'] for d in pickle.load(open('cs_viz_cache/cands.pkl','rb'))[:2]]
x = POOL[0].to(DEVICE); x_cf = POOL[1].to(DEVICE)
with torch.no_grad():
    p = F.softmax(model(x.unsqueeze(0))[0], 0); y1, y2 = [int(i) for i in p.topk(2).indices.tolist()]
print(f'image ok; y1={y1} y2={y2}', flush=True)
print('phi(x) shape:', tuple(phi(x.unsqueeze(0)).shape), flush=True)

for m in _ALL:
    print(f'--- trying: {m}', flush=True)
    try:
        A = attr_map(m, model, x, y1, x_cf=x_cf, phi=phi)
        print(f'    OK  {m}  shape={tuple(A.shape)}  finite={bool(torch.isfinite(A).all())}', flush=True)
    except Exception as e:
        print(f'    PYERR {m}: {type(e).__name__}: {str(e)[:120]}', flush=True)
print('ALL DONE', flush=True)
