"""
Segment-level gated CS_struct for ViT-B/16 (same pipeline as segment_occlusion.py, ViT model).
ViT-native: top-2 classes, occlusion, counterfactuals and attributions all use ViT.
phi = encoder.ln (ViT's penultimate features, analogue of ResNet layer4).
Run:  .venv/Scripts/python segment_occlusion_vit.py [N]   (default 100)
"""
import sys, pickle, warnings
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.stats import wilcoxon
warnings.filterwarnings('ignore')

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
FZ_SCALE, FZ_SIGMA, FZ_MINSIZE, SEED, EPS = 0.6, 0.8, 100, 0, 1e-8
OCC_CHUNK = 24                                    # smaller batch for ViT memory
OUT = Path('cs_viz_outputs'); OUT.mkdir(exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED); np.random.seed(SEED)
print(f'[setup] device={DEVICE}  N={N}  model=ViT-B/16')

import klig_methods as KM
from klig_methods import attr_map, METHODS as _ALL
from klig import make_phi_from_layer
from torchvision.models import vit_b_16, ViT_B_16_Weights
_w = ViT_B_16_Weights.IMAGENET1K_V1
model = vit_b_16(weights=_w).to(DEVICE).eval()
labels = _w.meta['categories']
phi = make_phi_from_layer(model, model.encoder.ln)   # ViT penultimate features (B,197,768)
KM.N_STEPS, KM.N_SAMPLES = 25, 3; KM.IG_STEPS = 25; KM.SG_SAMPLES = 25; KM.EG_SAMPLES = 25
KM.T_DESC, KM.N_MC_DESC, KM.N_MC_GRAD = 25, 8, 3
_rng = np.random.default_rng(SEED)
def attr_for(m, x1, cls, x_cf):
    if m == 'Random': return torch.from_numpy(_rng.standard_normal(x1.shape[-2:])).float()
    return attr_map(m, model, x1, int(cls), x_cf=x_cf, phi=phi)

# ── pool ──
srcs = ['cs_viz_cache/cands.pkl','klig2_dist_cache/klig2_dist_multiprob.pkl',
        'klig2_val_cache/klig2_dist_multiprob.pkl','cs_gate_cache/pool.pkl']
POOL = []; seen = set()
for s in srcs:
    if not Path(s).exists(): continue
    for d in pickle.load(open(s,'rb')):
        x = d['x']; x = x.squeeze(0) if x.dim()==4 else x
        fp = round(float(x.float().sum()),1)
        if fp in seen: continue
        seen.add(fp); POOL.append(x.cpu())
POOL_FILE = sys.argv[2] if len(sys.argv) > 2 else None
if POOL_FILE and Path(POOL_FILE).exists():
    POOL = [(d['x'].squeeze(0) if d['x'].dim()==4 else d['x']).cpu() for d in pickle.load(open(POOL_FILE,'rb'))]
    print(f'[pool] using {POOL_FILE}: {len(POOL)} images')
print(f'[pool] {len(POOL)} unique images')

# ── ViT top-2 for every pool image (ViT-native labels + cf map) ──
@torch.no_grad()
def vit_top2(x):
    p = F.softmax(model(x.unsqueeze(0).to(DEVICE))[0], 0)
    t = p.topk(2).indices.tolist(); return int(t[0]), int(t[1]), float(p[t[0]]), float(p[t[1]])
print('[vit] scoring pool for ViT top-2 ...')
from tqdm import tqdm
META = []
for x in tqdm(POOL, desc='vit top2'):
    t1, t2, p1, p2 = vit_top2(x); META.append((t1, t2, p1, p2))
by_class = {}
for x, (t1, _, _, _) in zip(POOL, META):
    by_class.setdefault(t1, x)                    # one exemplar per ViT top-1 class → counterfactual source
def cf_for(y2):
    return (by_class.get(y2, POOL[0])).to(DEVICE)

# diverse pick: dedup by ViT top-1 (unless explicit pool file)
order = list(range(len(POOL)))
if not (POOL_FILE and Path(POOL_FILE).exists()):
    import random as _r; _r.Random(SEED).shuffle(order)
sel, used = [], set()
for i in order:
    t1 = META[i][0]
    if not (POOL_FILE and Path(POOL_FILE).exists()):
        if t1 in used: continue
        used.add(t1)
    sel.append(i)
    if len(sel) >= N: break
print(f'[run] {len(sel)} images × {len(_ALL)} methods (ViT)')

# ── helpers (identical to segment_occlusion.py) ──
def get_segments(x, scale=FZ_SCALE, sigma=FZ_SIGMA, min_size=FZ_MINSIZE):
    img = x.cpu().numpy().transpose(1, 2, 0); img = (img - img.min()) / (np.ptp(img) + EPS)
    H, W, _ = img.shape; im = gaussian_filter(img, (sigma, sigma, 0)).reshape(-1, 3); Npx = H*W
    idx = np.arange(Npx).reshape(H, W)
    A = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()])
    B = np.concatenate([idx[:, 1:].ravel(),  idx[1:, :].ravel()])
    Wt = np.sqrt(((im[A]-im[B])**2).sum(1)); o = np.argsort(Wt); A, B, Wt = A[o], B[o], Wt[o]
    par = np.arange(Npx); rank = np.zeros(Npx, int); size = np.ones(Npx, int); intd = np.zeros(Npx); k = scale
    def find(z):
        r = z
        while par[r] != r: r = par[r]
        while par[z] != r: par[z], z = r, par[z]
        return r
    for a, b, w in zip(A.tolist(), B.tolist(), Wt.tolist()):
        ra, rb = find(a), find(b)
        if ra == rb: continue
        if w <= min(intd[ra]+k/size[ra], intd[rb]+k/size[rb]):
            if rank[ra] < rank[rb]: ra, rb = rb, ra
            par[rb] = ra; size[ra] += size[rb]; intd[ra] = max(intd[ra], intd[rb], w)
            if rank[ra] == rank[rb]: rank[ra] += 1
    for a, b in zip(A.tolist(), B.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb and (size[ra] < min_size or size[rb] < min_size):
            if rank[ra] < rank[rb]: ra, rb = rb, ra
            par[rb] = ra; size[ra] += size[rb]
    roots = np.array([find(i) for i in range(Npx)]); _, seg = np.unique(roots, return_inverse=True)
    return seg.reshape(H, W)

@torch.no_grad()
def segment_model_delta(x, y1, y2, seg, chunk=OCC_CHUNK):
    base = F.softmax(model(x.unsqueeze(0))[0], 0); b1, b2 = base[y1].item(), base[y2].item()
    labs = np.unique(seg); xb = x.unsqueeze(0).repeat(len(labs), 1, 1, 1).clone()
    for k, lab in enumerate(labs): xb[k][:, torch.from_numpy(seg == lab).to(x.device)] = 0
    d1 = np.zeros(len(labs)); d2 = np.zeros(len(labs))
    for s in range(0, len(labs), chunk):
        p = F.softmax(model(xb[s:s+chunk]), -1)
        d1[s:s+p.shape[0]] = b1 - p[:, y1].cpu().numpy(); d2[s:s+p.shape[0]] = b2 - p[:, y2].cpu().numpy()
    return labs, d1, d2

def segment_method_delta(A1, A2, seg, labs):
    a1, a2 = np.abs(A1), np.abs(A2); m1 = np.zeros(len(labs)); m2 = np.zeros(len(labs))
    for k, lab in enumerate(labs):
        mask = seg == lab; m1[k] = a1[mask].mean(); m2[k] = a2[mask].mean()
    return m1, m2

def _topseg(v, f=0.25):
    v = np.asarray(v, float)
    return v >= np.quantile(v, 1 - f) if np.ptp(v) > EPS else np.zeros(len(v), bool)
def cs_struct_gated(A1, A2, mask, sigma=4):
    D = (A1 - A2).astype(float) * mask; D = D / (np.abs(D).max() + EPS)
    coh = gaussian_filter(D, sigma); return float((coh ** 2).sum() / ((D ** 2).sum() + EPS))
def regime_ratio(d1, d2):
    return float(((d1-d2)**2).sum() / (((d1+d2)**2).sum() + EPS))

# ── main loop ──
import gc
METHODS = list(_ALL)
conf_ratio, rand_ratio, store = [], [], []
fail = {m: 0 for m in METHODS}
DBG = len(sel) <= 5                                # verbose per-stage flushing for small smoke runs
for ii, i in enumerate(tqdm(sel, desc='ViT segment occlusion')):
    if DBG: print(f'[img {ii}] start idx={i}', flush=True)
    x = POOL[i].to(DEVICE); y1, y2 = META[i][0], META[i][1]; x_cf = cf_for(y2)
    seg = get_segments(x)
    if DBG: print(f'[img {ii}] seg={len(np.unique(seg))} -> occlusion', flush=True)
    labs, d1, d2 = segment_model_delta(x, y1, y2, seg); conf_ratio.append(regime_ratio(d1, d2))
    disc = _topseg(np.abs(d1 - d2)); region_mask = np.isin(seg, labs[disc]).astype(float)
    yr1, yr2 = _rng.choice(1000, 2, replace=False)
    _, rd1, rd2 = segment_model_delta(x, int(yr1), int(yr2), seg); rand_ratio.append(regime_ratio(rd1, rd2))
    if DBG: print(f'[img {ii}] occlusion done -> methods', flush=True)
    mdel, gcs = {}, {}
    for m in METHODS:
        if DBG: print(f'[img {ii}]   {m}', flush=True)
        try:
            A1 = attr_for(m, x, y1, x_cf).detach().cpu().numpy(); A2 = attr_for(m, x, y2, x_cf).detach().cpu().numpy()
            mdel[m] = segment_method_delta(A1, A2, seg, labs); gcs[m] = cs_struct_gated(A1, A2, region_mask)
        except Exception as e:
            fail[m] += 1; mdel[m] = (np.zeros(len(labs)), np.zeros(len(labs))); gcs[m] = np.nan
            if fail[m] == 1: print(f'  [warn] {m} failed: {type(e).__name__}: {str(e)[:90]}')
    store.append(dict(idx=i, x=x.cpu(), seg=seg, labels=labs, y1=y1, y2=y2,
                      model_d1=d1, model_d2=d2, method_deltas=mdel, gated_cs=gcs))
    del x, x_cf; gc.collect(); torch.cuda.empty_cache()      # free GPU between images
    if (ii + 1) % 25 == 0:                                   # crash-safe checkpoint
        pickle.dump(store, open(OUT/'segment_store_vit.pkl', 'wb')); print(f'[ckpt] {ii+1} saved', flush=True)

conf_ratio, rand_ratio = np.array(conf_ratio), np.array(rand_ratio)
thr = rand_ratio.mean() + rand_ratio.std(); nsp = int((conf_ratio > thr).sum())
print('\n' + '='*50)
print(f'confusable ratio: mean={conf_ratio.mean():.3f}  random: mean={rand_ratio.mean():.3f}')
print(f'SPATIAL (ratio>{thr:.3f}): {nsp}/{len(conf_ratio)} ({100*nsp/len(conf_ratio):.0f}%)')
try: p = wilcoxon(conf_ratio, rand_ratio).pvalue
except Exception: p = float('nan')
print(f'Wilcoxon confusable vs random: p={p:.2e}')
if any(fail.values()): print('[fail counts]', {m: n for m, n in fail.items() if n})
pickle.dump(store, open(OUT/'segment_store_vit.pkl', 'wb'))
print(f'saved {OUT}/segment_store_vit.pkl  ({len(store)} images)', flush=True)
print('DONE — emit table with:  python emit_vit_table.py', flush=True)
# table emitted by emit_vit_table.py (separate, no-CUDA → avoids the at-exit teardown segfault)
