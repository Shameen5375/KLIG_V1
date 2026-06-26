"""
Day 1 — Segment-level contrastive occlusion (Felzenszwalb superpixels, pure NumPy — no skimage).
Per segment: (drop_y1 - drop_y2) = which SEGMENT flips y1<->y2. Robust to pixel noise.
Produces the spatial/featural split (+ random control) and per-segment Δ for model + each method.
Run:  .venv/Scripts/python segment_occlusion.py [N]   (default 100)
"""
import sys, pickle, warnings
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.stats import wilcoxon
warnings.filterwarnings('ignore')

N = int(sys.argv[1]) if len(sys.argv) > 1 else -1     # -1 = ALL images in the cache
FZ_SCALE, FZ_SIGMA, FZ_MINSIZE, SEED, EPS = 0.6, 0.8, 100, 0, 1e-8
OUT = Path('cs_viz_outputs'); OUT.mkdir(exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED); np.random.seed(SEED)
print(f'[setup] device={DEVICE}  N={N}')

import klig_methods as KM
from klig_methods import attr_map, METHODS as _ALL, make_phi
from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(DEVICE).eval()
labels = ResNet50_Weights.IMAGENET1K_V2.meta['categories']
phi = make_phi(model)
KM.N_STEPS, KM.N_SAMPLES = 25, 3; KM.IG_STEPS = 25; KM.SG_SAMPLES = 25; KM.EG_SAMPLES = 25
KM.T_DESC, KM.N_MC_DESC, KM.N_MC_GRAD = 25, 8, 3
_rng = np.random.default_rng(SEED)
def attr_for(m, x1, cls, x_cf):
    H, W = x1.shape[-2], x1.shape[-1]
    if m == 'Random': return torch.from_numpy(_rng.standard_normal((H, W))).float()
    return attr_map(m, model, x1, int(cls), x_cf=x_cf, phi=phi)

# pool + cf
srcs = ['cs_viz_cache/cands.pkl','klig2_dist_cache/klig2_dist_multiprob.pkl',
        'klig2_val_cache/klig2_dist_multiprob.pkl','cs_gate_cache/pool.pkl']
CANDS, seen = [], set()
for s in srcs:
    if not Path(s).exists(): continue
    for d in pickle.load(open(s,'rb')):
        if len(d.get('high_cls',[]))<2: continue
        x=d['x']; x=x.squeeze(0) if x.dim()==4 else x
        fp=round(float(x.float().sum()),1)
        if fp in seen: continue
        seen.add(fp); CANDS.append({'idx':len(CANDS),'x':x.cpu(),'high_cls':[int(c) for c in d['high_cls'][:2]]})
# optional 2nd arg = a pool .pkl (e.g. cs_viz_cache/pool1000.pkl from build_pool_1000.py).
# That pool is an already-random stream → use it directly, NO dedup.
POOL_FILE = sys.argv[2] if len(sys.argv) > 2 else None
NO_DEDUP = False
if POOL_FILE and Path(POOL_FILE).exists():
    _p = pickle.load(open(POOL_FILE, 'rb'))
    CANDS = [{'idx': i, 'x': (d['x'].squeeze(0) if d['x'].dim() == 4 else d['x']).cpu(),
              'high_cls': [int(c) for c in d['high_cls'][:2]]} for i, d in enumerate(_p)]
    NO_DEDUP = True
    print(f'[pool] using {POOL_FILE}: {len(CANDS)} random images (no dedup)')

def pick(n):
    if NO_DEDUP: return CANDS[:n]                 # random stream already → take all
    import random as _r; pool = list(CANDS); _r.Random(SEED).shuffle(pool)   # shuffle → not animal-front-loaded
    sel, used = [], set()
    for d in pool:                               # dedup by top-1 class → diverse spread
        c = int(d['high_cls'][0])
        if c in used: continue
        used.add(c); sel.append(d)
        if len(sel) >= n: break
    return sel
def cf_for(sel):
    need={int(d['high_cls'][1]) for d in sel}; cf={}
    for d in CANDS:
        for c in (int(d['high_cls'][0]),int(d['high_cls'][1])):
            if c in need and c not in cf: cf[c]=d['x'].to(DEVICE)
    fb=CANDS[0]['x'].to(DEVICE)
    for c in need-set(cf): cf[c]=fb
    return cf

# ── Felzenszwalb (pure NumPy) ────────────────────────────────────────────────
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
def segment_model_delta(x, y1, y2, seg, chunk=64):
    base = F.softmax(model(x.unsqueeze(0))[0], 0); b1, b2 = base[y1].item(), base[y2].item()
    labs = np.unique(seg); xb = x.unsqueeze(0).repeat(len(labs), 1, 1, 1).clone()
    for k, lab in enumerate(labs):
        xb[k][:, torch.from_numpy(seg == lab).to(x.device)] = 0
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
    # pixel CS_struct of the method's class-difference, GATED to the model's discriminative region.
    # coherent difference there survives the blur (high); scatter dies (low). Noise-robust.
    D = (A1 - A2).astype(float) * mask; D = D / (np.abs(D).max() + EPS)
    coh = gaussian_filter(D, sigma); return float((coh ** 2).sum() / ((D ** 2).sum() + EPS))

def regime_ratio(d1, d2):
    return float(((d1-d2)**2).sum() / (((d1+d2)**2).sum() + EPS))

METHODS = list(_ALL)              # full 11-method roster (no Random)
if N < 0 or N > len(CANDS): N = len(CANDS)       # default: all cached images
print(f'[run] {N} images × {len(METHODS)} methods')
sel = pick(N); cf = cf_for(sel)
conf_ratio, rand_ratio, store = [], [], []
from tqdm import tqdm
for d in tqdm(sel, desc='segment occlusion'):
    x = d['x'].squeeze(0).to(DEVICE); y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
    x_cf = cf.get(y2); x_cf = (x_cf.squeeze(0) if x_cf.dim() == 4 else x_cf).to(DEVICE)
    seg = get_segments(x)
    labs, d1, d2 = segment_model_delta(x, y1, y2, seg); conf_ratio.append(regime_ratio(d1, d2))
    disc = _topseg(np.abs(d1 - d2))                                  # model's discriminative segments
    region_mask = np.isin(seg, labs[disc]).astype(float)            # → pixel region (for gated CS_struct)
    yr1, yr2 = _rng.choice(1000, 2, replace=False)
    _, rd1, rd2 = segment_model_delta(x, int(yr1), int(yr2), seg); rand_ratio.append(regime_ratio(rd1, rd2))
    mdel, gcs = {}, {}
    for m in METHODS:
        A1 = attr_for(m, x, y1, x_cf).detach().cpu().numpy(); A2 = attr_for(m, x, y2, x_cf).detach().cpu().numpy()
        mdel[m] = segment_method_delta(A1, A2, seg, labs)
        gcs[m] = cs_struct_gated(A1, A2, region_mask)               # pixel CS_struct within the region
    store.append(dict(idx=d['idx'], x=x.cpu(), seg=seg, labels=labs, y1=y1, y2=y2,
                      model_d1=d1, model_d2=d2, method_deltas=mdel, gated_cs=gcs))

conf_ratio, rand_ratio = np.array(conf_ratio), np.array(rand_ratio)
thr = rand_ratio.mean() + rand_ratio.std()
nsp = int((conf_ratio > thr).sum())
print('\n' + '='*50)
print(f'confusable ratio: mean={conf_ratio.mean():.3f} median={np.median(conf_ratio):.3f}')
print(f'random     ratio: mean={rand_ratio.mean():.3f} median={np.median(rand_ratio):.3f}')
print(f'SPATIAL (ratio>{thr:.3f}): {nsp}/{len(conf_ratio)} ({100*nsp/len(conf_ratio):.0f}%) | '
      f'FEATURAL {len(conf_ratio)-nsp}')
try: p = wilcoxon(conf_ratio, rand_ratio).pvalue
except Exception: p = float('nan')
print(f'Wilcoxon confusable vs random: p={p:.2e}  → '
      + ('SEPARATES ✓ (segment test valid)' if p < 0.05 else 'CANNOT separate — STOP'))
pickle.dump(store, open(OUT/'segment_store.pkl', 'wb'))
print(f'\nsaved {OUT}/segment_store.pkl  ({len(store)} images, for Day 2 rendering)')

# ── auto-emit the method-comparison table (correctness + class-sensitivity ± SE) ──
try:
    import pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    PO = ['Vanilla Grad','SmoothGrad','IG-zero','Blur-IG','IDG','Guided IG','ExpGrad',
          'KLIG-Adaptive','KL-IG (linear)','KL-IG²','KL-IG² (adaptive)']
    tmethods = [m for m in PO if m in METHODS] + [m for m in METHODS if m not in PO]
    FM, FMeth, FSens = 0.25, 0.05, 0.15
    def _ts(v, f):
        v = np.asarray(v, float)
        return np.zeros(len(v), bool) if np.ptp(v) < 1e-12 else (v >= np.quantile(v, 1 - f))
    DR_FRAC = 0.25                                          # land+differ region size (stable region)
    def _prec(mv, dv): mt = _ts(mv, FMeth); mr = _ts(dv, FM); return (mt & mr).sum()/mt.sum() if mt.sum() else np.nan
    def _precf(ma, mo): return (ma & mo).sum() / (ma.sum() + 1e-12)     # precision: method area inside model area
    def _dr(a1, a2, d1, d2):                                # land + differ (precision, signed model region)
        D1, D2 = _ts(d1, DR_FRAC), _ts(d2, DR_FRAC); A1, A2 = _ts(np.abs(a1), DR_FRAC), _ts(np.abs(a2), DR_FRAC)
        return ((_precf(A1, D1) - _precf(A1, D2)) + (_precf(A2, D2) - _precf(A2, D1))) / 2
    def _mse(v): v = np.asarray(v, float); v = v[~np.isnan(v)]; return v.mean(), v.std()/np.sqrt(len(v))
    trows = []
    for m in tmethods:
        pc, dr, gc = [], [], []
        for R in store:
            a, b = R['method_deltas'][m]
            pc += [_prec(a, R['model_d1']), _prec(b, R['model_d2'])]
            dr.append(_dr(a, b, R['model_d1'], R['model_d2']))
            gc.append(R['gated_cs'][m])                            # pixel CS_struct within model region
        cm, cse = _mse(pc); dm, dse = _mse(dr); gm, gse = _mse(gc)
        trows.append(dict(method=m, correctness=cm, c_se=cse, differ_and_right=dm, dr_se=dse,
                          gated_CSstruct=gm, gc_se=gse))
    # null controls for differ_and_right (must floor): Random ≈ 0, class-blind = 0
    _rc = np.random.default_rng(0); rdr, cbdr = [], []
    _cbm = 'KL-IG² (adaptive)' if 'KL-IG² (adaptive)' in tmethods else tmethods[0]
    for R in store:
        ns = len(R['labels'])
        rdr.append(_dr(_rc.random(ns), _rc.random(ns), R['model_d1'], R['model_d2']))
        cb = R['method_deltas'][_cbm][0]; cbdr.append(_dr(cb, cb, R['model_d1'], R['model_d2']))
    print(f'[controls] differ_and_right  Random={np.nanmean(rdr):+.3f}±{np.nanstd(rdr)/np.sqrt(len(rdr)):.3f}  '
          f'ClassBlind={np.nanmean(cbdr):+.3f}  (both should ≈ 0)')
    tdf = pd.DataFrame(trows).sort_values('gated_CSstruct', ascending=False).reset_index(drop=True)
    tdf.to_csv(OUT/'segment_metric_table.csv', index=False)
    r2 = lambda v: (int(np.argsort(-np.asarray(v))[0]), int(np.argsort(-np.asarray(v))[1]))
    idx = {1: r2(tdf.correctness.values), 2: r2(tdf.differ_and_right.values), 3: r2(tdf.gated_CSstruct.values)}
    fig, ax = plt.subplots(figsize=(11, 0.5*len(tdf)+1.2), facecolor='white'); ax.axis('off')
    cells = [[r.method, f'{r.correctness:.3f} ± {r.c_se:.3f}', f'{r.differ_and_right:+.3f} ± {r.dr_se:.3f}',
              f'{r.gated_CSstruct:.3f} ± {r.gc_se:.3f}'] for r in tdf.itertuples()]
    tb = ax.table(cellText=cells, colLabels=['method', 'correctness', 'differ_and_right', 'gated CS_struct'],
                  cellLoc='center', loc='center')
    tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.5)
    for j in range(4): tb[0, j].set_facecolor('#34495e'); tb[0, j].set_text_props(color='white', fontweight='bold')
    for col, (i1, i2) in idx.items():
        tb[(i1+1, col)].set_text_props(fontweight='bold'); tb[(i1+1, col)].set_facecolor('#cfe3f5')
        tb[(i2+1, col)].set_text_props(fontstyle='italic'); tb[(i2+1, col)].set_facecolor('#eaf3fb')
    plt.title(f'Segment-level method comparison (n={len(store)}, ±SE)\ncorrectness=precision(5%↪25%) · '
              f'differ_and_right=land+differ(FRAC={DR_FRAC}) · gated CS_struct=pixel coherence in model region\n'
              'bold=highest  italic=2nd  ·  sorted by gated CS_struct', fontsize=9, fontweight='bold', pad=12)
    plt.tight_layout(); plt.savefig(OUT/'segment_metric_table.png', dpi=200, bbox_inches='tight'); plt.close()
    print(f'saved {OUT}/segment_metric_table.png  (n={len(store)}, headline=gated CS_struct)')
except Exception as e:
    print(f'[warn] table emit failed: {e}')
