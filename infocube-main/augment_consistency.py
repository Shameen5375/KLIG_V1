"""Augmentation-consistency check for gated CS_struct (COSE-style consistency-under-augmentation).

Claim: under label-PRESERVING transforms (flip, small rotate, brightness/contrast, crop) the
gated CS_struct score barely drifts (it is stable, not fragile); under a label-CHANGING change
(swap the confused class to a random one) it drifts much more (it is sensitive to the class pair).
Stable-for-preserving + moves-for-changing = the consistency/sensitivity balance COSE argues for.

Reuses the EXACT gated CS_struct machinery from segment_occlusion.py (Felzenszwalb segments,
segment occlusion delta, top-25% discriminative region, pixel coherence in region).
Run:  .venv/Scripts/python augment_consistency.py [n]   (default 60)
"""
import sys, pickle, warnings
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import torchvision.transforms.functional as TF
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr, wilcoxon
warnings.filterwarnings('ignore')

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
FZ_SCALE, FZ_SIGMA, FZ_MINSIZE, SEED, EPS = 0.6, 0.8, 100, 0, 1e-8
OUT = Path('cs_viz_outputs'); OUT.mkdir(exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED); np.random.seed(SEED)
HEAD = 'KL-IG² (adaptive)'                                    # headline method for the stability story

import klig_methods as KM
from klig_methods import attr_map, METHODS as _ALL, make_phi
from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(DEVICE).eval()
phi = make_phi(model)
KM.N_STEPS, KM.N_SAMPLES = 25, 3; KM.IG_STEPS = 25; KM.SG_SAMPLES = 25; KM.EG_SAMPLES = 25
KM.T_DESC, KM.N_MC_DESC, KM.N_MC_GRAD = 25, 8, 3
_rng = np.random.default_rng(SEED)
_mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1); _std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

# ── EXACT gated CS_struct machinery (copied verbatim from segment_occlusion.py) ──────────
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

def _topseg(v, f=0.25):
    v = np.asarray(v, float)
    return v >= np.quantile(v, 1 - f) if np.ptp(v) > EPS else np.zeros(len(v), bool)
def cs_struct_gated(A1, A2, mask, sigma=4):
    D = (A1 - A2).astype(float) * mask; D = D / (np.abs(D).max() + EPS)
    coh = gaussian_filter(D, sigma); return float((coh ** 2).sum() / ((D ** 2).sum() + EPS))

def attr_for(m, x1, cls, x_cf):
    return attr_map(m, model, x1, int(cls), x_cf=x_cf, phi=phi)

def region_for(x, y1, y2):
    """Segments + occlusion + top-25% discriminative region — computed ONCE, reused across methods."""
    seg = get_segments(x); labs, d1, d2 = segment_model_delta(x, y1, y2, seg)
    return np.isin(seg, labs[_topseg(np.abs(d1 - d2))]).astype(float)

def cs_method(x, y1, y2, x_cf, region, m=HEAD):
    """gated CS_struct for method m, reusing a precomputed region."""
    A1 = attr_for(m, x, y1, x_cf).detach().cpu().numpy()
    A2 = attr_for(m, x, y2, x_cf).detach().cpu().numpy()
    return cs_struct_gated(A1, A2, region)

# ── label-preserving transforms (image-space; renormalize for photometric) ──────────────
def denorm(x): return (x*_std.to(x.device)+_mean.to(x.device)).clamp(0,1)
def renorm(i): return (i - _mean.to(i.device)) / _std.to(i.device)
TRANSFORMS = {
    'hflip':        lambda x: torch.flip(x, [-1]),
    'rotate+10':    lambda x: TF.rotate(x, 10, fill=0.0),
    'brightness1.2':lambda x: renorm(TF.adjust_brightness(denorm(x), 1.2)),
    'contrast1.3':  lambda x: renorm(TF.adjust_contrast(denorm(x), 1.3)),
    'crop0.85':     lambda x: TF.resized_crop(x, 17, 17, 190, 190, [224, 224], antialias=True),
}

# ── pool of confused pairs (same sources as segment_occlusion) ──────────────────────────
srcs = ['cs_viz_cache/cands.pkl','klig2_dist_cache/klig2_dist_multiprob.pkl',
        'klig2_val_cache/klig2_dist_multiprob.pkl','cs_gate_cache/pool.pkl']
CANDS, seen = [], set()
for s in srcs:
    if not Path(s).exists(): continue
    for d in pickle.load(open(s,'rb')):
        if len(d.get('high_cls',[]))<2: continue
        x=d['x']; x=x.squeeze(0) if x.dim()==4 else x; fp=round(float(x.float().sum()),1)
        if fp in seen: continue
        seen.add(fp); CANDS.append({'x':x.cpu(),'high_cls':[int(c) for c in d['high_cls'][:2]]})
import random as _r; _r.Random(SEED).shuffle(CANDS)
sel, used = [], set()
for d in CANDS:                                              # dedup by top-1 -> diverse
    c = int(d['high_cls'][0])
    if c in used: continue
    used.add(c); sel.append(d)
    if len(sel) >= N: break
cf_by = {}
for d in CANDS: cf_by.setdefault(int(d['high_cls'][0]), d['x'])
def cf_for(y2): return (cf_by.get(y2, CANDS[0]['x'])).to(DEVICE)
print(f'[setup] {len(sel)} confused-pair images | device={DEVICE} | headline={HEAD}')

# ── run ─────────────────────────────────────────────────────────────────────────────────
CKPT = Path('cs_viz_cache/augment_consistency.pkl'); rows = []
if CKPT.exists(): rows = pickle.load(open(CKPT,'rb')); print(f'[resume] {len(rows)} cached')
from tqdm import tqdm
for i in tqdm(range(len(rows), len(sel)), desc='aug-consistency'):
    d = sel[i]; x0 = d['x'].to(DEVICE); y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
    xcf = cf_for(y2)
    top0 = int(model(x0.unsqueeze(0))[0].argmax())
    reg0 = region_for(x0, y1, y2)                             # region computed ONCE, reused by all methods
    base_all = {m: cs_method(x0, y1, y2, xcf, reg0, m) for m in _ALL}
    cs0 = base_all[HEAD]                                      # headline = its entry (no recompute)
    rec = dict(y1=y1, y2=y2, cs0=cs0, base_all=base_all, transforms={}, flip_all={})
    for name, t in TRANSFORMS.items():
        xt = t(x0)
        topt = int(model(xt.unsqueeze(0))[0].argmax())
        regt = region_for(xt, y1, y2)                         # one region per transformed image
        if name == 'hflip':
            rec['flip_all'] = {m: cs_method(xt, y1, y2, xcf, regt, m) for m in _ALL}
            cst = rec['flip_all'][HEAD]
        else:
            cst = cs_method(xt, y1, y2, xcf, regt, HEAD)
        rec['transforms'][name] = dict(cs=cst, pred_preserved=(topt == top0))
    # label-CHANGING control: swap y2 -> random class (sensitivity)
    yr = int(_rng.choice([c for c in range(1000) if c not in (y1, y2)]))
    csc = cs_method(x0, y1, yr, cf_for(yr), region_for(x0, y1, yr), HEAD)
    rec['control_swap'] = csc
    rows.append(rec)
    if (i+1) % 10 == 0: pickle.dump(rows, open(CKPT,'wb'))
pickle.dump(rows, open(CKPT,'wb'))

# ── aggregate (ABSOLUTE drift — CS_struct is already a bounded [0,1] ratio) ──────────────
def absdrift(a, b): return abs(a-b)
cs0 = np.array([r['cs0'] for r in rows]); n = len(rows)
print('\n' + '='*64)
print(f'headline gated CS_struct baseline: mean={cs0.mean():.3f} ± {cs0.std()/np.sqrt(n):.3f}  (n={n})')
print(f'{"transform":16s} {"mean CS":>9s} {"abs.drift":>10s} {"pred kept":>10s}')
drift_by, drift_pres = {}, {}                                # all cases / pred-preserved subset
for name in TRANSFORMS:
    cst = np.array([r['transforms'][name]['cs'] for r in rows])
    ad  = np.array([absdrift(r['transforms'][name]['cs'], r['cs0']) for r in rows])
    kept= np.array([r['transforms'][name]['pred_preserved'] for r in rows])
    drift_by[name] = ad; drift_pres[name] = ad[kept] if kept.any() else ad
    print(f'{name:16s} {cst.mean():9.3f} {ad.mean():10.3f} {100*kept.mean():9.0f}%')
ctrl = np.array([absdrift(r['control_swap'], r['cs0']) for r in rows]); drift_by['CTRL: class-swap'] = ctrl
pooled_pres = np.concatenate([drift_pres[k] for k in TRANSFORMS])   # pred-preserved only
print(f'{"CTRL class-swap":16s} {"":9s} {ctrl.mean():10.3f}  (label-CHANGING → should be larger)')
# paired per-image test: mean preserving abs-drift vs class-swap abs-drift
per_img_pres = np.array([np.mean([absdrift(r['transforms'][k]['cs'], r['cs0'])
                                  for k in TRANSFORMS if r['transforms'][k]['pred_preserved']] or
                                 [absdrift(r['transforms'][k]['cs'], r['cs0']) for k in TRANSFORMS])
                         for r in rows])
try: pv = wilcoxon(per_img_pres, ctrl).pvalue
except Exception: pv = float('nan')
print(f'preserving drift {pooled_pres.mean():.3f}  vs  class-swap drift {ctrl.mean():.3f}  '
      f'(ratio {ctrl.mean()/(pooled_pres.mean()+EPS):.1f}x)  paired Wilcoxon p={pv:.2e}')

# method ordering under flip
mb = np.array([[r['base_all'][m] for m in _ALL] for r in rows]).mean(0)
mf = np.array([[r['flip_all'][m] for m in _ALL] for r in rows]).mean(0)
rho = spearmanr(mb, mf).correlation
order_b = [ _ALL[k] for k in np.argsort(-mb) ]
print(f'\nmethod-ordering under hflip: Spearman(base, flip) = {rho:+.3f}  | top method base={order_b[0]}')

pickle.dump(dict(rows=rows, mb=mb, mf=mf, methods=list(_ALL)), open(OUT/'augment_consistency_summary.pkl','wb'))

# ── figure (2 panels) ───────────────────────────────────────────────────────────────────
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 5.2), facecolor='white')
# A: before vs after scatter (pooled preserving transforms) — tight to y=x = stable
cmap = plt.get_cmap('tab10')
for j, name in enumerate(TRANSFORMS):
    b = cs0; a = np.array([r['transforms'][name]['cs'] for r in rows])
    ax[0].scatter(b, a, s=18, alpha=0.6, color=cmap(j), label=name)
lim = [0, max(cs0.max(), 1.0)*1.05]
ax[0].plot(lim, lim, 'k--', lw=1, alpha=0.6); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].set_xlabel('CS_struct — original', fontsize=11); ax[0].set_ylabel('CS_struct — transformed', fontsize=11)
ax[0].set_title('Stability: transformed ≈ original (on y=x)', fontsize=12, fontweight='bold')
ax[0].legend(fontsize=9, loc='upper left'); ax[0].grid(alpha=0.3)
# B: absolute drift box — preserving (small) vs class-swap control (large)
keys = list(TRANSFORMS) + ['CTRL: class-swap']
data = [drift_by[k] for k in keys]
bp = ax[1].boxplot(data, vert=True, patch_artist=True, showfliers=False, widths=0.6)
for k, patch in enumerate(bp['boxes']): patch.set_facecolor('#b00020' if keys[k].startswith('CTRL') else '#4c72b0'); patch.set_alpha(0.7)
ax[1].set_xticklabels(keys, rotation=30, ha='right', fontsize=9)
ax[1].set_ylabel('absolute drift  |ΔCS|', fontsize=11)
ax[1].set_title(f'Label-preserving drift ≪ class-swap drift ({ctrl.mean()/(pooled_pres.mean()+EPS):.1f}×, '
                f'p={pv:.1e})', fontsize=12, fontweight='bold'); ax[1].grid(alpha=0.3, axis='y')
plt.suptitle(f'Gated CS_struct — augmentation-consistency check (n={n}, headline={HEAD}): '
             'stable under label-preserving transforms, sensitive to class change',
             fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.95]); out=OUT/'augment_consistency.png'
plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close(); print('saved', out)
