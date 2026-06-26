"""
Dissect how class sensitivity occurs — per-image, per-regime.  CHARACTERIZATION, not ranking.
  Part 1: model regime distribution (spatial vs featural, contrastive occlusion).
  Part 2: per-method alignment(M_delta, GT_delta) stratified by regime (tracks / imposes / misses).
  Part 3: clean vs messy (spread of method alignments) vs regime_ratio / semantic-dist / salience.
  Ladder: GT-oracle=1, Random~0, class-blind~0, Shuffle~0 (must hold).
  Framings: primary M=A_y1-A_y2 (all methods); secondary native CF-relative A_y1^{cf=y2} (KL-IG² family).
Run:  .venv/Scripts/python contrastive_dissect.py [N]   (default 100; use 10 for smoke)
"""
import sys, pickle, warnings
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
PATCH, STRIDE, CHUNK, SEED, EPS, THR = 32, 16, 64, 0, 1e-8, 0.63
OUT = Path('cs_viz_outputs'); OUT.mkdir(exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED); np.random.seed(SEED)
print(f'[setup] device={DEVICE}  N={N}')

import klig_methods as KM
from klig_methods import attr_map, METHODS, make_phi
from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(DEVICE).eval()
labels = ResNet50_Weights.IMAGENET1K_V2.meta['categories']
phi = make_phi(model)
KM.N_STEPS, KM.N_SAMPLES = 25, 3; KM.IG_STEPS = 25; KM.SG_SAMPLES = 25; KM.EG_SAMPLES = 25
KM.T_DESC, KM.N_MC_DESC, KM.N_MC_GRAD = 25, 8, 3
ROSTER = list(METHODS) + ['Random']; KLIG2A = 'KL-IG² (adaptive)'; KLIG2F = 'KL-IG²'
CF_FAMILY = [KLIG2A, KLIG2F]
VIZ_METHODS = [KLIG2A, 'Blur-IG', 'Vanilla Grad']
_rng = np.random.default_rng(SEED)
_MEAN = torch.tensor([0.485,0.456,0.406]).view(3,1,1); _STD = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
def denorm(t): return (t.detach().cpu()*_STD+_MEAN).clamp(0,1)
def attr_for(m, x1, cls, x_cf):
    H, W = x1.shape[-2], x1.shape[-1]
    if m == 'Random': return torch.from_numpy(_rng.standard_normal((H, W))).float()
    return attr_map(m, model, x1, int(cls), x_cf=x_cf, phi=phi)
def npm(A): return (A.detach().cpu().numpy() if torch.is_tensor(A) else np.asarray(A)).astype(float)

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
def pick(n):
    import random as _r; pool=list(CANDS); _r.Random(SEED).shuffle(pool); sel,used=[],set()
    for d in pool:
        c=int(d['high_cls'][0])
        if c in used: continue
        used.add(c); sel.append(d)
        if len(sel)>=n: break
    return sel
def cf_for(sel):
    need={int(d['high_cls'][1]) for d in sel}; cf={}
    for d in CANDS:
        for c in (int(d['high_cls'][0]),int(d['high_cls'][1])):
            if c in need and c not in cf: cf[c]=d['x'].to(DEVICE)
    fb=CANDS[0]['x'].to(DEVICE)
    for c in need-set(cf): cf[c]=fb
    return cf

# CLIP semantic distance (Part 3)
try:
    from transformers import CLIPModel, CLIPTokenizerFast
    _clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(DEVICE).eval()
    _tok = CLIPTokenizerFast.from_pretrained('openai/clip-vit-base-patch32')
    _emb_cache = {}
    @torch.no_grad()
    def _emb(c):
        if c not in _emb_cache:
            t = _tok([f'a photo of a {labels[c].split(",")[0]}'], return_tensors='pt',
                     padding=True, truncation=True, max_length=77).to(DEVICE)
            e = _clip.text_model(**t).pooler_output
            _emb_cache[c] = F.normalize(_clip.text_projection(e), dim=-1)[0].cpu().numpy()
        return _emb_cache[c]
    def sem_dist(y1, y2): return float(1 - _emb(y1) @ _emb(y2))
    HAVE_CLIP = True
except Exception as e:
    print(f'[warn] CLIP unavailable ({e}); skipping semantic-distance in Part 3'); HAVE_CLIP = False

COORDS = [(i, j) for i in range(0, 224-PATCH+1, STRIDE) for j in range(0, 224-PATCH+1, STRIDE)]
GS = int(len(COORDS) ** 0.5)
@torch.no_grad()
def gt_grids(x, y1, y2):
    base = F.softmax(model(x.unsqueeze(0))[0], -1); b1, b2 = float(base[y1]), float(base[y2])
    dl = np.zeros(len(COORDS)); sh = np.zeros(len(COORDS))
    for k in range(0, len(COORDS), CHUNK):
        bc = COORDS[k:k+CHUNK]; xb = x.unsqueeze(0).repeat(len(bc),1,1,1).clone()
        for b,(i,j) in enumerate(bc): xb[b,:,i:i+PATCH,j:j+PATCH]=0
        p = F.softmax(model(xb), -1); d1=(b1-p[:,y1]).cpu().numpy(); d2=(b2-p[:,y2]).cpu().numpy()
        for b in range(len(bc)): dl[k+b]=d1[b]-d2[b]; sh[k+b]=d1[b]+d2[b]
    return dl, sh
def to_grid(M): return np.array([M[i:i+PATCH, j:j+PATCH].mean() for (i,j) in COORDS])
def pear(a, b):
    a=a-a.mean(); b=b-b.mean(); d=np.linalg.norm(a)*np.linalg.norm(b)
    return 0.0 if d<1e-12 else float(a@b/d)

sel = pick(N); cf = cf_for(sel)
recs = []; VIZ = {}            # per-image record dicts; VIZ stores grids for a subset
from tqdm import tqdm
for d in tqdm(sel, desc='dissect'):
    x = d['x'].squeeze(0).to(DEVICE); y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
    x_cf = cf.get(y2); x_cf = (x_cf.squeeze(0) if x_cf.dim()==4 else x_cf).to(DEVICE)
    GTd, GTs = gt_grids(x, y1, y2)
    ratio = float((GTd**2).sum()/((GTs**2).sum()+EPS)); reg = 'spatial' if ratio > THR else 'featural'
    rec = {'idx': d['idx'], 'y1': y1, 'y2': y2, 'y1_label': labels[y1].split(',')[0],
           'regime_ratio': ratio, 'regime': reg, 'salience': float((GTs**2).sum())}
    if HAVE_CLIP: rec['sem_dist'] = sem_dist(y1, y2)
    A = {m: (npm(attr_for(m, x, y1, x_cf)), npm(attr_for(m, x, y2, x_cf))) for m in ROSTER}
    aG = np.abs(GTd)
    for m in ROSTER:
        mdg = to_grid(A[m][0] - A[m][1])                                     # primary M=A_y1-A_y2
        rec['align_' + m] = pear(mdg, GTd)                                   # signed (direction matters)
        rec['signblind_' + m] = pear(np.abs(mdg), aG)                        # location only (sign-blind)
    for m in CF_FAMILY:                                                       # secondary native (A_y1 vs y2-CF)
        nmg = to_grid(A[m][0])
        rec['native_' + m] = pear(nmg, GTd)
        rec['native_signblind_' + m] = pear(np.abs(nmg), aG)
    rec['align_GT-oracle'] = pear(GTd, GTd); rec['align_class-blind'] = pear(np.zeros_like(GTd), GTd)
    rec['align_Shuffle'] = pear(_rng.permutation(GTd), GTd)
    recs.append(rec)
    VIZ[d['idx']] = {'GT': GTd, **{m: to_grid(A[m][0]-A[m][1]) for m in VIZ_METHODS}}
df = pd.DataFrame(recs); df.to_csv(OUT/'dissect_per_image.csv', index=False)

# ── PART 1 ───────────────────────────────────────────────────────────────────
nsp = int((df.regime=='spatial').sum()); nft = len(df)-nsp
print('\n' + '='*60 + f'\nPART 1 — model regime (N={len(df)})')
print(f'  SPATIAL {nsp} ({100*nsp/len(df):.0f}%) | FEATURAL {nft} ({100*nft/len(df):.0f}%) '
      f'| regime_ratio median={df.regime_ratio.median():.2f}')

# ── PART 2 ───────────────────────────────────────────────────────────────────
print('\n' + '='*60 + '\nPART 2 — per-method alignment by regime (characterization)')
sp, ft = df[df.regime=='spatial'], df[df.regime=='featural']
def interp(a_sp, a_ft):
    if a_sp > 0.1 and abs(a_ft) < 0.1: return 'TRACKS (aligns on spatial, flat on featural)'
    if a_sp > 0.1 and abs(a_ft) > 0.1: return 'IMPOSES (structured even on featural)'
    if abs(a_sp) < 0.1: return 'MISSES (no alignment even on spatial)'
    return 'mixed'
rows = []
for m in ROSTER:
    a_sp = sp['align_'+m].mean(); a_ft = ft['align_'+m].mean() if len(ft) else float('nan')
    rows.append({'method': m, 'align_spatial': a_sp, 'align_featural': a_ft,
                 'signblind_spatial': sp['signblind_'+m].mean(),
                 'gap': a_sp - (a_ft if len(ft) else 0), 'interpretation': interp(a_sp, a_ft)})
p2 = pd.DataFrame(rows).sort_values('align_spatial', ascending=False).reset_index(drop=True)
p2.to_csv(OUT/'dissect_part2_by_regime.csv', index=False)
print(p2.round(3).to_string(index=False))

print('\nDECISIVE — KL-IG² sign-blind location test (spatial pairs):')
for m in CF_FAMILY:
    s = sp['align_'+m].mean(); sb = sp['signblind_'+m].mean()
    ns = sp['native_'+m].mean(); nsb = sp['native_signblind_'+m].mean()
    if sb > 0.10 and s < -0.03: v = 'RIGHT LOCATION, INVERTED SIGN (convention flip — not a faithfulness miss)'
    elif sb < 0.10:             v = 'GENUINE MISS (low even sign-blind → wrong location)'
    else:                       v = 'mixed / weak'
    print(f'  {m:20s} signed={s:+.3f}  sign-blind={sb:+.3f}  |  native signed={ns:+.3f}  '
          f'sign-blind={nsb:+.3f}\n      → {v}')
# reference: a positively-aligned method should have signed ≈ sign-blind (both high)
_ref = 'KL-IG (linear)'
print(f'  [ref] {_ref:18s} signed={sp["align_"+_ref].mean():+.3f}  sign-blind={sp["signblind_"+_ref].mean():+.3f}'
      '  (positive aligner: signed≈sign-blind)')

# ── PART 3 ───────────────────────────────────────────────────────────────────
print('\n' + '='*60 + '\nPART 3 — clean vs messy class sensitivity')
real = [m for m in ROSTER if m != 'Random']
df['align_spread'] = df[['align_'+m for m in real]].std(axis=1)
med = df['align_spread'].median()
df['cleanliness'] = np.where(df['align_spread'] <= med, 'clean', 'messy')
print(f'  spread median={med:.3f}; clean={int((df.cleanliness=="clean").sum())} '
      f'messy={int((df.cleanliness=="messy").sum())}  (low spread=methods agree=clean)')
for col, name in [('regime_ratio','regime_ratio'), ('salience','salience')] + ([('sem_dist','sem_dist')] if HAVE_CLIP else []):
    rho, pv = spearmanr(df['align_spread'], df[col])
    print(f'  spread vs {name:12s}: Spearman rho={rho:+.3f}  p={pv:.2g}')
df.to_csv(OUT/'dissect_per_image.csv', index=False)

# ── LADDER ───────────────────────────────────────────────────────────────────
print('\n' + '='*60 + '\nVALIDITY LADDER')
for k in ['GT-oracle','Random','class-blind','Shuffle']:
    print(f'  {k:12s} mean align = {df["align_"+k].mean():+.3f}')
ok = df['align_GT-oracle'].mean()>0.95 and abs(df['align_Random'].mean())<0.1 \
     and abs(df['align_Shuffle'].mean())<0.1 and abs(df['align_class-blind'].mean())<0.1
print('  →', 'LADDER HOLDS' if ok else 'LADDER ISSUE')

# ── VIZ: 6 images (3 spatial highest-ratio + 3 featural lowest-ratio) ─────────
pick_sp = df[df.regime=='spatial'].nlargest(3, 'regime_ratio')['idx'].tolist()
pick_ft = df[df.regime=='featural'].nsmallest(3, 'regime_ratio')['idx'].tolist()
vids = pick_sp + pick_ft
ncol = 2 + len(VIZ_METHODS)
fig, ax = plt.subplots(len(vids), ncol, figsize=(2.6*ncol, 2.7*len(vids)), facecolor='white', squeeze=False)
byidx = {d['idx']: d for d in sel}
for r, iid in enumerate(vids):
    rec = df[df.idx==iid].iloc[0]; d = byidx[iid]; x = d['x'].squeeze(0)
    G = VIZ[iid]['GT']; vg = max(abs(G).max(), 1e-9)
    ax[r,0].imshow(denorm(x).permute(1,2,0).numpy()); ax[r,0].axis('off')
    ax[r,1].imshow(G.reshape(GS,GS), cmap='bwr', vmin=-vg, vmax=vg); ax[r,1].axis('off')
    if r==0: ax[r,0].set_title('image', fontsize=9); ax[r,1].set_title('GT Δ (model)', fontsize=9)
    for c, m in enumerate(VIZ_METHODS):
        Mg = VIZ[iid][m]; vm = max(abs(Mg).max(), 1e-9)
        ax[r,2+c].imshow(Mg.reshape(GS,GS), cmap='bwr', vmin=-vm, vmax=vm); ax[r,2+c].axis('off')
        if r==0: ax[r,2+c].set_title(m, fontsize=8)
        ax[r,2+c].set_xlabel(f"r={rec['align_'+m]:+.2f}", fontsize=8)
    ax[r,0].axis('on'); ax[r,0].set_xticks([]); ax[r,0].set_yticks([])
    for sp_ in ax[r,0].spines.values(): sp_.set_visible(False)
    ax[r,0].set_ylabel(f"{rec['regime']}\nratio={rec['regime_ratio']:.2f}", fontsize=8, fontweight='bold',
                       color='#1a5fb4' if rec['regime']=='spatial' else '#a51d2d')
fig.suptitle('Dissection: model GT Δ vs method Δ (red=y1, blue=y2). r = alignment.',
             fontsize=12, fontweight='bold', y=1.0)
plt.tight_layout(); plt.savefig(OUT/'dissect_images.png', dpi=150, bbox_inches='tight'); plt.close()
print('\nsaved: dissect_per_image.csv, dissect_part2_by_regime.csv, dissect_images.png')
