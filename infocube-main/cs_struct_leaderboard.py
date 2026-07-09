"""Gated CS_struct leaderboard — rank ALL 11 attribution methods by class-sensitivity at n=1000.
Per image: Felzenszwalb segments -> discriminative region R (top-25% by |d1-d2| occlusion) ->
per method gated CS_struct of the class-difference (A_y1 - A_y2). Checkpointed every 20 images.
Run:  .venv/Scripts/python cs_struct_leaderboard.py [n] [pool.pkl]   (default 1000 balanced)
"""
import sys, pickle, warnings
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
warnings.filterwarnings('ignore')
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
POOL = sys.argv[2] if len(sys.argv) > 2 else 'cs_viz_cache/pool1000_balanced.pkl'
TAG = '_balanced' if 'balanced' in POOL else ''
FZ_SCALE, FZ_SIGMA, FZ_MINSIZE, SEED, EPS, DR_FRAC = 0.6, 0.8, 100, 0, 1e-8, 0.25
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.manual_seed(SEED); np.random.seed(SEED)
import klig_methods as KM
from klig_methods import attr_map, METHODS as ALL, make_phi
from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(DEVICE).eval(); phi = make_phi(model)
cats = ResNet50_Weights.IMAGENET1K_V2.meta['categories']
KM.N_STEPS, KM.N_SAMPLES = 25, 3; KM.IG_STEPS = 25; KM.SG_SAMPLES = 25; KM.EG_SAMPLES = 25
KM.T_DESC, KM.N_MC_DESC, KM.N_MC_GRAD = 25, 8, 3

def get_segments(x, scale=FZ_SCALE, sigma=FZ_SIGMA, min_size=FZ_MINSIZE):
    img = x.cpu().numpy().transpose(1,2,0); img = (img-img.min())/(np.ptp(img)+EPS)
    H,W,_ = img.shape; im = gaussian_filter(img,(sigma,sigma,0)).reshape(-1,3); Npx=H*W
    idx = np.arange(Npx).reshape(H,W)
    A = np.concatenate([idx[:,:-1].ravel(), idx[:-1,:].ravel()]); B = np.concatenate([idx[:,1:].ravel(), idx[1:,:].ravel()])
    Wt = np.sqrt(((im[A]-im[B])**2).sum(1)); o=np.argsort(Wt); A,B,Wt=A[o],B[o],Wt[o]
    par=np.arange(Npx); rank=np.zeros(Npx,int); size=np.ones(Npx,int); intd=np.zeros(Npx); k=scale
    def find(z):
        r=z
        while par[r]!=r: r=par[r]
        while par[z]!=r: par[z],z=r,par[z]
        return r
    for a,b,w in zip(A.tolist(),B.tolist(),Wt.tolist()):
        ra,rb=find(a),find(b)
        if ra==rb: continue
        if w<=min(intd[ra]+k/size[ra], intd[rb]+k/size[rb]):
            if rank[ra]<rank[rb]: ra,rb=rb,ra
            par[rb]=ra; size[ra]+=size[rb]; intd[ra]=max(intd[ra],intd[rb],w)
            if rank[ra]==rank[rb]: rank[ra]+=1
    for a,b in zip(A.tolist(),B.tolist()):
        ra,rb=find(a),find(b)
        if ra!=rb and (size[ra]<min_size or size[rb]<min_size):
            if rank[ra]<rank[rb]: ra,rb=rb,ra
            par[rb]=ra; size[ra]+=size[rb]
    roots=np.array([find(i) for i in range(Npx)]); _,seg=np.unique(roots,return_inverse=True)
    return seg.reshape(H,W)

@torch.no_grad()
def region_of(x, y1, y2, seg, chunk=64):
    base=F.softmax(model(x.unsqueeze(0))[0],0); b1,b2=base[y1].item(),base[y2].item()
    labs=np.unique(seg); xb=x.unsqueeze(0).repeat(len(labs),1,1,1).clone()
    for k,lab in enumerate(labs): xb[k][:, torch.from_numpy(seg==lab).to(x.device)]=0
    d1=np.zeros(len(labs)); d2=np.zeros(len(labs))
    for s in range(0,len(labs),chunk):
        p=F.softmax(model(xb[s:s+chunk]),-1)
        d1[s:s+p.shape[0]]=b1-p[:,y1].cpu().numpy(); d2[s:s+p.shape[0]]=b2-p[:,y2].cpu().numpy()
    dd = np.abs(d1-d2); disc = dd >= np.quantile(dd, 1-DR_FRAC) if np.ptp(dd)>EPS else np.zeros(len(labs),bool)
    return np.isin(seg, labs[disc]).astype(float)

def cs_struct_gated(A1, A2, mask, sigma=4):
    D = (A1 - A2).astype(float) * mask; D = D / (np.abs(D).max() + EPS)
    coh = gaussian_filter(D, sigma); return float((coh**2).sum() / ((D**2).sum() + EPS))

pool = pickle.load(open(POOL,'rb'))[:N]
def img_of(d): x=d['x']; return (x.squeeze(0) if x.dim()==4 else x)
cf_by = {}
for d in pool: cf_by.setdefault(int(d['high_cls'][0]), img_of(d))
def cf_for(y2): return (cf_by.get(y2, img_of(pool[0]))).to(DEVICE)
print(f'[setup] CS_struct leaderboard | pool={POOL} {len(pool)} imgs × {len(ALL)} methods | {DEVICE}')

CKPT = Path(f'cs_viz_cache/cs_leaderboard{TAG}.pkl'); rows = []
if CKPT.exists(): rows = pickle.load(open(CKPT,'rb')); print(f'[resume] {len(rows)} cached')
from tqdm import tqdm
for i in tqdm(range(len(rows), len(pool)), desc='CS_struct leaderboard'):
    d = pool[i]; x = img_of(d).to(DEVICE); y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
    with torch.no_grad():
        if F.softmax(model(x.unsqueeze(0))[0],0)[y2] > F.softmax(model(x.unsqueeze(0))[0],0)[y1]: y1, y2 = y2, y1
    xcf = cf_for(y2); seg = get_segments(x); region = region_of(x, y1, y2, seg)
    cs = {}
    for m in ALL:
        try:
            A1 = attr_map(m, model, x, y1, x_cf=xcf, phi=phi).detach().cpu().numpy()
            A2 = attr_map(m, model, x, y2, x_cf=xcf, phi=phi).detach().cpu().numpy()
            cs[m] = cs_struct_gated(A1, A2, region)
        except Exception: cs[m] = np.nan
    rows.append(dict(idx=i, y1=y1, y2=y2, cs=cs))
    if (i+1) % 20 == 0: pickle.dump(rows, open(CKPT,'wb'))
pickle.dump(rows, open(CKPT,'wb')); rows = rows[:len(pool)]; n = len(rows)

# ── rank ─────────────────────────────────────────────────────────────────────────────────
def mse(v): v = np.asarray(v, float); v = v[~np.isnan(v)]; return v.mean(), v.std()/np.sqrt(max(len(v),1))
PO = ['Vanilla Grad','SmoothGrad','IG-zero','Blur-IG','IDG','Guided IG','ExpGrad',
      'KLIG-Adaptive','KL-IG (linear)','KL-IG²','KL-IG² (adaptive)']
stats = []
for m in ALL:
    mean, se = mse([r['cs'][m] for r in rows]); stats.append((m, mean, se))
stats.sort(key=lambda t: -t[1])
pd.DataFrame([{'rank':i+1,'method':m,'gated_CSstruct':f'{mn:.4f}','SE':f'{se:.4f}'} for i,(m,mn,se) in enumerate(stats)]
             ).to_csv(f'cs_viz_outputs/cs_leaderboard{TAG}.csv', index=False)
print('\n' + '='*54 + f'\nGated CS_struct leaderboard (n={n})')
for i,(m,mn,se) in enumerate(stats): print(f'  {i+1:2d}. {m:20s} {mn:.4f} ± {se:.4f}')

fig, ax = plt.subplots(figsize=(9, 0.5*len(stats)+1.2), facecolor='white'); ax.axis('off')
cells = [[f'{i+1}', m, f'{mn:.4f} ± {se:.4f}'] for i,(m,mn,se) in enumerate(stats)]
tb = ax.table(cellText=cells, colLabels=['rank','method','gated CS_struct ± SE'], cellLoc='center', loc='center')
tb.auto_set_font_size(False); tb.set_fontsize(11); tb.scale(1, 1.6)
for j in range(3): tb[0,j].set_facecolor('#34495e'); tb[0,j].set_text_props(color='white', fontweight='bold')
for j in range(3): tb[1,j].set_facecolor('#cfe3f5'); tb[1,j].set_text_props(fontweight='bold')   # #1 highlighted
ax.set_title(f'Class-sensitivity leaderboard — gated CS_struct (n={n}, pool={Path(POOL).stem})\n'
             'higher = more spatially-coherent class discrimination · sorted, ±SE', fontsize=11, fontweight='bold', pad=12)
plt.tight_layout(); out=f'cs_viz_outputs/cs_leaderboard{TAG}_table.png'
plt.savefig(out, dpi=170, bbox_inches='tight'); plt.close()
print(f'saved {out} (+ .csv)  | #1 = {stats[0][0]}')

# bar chart
fig, ax = plt.subplots(figsize=(10, 5.5), facecolor='white')
ms = [m for m,_,_ in stats][::-1]; mn = [x for _,x,_ in stats][::-1]; se = [s for _,_,s in stats][::-1]
colors = ['#8b0000' if m=='KL-IG² (adaptive)' else '#4c72b0' for m in ms]
ax.barh(range(len(ms)), mn, xerr=se, color=colors, alpha=0.85, error_kw=dict(lw=1))
ax.set_yticks(range(len(ms))); ax.set_yticklabels(ms, fontsize=10)
ax.set_xlabel('gated CS_struct  (mean ± SE)', fontsize=11)
ax.set_title(f'Class-sensitivity ranking — gated CS_struct (n={n})', fontsize=13, fontweight='bold')
ax.grid(alpha=0.3, axis='x')
plt.tight_layout(); outb=f'cs_viz_outputs/cs_leaderboard{TAG}_bar.png'
plt.savefig(outb, dpi=150, bbox_inches='tight'); plt.close(); print('saved', outb)
