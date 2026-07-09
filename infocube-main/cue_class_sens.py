"""Cue ablation for CLASS SENSITIVITY — which cue (texture/edge/shape) carries the y1-vs-y2
DISCRIMINATIVE signal inside the discriminative region R?

We measure the collapse of the class DIFFERENTIAL d1-d2 (how differently the two classes rely
on R), not the top-1 confidence.  Cue removals are applied ONLY inside R; an R-specific control
(same removal on a random region of equal size) is subtracted to remove global perturbation
leakage.  Run for ResNet50 and ViT-B/16.
Run:  .venv/Scripts/python cue_class_sens.py [n]   (default 30)
"""
import sys, pickle, warnings
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter, map_coordinates
warnings.filterwarnings('ignore')
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
FZ_SCALE, FZ_SIGMA, FZ_MINSIZE, SEED, EPS, DR_FRAC = 0.6, 0.8, 100, 0, 1e-8, 0.25
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.manual_seed(SEED); np.random.seed(SEED)
_mean = np.array([0.485,0.456,0.406]).reshape(3,1,1); _std = np.array([0.229,0.224,0.225]).reshape(3,1,1)

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
def top2_and_region(model, x, seg, chunk=64):
    base = F.softmax(model(x.unsqueeze(0))[0], 0)
    y1, y2 = [int(c) for c in base.topk(2).indices]
    b1, b2 = base[y1].item(), base[y2].item()
    labs = np.unique(seg); xb = x.unsqueeze(0).repeat(len(labs),1,1,1).clone()
    for k,lab in enumerate(labs): xb[k][:, torch.from_numpy(seg==lab).to(x.device)] = 0
    d1=np.zeros(len(labs)); d2=np.zeros(len(labs))
    for s in range(0,len(labs),chunk):
        p=F.softmax(model(xb[s:s+chunk]),-1)
        d1[s:s+p.shape[0]]=b1-p[:,y1].cpu().numpy(); d2[s:s+p.shape[0]]=b2-p[:,y2].cpu().numpy()
    dd=np.abs(d1-d2); disc = dd>=np.quantile(dd,1-DR_FRAC) if np.ptp(dd)>EPS else np.zeros(len(labs),bool)
    return y1, y2, np.isin(seg, labs[disc])                                    # boolean region mask

@torch.no_grad()
def class_differential(model, x_np, Rb, y1, y2):
    """d1 - d2 : how differently y1 vs y2 rely on region R (drop when R is occluded)."""
    x = torch.from_numpy(x_np).float().to(DEVICE)
    p = F.softmax(model(x.unsqueeze(0))[0], 0)
    xo = x.clone(); xo[:, torch.from_numpy(Rb).to(DEVICE)] = 0
    po = F.softmax(model(xo.unsqueeze(0))[0], 0)
    d1 = (p[y1]-po[y1]).item(); d2 = (p[y2]-po[y2]).item()
    return d1 - d2

# ── cue removals (applied ONLY inside R) ─────────────────────────────────────────────────
def blend(img, alt, Rb): m = Rb[None].astype(float); return img*(1-m) + alt*m
def remove_texture(img, Rb): return blend(img, np.stack([median_filter(img[c], size=5) for c in range(3)]), Rb)
def remove_edge(img, Rb):    return blend(img, np.stack([gaussian_filter(img[c], sigma=4) for c in range(3)]), Rb)
def smooth_warp(img, rng, strength=15, sigma=8):
    H,W = img.shape[1:]; yy,xx = np.mgrid[0:H,0:W]
    dx = gaussian_filter(rng.standard_normal((H,W)), sigma)*strength
    dy = gaussian_filter(rng.standard_normal((H,W)), sigma)*strength
    crd = [np.clip(yy+dy,0,H-1), np.clip(xx+dx,0,W-1)]
    return np.stack([map_coordinates(img[c], crd, order=1, mode='reflect') for c in range(3)])
def remove_shape(img, Rb, rng): return blend(img, smooth_warp(img, rng), Rb)   # deform, no new edges
def photometric(img):                                                          # label-preserving jitter (NULL)
    im01 = np.clip(img*_std + _mean, 0, 1)
    im01 = np.clip(im01*1.2, 0, 1); im01 = np.clip((im01-0.5)*1.3 + 0.5, 0, 1) # brightness + contrast
    return (im01 - _mean)/_std
def remove_null(img, Rb): return blend(img, photometric(img), Rb)              # keeps texture/edge/shape intact
CUES = ['texture','edge','shape','null']                                       # 'null' = augmentation control
CUE_LABEL = {'texture':'texture','edge':'edge','shape':'shape','null':'null\n(augment)'}

def cue_collapse(model, x_np, Rb, y1, y2, rng):
    base = class_differential(model, x_np, Rb, y1, y2)
    out = {}
    for name in CUES:
        if name=='texture': xr = remove_texture(x_np, Rb)
        elif name=='edge':  xr = remove_edge(x_np, Rb)
        elif name=='shape': xr = remove_shape(x_np, Rb, rng)
        else:               xr = remove_null(x_np, Rb)
        new = class_differential(model, xr, Rb, y1, y2)
        out[name] = float(np.clip(1 - abs(new)/(abs(base)+EPS), 0.0, 1.0))     # fraction of differential lost [0,1]
    return out, abs(base)

def cue_class_sens_Rspecific(model, x_np, Rb, y1, y2, rng):
    inR, base_mag = cue_collapse(model, x_np, Rb, y1, y2, rng)
    # random region of EQUAL size: roll the region mask to a random location
    dy, dx = int(rng.integers(40,180)), int(rng.integers(40,180))
    Rrand = np.roll(Rb, (dy,dx), axis=(0,1))
    outR, _ = cue_collapse(model, x_np, Rrand, y1, y2, rng)
    return {k: inR[k]-outR[k] for k in CUES}, base_mag

# ── run both architectures ───────────────────────────────────────────────────────────────
pool = pickle.load(open('cs_viz_cache/pool1000_balanced.pkl','rb'))[:N]
def img_of(d): x=d['x']; return (x.squeeze(0) if x.dim()==4 else x)
print(f'[setup] cue class-sensitivity | {len(pool)} images | {DEVICE}')

from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights
ARCHS = {'ResNet50': lambda: resnet50(weights=ResNet50_Weights.IMAGENET1K_V2),
         'ViT-B/16': lambda: vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)}
CKPT = Path('cs_viz_cache/cue_class_sens.pkl')
results = pickle.load(open(CKPT,'rb')) if CKPT.exists() else {}
from tqdm import tqdm
for arch, build in ARCHS.items():
    if arch in results and len(results[arch]) >= len(pool): continue
    model = build().to(DEVICE).eval()
    recs = results.get(arch, [])
    for i in tqdm(range(len(recs), len(pool)), desc=arch):
        rng = np.random.default_rng(1000+i)
        x = img_of(pool[i]).to(DEVICE); xn = x.cpu().numpy()
        seg = get_segments(x); y1, y2, Rb = top2_and_region(model, x, seg)
        vals, base_mag = cue_class_sens_Rspecific(model, xn, Rb, y1, y2, rng)
        recs.append(dict(base=base_mag, **vals))
        if (i+1) % 10 == 0: results[arch]=recs; pickle.dump(results, open(CKPT,'wb'))
    results[arch] = recs; pickle.dump(results, open(CKPT,'wb'))
    del model; torch.cuda.empty_cache()

# ── aggregate (only images with a non-trivial base differential) ─────────────────────────
def agg(recs):
    out = {}
    for c in CUES:
        v = np.array([r[c] for r in recs if abs(r['base'])>0.02])
        out[c] = (v.mean(), v.std()/np.sqrt(max(len(v),1)), len(v))
    return out
print('\n' + '='*56)
stats = {}
for arch, recs in results.items():
    stats[arch] = agg(recs)
    print(f'{arch}  (n={sum(abs(r["base"])>0.02 for r in recs)} with usable differential)')
    for c in CUES: print(f'   {c:8s} collapse (R-specific) = {stats[arch][c][0]:+.3f} ± {stats[arch][c][1]:.3f}')

# ── figure ───────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 5.4), facecolor='white')
xx = np.arange(len(CUES)); w = 0.38; colors = {'ResNet50':'#4c72b0','ViT-B/16':'#dd8452'}
for j, arch in enumerate(stats):
    m = [stats[arch][c][0] for c in CUES]; se = [stats[arch][c][1] for c in CUES]
    ax.bar(xx + (j-0.5)*w, m, w, yerr=se, capsize=3, color=colors.get(arch,'#888'), alpha=0.88, label=arch)
ax.axhline(0, color='#888', lw=0.8)
ax.set_xticks(xx); ax.set_xticklabels([CUE_LABEL[c] for c in CUES], fontsize=11)
ax.set_ylabel('collapse of class differential  (R-specific)\nfraction of y1-vs-y2 signal that cue carries', fontsize=10.5)
ax.set_title('Which cue carries the class-discriminative signal?\ncue removed inside R; R-specific control · null = label-preserving jitter (should not collapse)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=0.3, axis='y')
plt.tight_layout(); out='cs_viz_outputs/cue_class_sens.png'
plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close(); print('saved', out)

# ── clean table ──────────────────────────────────────────────────────────────────────────
import pandas as pd
archs = list(stats)
df = pd.DataFrame([{'cue': c, **{a: stats[a][c][0] for a in archs}} for c in CUES])
df.to_csv('cs_viz_outputs/cue_class_sens.csv', index=False)
REAL = ['texture','edge','shape']                                           # 'null' is the control, not a winner
dom = {a: REAL[int(np.argmax([stats[a][c][0] for c in REAL]))] for a in archs}
n_used = {a: sum(abs(r['base'])>0.02 for r in results[a]) for a in archs}
figt, axt = plt.subplots(figsize=(8.2, 2.4), facecolor='white'); axt.axis('off')
cells = [[c] + [f'{stats[a][c][0]:+.3f} ± {stats[a][c][1]:.3f}' for a in archs] for c in CUES]
tb = axt.table(cellText=cells, colLabels=['cue']+[f'{a}\n(n={n_used[a]})' for a in archs], cellLoc='center', loc='center')
tb.auto_set_font_size(False); tb.set_fontsize(12); tb.scale(1, 2.0)
for j in range(len(archs)+1): tb[0,j].set_facecolor('#34495e'); tb[0,j].set_text_props(color='white', fontweight='bold')
for ci, c in enumerate(CUES):                                               # bold the dominant cue per arch
    for aj, a in enumerate(archs):
        if dom[a] == c: tb[ci+1, aj+1].set_text_props(fontweight='bold'); tb[ci+1, aj+1].set_facecolor('#cfe3f5')
axt.set_title('Cue carrying the y1-vs-y2 class-discriminative signal (R-specific collapse ± SE)\n'
              + ' · '.join(f'{a}: {dom[a]}' for a in archs), fontsize=11, fontweight='bold', pad=10)
plt.tight_layout(); outt='cs_viz_outputs/cue_class_sens_table.png'
plt.savefig(outt, dpi=170, bbox_inches='tight'); plt.close()
print('saved', outt, '(+ .csv) | dominant cue:', dom)
