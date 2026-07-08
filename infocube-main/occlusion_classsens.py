"""Model-based class sensitivity via SEGMENT OCCLUSION — highlight the region the MODEL uses
for its top-1 vs top-2 prediction (no attribution method involved).
Per image: occlude each Felzenszwalb superpixel, measure the prob drop for top-1 (d1) and top-2 (d2).
  - small n (<=12): detailed 4-col view  [image | top-1 imp | top-2 imp | discriminative R]
  - large n:        10-wide MONTAGE of the discriminative overlay (red=top-1, blue=top-2)
Run:  .venv/Scripts/python occlusion_classsens.py [n]   (default 100)
"""
import sys, pickle, warnings
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.cm as cm
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter
warnings.filterwarnings('ignore')
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
FZ_SCALE, FZ_SIGMA, FZ_MINSIZE, SEED, EPS = 0.6, 0.8, 100, 0, 1e-8
DR_FRAC = 0.25                                          # top-25% discriminative segments (region R)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.manual_seed(SEED); np.random.seed(SEED)
from torchvision.models import resnet50, ResNet50_Weights
_w = ResNet50_Weights.IMAGENET1K_V2; model = resnet50(weights=_w).to(DEVICE).eval(); cats = _w.meta['categories']
_mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1); _std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

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
def segment_model_delta(x, y1, y2, seg, chunk=64):
    base=F.softmax(model(x.unsqueeze(0))[0],0); b1,b2=base[y1].item(),base[y2].item()
    labs=np.unique(seg); xb=x.unsqueeze(0).repeat(len(labs),1,1,1).clone()
    for k,lab in enumerate(labs): xb[k][:, torch.from_numpy(seg==lab).to(x.device)]=0
    d1=np.zeros(len(labs)); d2=np.zeros(len(labs))
    for s in range(0,len(labs),chunk):
        p=F.softmax(model(xb[s:s+chunk]),-1)
        d1[s:s+p.shape[0]]=b1-p[:,y1].cpu().numpy(); d2[s:s+p.shape[0]]=b2-p[:,y2].cpu().numpy()
    return labs,d1,d2

# ── pool ─────────────────────────────────────────────────────────────────────────────────
srcs=['cs_viz_cache/cands.pkl','klig2_dist_cache/klig2_dist_multiprob.pkl',
      'klig2_val_cache/klig2_dist_multiprob.pkl','cs_gate_cache/pool.pkl']
CANDS,seen=[],set()
for s in srcs:
    if not Path(s).exists(): continue
    for d in pickle.load(open(s,'rb')):
        if len(d.get('high_cls',[]))<2: continue
        x=d['x']; x=x.squeeze(0) if x.dim()==4 else x; fp=round(float(x.float().sum()),1)
        if fp in seen: continue
        seen.add(fp); CANDS.append({'x':x.cpu(),'high_cls':[int(c) for c in d['high_cls'][:2]]})
import random as _r; _r.Random(SEED).shuffle(CANDS)
sel,used=[],set()
for d in CANDS:
    c=int(d['high_cls'][0])
    if c in used: continue
    used.add(c); sel.append(d)
    if len(sel)>=N: break
print(f'[setup] {len(sel)} confused-pair images | occlusion-based model class sensitivity | {DEVICE}')

# ── compute (checkpointed) ──────────────────────────────────────────────────────────────
CKPT = Path('cs_viz_cache/occlusion_classsens.pkl'); recs = []
if CKPT.exists(): recs = pickle.load(open(CKPT,'rb')); print(f'[resume] {len(recs)} cached')
from tqdm import tqdm
for i in tqdm(range(len(recs), len(sel)), desc='occlusion class-sens'):
    d = sel[i]; x = d['x'].to(DEVICE); y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
    with torch.no_grad(): probs = F.softmax(model(x.unsqueeze(0))[0], 0)
    if probs[y2] > probs[y1]: y1, y2 = y2, y1                        # y1 = top-1, y2 = top-2
    seg = get_segments(x); labs, d1, d2 = segment_model_delta(x, y1, y2, seg)
    recs.append(dict(x=d['x'].half().numpy(), y1=y1, y2=y2, p1=float(probs[y1]), p2=float(probs[y2]),
                     seg=seg.astype(np.int16), d1=d1.astype(np.float32), d2=d2.astype(np.float32)))
    if (i+1) % 20 == 0: pickle.dump(recs, open(CKPT,'wb'))
pickle.dump(recs, open(CKPT,'wb'))
recs = recs[:len(sel)]; n = len(recs)

def denorm_np(xh):
    x = torch.from_numpy(xh.astype(np.float32))
    return (x*_std+_mean).clamp(0,1).permute(1,2,0).numpy()
def disc_overlay(rec):
    seg, d1, d2 = rec['seg'].astype(int), rec['d1'], rec['d2']; diff = d1 - d2
    disc = np.abs(diff) >= np.quantile(np.abs(diff), 1-DR_FRAC)
    im = denorm_np(rec['x'])
    sgn = (np.where(disc, np.sign(diff), 0)*np.abs(diff))[seg]
    mag = np.abs(sgn)/(np.abs(sgn).max()+EPS); al = (0.55*mag)[...,None]
    col = np.where((sgn>0)[...,None], np.array([0.85,0.1,0.1]), np.array([0.1,0.3,0.9]))
    return np.clip(im*(1-al)+col*al, 0, 1)
def heat(im, m, cmap):
    mn = np.clip(m/(m.max()+EPS),0,1); h = cm.get_cmap(cmap)(mn)[...,:3]; al=(0.30+0.5*mn)[...,None]
    return np.clip(im*(1-al)+h*al,0,1)

legend = [Patch(facecolor=(0.85,0.1,0.1), label='region drives top-1'),
          Patch(facecolor=(0.1,0.3,0.9), label='region drives top-2')]

if n <= 12:
    # detailed 4-column view
    fig, ax = plt.subplots(n, 4, figsize=(13, 3.3*n), facecolor='white')
    if n==1: ax = ax[None,:]
    for r, rec in enumerate(recs):
        im = denorm_np(rec['x']); seg = rec['seg'].astype(int)
        ax[r,0].imshow(im); ax[r,0].axis('off')
        ax[r,0].text(-0.06,0.5,f"top-1: {cats[rec['y1']].split(',')[0]} (p={rec['p1']:.2f})\n"
                     f"top-2: {cats[rec['y2']].split(',')[0]} (p={rec['p2']:.2f})", transform=ax[r,0].transAxes,
                     rotation=90, va='center', ha='center', fontsize=9, fontweight='bold')
        ax[r,1].imshow(heat(im, rec['d1'][seg], 'Reds')); ax[r,1].axis('off')
        ax[r,2].imshow(heat(im, rec['d2'][seg], 'Blues')); ax[r,2].axis('off')
        ax[r,3].imshow(disc_overlay(rec)); ax[r,3].axis('off')
    for j,t in enumerate(['input (top-1 / top-2)','top-1 importance (d₁)','top-2 importance (d₂)',
                          'discriminative region R\nred=top-1, blue=top-2']):
        ax[0,j].set_title(t, fontsize=11, fontweight='bold')
    fig.legend(handles=legend, loc='upper right', fontsize=9)
    plt.suptitle('Model class sensitivity by SEGMENT OCCLUSION — where the model looks for top-1 vs top-2',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.97]); out='cs_viz_outputs/occlusion_classsens.png'
else:
    # montage of discriminative overlays (red=top-1, blue=top-2)
    cols = 10; rows = int(np.ceil(n/cols))
    fig, ax = plt.subplots(rows, cols, figsize=(1.7*cols, 1.7*rows), facecolor='white')
    ax = np.atleast_2d(ax)
    for k in range(rows*cols):
        a = ax[k//cols, k%cols]; a.axis('off')
        if k < n:
            a.imshow(disc_overlay(recs[k]))
            a.set_title(f"{cats[recs[k]['y1']].split(',')[0][:10]} / {cats[recs[k]['y2']].split(',')[0][:10]}",
                        fontsize=5.5)
    fig.legend(handles=legend, loc='lower center', ncol=2, fontsize=11, bbox_to_anchor=(0.5, -0.01))
    plt.suptitle(f'Model class sensitivity by SEGMENT OCCLUSION (n={n}) — discriminative region per image '
                 '(red = drives top-1, blue = drives top-2). Cell title: top-1 / top-2 class.',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0,0.02,1,0.97]); out='cs_viz_outputs/occlusion_classsens.png'
plt.savefig(out, dpi=140, bbox_inches='tight'); plt.close(); print('saved', out, f'(n={n})')
