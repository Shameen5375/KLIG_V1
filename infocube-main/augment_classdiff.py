"""Explain the 'class difference' in CS_struct: it is the SAME image attributed toward the
model's top-1 vs its top-2 class (the confused pair) — NOT a difference against another image.
Panels: image (top-1/top-2 + probs) | A_top1 | A_top2 | D = A_top1 - A_top2 | gated coherence.
Run:  .venv/Scripts/python augment_classdiff.py [idx]
"""
import sys, pickle, warnings
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
warnings.filterwarnings('ignore')
FZ_SCALE, FZ_SIGMA, FZ_MINSIZE, SEED, EPS = 0.6, 0.8, 100, 0, 1e-8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.manual_seed(SEED); np.random.seed(SEED)
HEAD = 'KL-IG² (adaptive)'
import klig_methods as KM
from klig_methods import attr_map, make_phi
from torchvision.models import resnet50, ResNet50_Weights
_w = ResNet50_Weights.IMAGENET1K_V2; model = resnet50(weights=_w).to(DEVICE).eval(); cats = _w.meta['categories']
phi = make_phi(model)
KM.N_STEPS, KM.N_SAMPLES = 25, 3; KM.IG_STEPS = 25; KM.SG_SAMPLES = 25; KM.EG_SAMPLES = 25
KM.T_DESC, KM.N_MC_DESC, KM.N_MC_GRAD = 25, 8, 3
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
def _topseg(v,f=0.25):
    v=np.asarray(v,float); return v>=np.quantile(v,1-f) if np.ptp(v)>EPS else np.zeros(len(v),bool)

# ── pool + same representative pick as augment_example ──────────────────────────────────
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
    if len(sel)>=60: break
cf_by={}
for d in CANDS: cf_by.setdefault(int(d['high_cls'][0]), d['x'])

idx=int(sys.argv[1]) if len(sys.argv)>1 else None
if idx is None and Path('cs_viz_cache/augment_consistency.pkl').exists():
    rows=pickle.load(open('cs_viz_cache/augment_consistency.pkl','rb'))
    drift=[np.mean([abs(t['cs']-r['cs0']) for t in r['transforms'].values()]) for r in rows]; med=np.median(drift)
    cand=[(abs(drift[i]-med), i) for i,r in enumerate(rows)
          if all(t['pred_preserved'] for t in r['transforms'].values()) and r['cs0']>0.06]
    idx=min(cand)[1] if cand else 0
idx=idx if idx is not None else 0
d=sel[idx]; x0=d['x'].to(DEVICE); y1,y2=int(d['high_cls'][0]),int(d['high_cls'][1])
xcf=(cf_by.get(y2,CANDS[0]['x'])).to(DEVICE)

# order by the model so y1=top-1, y2=top-2 among the pair
with torch.no_grad(): probs=F.softmax(model(x0.unsqueeze(0))[0],0)
if probs[y2]>probs[y1]: y1,y2=y2,y1
p1,p2=float(probs[y1]),float(probs[y2])
print(f'[idx={idx}] top-1={cats[y1].split(",")[0]} p={p1:.3f} | top-2={cats[y2].split(",")[0]} p={p2:.3f}')

A1=attr_map(HEAD,model,x0,int(y1),x_cf=xcf,phi=phi).detach().cpu().numpy()
A2=attr_map(HEAD,model,x0,int(y2),x_cf=xcf,phi=phi).detach().cpu().numpy()
seg=get_segments(x0); labs,d1,d2=segment_model_delta(x0,y1,y2,seg)
region=np.isin(seg,labs[_topseg(np.abs(d1-d2))]).astype(float)
D=(A1-A2)*region; D=D/(np.abs(D).max()+EPS); coh=gaussian_filter(D,4)
cs=float((coh**2).sum()/(( (A1-A2)*region/ (np.abs((A1-A2)*region).max()+EPS) )**2).sum()+EPS)

def nabs(a): a=np.abs(a); return np.clip(a/(np.percentile(a,99)+EPS),0,1)
im=(x0.cpu()*_std+_mean).clamp(0,1).permute(1,2,0).numpy()
fig,ax=plt.subplots(1,5,figsize=(19,4.2),facecolor='white')
ax[0].imshow(im); ax[0].axis('off')
ax[0].set_title(f'input\ntop-1: {cats[y1].split(",")[0]} (p={p1:.2f})\ntop-2: {cats[y2].split(",")[0]} (p={p2:.2f})',
                fontsize=11, fontweight='bold')
ax[1].imshow(nabs(A1),cmap='inferno'); ax[1].axis('off')
ax[1].set_title(f'A₁ = attribution → top-1\n({cats[y1].split(",")[0]})', fontsize=11, fontweight='bold')
ax[2].imshow(nabs(A2),cmap='inferno'); ax[2].axis('off')
ax[2].set_title(f'A₂ = attribution → top-2\n({cats[y2].split(",")[0]})', fontsize=11, fontweight='bold')
v=np.percentile(np.abs(A1-A2),99)+EPS
ax[3].imshow((A1-A2),cmap='RdBu_r',vmin=-v,vmax=v); ax[3].axis('off')
ax[3].set_title('class difference\nD = A₁ − A₂', fontsize=11, fontweight='bold', color='#b00020')
vc=np.percentile(np.abs(coh),99.5)+EPS
ax[4].imshow(coh,cmap='RdBu_r',vmin=-vc,vmax=vc); ax[4].axis('off')
ax[4].set_title(f'D gated to R + blurred\n(scored) CS_struct={cs:.3f}', fontsize=11, fontweight='bold', color='#b00020')
plt.suptitle('What "class difference" means: the SAME image attributed to top-1 vs top-2 (the confused pair). '
             'red = leans top-1, blue = leans top-2.  It is NOT a difference against another image.',
             fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.94]); out='cs_viz_outputs/augment_classdiff.png'
plt.savefig(out,dpi=150,bbox_inches='tight'); plt.close(); print('saved',out)
