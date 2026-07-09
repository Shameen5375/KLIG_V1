"""Visual example for the augmentation-consistency check: pick one confused-pair image, show
each label-preserving transform with its discriminative region R and class-difference map D,
and the CS_struct score — so you can SEE R and D move with the image while CS stays stable.
Run:  .venv/Scripts/python augment_example.py
"""
import sys, pickle, warnings
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import torchvision.transforms.functional as TF
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

# ── machinery (same as segment_occlusion / augment_consistency) ──────────────────────────
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
def cs_struct_gated(A1,A2,mask,sigma=4):
    D=(A1-A2).astype(float)*mask; D=D/(np.abs(D).max()+EPS)
    coh=gaussian_filter(D,sigma); return float((coh**2).sum()/((D**2).sum()+EPS))
def compute(x,y1,y2,x_cf):
    seg=get_segments(x); labs,d1,d2=segment_model_delta(x,y1,y2,seg)
    region=np.isin(seg,labs[_topseg(np.abs(d1-d2))]).astype(float)
    A1=attr_map(HEAD,model,x,int(y1),x_cf=x_cf,phi=phi).detach().cpu().numpy()
    A2=attr_map(HEAD,model,x,int(y2),x_cf=x_cf,phi=phi).detach().cpu().numpy()
    cs=cs_struct_gated(A1,A2,region)
    D=(A1-A2)*region; D=D/(np.abs(D).max()+EPS)
    coh=gaussian_filter(D,4)                                   # the coherence map CS_struct scores
    import torch.nn.functional as _F
    pr = _F.softmax(model(x.unsqueeze(0))[0], 0); t2 = pr.topk(2).indices.tolist()
    return dict(cs=cs, region=region, D=D, coh=coh, top=int(t2[0]),
                t1=int(t2[0]), p1=float(pr[t2[0]]), t2c=int(t2[1]), p2=float(pr[t2[1]]))

def denorm(x): return (x*_std.to(x.device)+_mean.to(x.device)).clamp(0,1)
def renorm(i): return (i-_mean.to(i.device))/_std.to(i.device)
TRANSFORMS = {
    'original':     lambda x: x,
    'hflip':        lambda x: torch.flip(x,[-1]),
    'rotate+10':    lambda x: TF.rotate(x,10,fill=0.0),
    'brightness1.2':lambda x: renorm(TF.adjust_brightness(denorm(x),1.2)),
    'crop0.85':     lambda x: TF.resized_crop(x,17,17,190,190,[224,224],antialias=True),
}

# ── pool + choose a clear example (pred preserved across transforms, non-trivial CS) ─────
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
    if len(sel)>=100: break                            # match augment_consistency n=100 cache
cf_by={}
for d in CANDS: cf_by.setdefault(int(d['high_cls'][0]), d['x'])

# pick a REPRESENTATIVE example: prediction preserved on all transforms, and per-image drift
# closest to the population median (not an outlier), with non-trivial CS for a visible map.
idx = int(sys.argv[1]) if len(sys.argv)>1 else None
if idx is None and Path('cs_viz_cache/augment_consistency.pkl').exists():
    rows=pickle.load(open('cs_viz_cache/augment_consistency.pkl','rb'))
    drift=[np.mean([abs(t['cs']-r['cs0']) for t in r['transforms'].values()]) for r in rows]
    med=np.median(drift)
    mean_drift=np.mean([np.mean([abs(t['cs']-r['cs0']) for t in r['transforms'].values()]) for r in rows])
    mean_cs=np.mean([r['cs0'] for r in rows])
    # representative of the AGGREGATE stability result: drift near population mean AND cs0 near mean;
    # require top-1 preserved on >=4/5 transforms; exclude prior example (20)
    cand=sorted((abs(drift[i]-mean_drift)+abs(rows[i]['cs0']-mean_cs), i) for i,r in enumerate(rows)
          if sum(t['pred_preserved'] for t in r['transforms'].values())>=4 and r['cs0']>0.06 and i!=20)
    idx=cand[0][1] if cand else int(np.argmin([abs(d-med) for d in drift]))
idx = idx if idx is not None else 0
d=sel[idx]; x0=d['x'].to(DEVICE); y1,y2=int(d['high_cls'][0]),int(d['high_cls'][1])
xcf=(cf_by.get(y2,CANDS[0]['x'])).to(DEVICE)
print(f'[example] idx={idx}  y1={cats[y1].split(",")[0]}  y2={cats[y2].split(",")[0]}')

# ── figure: rows=transforms, cols=[image, region R, class-diff D] ────────────────────────
names=list(TRANSFORMS); fig,ax=plt.subplots(len(names),3,figsize=(9,3.0*len(names)),facecolor='white')
for r,name in enumerate(names):
    xt=TRANSFORMS[name](x0); res=compute(xt,y1,y2,xcf); im=denorm(xt).permute(1,2,0).cpu().numpy()
    ax[r,0].imshow(im); ax[r,0].axis('off')
    ax[r,0].text(-0.12,0.5,f"{name}\nCS={res['cs']:.3f}", transform=ax[r,0].transAxes, rotation=90,
                 va='center', ha='center', fontsize=10, fontweight='bold',
                 color='#b00020' if name=='original' else 'black')
    ax[r,0].text(0.5,-0.03, f"top-1: {cats[res['t1']].split(',')[0]} (p={res['p1']:.2f})   "
                 f"top-2: {cats[res['t2c']].split(',')[0]} (p={res['p2']:.2f})",
                 transform=ax[r,0].transAxes, ha='center', va='top', fontsize=8)
    # region R as bright outline + light fill on image
    ax[r,1].imshow(im)
    ax[r,1].contourf(res['region'], levels=[0.5,1.5], colors=['#00bfff'], alpha=0.28)
    ax[r,1].contour(res['region'], levels=[0.5], colors=['#0050ff'], linewidths=1.6)
    ax[r,1].axis('off')
    # coherence map the metric scores (percentile-scaled for contrast), diverging
    coh=res['coh']; v=np.percentile(np.abs(coh),99.5)+EPS
    ax[r,2].imshow(coh, cmap='RdBu_r', vmin=-v, vmax=v); ax[r,2].axis('off')
for j,t in enumerate(['transformed image', 'discriminative region R', 'class-diff coherence  (scored)']):
    ax[0,j].set_title(t, fontsize=11, fontweight='bold')
plt.suptitle(f'Augmentation-consistency — visual example: {cats[y1].split(",")[0]} vs {cats[y2].split(",")[0]}  '
             f'(HEAD={HEAD})\nregion R and class-diff coherence move WITH the image; CS_struct is largely '
             'preserved (per-transform drift in Table 1)', fontsize=11, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.96]); out='cs_viz_outputs/augment_example.png'
plt.savefig(out,dpi=150,bbox_inches='tight'); plt.close(); print('saved',out)
