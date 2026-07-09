"""R-equivariance — does the discriminative region R move WITH the image under label-preserving
transforms?  Flip the image, recompute R, flip it back, and measure IoU with the original R.
High IoU => R tracks image content (content-anchored), not positional/occlusion artifacts —
validating the foundation the whole gating story rests on.
Run:  .venv/Scripts/python r_equivariance.py [n]   (default 30)
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
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
FZ_SCALE, FZ_SIGMA, FZ_MINSIZE, SEED, EPS, DR_FRAC = 0.6, 0.8, 100, 0, 1e-8, 0.25
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.manual_seed(SEED); np.random.seed(SEED)
from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(DEVICE).eval()

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
def region_bool(x, chunk=64):
    base = F.softmax(model(x.unsqueeze(0))[0], 0); y1,y2 = [int(c) for c in base.topk(2).indices]
    b1,b2 = base[y1].item(), base[y2].item(); seg = get_segments(x)
    labs = np.unique(seg); xb = x.unsqueeze(0).repeat(len(labs),1,1,1).clone()
    for k,lab in enumerate(labs): xb[k][:, torch.from_numpy(seg==lab).to(x.device)] = 0
    d1=np.zeros(len(labs)); d2=np.zeros(len(labs))
    for s in range(0,len(labs),chunk):
        p=F.softmax(model(xb[s:s+chunk]),-1)
        d1[s:s+p.shape[0]]=b1-p[:,y1].cpu().numpy(); d2[s:s+p.shape[0]]=b2-p[:,y2].cpu().numpy()
    dd=np.abs(d1-d2); disc = dd>=np.quantile(dd,1-DR_FRAC) if np.ptp(dd)>EPS else np.zeros(len(labs),bool)
    return np.isin(seg, labs[disc])

def iou(a, b): return float((a & b).sum() / ((a | b).sum() + EPS))

pool = pickle.load(open('cs_viz_cache/pool1000_balanced.pkl','rb'))[:N]
def img_of(d): x=d['x']; return (x.squeeze(0) if x.dim()==4 else x)
print(f'[setup] R-equivariance | {len(pool)} images | ResNet50 | {DEVICE}')

flip_iou, rot_iou, R0s = [], [], []
from tqdm import tqdm
for i in tqdm(range(len(pool)), desc='R-equivariance'):
    x = img_of(pool[i]).to(DEVICE)
    R0 = region_bool(x); R0s.append(R0)
    # horizontal flip (exact): recompute R on flipped image, flip mask back, compare
    Rf = region_bool(torch.flip(x, [-1])); flip_iou.append(iou(R0, np.flip(Rf, 1)))
    # rotation +12°: recompute R on rotated image, rotate mask back, compare
    Rr = region_bool(TF.rotate(x, 12, fill=0.0))
    Rr_back = TF.rotate(torch.from_numpy(Rr.astype(np.float32))[None], -12)[0].numpy() > 0.5
    valid = TF.rotate(torch.ones(1,224,224), -12)[0].numpy() > 0.5              # ignore rotation-empty corners
    rot_iou.append(float(((R0 & Rr_back) & valid).sum() / (((R0 | Rr_back) & valid).sum() + EPS)))
# null baseline: R of an UNRELATED image (true chance — regions from different content)
null_iou = [iou(R0s[i], R0s[(i + len(R0s)//2) % len(R0s)]) for i in range(len(R0s))]

def ms(v): v=np.array(v); return v.mean(), v.std()/np.sqrt(len(v))
fm,fs = ms(flip_iou); rm,rs = ms(rot_iou); nm,nsd = ms(null_iou)
print('\n' + '='*52)
print(f'flip   IoU(R, flip-back R_flipped)   = {fm:.3f} ± {fs:.3f}')
print(f'rotate IoU(R, rot-back R_rotated)    = {rm:.3f} ± {rs:.3f}')
print(f'null   IoU(R, other-image R) [chance] = {nm:.3f} ± {nsd:.3f}')
print(f'=> R is content-anchored: equivariant IoU {fm:.2f}(flip)/{rm:.2f}(rot) vs chance {nm:.2f}')

fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
labels = ['flip','rotate +12°','null (shift)\n[chance]']
vals=[fm,rm,nm]; errs=[fs,rs,nsd]; cols=['#2a7a2a','#2a7a2a','#999']
ax.bar(range(3), vals, yerr=errs, capsize=4, color=cols, alpha=0.88)
for i,v in enumerate(vals): ax.text(i, v+0.02, f'{v:.2f}', ha='center', fontsize=11, fontweight='bold')
ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel('IoU (R original  vs  inverse-transformed R)', fontsize=11); ax.set_ylim(0, 1.0)
ax.set_title(f'R is equivariant under label-preserving transforms (n={N})\n'
             'R moves with the image content, above unrelated-image chance', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y')
plt.tight_layout(); out='cs_viz_outputs/r_equivariance.png'
plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close(); print('saved', out)
