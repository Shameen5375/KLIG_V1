"""Render the contrastive step panel for the most FEATURAL pairs (lowest ratio) from the n=100 run."""
import sys, pickle, warnings
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
PATCH, STRIDE, CHUNK, EPS, SEED = 32, 16, 64, 1e-8, 0
NEX = int(sys.argv[1]) if len(sys.argv) > 1 else 2
OUT = Path('cs_viz_outputs'); DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED)

from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(DEVICE).eval()
labels = ResNet50_Weights.IMAGENET1K_V2.meta['categories']
_MEAN = torch.tensor([0.485,0.456,0.406]).view(3,1,1); _STD = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
def denorm(t): return (t.detach().cpu()*_STD+_MEAN).clamp(0,1)

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
by_idx = {d['idx']: d for d in CANDS}

df = pd.read_csv(OUT/'contrastive_regime.csv').sort_values('ratio_conf')
feat = df.head(NEX)
print('most-featural pairs (lowest ratio):')
print(feat[['idx','ratio_conf','ratio_rand']].to_string(index=False))

@torch.no_grad()
def cmaps(x, y1, y2):
    H,W = x.shape[1], x.shape[2]
    base = F.softmax(model(x.unsqueeze(0))[0],-1); b1,b2 = float(base[y1]), float(base[y2])
    coords=[(i,j) for i in range(0,H-PATCH+1,STRIDE) for j in range(0,W-PATCH+1,STRIDE)]
    D1,D2,cnt=np.zeros((H,W)),np.zeros((H,W)),np.zeros((H,W))
    for k in range(0,len(coords),CHUNK):
        bc=coords[k:k+CHUNK]; xb=x.unsqueeze(0).repeat(len(bc),1,1,1).clone()
        for b,(i,j) in enumerate(bc): xb[b,:,i:i+PATCH,j:j+PATCH]=0
        p=F.softmax(model(xb),-1); d1=(b1-p[:,y1]).cpu().numpy(); d2=(b2-p[:,y2]).cpu().numpy()
        for b,(i,j) in enumerate(bc):
            D1[i:i+PATCH,j:j+PATCH]+=d1[b]; D2[i:i+PATCH,j:j+PATCH]+=d2[b]; cnt[i:i+PATCH,j:j+PATCH]+=1
    D1/= (cnt+EPS); D2/=(cnt+EPS); dl=D1-D2; sh=D1+D2
    return D1,D2,dl,sh, float((dl**2).sum()/((sh**2).sum()+EPS))

cols=['image','drop y1','drop y2','Δ = y1−y2','shared = y1+y2']
fig,ax=plt.subplots(NEX,5,figsize=(3.0*5,3.2*NEX),facecolor='white',squeeze=False)
for r,(_,row) in enumerate(feat.iterrows()):
    d=by_idx[int(row['idx'])]; x=d['x'].squeeze(0).to(DEVICE)
    y1,y2=int(d['high_cls'][0]),int(d['high_cls'][1])
    D1,D2,dl,sh,ratio=cmaps(x,y1,y2)
    vpos=max(D1.max(),D2.max(),sh.max(),1e-9); vd=max(abs(dl).max(),1e-9)
    ims=[(denorm(x).permute(1,2,0).numpy(),None,None),(D1,'magma',(0,vpos)),(D2,'magma',(0,vpos)),
         (dl,'bwr',(-vd,vd)),(sh,'magma',(0,vpos))]
    for c,(im,cm,vr) in enumerate(ims):
        a=ax[r,c]
        a.imshow(im) if cm is None else a.imshow(im,cmap=cm,vmin=vr[0],vmax=vr[1])
        a.set_xticks([]); a.set_yticks([])
        if r==0: a.set_title(cols[c],fontsize=11)
    ax[r,0].set_ylabel(f'{labels[y1].split(",")[0]}\nvs {labels[y2].split(",")[0]}\nratio={ratio:.2f}\nFEATURAL',
                       fontsize=9,fontweight='bold',rotation=0,ha='right',labelpad=42,color='#a51d2d')
plt.suptitle('FEATURAL examples — Δ is flat/weak (no location favors one class over the other)',
             fontsize=13,fontweight='bold',y=1.0)
plt.tight_layout(); plt.savefig(OUT/'contrastive_featural_examples.png',dpi=150,bbox_inches='tight'); plt.close()
print('saved', OUT/'contrastive_featural_examples.png')
