"""
Spatial vs Featural split over the whole pool, with a per-category breakdown.
ratio = energy(Δ)/energy(shared) from contrastive occlusion; ratio > THR → SPATIAL else FEATURAL.
Run:  .venv/Scripts/python contrastive_regime_classes.py [N]   (default = all images)
"""
import sys, pickle, warnings
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
PATCH, STRIDE, CHUNK, EPS, SEED, THR = 32, 16, 64, 1e-8, 0, 0.63
OUT = Path('cs_viz_outputs'); DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED)

from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(DEVICE).eval()
labels = ResNet50_Weights.IMAGENET1K_V2.meta['categories']

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
N = int(sys.argv[1]) if len(sys.argv)>1 else len(CANDS)
CANDS = CANDS[:N]
print(f'[pool] running on {len(CANDS)} images')

# ── coarse category from the top-1 class label (keyword buckets; animal also = ImageNet 0-397) ──
KW = {
 'animal': ['dog','cat','bird','fish','shark','snake','lizard','frog','turtle','tortoise','spider',
            'beetle','bee','butterfly','ant','crab','lobster','bear','wolf','fox','lion','tiger',
            'leopard','elephant','monkey','ape','panda','zebra','horse','cow','sheep','pig','deer',
            'rabbit','squirrel','owl','eagle','penguin','whale','seal','frog','newt','salamander','hen',
            'cock','ostrich','goose','duck','crocodile','iguana','chameleon','scorpion','toad','hog','ox'],
 'vehicle': ['car','truck','bus','train','boat','ship','airplane','aircraft','bicycle','motor','cab',
             'wagon','trailer','scooter','ambulance','tank','jeep','van','locomotive','submarine','canoe','raft'],
 'food': ['bread','pizza','cake','apple','banana','orange','broccoli','sandwich','hotdog','donut',
          'cheese','mushroom','strawberry','lemon','ice','soup','burrito','pretzel','bagel','guacamole','espresso','wine'],
 'instrument': ['guitar','violin','piano','drum','flute','trumpet','sax','harp','accordion','banjo','cello','organ'],
 'clothing': ['shirt','suit','jersey','coat','gown','dress','jean','sock','sandal','boot','sneaker','hat','helmet','mask','glove','bow tie','swimsuit'],
 'furniture': ['chair','table','couch','sofa','bed','desk','bench','wardrobe','bookcase','cradle','crib'],
}
def category(cls):
    lab = labels[cls].lower()
    for cat, kws in KW.items():
        if any(k in lab for k in kws): return cat
    return 'animal' if cls < 398 else 'other'

@torch.no_grad()
def ratio_of(x, y1, y2):
    H,W = x.shape[1], x.shape[2]
    base = F.softmax(model(x.unsqueeze(0))[0],-1); b1,b2=float(base[y1]),float(base[y2])
    coords=[(i,j) for i in range(0,H-PATCH+1,STRIDE) for j in range(0,W-PATCH+1,STRIDE)]
    D1,D2,cnt=np.zeros((H,W)),np.zeros((H,W)),np.zeros((H,W))
    for k in range(0,len(coords),CHUNK):
        bc=coords[k:k+CHUNK]; xb=x.unsqueeze(0).repeat(len(bc),1,1,1).clone()
        for b,(i,j) in enumerate(bc): xb[b,:,i:i+PATCH,j:j+PATCH]=0
        p=F.softmax(model(xb),-1); d1=(b1-p[:,y1]).cpu().numpy(); d2=(b2-p[:,y2]).cpu().numpy()
        for b,(i,j) in enumerate(bc):
            D1[i:i+PATCH,j:j+PATCH]+=d1[b]; D2[i:i+PATCH,j:j+PATCH]+=d2[b]; cnt[i:i+PATCH,j:j+PATCH]+=1
    D1/= (cnt+EPS); D2/=(cnt+EPS); dl=D1-D2; sh=D1+D2
    return float((dl**2).sum()/((sh**2).sum()+EPS))

from tqdm import tqdm
rows=[]
for d in tqdm(CANDS, desc='regime split'):
    x=d['x'].squeeze(0).to(DEVICE); y1,y2=int(d['high_cls'][0]),int(d['high_cls'][1])
    r=ratio_of(x,y1,y2)
    rows.append({'idx':d['idx'],'y1':y1,'y1_label':labels[y1].split(',')[0],
                 'category':category(y1),'ratio':r,'regime':'spatial' if r>THR else 'featural'})
df=pd.DataFrame(rows); df.to_csv(OUT/'regime_split_per_image.csv',index=False)

nsp=int((df['regime']=='spatial').sum()); nft=len(df)-nsp
print('\n'+'='*50)
print(f'OVERALL (n={len(df)}, threshold ratio>{THR}):')
print(f'  SPATIAL : {nsp}/{len(df)} ({100*nsp/len(df):.1f}%)')
print(f'  FEATURAL: {nft}/{len(df)} ({100*nft/len(df):.1f}%)')

cat = (df.groupby('category')['regime'].value_counts().unstack(fill_value=0))
for c in ['spatial','featural']:
    if c not in cat.columns: cat[c]=0
cat['n']=cat['spatial']+cat['featural']; cat['%featural']=(100*cat['featural']/cat['n']).round(1)
cat=cat.sort_values('n',ascending=False)
print('\nBY CATEGORY (spatial | featural | n | %featural):')
print(cat[['spatial','featural','n','%featural']].to_string())
cat.to_csv(OUT/'regime_split_by_category.csv')

# classes that ARE featural (rare → list them)
print('\nFEATURAL pairs (the minority):')
print(df[df['regime']=='featural'][['idx','y1_label','category','ratio']].to_string(index=False))

fig,ax=plt.subplots(figsize=(9,4.5),facecolor='white')
cc=cat.index.tolist()
ax.bar(cc,cat['spatial'],label='spatial',color='#1a5fb4')
ax.bar(cc,cat['featural'],bottom=cat['spatial'],label='featural',color='#a51d2d')
for i,c in enumerate(cc): ax.text(i,cat['n'].iloc[i]+0.5,f"{cat['%featural'].iloc[i]:.0f}% ft",ha='center',fontsize=8)
ax.set_ylabel('# image-pairs'); ax.set_title(f'Spatial vs Featural by category (n={len(df)})')
ax.legend(); plt.xticks(rotation=20,ha='right')
plt.tight_layout(); plt.savefig(OUT/'regime_split_by_category.png',dpi=150,bbox_inches='tight'); plt.close()
print('\nsaved: regime_split_per_image.csv, regime_split_by_category.csv, regime_split_by_category.png')
