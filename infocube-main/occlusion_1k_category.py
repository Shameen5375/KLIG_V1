"""Spatial vs featural per SEMANTIC category (dog, bird, snake, fish, food, vehicle, ...),
bucketed via the WordNet hypernym hierarchy. Reuses cached occlusion_1k.pkl (no recompute).
Also emits examples with Felzenszwalb segment boundaries drawn (highlight IS per-superpixel).
Run:  .venv/Scripts/python occlusion_1k_category.py
"""
import sys, pickle, warnings
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from nltk.corpus import wordnet as wn
warnings.filterwarnings('ignore')
EPS, SEED, DR_FRAC = 1e-8, 0, 0.25
POOL = sys.argv[1] if len(sys.argv) > 1 else 'cs_viz_cache/pool1000.pkl'
TAG = '_balanced' if 'balanced' in POOL else ''
def O(x): return f'cs_viz_outputs/occlusion_1k{TAG}_{x}'
FZ_SCALE, FZ_SIGMA, FZ_MINSIZE = 0.6, 0.8, 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torchvision.models import resnet50, ResNet50_Weights
_w = ResNet50_Weights.IMAGENET1K_V2; model = resnet50(weights=_w).to(DEVICE).eval(); cats = _w.meta['categories']
_mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1); _std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
rows = pickle.load(open(f'cs_viz_cache/occlusion_1k{TAG}.pkl','rb'))
pool = pickle.load(open(POOL,'rb'))
conf = np.array([r['conf'] for r in rows]); rand = np.array([r['rand'] for r in rows])
thr = rand.mean() + rand.std(); spatial = conf > thr; n = len(rows)
print(f'[setup] {n} cached rows | threshold={thr:.3f} | WordNet semantic buckets')

# ── WordNet semantic bucketing (priority order; first ancestor match wins) ───────────────
BUCKETS = [('dog', {'dog'}), ('bird', {'bird'}), ('snake', {'snake'}),
           ('reptile', {'reptile', 'diapsid', 'turtle', 'crocodilian_reptile'}),
           ('amphibian', {'amphibian'}), ('fish', {'fish'}),
           ('insect/arthropod', {'arthropod'}), ('primate', {'primate'}),
           ('other mammal', {'mammal'}), ('invertebrate', {'invertebrate', 'mollusk', 'coelenterate'}),
           ('food/produce', {'food', 'foodstuff', 'produce', 'fruit', 'vegetable'}),
           ('vehicle', {'vehicle', 'craft'}), ('musical instrument', {'musical_instrument'}),
           ('clothing', {'clothing', 'garment'}), ('container', {'container'}),
           ('furniture', {'furniture'}), ('structure/building', {'structure', 'building'}),
           ('device/appliance', {'device', 'appliance', 'machine'}), ('tool', {'tool'})]
_cache = {}
def cat_of(name):
    if name in _cache: return _cache[name]
    base = name.split(',')[0].strip().lower()
    ss = wn.synsets(base.replace(' ', '_'), pos='n') or wn.synsets(base.split()[-1], pos='n')
    lab = 'other'
    if ss:
        anc = set()
        for path in ss[0].hypernym_paths():
            for syn in path: anc.add(syn.name().split('.')[0])
        for label, keys in BUCKETS:
            if anc & keys: lab = label; break
        else:
            if 'animal' in anc or 'organism' in anc: lab = 'other animal'
            elif 'artifact' in anc or 'instrumentality' in anc: lab = 'other object'
    _cache[name] = lab; return lab

df = pd.DataFrame([dict(category=cat_of(cats[r['y1']]), spatial=bool(s)) for r, s in zip(rows, spatial)])
g = df.groupby('category')['spatial'].agg(n='count', spatial='sum'); g['featural'] = g['n'] - g['spatial']
g['pct_spatial'] = (100*g['spatial']/g['n']).round(0).astype(int)
g = g.sort_values('n', ascending=False).reset_index()
g[['category','n','spatial','featural','pct_spatial']].to_csv(O('semantic.csv'), index=False)
print(f'[categories] {len(g)} semantic buckets')
for r in g.itertuples(): print(f'  {r.category:20s} n={r.n:4d}  spatial={int(r.spatial):4d}  featural={int(r.featural):3d}  ({r.pct_spatial}% spatial)')

# ── table figure ─────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 0.42*len(g)+1.5), facecolor='white'); ax.axis('off')
cells = [[r.category, str(r.n), str(int(r.spatial)), str(int(r.featural)), f'{r.pct_spatial}%'] for r in g.itertuples()]
tb = ax.table(cellText=cells, colLabels=['semantic category','n','spatial','featural','%spatial'],
              cellLoc='center', loc='center')
tb.auto_set_font_size(False); tb.set_fontsize(10.5); tb.scale(1, 1.55)
for j in range(5): tb[0,j].set_facecolor('#34495e'); tb[0,j].set_text_props(color='white', fontweight='bold')
for i in range(len(g)):
    for j in range(5): tb[i+1,j].set_facecolor('#fbe9ea' if int(cells[i][2])>=int(cells[i][3]) else '#e9f0fb')
ax.set_title(f'Spatial vs featural per SEMANTIC category (WordNet buckets, n={n})\n'
             'featural = structural / same-region · row red if mostly spatial, blue if mostly featural',
             fontsize=11, fontweight='bold', pad=10)
plt.tight_layout(); plt.savefig(O('semantic_table.png'), dpi=170, bbox_inches='tight'); plt.close()
print('saved', O('semantic_table.png'), '(+ .csv)')

# ── examples WITH Felzenszwalb boundaries drawn (proof the highlight is per-superpixel) ──
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
def seg_delta(x, y1, y2, seg, chunk=64):
    base=F.softmax(model(x.unsqueeze(0))[0],0); b1,b2=base[y1].item(),base[y2].item()
    labs=np.unique(seg); xb=x.unsqueeze(0).repeat(len(labs),1,1,1).clone()
    for k,lab in enumerate(labs): xb[k][:, torch.from_numpy(seg==lab).to(x.device)]=0
    d1=np.zeros(len(labs)); d2=np.zeros(len(labs))
    for s in range(0,len(labs),chunk):
        p=F.softmax(model(xb[s:s+chunk]),-1)
        d1[s:s+p.shape[0]]=b1-p[:,y1].cpu().numpy(); d2[s:s+p.shape[0]]=b2-p[:,y2].cpu().numpy()
    return d1,d2
def boundaries(seg):
    b = np.zeros(seg.shape, bool)
    b[:-1,:] |= seg[:-1,:]!=seg[1:,:]; b[1:,:] |= seg[:-1,:]!=seg[1:,:]
    b[:,:-1] |= seg[:,:-1]!=seg[:,1:]; b[:,1:] |= seg[:,:-1]!=seg[:,1:]
    return b
def img_of(d): x=d['x']; return (x.squeeze(0) if x.dim()==4 else x)
def overlay_bd(i, y1, y2):
    x = img_of(pool[i]).to(DEVICE); seg = get_segments(x); d1,d2 = seg_delta(x, y1, y2, seg)
    diff = d1 - d2; disc = np.abs(diff) >= np.quantile(np.abs(diff), 1-DR_FRAC)
    im = (x.cpu()*_std+_mean).clamp(0,1).permute(1,2,0).numpy()
    sgn = (np.where(disc, np.sign(diff), 0)*np.abs(diff))[seg]; mag = np.abs(sgn)/(np.abs(sgn).max()+EPS)
    al=(0.55*mag)[...,None]; col=np.where((sgn>0)[...,None], np.array([0.85,0.1,0.1]), np.array([0.1,0.3,0.9]))
    o = np.clip(im*(1-al)+col*al, 0, 1); o[boundaries(seg)] = [0.82,0.82,0.82]   # yellow superpixel edges
    return o
order = np.argsort(conf); pick = list(order[::-1][:2]) + list(order[:2])          # 2 spatial + 2 featural
fig, ax = plt.subplots(1, 4, figsize=(15, 4.2), facecolor='white')
for c, k in enumerate(pick):
    r = rows[k]; ax[c].imshow(overlay_bd(r['idx'], r['y1'], r['y2'])); ax[c].axis('off')
    tag = 'SPATIAL' if c < 2 else 'FEATURAL'
    ax[c].set_title(f"{tag}  ratio={r['conf']:.2f}\n{cats[r['y1']].split(',')[0]} / {cats[r['y2']].split(',')[0]}",
                    fontsize=9, color='#b00020' if c<2 else '#1f6fd6', fontweight='bold')
plt.suptitle('Highlight IS per-Felzenszwalb-superpixel — yellow = segment boundaries; '
             'each superpixel painted uniformly (red=drives top-1, blue=top-2)', fontsize=11, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.93]); plt.savefig(O('boundaries.png'), dpi=150, bbox_inches='tight'); plt.close()
print('saved', O('boundaries.png'))
