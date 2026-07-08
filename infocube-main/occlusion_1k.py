"""Spatial vs Featural separation at scale (n=1000) via segment occlusion, using pool1000.pkl.
regime_ratio = Σ(d1-d2)² / Σ(d1+d2)²  (HIGH => different regions = SPATIAL; LOW => same region = FEATURAL).
Threshold from a random-class-pair control. Lightweight checkpoint (index + classes + ratios).
Outputs: overall table (+csv), per-category table (+csv), stats figure, examples figure.
Run:  .venv/Scripts/python occlusion_1k.py [n] [pool.pkl]   (default 1000 cs_viz_cache/pool1000.pkl)
"""
import sys, pickle, warnings
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import wilcoxon
warnings.filterwarnings('ignore')
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
POOL = sys.argv[2] if len(sys.argv) > 2 else 'cs_viz_cache/pool1000.pkl'
FZ_SCALE, FZ_SIGMA, FZ_MINSIZE, SEED, EPS, DR_FRAC = 0.6, 0.8, 100, 0, 1e-8, 0.25
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); np.random.seed(SEED)
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
def seg_delta(x, y1, y2, seg, chunk=64):
    base=F.softmax(model(x.unsqueeze(0))[0],0); b1,b2=base[y1].item(),base[y2].item()
    labs=np.unique(seg); xb=x.unsqueeze(0).repeat(len(labs),1,1,1).clone()
    for k,lab in enumerate(labs): xb[k][:, torch.from_numpy(seg==lab).to(x.device)]=0
    d1=np.zeros(len(labs)); d2=np.zeros(len(labs))
    for s in range(0,len(labs),chunk):
        p=F.softmax(model(xb[s:s+chunk]),-1)
        d1[s:s+p.shape[0]]=b1-p[:,y1].cpu().numpy(); d2[s:s+p.shape[0]]=b2-p[:,y2].cpu().numpy()
    return d1,d2,base
def regime_ratio(d1,d2): return float(((d1-d2)**2).sum()/(((d1+d2)**2).sum()+EPS))

pool = pickle.load(open(POOL,'rb'))[:N]
def img_of(d): x=d['x']; return (x.squeeze(0) if x.dim()==4 else x)
print(f'[setup] pool={POOL} using {len(pool)} images | {DEVICE}')

# ── compute (lightweight checkpoint: pool_idx, y1, y2, conf, rand) ───────────────────────
CKPT = Path('cs_viz_cache/occlusion_1k.pkl'); rows = []
if CKPT.exists(): rows = pickle.load(open(CKPT,'rb')); print(f'[resume] {len(rows)} cached')
rng = np.random.default_rng(SEED)
from tqdm import tqdm
for i in tqdm(range(len(rows), len(pool)), desc='occlusion 1k'):
    d = pool[i]; x = img_of(d).to(DEVICE); y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
    seg = get_segments(x)
    d1, d2, base = seg_delta(x, y1, y2, seg)
    if base[y2] > base[y1]: y1, y2, d1, d2 = y2, y1, d2, d1                # y1 = higher-prob (top of pair)
    conf = regime_ratio(d1, d2)
    yr1, yr2 = [int(c) for c in rng.choice([c for c in range(1000) if c not in (y1,y2)], 2, replace=False)]
    rd1, rd2, _ = seg_delta(x, yr1, yr2, seg); rnd = regime_ratio(rd1, rd2)
    rows.append(dict(idx=i, y1=y1, y2=y2, p1=float(base[y1]), p2=float(base[y2]), conf=conf, rand=rnd))
    if (i+1) % 50 == 0: pickle.dump(rows, open(CKPT,'wb'))
pickle.dump(rows, open(CKPT,'wb'))
rows = rows[:len(pool)]; n = len(rows)

conf = np.array([r['conf'] for r in rows]); rand = np.array([r['rand'] for r in rows])
thr = rand.mean() + rand.std(); spatial = conf > thr
nsp, nfe = int(spatial.sum()), int((~spatial).sum())
try: pv = wilcoxon(conf, rand).pvalue
except Exception: pv = float('nan')
print('\n' + '='*60)
print(f'confusable regime ratio: mean={conf.mean():.3f} median={np.median(conf):.3f}')
print(f'random     regime ratio: mean={rand.mean():.3f} median={np.median(rand):.3f}')
print(f'threshold = {thr:.3f} | SPATIAL {nsp} ({100*nsp/n:.0f}%) | FEATURAL {nfe} ({100*nfe/n:.0f}%) | p={pv:.2e}')

# ── overall table (PNG + CSV) ────────────────────────────────────────────────────────────
overall = [['images (n)', f'{n}'],
           ['SPATIAL — different regions', f'{nsp}  ({100*nsp/n:.1f}%)'],
           ['FEATURAL — same region', f'{nfe}  ({100*nfe/n:.1f}%)'],
           ['confusable ratio (mean / median)', f'{conf.mean():.3f} / {np.median(conf):.3f}'],
           ['random ratio (mean / median)', f'{rand.mean():.3f} / {np.median(rand):.3f}'],
           ['threshold (rand mean+std)', f'{thr:.3f}'],
           ['confusable > random (Wilcoxon)', f'p = {pv:.1e}']]
pd.DataFrame(overall, columns=['metric','value']).to_csv('cs_viz_outputs/occlusion_1k_overall.csv', index=False)
# animal (class<398) vs object (>=398) split
anim = np.array([r['y1'] < 398 for r in rows])
def split_counts(mask):
    m = mask; return int((spatial & m).sum()), int((~spatial & m).sum())
asp, afe = split_counts(anim); osp, ofe = split_counts(~anim)
sup = [['animal (top-1 class < 398)', f'{anim.sum()}', f'{asp}', f'{afe}', f'{100*asp/max(anim.sum(),1):.0f}%'],
       ['object (top-1 class ≥ 398)', f'{(~anim).sum()}', f'{osp}', f'{ofe}', f'{100*osp/max((~anim).sum(),1):.0f}%']]

fig, ax = plt.subplots(2, 1, figsize=(8.5, 6.4), facecolor='white'); [a.axis('off') for a in ax]
t1 = ax[0].table(cellText=overall, colLabels=['metric','value'], cellLoc='left', loc='center')
t1.auto_set_font_size(False); t1.set_fontsize(11); t1.scale(1,1.6)
for j in range(2): t1[0,j].set_facecolor('#34495e'); t1[0,j].set_text_props(color='white',fontweight='bold')
ax[0].set_title(f'Overall — spatial vs featural (segment occlusion, n={n})', fontsize=12, fontweight='bold', pad=8)
t2 = ax[1].table(cellText=sup, colLabels=['super-category','n','spatial','featural','%spatial'], cellLoc='center', loc='center')
t2.auto_set_font_size(False); t2.set_fontsize(11); t2.scale(1,1.6)
for j in range(5): t2[0,j].set_facecolor('#34495e'); t2[0,j].set_text_props(color='white',fontweight='bold')
ax[1].set_title('By super-category (animal vs object)', fontsize=12, fontweight='bold', pad=8)
plt.tight_layout(); plt.savefig('cs_viz_outputs/occlusion_1k_overall_table.png', dpi=170, bbox_inches='tight'); plt.close()
print('saved cs_viz_outputs/occlusion_1k_overall_table.png (+ .csv)')

# ── per-category table (top-1 class): spatial vs featural counts (PNG top-N + full CSV) ──
df = pd.DataFrame([dict(category=cats[r['y1']].split(',')[0], spatial=bool(s)) for r,s in zip(rows,spatial)])
g = df.groupby('category')['spatial'].agg(n='count', spatial='sum'); g['featural'] = g['n'] - g['spatial']
g['pct_spatial'] = (100*g['spatial']/g['n']).round(0).astype(int)
g = g.sort_values(['n','spatial'], ascending=False).reset_index()
g[['category','n','spatial','featural','pct_spatial']].to_csv('cs_viz_outputs/occlusion_1k_percategory.csv', index=False)
print(f'per-category: {len(g)} distinct top-1 categories | saved cs_viz_outputs/occlusion_1k_percategory.csv')
TOPN = min(30, len(g)); gt = g.head(TOPN)
fig, ax = plt.subplots(figsize=(9, 0.34*TOPN+1.4), facecolor='white'); ax.axis('off')
cells = [[r.category, str(r.n), str(int(r.spatial)), str(int(r.featural)), f'{r.pct_spatial}%'] for r in gt.itertuples()]
tb = ax.table(cellText=cells, colLabels=['category (top-1 class)','n','spatial','featural','%spatial'],
              cellLoc='center', loc='center')
tb.auto_set_font_size(False); tb.set_fontsize(9.5); tb.scale(1,1.45)
for j in range(5): tb[0,j].set_facecolor('#34495e'); tb[0,j].set_text_props(color='white',fontweight='bold')
for i in range(TOPN):
    for j in range(5): tb[i+1,j].set_facecolor('#fbe9ea' if int(cells[i][2])>=int(cells[i][3]) else '#e9f0fb')
ax.set_title(f'Spatial vs featural per category — top {TOPN} of {len(g)} categories by count (n={n})\n'
             'row shaded red if mostly SPATIAL, blue if mostly FEATURAL · full list in occlusion_1k_percategory.csv',
             fontsize=10.5, fontweight='bold', pad=10)
plt.tight_layout(); plt.savefig('cs_viz_outputs/occlusion_1k_percategory_table.png', dpi=170, bbox_inches='tight'); plt.close()
print('saved cs_viz_outputs/occlusion_1k_percategory_table.png')

# ── stats figure ─────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(13, 5), facecolor='white')
b = np.linspace(0, np.percentile(np.concatenate([conf,rand]), 97), 30)
ax[0].hist(rand, bins=b, alpha=0.5, color='#888', label=f'random pairs (mean {rand.mean():.2f})')
ax[0].hist(conf[~spatial], bins=b, alpha=0.75, color='#1f6fd6', label=f'FEATURAL — same region ({nfe})')
ax[0].hist(conf[spatial], bins=b, alpha=0.75, color='#b00020', label=f'SPATIAL — diff regions ({nsp})')
ax[0].axvline(thr, color='k', ls='--', lw=1.5, label=f'threshold {thr:.2f}')
ax[0].set_xlabel('regime ratio  Σ(d₁−d₂)² / Σ(d₁+d₂)²', fontsize=11); ax[0].set_ylabel('images', fontsize=11)
ax[0].set_title('Spatial vs featural split (confusable pairs)', fontsize=12, fontweight='bold'); ax[0].legend(fontsize=9)
bp = ax[1].boxplot([conf, rand], vert=True, patch_artist=True, showfliers=False, widths=0.6, labels=['confusable','random'])
for patch,c in zip(bp['boxes'], ['#4c72b0','#999']): patch.set_facecolor(c); patch.set_alpha(0.7)
ax[1].set_ylabel('regime ratio', fontsize=11); ax[1].grid(alpha=0.3, axis='y')
ax[1].set_title(f'Confusable > random (Wilcoxon p={pv:.1e})', fontsize=12, fontweight='bold')
plt.suptitle(f'Segment-occlusion regime analysis (n={n}): SAME region or DIFFERENT regions for the two classes?',
             fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig('cs_viz_outputs/occlusion_1k_stats.png', dpi=150, bbox_inches='tight'); plt.close()
print('saved cs_viz_outputs/occlusion_1k_stats.png')

# ── examples (3 most-spatial + 3 most-featural), recomputed on demand ────────────────────
def overlay(i, y1, y2):
    x = img_of(pool[i]).to(DEVICE); seg = get_segments(x); d1,d2,_ = seg_delta(x, y1, y2, seg)
    diff = d1 - d2; disc = np.abs(diff) >= np.quantile(np.abs(diff), 1-DR_FRAC)
    im = (x.cpu()*_std+_mean).clamp(0,1).permute(1,2,0).numpy()
    sgn = (np.where(disc, np.sign(diff), 0)*np.abs(diff))[seg]; mag = np.abs(sgn)/(np.abs(sgn).max()+EPS)
    al=(0.55*mag)[...,None]; col=np.where((sgn>0)[...,None], np.array([0.85,0.1,0.1]), np.array([0.1,0.3,0.9]))
    return np.clip(im*(1-al)+col*al,0,1)
order = np.argsort(conf); featural_i = order[:3]; spatial_i = order[::-1][:3]
fig, ax = plt.subplots(2, 3, figsize=(11, 7.6), facecolor='white')
for c, k in enumerate(spatial_i):
    r = rows[k]; ax[0,c].imshow(overlay(r['idx'], r['y1'], r['y2'])); ax[0,c].axis('off')
    ax[0,c].set_title(f"{cats[r['y1']].split(',')[0]} / {cats[r['y2']].split(',')[0]}\nratio={r['conf']:.2f}", fontsize=9)
for c, k in enumerate(featural_i):
    r = rows[k]; ax[1,c].imshow(overlay(r['idx'], r['y1'], r['y2'])); ax[1,c].axis('off')
    ax[1,c].set_title(f"{cats[r['y1']].split(',')[0]} / {cats[r['y2']].split(',')[0]}\nratio={r['conf']:.2f}", fontsize=9)
ax[0,0].text(-0.12,0.5,'SPATIAL\n(different regions)', transform=ax[0,0].transAxes, rotation=90, va='center', ha='center',
             fontsize=11, fontweight='bold', color='#b00020')
ax[1,0].text(-0.12,0.5,'FEATURAL\n(same region)', transform=ax[1,0].transAxes, rotation=90, va='center', ha='center',
             fontsize=11, fontweight='bold', color='#1f6fd6')
plt.suptitle('Examples — red = region drives top-1, blue = drives top-2.  '
             'SPATIAL: different places · FEATURAL: same object, different features.', fontsize=11, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.94]); plt.savefig('cs_viz_outputs/occlusion_1k_examples.png', dpi=150, bbox_inches='tight'); plt.close()
print('saved cs_viz_outputs/occlusion_1k_examples.png')
