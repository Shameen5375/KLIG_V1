"""Spatial vs Featural separation based on SEGMENT-RISE (Option B), on pool1000.
regime_ratio = Σ(r1-r2)² / Σ(r1+r2)²  with r = centered RISE importance (imp_on - imp_off).
The random-pair control is accumulated from the SAME masks (free), so one RISE pass per image.
Outputs: overall table (+csv), per-semantic-category table (+csv), stats figure, examples.
Run:  .venv/Scripts/python segment_rise_separation.py [n] [N_masks]   (default 1000 2000)
"""
import sys, pickle, warnings
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from nltk.corpus import wordnet as wn
warnings.filterwarnings('ignore')
import segment_rise as SR                                    # reuses its model, get_segments, segment_rise_region
N_IMG = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
N_MASK = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
POOL = sys.argv[3] if len(sys.argv) > 3 else 'cs_viz_cache/pool1000.pkl'
TAG = '_balanced' if 'balanced' in POOL else ''
def O(x): return f'cs_viz_outputs/segment_rise_sep{TAG}_{x}'
EPS, SEED, DR_FRAC, P_ON, BATCH = 1e-8, 0, 0.25, 0.5, 128
model, cats, DEVICE = SR.model, SR.cats, SR.DEVICE
_mean, _std = SR._mean, SR._std
rng_master = np.random.default_rng(SEED)

@torch.no_grad()
def rise_regimes(x, seg, y1, y2, yr1, yr2, N=N_MASK, p_on=P_ON, batch=BATCH, seed=SEED):
    """One RISE pass; accumulate centered importances for the confusable pair AND a random pair."""
    dev = x.device; K = int(seg.max()) + 1
    seg_t = torch.from_numpy(seg.astype(np.int64)).to(dev)
    rng = np.random.default_rng(seed)
    cls = [y1, y2, yr1, yr2]
    on = [np.zeros(K) for _ in cls]; of = [np.zeros(K) for _ in cls]; cn = np.zeros(K); cf = np.zeros(K); done = 0
    while done < N:
        b = min(batch, N - done)
        m = rng.random((b, K)) < p_on; mo = ~m
        masks = torch.from_numpy(m).float().to(dev)[:, seg_t]
        p = F.softmax(model(x.unsqueeze(0) * masks.unsqueeze(1)), -1)
        for j, c in enumerate(cls):
            s = p[:, c].cpu().numpy()
            on[j] += (s[:,None]*m).sum(0); of[j] += (s[:,None]*mo).sum(0)
        cn += m.sum(0); cf += mo.sum(0); done += b
    r = [on[j]/np.maximum(cn,1) - of[j]/np.maximum(cf,1) for j in range(4)]     # centered importances
    def regime(a, b): return float(((a-b)**2).sum()/(((a+b)**2).sum()+EPS))
    return regime(r[0], r[1]), regime(r[2], r[3])                              # confusable, random

pool = pickle.load(open(POOL,'rb'))[:N_IMG]
def img_of(d): x=d['x']; return (x.squeeze(0) if x.dim()==4 else x)
print(f'[setup] segment-RISE separation | {len(pool)} images | N_masks={N_MASK} | {DEVICE}')

CKPT = Path(f'cs_viz_cache/segment_rise_sep{TAG}.pkl'); rows = []
if CKPT.exists(): rows = pickle.load(open(CKPT,'rb')); print(f'[resume] {len(rows)} cached')
from tqdm import tqdm
for i in tqdm(range(len(rows), len(pool)), desc='RISE separation'):
    d = pool[i]; x = img_of(d).to(DEVICE); y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
    with torch.no_grad(): base = F.softmax(model(x.unsqueeze(0))[0], 0)
    if base[y2] > base[y1]: y1, y2 = y2, y1
    yr1, yr2 = [int(c) for c in rng_master.choice([c for c in range(1000) if c not in (y1,y2)], 2, replace=False)]
    seg = SR.get_segments(x)
    conf, rnd = rise_regimes(x, seg, y1, y2, yr1, yr2)
    rows.append(dict(idx=i, y1=y1, y2=y2, conf=conf, rand=rnd))
    if (i+1) % 50 == 0: pickle.dump(rows, open(CKPT,'wb'))
pickle.dump(rows, open(CKPT,'wb')); rows = rows[:len(pool)]; n = len(rows)

conf = np.array([r['conf'] for r in rows]); rand = np.array([r['rand'] for r in rows])
thr = rand.mean() + rand.std(); spatial = conf > thr; nsp, nfe = int(spatial.sum()), int((~spatial).sum())
try: pv = wilcoxon(conf, rand).pvalue
except Exception: pv = float('nan')
print('\n' + '='*60)
print(f'confusable regime (RISE): mean={conf.mean():.3f} median={np.median(conf):.3f}')
print(f'random     regime (RISE): mean={rand.mean():.3f} median={np.median(rand):.3f}')
print(f'threshold={thr:.3f} | SPATIAL {nsp} ({100*nsp/n:.0f}%) | FEATURAL {nfe} ({100*nfe/n:.0f}%) | p={pv:.2e}')

# ── overall table ────────────────────────────────────────────────────────────────────────
overall = [['images (n)', f'{n}'], ['method', 'segment-RISE (interactions)'],
           ['SPATIAL — different regions', f'{nsp}  ({100*nsp/n:.1f}%)'],
           ['FEATURAL — same region', f'{nfe}  ({100*nfe/n:.1f}%)'],
           ['confusable ratio (mean/median)', f'{conf.mean():.3f} / {np.median(conf):.3f}'],
           ['random ratio (mean/median)', f'{rand.mean():.3f} / {np.median(rand):.3f}'],
           ['threshold (rand mean+std)', f'{thr:.3f}'], ['confusable > random (Wilcoxon)', f'p = {pv:.1e}']]
pd.DataFrame(overall, columns=['metric','value']).to_csv(O('overall.csv'), index=False)

# ── WordNet semantic buckets ─────────────────────────────────────────────────────────────
BUCKETS = [('dog',{'dog'}),('bird',{'bird'}),('snake',{'snake'}),
           ('reptile',{'reptile','diapsid','turtle','crocodilian_reptile'}),('amphibian',{'amphibian'}),
           ('fish',{'fish'}),('insect/arthropod',{'arthropod'}),('primate',{'primate'}),
           ('other mammal',{'mammal'}),('invertebrate',{'invertebrate','mollusk','coelenterate'}),
           ('food/produce',{'food','foodstuff','produce','fruit','vegetable'}),('vehicle',{'vehicle','craft'}),
           ('musical instrument',{'musical_instrument'}),('clothing',{'clothing','garment'}),
           ('container',{'container'}),('furniture',{'furniture'}),('structure/building',{'structure','building'}),
           ('device/appliance',{'device','appliance','machine'}),('tool',{'tool'})]
_cc = {}
def cat_of(name):
    if name in _cc: return _cc[name]
    base = name.split(',')[0].strip().lower()
    ss = wn.synsets(base.replace(' ','_'), pos='n') or wn.synsets(base.split()[-1], pos='n'); lab='other'
    if ss:
        anc=set()
        for path in ss[0].hypernym_paths():
            for syn in path: anc.add(syn.name().split('.')[0])
        for label,keys in BUCKETS:
            if anc & keys: lab=label; break
        else:
            if 'animal' in anc or 'organism' in anc: lab='other animal'
            elif 'artifact' in anc or 'instrumentality' in anc: lab='other object'
    _cc[name]=lab; return lab
df = pd.DataFrame([dict(category=cat_of(cats[r['y1']]), spatial=bool(s)) for r,s in zip(rows,spatial)])
g = df.groupby('category')['spatial'].agg(n='count', spatial='sum'); g['featural']=g['n']-g['spatial']
g['pct_spatial']=(100*g['spatial']/g['n']).round(0).astype(int); g=g.sort_values('n',ascending=False).reset_index()
g[['category','n','spatial','featural','pct_spatial']].to_csv(O('semantic.csv'), index=False)

fig, ax = plt.subplots(2, 1, figsize=(9, 0.42*len(g)+3.2), facecolor='white',
                       gridspec_kw={'height_ratios':[len(overall)+1, len(g)+1]}); [a.axis('off') for a in ax]
t1 = ax[0].table(cellText=overall, colLabels=['metric','value'], cellLoc='left', loc='center')
t1.auto_set_font_size(False); t1.set_fontsize(10.5); t1.scale(1,1.55)
for j in range(2): t1[0,j].set_facecolor('#34495e'); t1[0,j].set_text_props(color='white',fontweight='bold')
ax[0].set_title(f'Segment-RISE — spatial vs featural (n={n}, N_masks={N_MASK})', fontsize=12, fontweight='bold', pad=8)
cells=[[r.category,str(r.n),str(int(r.spatial)),str(int(r.featural)),f'{r.pct_spatial}%'] for r in g.itertuples()]
tb = ax[1].table(cellText=cells, colLabels=['semantic category','n','spatial','featural','%spatial'], cellLoc='center', loc='center')
tb.auto_set_font_size(False); tb.set_fontsize(10); tb.scale(1,1.5)
for j in range(5): tb[0,j].set_facecolor('#34495e'); tb[0,j].set_text_props(color='white',fontweight='bold')
for i in range(len(g)):
    for j in range(5): tb[i+1,j].set_facecolor('#fbe9ea' if int(cells[i][2])>=int(cells[i][3]) else '#e9f0fb')
ax[1].set_title('By semantic category (WordNet) · red=mostly spatial, blue=mostly featural', fontsize=11, fontweight='bold', pad=8)
plt.tight_layout(); plt.savefig(O('table.png'), dpi=170, bbox_inches='tight'); plt.close()
print('saved', O('table.png'), f'| {len(g)} categories')

# ── stats figure ─────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(13, 5), facecolor='white')
b = np.linspace(0, np.percentile(np.concatenate([conf,rand]), 97), 30)
ax[0].hist(rand, bins=b, alpha=0.5, color='#888', label=f'random pairs (mean {rand.mean():.2f})')
ax[0].hist(conf[~spatial], bins=b, alpha=0.75, color='#1f6fd6', label=f'FEATURAL — same region ({nfe})')
ax[0].hist(conf[spatial], bins=b, alpha=0.75, color='#b00020', label=f'SPATIAL — diff regions ({nsp})')
ax[0].axvline(thr, color='k', ls='--', lw=1.5, label=f'threshold {thr:.2f}')
ax[0].set_xlabel('RISE regime ratio  Σ(r₁−r₂)² / Σ(r₁+r₂)²', fontsize=11); ax[0].set_ylabel('images', fontsize=11)
ax[0].set_title('Spatial vs featural split (segment-RISE)', fontsize=12, fontweight='bold'); ax[0].legend(fontsize=9)
bp = ax[1].boxplot([conf, rand], vert=True, patch_artist=True, showfliers=False, widths=0.6, labels=['confusable','random'])
for patch,c in zip(bp['boxes'], ['#4c72b0','#999']): patch.set_facecolor(c); patch.set_alpha(0.7)
ax[1].set_ylabel('RISE regime ratio', fontsize=11); ax[1].grid(alpha=0.3, axis='y')
ax[1].set_title(f'Confusable > random (Wilcoxon p={pv:.1e})', fontsize=12, fontweight='bold')
plt.suptitle(f'Segment-RISE regime analysis (n={n}): SAME or DIFFERENT regions for the two classes?',
             fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig(O('stats.png'), dpi=150, bbox_inches='tight'); plt.close()
print('saved', O('stats.png'))

# ── examples: 3 spatial + 3 featural, region overlay (red=top-1, blue=top-2, light borders) ──
order = np.argsort(conf)
picks = [('SPATIAL','#b00020', order[::-1][:3]), ('FEATURAL','#1f6fd6', order[:3])]
figE, axE = plt.subplots(2, 3, figsize=(11, 7.6), facecolor='white')
for ri,(tag,color,idxs) in enumerate(picks):
    for ci,k in enumerate(idxs):
        rr=rows[k]; x=img_of(pool[rr['idx']]).to(DEVICE); seg=SR.get_segments(x)
        disc,_,_,_=SR.segment_rise_region(model,x,seg,rr['y1'],rr['y2'],N=N_MASK)
        im=(x.cpu()*_std+_mean).clamp(0,1).permute(1,2,0).numpy()
        axE[ri,ci].imshow(SR.region_overlay(im,seg,disc)); axE[ri,ci].axis('off')
        axE[ri,ci].set_title(cats[rr['y1']].split(',')[0]+' / '+cats[rr['y2']].split(',')[0]+f"\nratio={rr['conf']:.2f}", fontsize=9)
    axE[ri,0].text(-0.12,0.5,tag,transform=axE[ri,0].transAxes,rotation=90,va='center',ha='center',fontsize=11,fontweight='bold',color=color)
figE.suptitle('Segment-RISE discriminative regions — red = drives top-1, blue = top-2, light gray = superpixel borders',fontsize=11,fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.94]); plt.savefig(O('examples.png'),dpi=150,bbox_inches='tight'); plt.close()
print('saved', O('examples.png'))
