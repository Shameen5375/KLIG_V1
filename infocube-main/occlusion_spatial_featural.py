"""Spatial vs Featural split from segment occlusion (reuses cached occlusion_classsens.pkl).
regime_ratio = Σ(d1-d2)² / Σ(d1+d2)² :  HIGH => classes use DIFFERENT regions (SPATIAL);
LOW => classes use the SAME region, differ by feature/texture (FEATURAL).
Threshold from a random-class-pair control (mean+std). Emits stats + two figures.
Run:  .venv/Scripts/python occlusion_spatial_featural.py
"""
import sys, pickle, warnings
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
warnings.filterwarnings('ignore')
EPS, SEED = 1e-8, 0; DR_FRAC = 0.25
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); np.random.seed(SEED)
from torchvision.models import resnet50, ResNet50_Weights
_w = ResNet50_Weights.IMAGENET1K_V2; model = resnet50(weights=_w).to(DEVICE).eval(); cats = _w.meta['categories']
_mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1); _std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
recs = pickle.load(open('cs_viz_cache/occlusion_classsens.pkl','rb'))
print(f'[setup] {len(recs)} cached images | spatial vs featural via segment occlusion')

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
def regime_ratio(d1,d2): return float(((d1-d2)**2).sum()/(((d1+d2)**2).sum()+EPS))
def to_x(rec): return torch.from_numpy(rec['x'].astype(np.float32)).to(DEVICE)

# ── confusable ratio (cached) + random-pair control (recompute, reuse cached seg) ────────
CKPT = Path('cs_viz_cache/occlusion_spatial_featural.pkl'); cache = {}
if CKPT.exists(): cache = pickle.load(open(CKPT,'rb'))
rng = np.random.default_rng(SEED); conf, rand = [], []
from tqdm import tqdm
for i, rec in enumerate(tqdm(recs, desc='regime ratios')):
    seg = rec['seg'].astype(int); conf.append(regime_ratio(rec['d1'], rec['d2']))
    if i in cache: rand.append(cache[i]); continue
    yr1, yr2 = [int(c) for c in rng.choice([c for c in range(1000) if c not in (rec['y1'],rec['y2'])], 2, replace=False)]
    rd1, rd2 = seg_delta(to_x(rec), yr1, yr2, seg)
    cache[i] = regime_ratio(rd1, rd2); rand.append(cache[i])
    if (i+1) % 20 == 0: pickle.dump(cache, open(CKPT,'wb'))
pickle.dump(cache, open(CKPT,'wb'))
conf, rand = np.array(conf), np.array(rand); n = len(conf)
thr = rand.mean() + rand.std()
spatial = conf > thr; nsp = int(spatial.sum()); nfe = n - nsp
try: pv = wilcoxon(conf, rand).pvalue
except Exception: pv = float('nan')
print('\n' + '='*60)
print(f'confusable regime ratio: mean={conf.mean():.3f} median={np.median(conf):.3f}')
print(f'random     regime ratio: mean={rand.mean():.3f} median={np.median(rand):.3f}')
print(f'threshold (rand mean+std) = {thr:.3f}')
print(f'SPATIAL  (different regions, ratio>thr): {nsp}/{n} ({100*nsp/n:.0f}%)')
print(f'FEATURAL (same region, ratio<=thr):      {nfe}/{n} ({100*nfe/n:.0f}%)')
print(f'Wilcoxon confusable vs random: p={pv:.2e}  '
      + ('SEPARATES ✓' if pv < 0.05 else 'does not separate'))

# ── figure 1: stats ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(13, 5), facecolor='white')
b = np.linspace(0, np.percentile(np.concatenate([conf,rand]), 97), 26)
ax[0].hist(rand, bins=b, alpha=0.5, color='#888', label=f'random pairs (mean {rand.mean():.2f})')
ax[0].hist(conf[~spatial], bins=b, alpha=0.75, color='#1f6fd6', label=f'FEATURAL — same region ({nfe})')
ax[0].hist(conf[spatial], bins=b, alpha=0.75, color='#b00020', label=f'SPATIAL — diff regions ({nsp})')
ax[0].axvline(thr, color='k', ls='--', lw=1.5, label=f'threshold {thr:.2f}')
ax[0].set_xlabel('regime ratio  Σ(d₁−d₂)² / Σ(d₁+d₂)²', fontsize=11); ax[0].set_ylabel('images', fontsize=11)
ax[0].set_title('Spatial vs featural split (confusable pairs)', fontsize=12, fontweight='bold'); ax[0].legend(fontsize=9)
bp = ax[1].boxplot([conf, rand], vert=True, patch_artist=True, showfliers=False, widths=0.6, labels=['confusable','random'])
for patch,c in zip(bp['boxes'], ['#4c72b0','#999']): patch.set_facecolor(c); patch.set_alpha(0.7)
ax[1].set_ylabel('regime ratio', fontsize=11)
ax[1].set_title(f'Confusable > random  (Wilcoxon p={pv:.1e})', fontsize=12, fontweight='bold'); ax[1].grid(alpha=0.3, axis='y')
plt.suptitle(f'Segment-occlusion regime analysis (n={n}): do the two classes rely on the SAME region or DIFFERENT regions?',
             fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig('cs_viz_outputs/occlusion_spatial_featural_stats.png', dpi=150, bbox_inches='tight'); plt.close()
print('saved cs_viz_outputs/occlusion_spatial_featural_stats.png')

# ── figure 2: examples (3 most-spatial, 3 most-featural) with regions highlighted ────────
def overlay(rec):
    seg, d1, d2 = rec['seg'].astype(int), rec['d1'], rec['d2']; diff = d1 - d2
    disc = np.abs(diff) >= np.quantile(np.abs(diff), 1-DR_FRAC)
    im = (torch.from_numpy(rec['x'].astype(np.float32))*_std+_mean).clamp(0,1).permute(1,2,0).numpy()
    sgn = (np.where(disc, np.sign(diff), 0)*np.abs(diff))[seg]
    mag = np.abs(sgn)/(np.abs(sgn).max()+EPS); al=(0.55*mag)[...,None]
    col = np.where((sgn>0)[...,None], np.array([0.85,0.1,0.1]), np.array([0.1,0.3,0.9]))
    return np.clip(im*(1-al)+col*al,0,1)
order = np.argsort(conf)
featural_idx = order[:3]; spatial_idx = order[::-1][:3]
fig, ax = plt.subplots(2, 3, figsize=(11, 7.6), facecolor='white')
for c, i in enumerate(spatial_idx):
    ax[0,c].imshow(overlay(recs[i])); ax[0,c].axis('off')
    ax[0,c].set_title(f"{cats[recs[i]['y1']].split(',')[0]} / {cats[recs[i]['y2']].split(',')[0]}\nratio={conf[i]:.2f}", fontsize=9)
for c, i in enumerate(featural_idx):
    ax[1,c].imshow(overlay(recs[i])); ax[1,c].axis('off')
    ax[1,c].set_title(f"{cats[recs[i]['y1']].split(',')[0]} / {cats[recs[i]['y2']].split(',')[0]}\nratio={conf[i]:.2f}", fontsize=9)
ax[0,0].text(-0.12,0.5,'SPATIAL\n(different regions)', transform=ax[0,0].transAxes, rotation=90, va='center', ha='center',
             fontsize=11, fontweight='bold', color='#b00020')
ax[1,0].text(-0.12,0.5,'FEATURAL\n(same region)', transform=ax[1,0].transAxes, rotation=90, va='center', ha='center',
             fontsize=11, fontweight='bold', color='#1f6fd6')
plt.suptitle('Examples — red = region drives top-1, blue = drives top-2.\n'
             'SPATIAL: red & blue in DIFFERENT places.  FEATURAL: red & blue OVERLAP (same object, different features).',
             fontsize=11, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.94]); plt.savefig('cs_viz_outputs/occlusion_spatial_featural_examples.png', dpi=150, bbox_inches='tight'); plt.close()
print('saved cs_viz_outputs/occlusion_spatial_featural_examples.png')
