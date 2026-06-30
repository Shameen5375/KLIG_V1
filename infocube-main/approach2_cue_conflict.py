"""
APPROACH 2 — cue-conflict (Geirhos-style), frequency-hybrid version (no style-transfer net needed).
Build hybrid = low-freq(shape image S) + high-freq(texture image T), with class(S) != class(T).
Whichever class the model predicts = the cue it trusts. Geirhos shape-bias = #shape/(#shape+#texture).
Compare ResNet50 vs ViT-B/16. Forward-only.

Caveat: low/high frequency is a PROXY for AdaIN texture-vs-shape (the publishable version uses
stylized ImageNet). Reported with controls; treat as indicative of the architecture difference.
Run:  .venv/Scripts/python approach2_cue_conflict.py [N]   (default 50)
"""
import sys, pickle, warnings
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
import numpy as np, torch, torch.nn.functional as F, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
warnings.filterwarnings('ignore')
N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
EPS = 1e-8; DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rng = np.random.default_rng(0)
from tqdm import tqdm

from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights
resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(DEVICE).eval()
vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to(DEVICE).eval()
MODELS = {'ResNet50': resnet, 'ViT-B/16': vit}
cats = ViT_B_16_Weights.IMAGENET1K_V1.meta['categories']
store = pickle.load(open('cs_viz_outputs/segment_store_vit.pkl', 'rb'))
imgs = [R['x'].squeeze(0) for R in store]
print(f'[setup] device={DEVICE}  pool={len(imgs)}  models={list(MODELS)}')

@torch.no_grad()
def top1(model, xnp):
    x = torch.from_numpy(xnp).float().to(DEVICE)
    return int(model(x.unsqueeze(0))[0].argmax())
def lowpass(a, s): return gaussian_filter(a, sigma=(0, s, s))
def hybrid(S, T, s): return lowpass(S, s) + (T - lowpass(T, s))     # low(shape) + high(texture)
def denorm(a):
    m = np.array([0.485,0.456,0.406])[:,None,None]; sd = np.array([0.229,0.224,0.225])[:,None,None]
    return np.clip((a*sd+m).transpose(1,2,0), 0, 1)

# clean-image top1 per model (the "shape class" S = cs, "texture class" T = ct)
clean = {mn: [top1(MODELS[mn], im.numpy()) for im in imgs] for mn in MODELS}

# build N cue-conflict pairs with distinct clean classes (use ViT labels to pick partners)
base_cls = clean['ViT-B/16']
pairs = []
order = list(rng.permutation(len(imgs)))
for i in order:
    for j in order:
        if j != i and base_cls[j] != base_cls[i] and (i, j) not in pairs:
            pairs.append((i, j)); break
    if len(pairs) >= N: break
print(f'[pairs] {len(pairs)} cue-conflict trials')

SIGMAS = [2, 3, 4]                                              # valid regime: low-freq shape stays recognizable
results = {mn: {s: {'shape': 0, 'texture': 0, 'other': 0} for s in SIGMAS} for mn in MODELS}
san = {mn: 0 for mn in MODELS}                                   # sanity: lowpass(S) still -> cs ?
for (i, j) in tqdm(pairs, desc='cue conflict'):
    S, T = imgs[i].numpy(), imgs[j].numpy()
    for mn, model in MODELS.items():
        cs, ct = clean[mn][i], clean[mn][j]
        if cs == ct: continue
        if top1(model, lowpass(S, 2)) == cs: san[mn] += 1        # shape channel alone recovers cs (at valid sigma=2)
        for s in SIGMAS:
            pred = top1(model, hybrid(S, T, s))
            k = 'shape' if pred == cs else 'texture' if pred == ct else 'other'
            results[mn][s][k] += 1

def shape_bias(d): tot = d['shape'] + d['texture']; return d['shape']/tot if tot else np.nan, tot
print('\n=== Geirhos shape-bias = #shape / (#shape + #texture) ;  higher = more shape-reliant ===')
for mn in MODELS:
    print(f'\n{mn}   (lowpass-shape sanity: {san[mn]}/{len(pairs)} recovered the shape class)')
    for s in SIGMAS:
        d = results[mn][s]; sb, tot = shape_bias(d)
        print(f'  sigma={s:2d}:  shape-bias={sb:.2f}  (shape={d["shape"]} texture={d["texture"]} other={d["other"]}, decided={tot})')

# ── figure: shape-bias per model (main sigma=7) + trend vs sigma + example hybrids ──
fig = plt.figure(figsize=(15, 4.6), facecolor='white')
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.4])
cols = {'ResNet50': '#2c7fb8', 'ViT-B/16': '#d95f0e'}
S0 = 3
axA = fig.add_subplot(gs[0, 0])
sbs = [shape_bias(results[mn][S0])[0] for mn in MODELS]
axA.bar(range(len(MODELS)), sbs, color=[cols[m] for m in MODELS])
for k, v in enumerate(sbs): axA.text(k, v+0.02, f'{v:.2f}', ha='center', fontweight='bold')
axA.axhline(0.5, ls=':', color='gray'); axA.set_ylim(0, 1)
axA.set_xticks(range(len(MODELS))); axA.set_xticklabels(list(MODELS)); axA.set_ylabel('shape-bias')
axA.set_title(f'Shape-bias (sigma={S0})\n1=pure shape, 0=pure texture', fontsize=10, fontweight='bold')
axB = fig.add_subplot(gs[0, 1])
for mn in MODELS:
    axB.plot(SIGMAS, [shape_bias(results[mn][s])[0] for s in SIGMAS], 'o-', color=cols[mn], lw=2, label=mn)
axB.axhline(0.5, ls=':', color='gray'); axB.set_ylim(0, 1)
axB.set_xlabel('low/high split sigma (px)'); axB.set_ylabel('shape-bias'); axB.set_title('Robustness vs split', fontsize=10, fontweight='bold'); axB.legend(fontsize=8)
# example hybrids
axC = fig.add_subplot(gs[0, 2]); axC.axis('off')
ex = fig.add_gridspec(2, 3, left=0.66, right=0.99, top=0.84, bottom=0.12, wspace=0.05, hspace=0.35)
for r, (i, j) in enumerate(pairs[:2]):
    S, T = imgs[i].numpy(), imgs[j].numpy(); h = hybrid(S, T, S0)
    cs, ct = clean['ViT-B/16'][i], clean['ViT-B/16'][j]
    for cidx, (im, ti) in enumerate([(denorm(S), f'shape: {cats[cs].split(",")[0]}'),
                                     (denorm(T), f'texture: {cats[ct].split(",")[0]}'),
                                     (denorm(h), 'hybrid')]):
        a = fig.add_subplot(ex[r, cidx]); a.imshow(im); a.axis('off'); a.set_title(ti, fontsize=7.5)
fig.suptitle('Approach 2 — frequency-hybrid cue conflict (low-freq shape vs high-freq texture)', fontsize=12, fontweight='bold')
plt.savefig('cs_viz_outputs/approach2_cue_conflict.png', dpi=160, bbox_inches='tight'); plt.close()
print('\nsaved cs_viz_outputs/approach2_cue_conflict.png')
