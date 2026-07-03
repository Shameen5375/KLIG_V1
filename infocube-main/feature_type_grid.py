"""
Texture-vs-shape, shown on images. One ROW per cue-ablation method (original / texture / edge / shape),
columns = example images. Ablation applied inside ViT's discriminative region R (green outline);
each panel annotated with BOTH models' y1-vs-y2 margin drop (how much that architecture loses).
Run:  .venv/Scripts/python feature_type_grid.py [n_images]   (default 5)
"""
import sys, pickle, warnings
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
import numpy as np, torch, torch.nn.functional as F, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter
warnings.filterwarnings('ignore')
K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
EPS = 1e-8; DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rng = np.random.default_rng(0)

from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights
resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(DEVICE).eval()
vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to(DEVICE).eval()
cats = ViT_B_16_Weights.IMAGENET1K_V1.meta['categories']
store = pickle.load(open('cs_viz_outputs/segment_store_vit.pkl', 'rb'))

@torch.no_grad()
def top2(m, x): p = F.softmax(m(x.unsqueeze(0))[0], 0); t = p.topk(2).indices.tolist(); return int(t[0]), int(t[1])
@torch.no_grad()
def margin(m, x, a, b): p = F.softmax(m(x.unsqueeze(0))[0], 0); return float(p[a] - p[b])
@torch.no_grad()
def disc(m, x, a, b, seg, labs, chunk=24):
    base = F.softmax(m(x.unsqueeze(0))[0], 0); b1, b2 = base[a].item(), base[b].item()
    xb = x.unsqueeze(0).repeat(len(labs), 1, 1, 1).clone()
    for k, lab in enumerate(labs): xb[k][:, torch.from_numpy(seg == lab).to(x.device)] = 0
    d = np.zeros(len(labs))
    for s in range(0, len(labs), chunk):
        p = F.softmax(m(xb[s:s+chunk]), -1)
        d[s:s+p.shape[0]] = np.abs((b1 - p[:, a].cpu().numpy()) - (b2 - p[:, b].cpu().numpy()))
    dd = d >= np.quantile(d, 0.75) if np.ptp(d) > EPS else np.zeros(len(d), bool)
    return np.isin(seg, labs[dd])
def grid_shuffle(a, tile=28):
    C, H, W = a.shape; nh, nw = H//tile, W//tile
    t = a[:, :nh*tile, :nw*tile].reshape(C, nh, tile, nw, tile).transpose(0,1,3,2,4).reshape(C, nh*nw, tile, tile)
    t = t[:, rng.permutation(nh*nw)]
    return t.reshape(C, nh, nw, tile, tile).transpose(0,1,3,2,4).reshape(C, nh*tile, nw*tile)
def perturbed(a, kind):
    if kind == 'texture': return median_filter(a, size=(1, 5, 5))
    if kind == 'edge':    return gaussian_filter(a, sigma=(0, 4, 4))
    if kind == 'shape':   return grid_shuffle(a, 28)
def apply_in(a, M, kind):
    p = perturbed(a, kind); m = M.astype(float)[None]; return a*(1-m) + p*m
def denorm(a):
    m = np.array([0.485,0.456,0.406])[:,None,None]; sd = np.array([0.229,0.224,0.225])[:,None,None]
    return np.clip((a*sd+m).transpose(1,2,0), 0, 1)

# pick K clear examples (good ViT margin + localized region), RANDOM order each run
sel = []
for idx in np.random.default_rng().permutation(len(store)):     # unseeded -> different images each run
    R = store[int(idx)]
    x = R['x'].squeeze(0).to(DEVICE); seg = R['seg']; labs = np.asarray(R['labels'])
    y1v, y2v = top2(vit, x); bv = margin(vit, x, y1v, y2v)
    if bv < 0.12: continue
    Rm = disc(vit, x, y1v, y2v, seg, labs)
    if not (0.10 < Rm.mean() < 0.60): continue
    y1r, y2r = top2(resnet, x); br = margin(resnet, x, y1r, y2r)
    sel.append((x, seg, labs, y1v, y2v, bv, y1r, y2r, br, Rm))
    if len(sel) >= K: break
K = len(sel); print(f'selected {K} images')

ROWS = ['original', 'texture', 'edge', 'shape']
fig, ax = plt.subplots(len(ROWS), K, figsize=(3.0*K, 3.3*len(ROWS)), facecolor='white')
if K == 1: ax = ax[:, None]
for col, (x, seg, labs, y1v, y2v, bv, y1r, y2r, br, Rm) in enumerate(sel):
    xnp = x.cpu().numpy(); img = denorm(xnp)
    n1, n2 = cats[y1v].split(',')[0], cats[y2v].split(',')[0]
    for row, kind in enumerate(ROWS):
        a = ax[row, col]; a.axis('off')
        if kind == 'original':
            a.imshow(img); a.contour(Rm, levels=[0.5], colors='#1f77ff', linewidths=1.0)  # thin blue outline
            a.set_title(f'{n1} vs {n2}\nViT m={bv:.2f}  RN m={br:.2f}', fontsize=8.5)
        else:
            xi = apply_in(xnp, Rm, kind); xt = torch.from_numpy(xi).float().to(DEVICE)
            vd = bv - margin(vit, xt, y1v, y2v); rd = br - margin(resnet, xt, y1r, y2r)
            a.imshow(denorm(xi)); a.contour(Rm, levels=[0.5], colors='#1f77ff', linewidths=1.0)  # action stays visible
            a.set_title(f'ViT Δ{vd:+.2f}   RN Δ{rd:+.2f}', fontsize=9,
                        color=('#1a7f37' if vd > rd else '#2c7fb8'), fontweight='bold')
    ax[0, col].set_ylabel('')
# row labels on the left
for row, kind in enumerate(ROWS):
    ax[row, 0].text(-0.18, 0.5, kind.upper() + ('\nremoved in R' if kind != 'original' else ''),
                    transform=ax[row, 0].transAxes, rotation=90, va='center', ha='center',
                    fontsize=12, fontweight='bold')
plt.suptitle('Texture vs Shape on images — cue removed inside ViT\'s discriminative region R (blue outline)\n'
             'each panel: ViT Δ vs ResNet Δ margin drop  (green title = ViT loses more, blue = ResNet loses more)',
             fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig('cs_viz_outputs/feature_type_grid.png', dpi=150, bbox_inches='tight'); plt.close()
print('saved cs_viz_outputs/feature_type_grid.png')
