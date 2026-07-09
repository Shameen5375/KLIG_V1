"""
Visual illustration of the cue-ablation experiment: for sample images, show the model's
discriminative region R highlighted, then each cue removed INSIDE R (texture/edge/shape),
plus the outside-R control region. Annotated with the ViT margin drop per cue.
Run:  .venv/Scripts/python feature_type_viz.py [n_images]   (default 4)
"""
import sys, pickle, warnings
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
import numpy as np, torch, torch.nn.functional as F, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter, map_coordinates
warnings.filterwarnings('ignore')
K = int(sys.argv[1]) if len(sys.argv) > 1 else 4
EPS = 1e-8; DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rng = np.random.default_rng(0)

from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights
resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(DEVICE).eval()
vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to(DEVICE).eval()
cats = ViT_B_16_Weights.IMAGENET1K_V1.meta['categories']
store = pickle.load(open('cs_viz_outputs/segment_store_vit.pkl', 'rb'))

@torch.no_grad()
def top2(model, x):
    p = F.softmax(model(x.unsqueeze(0))[0], 0); t = p.topk(2).indices.tolist(); return int(t[0]), int(t[1])
@torch.no_grad()
def margin(model, x, y1, y2):
    p = F.softmax(model(x.unsqueeze(0))[0], 0); return float(p[y1] - p[y2])
@torch.no_grad()
def disc(model, x, y1, y2, seg, labs, chunk=24):
    base = F.softmax(model(x.unsqueeze(0))[0], 0); b1, b2 = base[y1].item(), base[y2].item()
    xb = x.unsqueeze(0).repeat(len(labs), 1, 1, 1).clone()
    for k, lab in enumerate(labs): xb[k][:, torch.from_numpy(seg == lab).to(x.device)] = 0
    d = np.zeros(len(labs))
    for s in range(0, len(labs), chunk):
        p = F.softmax(model(xb[s:s+chunk]), -1)
        d[s:s+p.shape[0]] = np.abs((b1 - p[:, y1].cpu().numpy()) - (b2 - p[:, y2].cpu().numpy()))
    dd = d >= np.quantile(d, 0.75) if np.ptp(d) > EPS else np.zeros(len(d), bool)
    return np.isin(seg, labs[dd]), dd

def out_region(dd, labs, seg, area):
    out = [l for l, x in zip(labs, dd) if not x]; rng.shuffle(out); M = np.zeros(seg.shape, bool)
    for l in out:
        M |= (seg == l)
        if M.sum() >= area: break
    return M
def smooth_warp(a, strength=15, sigma=8):
    C, H, W = a.shape; yy, xx = np.mgrid[0:H, 0:W]
    dx = gaussian_filter(rng.standard_normal((H, W)), sigma) * strength
    dy = gaussian_filter(rng.standard_normal((H, W)), sigma) * strength
    crd = [np.clip(yy + dy, 0, H - 1), np.clip(xx + dx, 0, W - 1)]
    return np.stack([map_coordinates(a[c], crd, order=1, mode='reflect') for c in range(C)])
def perturbed(a, kind):
    if kind == 'texture': return median_filter(a, size=(1, 5, 5))
    if kind == 'edge':    return gaussian_filter(a, sigma=(0, 4, 4))
    if kind == 'shape':   return smooth_warp(a)
def apply_in(a, M, kind):
    p = perturbed(a, kind); m = M.astype(float)[None]; return a * (1 - m) + p * m
def denorm(a):
    mean = np.array([0.485, 0.456, 0.406])[:, None, None]; std = np.array([0.229, 0.224, 0.225])[:, None, None]
    return np.clip((a * std + mean).transpose(1, 2, 0), 0, 1)

# pick K clear examples: decent ViT margin, localized region
sel = []
for R in store:
    x = R['x'].squeeze(0).to(DEVICE); seg = R['seg']; labs = np.asarray(R['labels'])
    y1, y2 = top2(vit, x); b = margin(vit, x, y1, y2)
    if b < 0.12: continue
    Rm, dd = disc(vit, x, y1, y2, seg, labs)
    if not (0.10 < Rm.mean() < 0.60): continue
    sel.append((R, x, seg, labs, y1, y2, b, Rm, dd))
    if len(sel) >= K: break
print(f'selected {len(sel)} images')

CUES = ['texture', 'edge', 'shape']
ncol = 2 + len(CUES) + 1
fig, ax = plt.subplots(len(sel), ncol, figsize=(3.0*ncol, 3.1*len(sel)), facecolor='white')
if len(sel) == 1: ax = ax[None, :]
for r, (R, x, seg, labs, y1, y2, b, Rm, dd) in enumerate(sel):
    xnp = x.cpu().numpy(); img = denorm(xnp)
    # ResNet region on same image
    yr1, yr2 = top2(resnet, x); Rr, _ = disc(resnet, x, yr1, yr2, seg, labs)
    Om = out_region(dd, labs, seg, Rm.sum())
    n1, n2 = cats[y1].split(',')[0], cats[y2].split(',')[0]
    def show(a, im, title, contour=None, ccol='lime', extra=None, ecol='blue'):
        a.imshow(im); a.set_title(title, fontsize=9); a.axis('off')
        if contour is not None: a.contour(contour, levels=[0.5], colors=ccol, linewidths=2.2)
        if extra is not None: a.contour(extra, levels=[0.5], colors=ecol, linewidths=1.6, linestyles='dashed')
    show(ax[r, 0], img, f'{n1} vs {n2}\n(ViT margin={b:.2f})')
    show(ax[r, 1], img, 'discriminative region\nViT=green  ResNet=blue', contour=Rm, ccol='lime', extra=Rr, ecol='deepskyblue')
    for j, c in enumerate(CUES):
        xi = apply_in(xnp, Rm, c); drop = b - margin(vit, torch.from_numpy(xi).float().to(DEVICE), y1, y2)
        show(ax[r, 2+j], denorm(xi), f'{c.upper()} removed in R\nΔViT margin = {drop:+.2f}', contour=Rm, ccol='lime')
    show(ax[r, ncol-1], img, 'outside-R control\n(equal area)', contour=Om, ccol='orange')
plt.suptitle('Cue ablation inside the discriminative region R  (green=ViT R, blue=ResNet R, orange=control)',
             fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97]); plt.savefig('cs_viz_outputs/feature_type_viz.png', dpi=150, bbox_inches='tight'); plt.close()
print('saved cs_viz_outputs/feature_type_viz.png')
