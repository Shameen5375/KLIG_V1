"""
WHAT FEATURE TYPE does each architecture key on in its discriminative region?
Cue-ablation sensitivity (Approach 1): destroy ONE cue (texture/edge/shape) INSIDE the
model's discriminative region R and measure how much the y1-vs-y2 margin collapses.
Forward-only (no gradients) -> fast, no VRAM spill. Same images through ResNet AND ViT.

Controls (mandatory):
  - random-region: same perturbation in an equal-area NON-discriminative region (must be < in-R).
  - per-perturbation sanity: confirm each op removes its target cue (gradient-energy proxy).
Run:  .venv/Scripts/python feature_type_probe.py [N]   (default 50)
"""
import sys, pickle, warnings
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
import numpy as np, torch, torch.nn.functional as F
from scipy.ndimage import gaussian_filter, median_filter
warnings.filterwarnings('ignore')
N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
EPS = 1e-8; DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rng = np.random.default_rng(0)
from tqdm import tqdm

from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights
resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(DEVICE).eval()
vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to(DEVICE).eval()
MODELS = {'ResNet50': resnet, 'ViT-B/16': vit}
CUES = ['texture', 'edge', 'shape']

# images + segments (segments are model-independent — reuse from the ViT store)
store = pickle.load(open('cs_viz_outputs/segment_store_vit.pkl', 'rb'))[:N]
print(f'[setup] device={DEVICE}  N={len(store)}  models={list(MODELS)}')

@torch.no_grad()
def top2(model, x):
    p = F.softmax(model(x.unsqueeze(0))[0], 0); t = p.topk(2).indices.tolist(); return int(t[0]), int(t[1])
@torch.no_grad()
def margin(model, x, y1, y2):
    p = F.softmax(model(x.unsqueeze(0))[0], 0); return float(p[y1] - p[y2])
@torch.no_grad()
def disc_segments(model, x, y1, y2, seg, labs, chunk=24):
    base = F.softmax(model(x.unsqueeze(0))[0], 0); b1, b2 = base[y1].item(), base[y2].item()
    xb = x.unsqueeze(0).repeat(len(labs), 1, 1, 1).clone()
    for k, lab in enumerate(labs): xb[k][:, torch.from_numpy(seg == lab).to(x.device)] = 0
    d = np.zeros(len(labs))
    for s in range(0, len(labs), chunk):
        p = F.softmax(model(xb[s:s+chunk]), -1)
        d[s:s+p.shape[0]] = np.abs((b1 - p[:, y1].cpu().numpy()) - (b2 - p[:, y2].cpu().numpy()))
    return d                                                     # |d1-d2| per segment

def topmask(d, labs, seg, f=0.25):                              # pixel mask of top-f segments by score
    disc = d >= np.quantile(d, 1 - f) if np.ptp(d) > EPS else np.zeros(len(d), bool)
    return np.isin(seg, labs[disc]), disc
def out_region(disc, labs, seg, area):                         # equal-area NON-discriminative region (control)
    out = [l for l, dd in zip(labs, disc) if not dd]; rng.shuffle(out)
    M = np.zeros(seg.shape, bool)
    for l in out:
        M |= (seg == l)
        if M.sum() >= area: break
    return M

# ── cue-specific perturbations (numpy, C,H,W) ──
def grid_shuffle(a, tile=28):
    C, H, W = a.shape; nh, nw = H // tile, W // tile
    t = a[:, :nh*tile, :nw*tile].reshape(C, nh, tile, nw, tile).transpose(0, 1, 3, 2, 4).reshape(C, nh*nw, tile, tile)
    t = t[:, rng.permutation(nh*nw)]
    return t.reshape(C, nh, nw, tile, tile).transpose(0, 1, 3, 2, 4).reshape(C, nh*tile, nw*tile)
def perturbed(a, kind):
    if kind == 'texture': return median_filter(a, size=(1, 5, 5))   # removes fine texture, KEEPS edges/shape
    if kind == 'edge':    return gaussian_filter(a, sigma=(0, 4, 4)) # removes edges, keeps coarse shape/colour
    if kind == 'shape':   return grid_shuffle(a, 28)                 # destroys global shape, keeps local texture
def apply_in(a, M, kind):
    p = perturbed(a, kind); m = M.astype(float)[None]
    return a * (1 - m) + p * m
def grad_energy(a, M):                                          # high-freq (edge/texture) proxy inside M
    gx = np.zeros_like(a); gy = np.zeros_like(a)
    gx[:, :, :-1] = np.diff(a, axis=2); gy[:, :-1, :] = np.diff(a, axis=1)
    e = (gx**2 + gy**2).sum(0); return float(e[M].mean()) if M.any() else 0.0

# ── main: per model, per image, per cue → fractional margin collapse in-R vs out-R ──
res = {mn: {c: {'in': [], 'out': []} for c in CUES} for mn in MODELS}
san = {c: {'kept': [], 'base': []} for c in CUES}               # sanity: grad-energy retained inside R
for R in tqdm(store, desc='cue ablation'):
    x = R['x'].squeeze(0).to(DEVICE); seg = R['seg']; labs = np.asarray(R['labels'])
    xnp = x.cpu().numpy()
    for mn, model in MODELS.items():
        y1, y2 = top2(model, x)
        d = disc_segments(model, x, y1, y2, seg, labs)
        Rmask, disc = topmask(d, labs, seg);
        if Rmask.sum() == 0: continue
        Omask = out_region(disc, labs, seg, Rmask.sum())
        base = margin(model, x, y1, y2)
        if base <= EPS: continue
        for c in CUES:
            xi = torch.from_numpy(apply_in(xnp, Rmask, c)).float().to(DEVICE)
            xo = torch.from_numpy(apply_in(xnp, Omask, c)).float().to(DEVICE)
            # ABSOLUTE margin drop (stable for confusable pairs; fractional explodes when base is small)
            res[mn][c]['in'].append(base - margin(model, xi, y1, y2))
            res[mn][c]['out'].append(base - margin(model, xo, y1, y2))
            if mn == 'ResNet50':                               # sanity once per image (cue-removal proxy)
                san[c]['kept'].append(grad_energy(perturbed(xnp, c), Rmask)); san[c]['base'].append(grad_energy(xnp, Rmask))

def ms(v): v = np.asarray(v, float); v = v[np.isfinite(v)]; return (v.mean(), v.std()/np.sqrt(len(v))) if len(v) else (np.nan, np.nan)
print('\n=== cue sanity (grad-energy RETAINED inside R after each perturbation; lower = more high-freq removed) ===')
for c in CUES:
    k = np.mean(san[c]['kept']) / (np.mean(san[c]['base']) + EPS)
    print(f'  {c:8s}: {k*100:4.0f}% of edge/texture energy retained')
print('\n=== absolute y1-vs-y2 margin drop when cue removed (mean ± SE) ===')
print(f'{"":10s} | ' + ' | '.join(f'{c:^22s}' for c in CUES))
for mn in MODELS:
    cells = []
    for c in CUES:
        im, ise = ms(res[mn][c]['in']); om, ose = ms(res[mn][c]['out'])
        cells.append(f'inR {im:.2f}±{ise:.2f} / out {om:.2f}')
    print(f'{mn:10s} | ' + ' | '.join(cells))

# ── figure: grouped bars, cues × (ResNet/ViT), in-R solid + out-R hatched control ──
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(11, 5.5), facecolor='white')
xc = np.arange(len(CUES)); w = 0.2
cols = {'ResNet50': '#2c7fb8', 'ViT-B/16': '#d95f0e'}
for j, mn in enumerate(MODELS):
    im = [ms(res[mn][c]['in'])[0] for c in CUES];  ise = [ms(res[mn][c]['in'])[1] for c in CUES]
    om = [ms(res[mn][c]['out'])[0] for c in CUES]; ose = [ms(res[mn][c]['out'])[1] for c in CUES]
    ax.bar(xc + (2*j-1.5)*w, im, w, yerr=ise, color=cols[mn], capsize=3, label=f'{mn}  (inside R)')
    ax.bar(xc + (2*j-0.5)*w, om, w, yerr=ose, color=cols[mn], alpha=0.35, hatch='///', capsize=3,
           label=f'{mn}  (outside R — control)')
ax.set_xticks(xc); ax.set_xticklabels([c.upper() for c in CUES], fontsize=11)
ax.set_ylabel('absolute margin drop  Δ[p(y1)−p(y2)]  when cue removed')
ax.set_title('What cue does each architecture rely on in its discriminative region?\n'
             'cue removed INSIDE R (solid) vs equal-area region OUTSIDE R (hatched control)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=8.5, ncol=2); ax.axhline(0, color='k', lw=0.6)
plt.tight_layout(); plt.savefig('cs_viz_outputs/feature_type_probe.png', dpi=170, bbox_inches='tight'); plt.close()
print('\nsaved cs_viz_outputs/feature_type_probe.png')
