import sys
sys.path.insert(0, r'C:\Users\saame\KLIG_V1\infocube-main')
import numpy as np, torch, torch.nn.functional as F, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights
from klig.image.stopping import find_sigma_stop

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MEAN = torch.tensor([0.485,0.456,0.406]); STD = torch.tensor([0.229,0.224,0.225])
w = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=w).to(DEVICE).eval(); labels = w.meta['categories']
def denorm(x): return (x.detach().cpu()*STD.view(-1,1,1)+MEAN.view(-1,1,1)).clamp(0,1)

x = torch.load(r'C:\Users\saame\KLIG_V1\infocube-main\_yorkie_x.pt').to(DEVICE)
x_raw = x.squeeze(0); C, H, W = x_raw.shape
with torch.no_grad(): tgt = int(model(x).argmax(1)); conf = float(model(x).softmax(-1)[0,tgt])
sf = min(max(find_sigma_stop(model, x_raw, target=tgt, tau=0.95), 1/256), 1.0)
print(f'image: {labels[tgt][:30]} cls={tgt} conf={conf:.2f}  sigma_final={sf:.3f}', flush=True)

import pickle
g = torch.Generator(device=DEVICE).manual_seed(0)
eps = torch.randn(C, H, W, generator=g, device=DEVICE)          # fixed noise for KLIG
# ExpGrad baseline = a REAL dataset image (EG averages over several such images);
# show the path from one representative real baseline -> input.
_ds = pickle.load(open('klig2_dist_cache/dataset.pkl', 'rb'))
bg  = _ds[7]['x'].squeeze(0).to(DEVICE)                          # a real image (not the dog)
bg_lbl = labels[_ds[7]['target']][:14]
blur_k = 51
coords = torch.arange(blur_k, dtype=torch.float32, device=DEVICE) - blur_k//2
kk = torch.exp(-0.5*(coords/16.0)**2); kk = kk/kk.sum()
kh = kk.view(1,1,-1,1).expand(3,-1,-1,-1); kw = kk.view(1,1,1,-1).expand(3,-1,-1,-1)
blur = F.conv2d(F.conv2d(x_raw.unsqueeze(0), kh, padding=(blur_k//2,0), groups=3), kw, padding=(0,blur_k//2), groups=3).squeeze(0)
zero = torch.zeros_like(x_raw)                                  # IG-zero baseline (normalized 0)

T = [0.0, 0.25, 0.5, 0.75, 1.0]

def path_img(method, t):
    if method == 'KLIG (adaptive)':
        sig = sf ** t                       # geometric: sig(0)=1, sig(1)=sf
        return t * x_raw + sig * eps, f'sigma={sig:.3f}'
    if method == 'ExpGrad':
        return bg + t * (x_raw - bg), (f'baseline: {bg_lbl}' if t == 0 else f't={t:.2f}')
    if method == 'Blur-IG':
        return blur + t * (x_raw - blur), f't={t:.2f}'
    if method == 'IG-zero':
        return zero + t * (x_raw - zero), f't={t:.2f}'

METHODS = [f'KLIG (adaptive)\nsigma_final={sf:.3f}', 'Blur-IG', 'IG-zero']
KEYS    = ['KLIG (adaptive)', 'Blur-IG', 'IG-zero']

fig, axes = plt.subplots(len(KEYS), len(T), figsize=(2.4*len(T), 2.4*len(KEYS)),
                         facecolor='white', gridspec_kw={'wspace':0.04,'hspace':0.08})
for r, key in enumerate(KEYS):
    for c, t in enumerate(T):
        img, ann = path_img(key, t)
        ax = axes[r, c]
        ax.imshow(np.clip(denorm(img).permute(1,2,0).numpy(), 0, 1))
        ax.text(0.03, 0.94, ann, transform=ax.transAxes, fontsize=7, va='top',
                color='white', bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.5, ec='none'))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        if r == 0: ax.set_title(f'x(t={t:.2f})', fontsize=10, fontweight='bold')
    axes[r, 0].set_ylabel(METHODS[r], fontsize=9, fontweight='bold', rotation=0,
                          labelpad=46, va='center')
fig.suptitle(f'Path images x(t): baseline -> input  ({labels[tgt][:20]})',
             fontsize=12, fontweight='bold', y=1.01)
plt.savefig(r'C:\Users\saame\KLIG_V1\infocube-main\_pathviz_yorkie.png', dpi=160, bbox_inches='tight')
print('OK saved _pathviz_yorkie.png', flush=True)
