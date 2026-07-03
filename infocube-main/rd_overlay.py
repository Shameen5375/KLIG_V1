"""
R-D Path Attribution shown the way region methods should be shown: a SMOOTH heatmap OVERLAID on
the image (Grad-CAM style), using soft Gaussian windows so it isn't patchy.
Run:  .venv/Scripts/python rd_overlay.py [n_images]   (default 5)
"""
import sys, pickle, warnings
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
from pathlib import Path
import numpy as np, torch, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.cm as cm
warnings.filterwarnings('ignore')
K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import rd_attribution as RD
from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(DEVICE).eval()
cats = ResNet50_Weights.IMAGENET1K_V2.meta['categories']
_mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1); _std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
store = pickle.load(open('cs_viz_outputs/segment_store_vit.pkl','rb'))
imgs = [ ((d['x'].squeeze(0) if d['x'].dim()==4 else d['x'])*_std+_mean).clamp(0,1) for d in store[:K] ]

def overlay(img, amap, cmap='jet'):
    a = amap - amap.min(); a = a/(np.percentile(a,99)+1e-8); a = np.clip(a,0,1)**0.85
    heat = cm.get_cmap(cmap)(a)[..., :3]; al = (0.25 + 0.55*a)[..., None]
    return np.clip(img*(1-al) + heat*al, 0, 1)

# RISE-style random-mask smooth map (pixel-level, not patchy)
wrap = RD.ModelWrapper(model, DEVICE); op = RD.NoiseOperator()
fig, ax = plt.subplots(2, K, figsize=(3.0*K, 6.2), facecolor='white')
if K == 1: ax = ax[:, None]
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
for c, img01 in enumerate(tqdm(imgs, desc='overlay')):
    x = img01.unsqueeze(0).to(DEVICE); cls = int(wrap.logits(x)[0].argmax())
    amap = RD.rise_smooth_map(wrap, x, op, 0.5, cls, n_masks=3000, s=8, p=0.5, batch=64, seed=0)
    amap = gaussian_filter(np.clip(amap, 0, None), 4)
    im = img01.permute(1,2,0).cpu().numpy()
    ax[0,c].imshow(im); ax[0,c].set_title(cats[cls].split(',')[0], fontsize=9); ax[0,c].axis('off')
    ax[1,c].imshow(overlay(im, amap)); ax[1,c].axis('off')
ax[0,0].text(-0.12,0.5,'image', transform=ax[0,0].transAxes, rotation=90, va='center', ha='center', fontsize=11, fontweight='bold')
ax[1,0].text(-0.12,0.5,'R-D overlay', transform=ax[1,0].transAxes, rotation=90, va='center', ha='center', fontsize=11, fontweight='bold', color='#b00020')
plt.suptitle('R-D Path Attribution — smooth pixel-level heatmap (RISE-style random-mask averaging, ~3000 masks)',
             fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig('cs_viz_outputs/rd_overlay.png', dpi=150, bbox_inches='tight'); plt.close()
print('saved cs_viz_outputs/rd_overlay.png')
