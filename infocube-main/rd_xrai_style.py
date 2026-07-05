"""R-D attribution shown in the XRAI figure format (Kapishnikov et al., ICCV 2019, arXiv:1906.02825):
heatmap overlay + top-k% most-salient region reveal (rest grayed)."""
import sys, pickle, warnings
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
import numpy as np, torch, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.cm as cm
from scipy.ndimage import gaussian_filter
warnings.filterwarnings('ignore')
K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
ARCH = sys.argv[2] if len(sys.argv) > 2 else 'vit'          # 'vit' or 'resnet'
OP = sys.argv[3] if len(sys.argv) > 3 else 'noise'          # 'noise' (amplitude) or 'blur' (spatial)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import rd_attribution as RD
if ARCH == 'vit':
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    _w = ViT_B_16_Weights.IMAGENET1K_V1; model = vit_b_16(weights=_w).to(DEVICE).eval()
else:
    from torchvision.models import resnet50, ResNet50_Weights
    _w = ResNet50_Weights.IMAGENET1K_V2; model = resnet50(weights=_w).to(DEVICE).eval()
cats = _w.meta['categories']; wrap = RD.ModelWrapper(model, DEVICE)
print(f'[setup] arch={ARCH}')
_mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1); _std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
store = pickle.load(open('cs_viz_outputs/segment_store_vit.pkl','rb'))
imgs = [((d['x'].squeeze(0) if d['x'].dim()==4 else d['x'])*_std+_mean).clamp(0,1) for d in store[:K]]

def heat_overlay(img, a):
    a = (a-a.min())/(np.percentile(a,99)-a.min()+1e-8); a = np.clip(a,0,1)
    heat = cm.get_cmap('jet')(a)[...,:3]; al=(0.30+0.55*a)[...,None]
    return np.clip(img*(1-al)+heat*al,0,1)
def reveal(img, a, frac):                                   # XRAI: keep top-frac salient, FULLY gray the rest
    thr = np.quantile(a, 1-frac); m = (a>=thr)[...,None]
    gray = np.full_like(img, 0.5)                           # solid flat gray for non-salient
    return np.where(m, img, gray)

FRACS = [0.05, 0.15, 0.40]
ncol = 2 + len(FRACS)
fig, ax = plt.subplots(K, ncol, figsize=(2.7*ncol, 2.7*K), facecolor='white')
if K == 1: ax = ax[None, :]
from tqdm import tqdm
for r, img01 in enumerate(tqdm(imgs, desc='xrai-style')):
    x = img01.unsqueeze(0).to(DEVICE); cls = int(wrap.logits(x)[0].argmax())
    op = RD.NoiseOperator() if OP == 'noise' else RD.BlurOperator(); lvl = 0.5 if OP == 'noise' else 10.0
    a = gaussian_filter(np.clip(RD.rise_smooth_map(wrap, x, op, lvl, cls, n_masks=3000, s=8, seed=0),0,None), 4)
    im = img01.permute(1,2,0).cpu().numpy()
    ax[r,0].imshow(im); ax[r,0].axis('off'); ax[r,0].set_title(cats[cls].split(',')[0] if r==0 else '', fontsize=9)
    if r==0: ax[r,0].set_ylabel('')
    ax[r,1].imshow(heat_overlay(im,a)); ax[r,1].axis('off')
    for j,f in enumerate(FRACS):
        ax[r,2+j].imshow(reveal(im,a,f)); ax[r,2+j].axis('off')
titles = ['input','R-D heatmap']+[f'top {int(f*100)}%' for f in FRACS]
for j,t in enumerate(titles): ax[0,j].set_title(t, fontsize=11, fontweight='bold')
plt.suptitle(f'R-D attribution ({ARCH.upper()}) in XRAI format (Kapishnikov et al., ICCV 2019, arXiv:1906.02825): heatmap + top-k% region reveal',
             fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.97]); out = f'cs_viz_outputs/rd_xrai_style_{ARCH}_{OP}.png'
plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
print('saved', out)
