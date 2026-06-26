import sys, os
sys.path.insert(0, r'C:\Users\saame\KLIG_V1\infocube-main')
import numpy as np, torch, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter
from torchvision.models import resnet50, ResNet50_Weights
from captum.attr import Saliency
from klig.image.attribution import ImageAttributor
from klig.image.stopping import find_sigma_stop

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_STEPS, N_SAMPLES = 50, 10
MEAN = torch.tensor([0.485,0.456,0.406]); STD = torch.tensor([0.229,0.224,0.225])
w = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=w).to(DEVICE).eval(); labels = w.meta['categories']

def denormalize(x): return (x.detach().cpu()*STD.view(-1,1,1)+MEAN.view(-1,1,1)).clamp(0,1)
def collapse(raw):                          # (C,H,W)/(1,C,H,W) -> non-neg (H,W)
    if raw.dim() == 4: raw = raw.squeeze(0)
    return raw.clamp(min=0).sum(0).detach().cpu().numpy()
def raw_klig(x,t,sf): return ImageAttributor(model,n_steps=N_STEPS,n_samples=N_SAMPLES,sigma_final=sf).attribute(x,target=t).attr
def raw_vanilla(x,t):
    xb = (x if x.dim()==4 else x.unsqueeze(0)).clone().detach().requires_grad_(True)
    return Saliency(model).attribute(xb, target=t, abs=False).detach().squeeze(0)

def estimate_object_mask(x, a):             # a: non-neg (H,W); GrabCut seeded by attr
    H, W = a.shape
    seed = (a >= np.percentile(a, 80)).astype(np.uint8)
    img_rgb = denormalize(x[0]).permute(1,2,0).numpy()
    img_bgr = (img_rgb*255).clip(0,255).astype(np.uint8)[:,:,::-1].copy()
    gc = np.where(seed, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    gc[a >= np.percentile(a, 95)] = cv2.GC_FGD
    edge = np.zeros((H,W), np.uint8); b = max(H,W)//10
    edge[:b,:]=1; edge[-b:,:]=1; edge[:,:b]=1; edge[:,-b:]=1
    gc[(edge==1) & (a < np.percentile(a,10))] = cv2.GC_BGD
    try:
        bg = np.zeros((1,65),np.float64); fg = np.zeros((1,65),np.float64)
        cv2.grabCut(img_bgr, gc, None, bg, fg, 5, cv2.GC_INIT_WITH_MASK)
        return np.where((gc==cv2.GC_FGD)|(gc==cv2.GC_PR_FGD),1,0).astype(np.uint8)
    except Exception:
        return seed
def object_focus_ratio(a, mask):
    tot = a.sum()
    return float(a[mask==1].sum()/tot) if tot > 1e-12 else 0.0

# ── same Yorkie image as sens-n / ins-del ────────────────────────────────────
x = torch.load(r'C:\Users\saame\KLIG_V1\infocube-main\_yorkie_x.pt').to(DEVICE)
with torch.no_grad(): tgt = int(model(x).argmax(1)); conf = float(model(x).softmax(-1)[0,tgt])
print(f'image: {labels[tgt][:30]} cls={tgt} conf={conf:.2f}', flush=True)

sf = min(max(find_sigma_stop(model, x, target=tgt, tau=0.95), 1/256), 1.0)
a_klig = collapse(raw_klig(x, tgt, sf))
a_van  = collapse(raw_vanilla(x, tgt))

# shared object mask (GrabCut seeded by KLIG-Adaptive) — same mask scores both
mask = estimate_object_mask(x, gaussian_filter(a_klig, 2))
ofr_klig = object_focus_ratio(a_klig, mask)
ofr_van  = object_focus_ratio(a_van,  mask)
print(f'OFR  KLIG-Adaptive={ofr_klig:.3f}   Vanilla Grad={ofr_van:.3f}', flush=True)

def contour(ax, m, color):
    cnts,_ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if len(c) < 3: continue
        p = c[:,0,:]
        ax.plot(np.append(p[:,0],p[0,0]), np.append(p[:,1],p[0,1]), color=color, lw=1.4)

fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), facecolor='white')
img = np.clip(denormalize(x[0]).permute(1,2,0).numpy(), 0, 1)
axes[0].imshow(img); contour(axes[0], mask, 'lime'); axes[0].axis('off')
axes[0].set_title(f'{labels[tgt][:18]}\n(object mask)', fontsize=10, fontweight='bold')
for ax, a, name, ofr in [(axes[1], a_klig, 'KLIG-Adaptive', ofr_klig),
                         (axes[2], a_van,  'Vanilla Grad',  ofr_van)]:
    disp = gaussian_filter(a, 2); vmax = max(np.percentile(disp,99), 1e-9)
    ax.imshow(np.clip(disp/vmax, 0, 1), cmap='cividis'); contour(ax, mask, 'white'); ax.axis('off')
    ax.set_title(f'{name}\nOFR={ofr:.3f}', fontsize=10, fontweight='bold')
fig.suptitle(f'Object Focus Ratio - KLIG-Adaptive vs Vanilla Grad ({labels[tgt][:24]})',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(r'C:\Users\saame\KLIG_V1\infocube-main\_ofr_yorkie.png', dpi=180, bbox_inches='tight')
print('OK saved _ofr_yorkie.png', flush=True)
