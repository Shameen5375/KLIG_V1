import sys, pickle, os
sys.path.insert(0, r'C:\Users\saame\KLIG_V1\infocube-main')
import numpy as np, torch, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from torchvision.models import resnet50, ResNet50_Weights
from klig.image.attribution import ImageAttributor
from klig.image.stopping import find_sigma_stop

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_STEPS, N_SAMPLES, SG_SAMPLES = 50, 10, 50
MEAN = torch.tensor([0.485,0.456,0.406]); STD = torch.tensor([0.229,0.224,0.225])
w = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=w).to(DEVICE).eval(); labels = w.meta['categories']

def denorm(x): return (x.detach().cpu()*STD.view(-1,1,1)+MEAN.view(-1,1,1)).clamp(0,1)
def absmax(a):
    if a.dim()==4: a=a.squeeze(0)
    i=a.abs().argmax(0,keepdim=True); return a.gather(0,i).squeeze(0)
def raw_klig(x,t,sf): return ImageAttributor(model,n_steps=N_STEPS,n_samples=N_SAMPLES,sigma_final=sf).attribute(x,target=t).attr
def raw_smoothgrad(x,t):
    xb = x if x.dim()==4 else x.unsqueeze(0)
    std = 0.15 * float((xb.max()-xb.min()).item())
    noisy = (xb + torch.randn(SG_SAMPLES, *xb.shape[1:], device=xb.device)*std).requires_grad_(True)
    return torch.autograd.grad(model(noisy)[:, t].sum(), noisy)[0].detach().mean(0)

def gini(a2d):                              # sparsity: higher = sparser/more concentrated
    a = np.abs(np.asarray(a2d).ravel()); a = a[a > 0]
    if a.size == 0: return 0.0
    a = np.sort(a); n = a.size
    return float((2*np.sum(np.arange(1, n+1)*a))/(n*a.sum()) - (n+1)/n)

# same Irish wolfhound
x = torch.load(r'C:\Users\saame\KLIG_V1\infocube-main\_yorkie_x.pt').to(DEVICE)
with torch.no_grad(): tgt = int(model(x).argmax(1)); conf = float(model(x).softmax(-1)[0,tgt])
print(f'image: {labels[tgt][:30]} cls={tgt} conf={conf:.2f}', flush=True)

sf = min(max(find_sigma_stop(model, x, target=tgt, tau=0.95), 1/256), 1.0)
a_klig = absmax(raw_klig(x, tgt, sf)).detach().cpu().numpy()

a_sg = absmax(raw_smoothgrad(x, tgt)).detach().cpu().numpy()

g_klig, g_sg = gini(a_klig), gini(a_sg)
print(f'Gini (sparsity)  KLIG-Adaptive={g_klig:.3f}   SmoothGrad={g_sg:.3f}', flush=True)

fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), facecolor='white')
axes[0].imshow(np.clip(denorm(x[0]).permute(1,2,0).numpy(),0,1)); axes[0].axis('off')
axes[0].set_title(f'{labels[tgt][:18]}', fontsize=10, fontweight='bold')
for ax, a, name, g in [(axes[1], a_klig, 'KLIG-Adaptive', g_klig),
                       (axes[2], a_sg,  'SmoothGrad',     g_sg)]:
    disp = gaussian_filter(np.abs(a), 2); vmax = max(np.percentile(disp,99), 1e-9)
    ax.imshow(np.clip(disp/vmax,0,1), cmap='cividis'); ax.axis('off')
    ax.set_title(f'{name}\nGini={g:.3f}', fontsize=10, fontweight='bold')
fig.suptitle(f'Attribution Sparsity (Gini, higher=sparser) - KLIG-Adaptive vs SmoothGrad ({labels[tgt][:20]})',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(r'C:\Users\saame\KLIG_V1\infocube-main\_sparsity_yorkie.png', dpi=180, bbox_inches='tight')
print('OK saved _sparsity_yorkie.png', flush=True)
