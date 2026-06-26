import sys
sys.path.insert(0, r'C:\Users\saame\KLIG_V1\infocube-main')
import numpy as np, torch, torch.nn.functional as F, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from torchvision.models import resnet50, ResNet50_Weights
from captum.attr import IntegratedGradients, Saliency
from klig.image.attribution import ImageAttributor
from klig.image.stopping import find_sigma_stop

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_STEPS, N_SAMPLES, IG_STEPS, SG_SAMPLES, EG_SAMPLES = 50, 10, 50, 50, 50
MEAN = torch.tensor([0.485,0.456,0.406]); STD = torch.tensor([0.229,0.224,0.225])
w = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=w).to(DEVICE).eval(); labels = w.meta['categories']

def denorm(x): return (x.detach().cpu()*STD.view(-1,1,1)+MEAN.view(-1,1,1)).clamp(0,1)
def _xb(x): return (x if x.dim()==4 else x.unsqueeze(0)).to(DEVICE)
def absmax(a):
    if a.dim()==4: a=a.squeeze(0)
    i=a.abs().argmax(0,keepdim=True); return a.gather(0,i).squeeze(0)
def gini(a2d):
    a = np.abs(np.asarray(a2d).ravel()); a = a[a>0]
    if a.size==0: return 0.0
    a = np.sort(a); n=a.size
    return float((2*np.sum(np.arange(1,n+1)*a))/(n*a.sum()) - (n+1)/n)

def make_blur(xb, ks=51, sigma=16.0):
    c = torch.arange(ks, dtype=torch.float32, device=xb.device) - ks//2
    k = torch.exp(-0.5*(c/sigma)**2); k = k/k.sum()
    kh = k.view(1,1,-1,1).expand(3,-1,-1,-1); kw = k.view(1,1,1,-1).expand(3,-1,-1,-1)
    o = F.conv2d(xb, kh, padding=(ks//2,0), groups=3); return F.conv2d(o, kw, padding=(0,ks//2), groups=3)

def raw_klig(x,t,sf): return ImageAttributor(model,n_steps=N_STEPS,n_samples=N_SAMPLES,sigma_final=sf).attribute(x,target=t).attr
def raw_ig_zero(x,t):
    xb=_xb(x); return IntegratedGradients(model).attribute(xb,baselines=torch.zeros_like(xb),target=t,n_steps=IG_STEPS,method='gausslegendre',internal_batch_size=IG_STEPS).detach().squeeze(0)
def raw_blur_ig(x,t):
    xb=_xb(x); return IntegratedGradients(model).attribute(xb,baselines=make_blur(xb),target=t,n_steps=IG_STEPS,method='gausslegendre',internal_batch_size=IG_STEPS).detach().squeeze(0)
def raw_smoothgrad(x,t):
    xb=_xb(x); std=0.15*float((xb.max()-xb.min()).item())
    noisy=(xb+torch.randn(SG_SAMPLES,*xb.shape[1:],device=xb.device)*std).requires_grad_(True)
    return torch.autograd.grad(model(noisy)[:,t].sum(),noisy)[0].detach().mean(0)
def raw_vanilla(x,t):
    xb=_xb(x).clone().detach().requires_grad_(True); return Saliency(model).attribute(xb,target=t,abs=False).detach().squeeze(0)
def raw_expgrad(x,t):
    xb=_xb(x); bg=torch.randn(EG_SAMPLES,*xb.shape[1:],device=xb.device); al=torch.rand(EG_SAMPLES,1,1,1,device=xb.device)
    interp=(bg+al*(xb-bg)).requires_grad_(True); g=torch.autograd.grad(model(interp)[:,t].sum(),interp)[0]
    return (g.detach()*(xb-bg)).mean(0)
def raw_idg(x,t):
    xb=_xb(x).clone().detach().requires_grad_(True); g=torch.autograd.grad(model(xb)[:,t].sum(),xb)[0]; return (xb*g).detach().squeeze(0)

x = torch.load(r'C:\Users\saame\KLIG_V1\infocube-main\_yorkie_x.pt').to(DEVICE)
with torch.no_grad(): tgt=int(model(x).argmax(1)); conf=float(model(x).softmax(-1)[0,tgt])
sf = min(max(find_sigma_stop(model, x, target=tgt, tau=0.95), 1/256), 1.0)
print(f'image: {labels[tgt][:30]} cls={tgt} conf={conf:.2f}', flush=True)

METHODS = {
    'KL-IG (adaptive)': lambda: raw_klig(x,tgt,sf),
    'KL-IG (sig=0.25)': lambda: raw_klig(x,tgt,0.25),
    'IDG':              lambda: raw_idg(x,tgt),
    'ExpGrad':          lambda: raw_expgrad(x,tgt),
    'IG-zero':          lambda: raw_ig_zero(x,tgt),
    'Blur-IG':          lambda: raw_blur_ig(x,tgt),
    'SmoothGrad':       lambda: raw_smoothgrad(x,tgt),
    'Vanilla Grad':     lambda: raw_vanilla(x,tgt),
}
results = {}
for name, fn in METHODS.items():
    a = absmax(fn()).detach().cpu().numpy()
    results[name] = (gini(a), a)
    print(f'  {name:18s} Gini={results[name][0]:.3f}', flush=True)

ranked = sorted(results.items(), key=lambda kv: kv[1][0], reverse=True)
print('\n=== Gini ranking (higher = sparser) ===', flush=True)
for r,(name,(g,_)) in enumerate(ranked,1):
    print(f'  {r}. {name:18s} {g:.3f}', flush=True)
hi_name,(hi_g,hi_a) = ranked[0]; lo_name,(lo_g,lo_a) = ranked[-1]

fig, axes = plt.subplots(1, 3, figsize=(11, 4.4), facecolor='white')
axes[0].imshow(np.clip(denorm(x[0]).permute(1,2,0).numpy(),0,1)); axes[0].axis('off')
axes[0].set_title(f'{labels[tgt][:18]}', fontsize=10, fontweight='bold')
for ax,(name,a,g,tag) in [(axes[1],(hi_name,hi_a,hi_g,'HIGHEST (sparsest)')),
                          (axes[2],(lo_name,lo_a,lo_g,'LOWEST (densest)'))]:
    disp=gaussian_filter(np.abs(a),2); vmax=max(np.percentile(disp,99),1e-9)
    ax.imshow(np.clip(disp/vmax,0,1),cmap='cividis'); ax.axis('off')
    ax.set_title(f'{tag}\n{name}  Gini={g:.3f}', fontsize=10, fontweight='bold')
fig.suptitle(f'Attribution Sparsity (Gini) - all 8 methods, highest vs lowest ({labels[tgt][:20]})',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(r'C:\Users\saame\KLIG_V1\infocube-main\_sparsity_all.png', dpi=180, bbox_inches='tight')
print('OK saved _sparsity_all.png', flush=True)
