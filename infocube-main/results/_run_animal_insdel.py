import sys
sys.path.insert(0, r'C:\Users\saame\KLIG_V1\infocube-main')
from datasets import load_dataset          # MUST be first (segfaults if after klig/captum)
import numpy as np, torch, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from klig.image.attribution import ImageAttributor
from klig.image.stopping import find_sigma_stop

_trapz = np.trapezoid
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_STEPS, N_SAMPLES = 50, 10
MEAN = torch.tensor([0.485, 0.456, 0.406]); STD = torch.tensor([0.229, 0.224, 0.225])
w = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=w).to(DEVICE).eval(); labels = w.meta['categories']
tfm = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                 T.Normalize(MEAN.tolist(), STD.tolist())])

def denorm(x): return (x.cpu() * STD.view(-1,1,1) + MEAN.view(-1,1,1)).clamp(0,1)
def absmax(a):
    if a.dim() == 4: a = a.squeeze(0)
    i = a.abs().argmax(0, keepdim=True); return a.gather(0, i).squeeze(0)
def raw_klig(x, t, sf):
    return ImageAttributor(model, n_steps=N_STEPS, n_samples=N_SAMPLES,
                           sigma_final=sf).attribute(x, target=t).attr

print('streaming for a confident animal image...', flush=True)
ds = load_dataset('evanarlian/imagenet_1k_resized_256', split='train', streaming=True)
x = None
for n, it in enumerate(ds):
    im = it['image']
    if im.mode != 'RGB': im = im.convert('RGB')
    xx = tfm(im).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p = model(xx).softmax(-1)[0]; c = int(p.argmax()); cf = float(p[c])
    if c < 398 and cf > 0.5:
        x = xx; tgt = c; break
    if n > 800: break
print(f'animal: {labels[tgt][:30]}  cls={tgt}  conf={cf:.2f}', flush=True)

x_raw = x.squeeze(0)
sf = min(max(find_sigma_stop(model, x_raw, target=tgt, tau=0.95), 1/256), 1.0)
attr = absmax(raw_klig(x_raw, tgt, sf)).detach().cpu().numpy()
H, W = attr.shape; C = x_raw.shape[0]; Npix = H * W
order = np.argsort(attr.ravel())[::-1]
black = ((torch.zeros(3,1,1) - MEAN.view(-1,1,1)) / STD.view(-1,1,1)).to(DEVICE).expand(C, H, W).contiguous()
STEPS = [0.0, 0.1, 0.3, 0.6, 1.0]; NC = 50

def Fp(img):
    with torch.no_grad(): return float(model(img.unsqueeze(0)).softmax(-1)[0, tgt])

def run(mode):
    xf, bf = x_raw.view(C, -1), black.view(C, -1); fr, lg, fm = [], [], {}
    for frac in np.linspace(0, 1, NC):
        k = int(frac * Npix); img = (x_raw if mode == 'delete' else black).clone().view(C, -1)
        src = bf if mode == 'delete' else xf
        if k > 0:
            pix = torch.tensor(order[:k].copy(), device=DEVICE); img[:, pix] = src[:, pix]
        img = img.view(C, H, W); fr.append(frac); lg.append(Fp(img))
        for s in STEPS:
            if abs(frac - s) < (0.5 / NC): fm[s] = denorm(img).permute(1, 2, 0).numpy()
    return fr, lg, fm

dfx, dlg, dF = run('delete'); ifx, ilg, iF = run('insert')
fig = plt.figure(figsize=(3*len(STEPS), 9), facecolor='white')
gs = fig.add_gridspec(3, len(STEPS), height_ratios=[1, 1, 1.1])
for ci, s in enumerate(STEPS):
    li = min(range(len(dfx)), key=lambda j: abs(dfx[j] - s))
    a = fig.add_subplot(gs[0, ci]); a.imshow(dF[s]); a.axis('off')
    a.set_title(f'{int(s*100)}% removed\nF={dlg[li]:.2f}', fontsize=9)
    a2 = fig.add_subplot(gs[1, ci]); a2.imshow(iF[s]); a2.axis('off')
    a2.set_title(f'{int(s*100)}% revealed\nF={ilg[li]:.2f}', fontsize=9)
cv = fig.add_subplot(gs[2, :])
cv.plot(dfx, dlg, color='red', lw=2, label='Deletion (AUC low=good)')
cv.plot(ifx, ilg, color='green', lw=2, label='Insertion (AUC high=good)')
for s in STEPS: cv.axvline(s, ls=':', color='gray', alpha=.6)
cv.set_xlabel('fraction of pixels'); cv.set_ylabel('target prob F'); cv.legend(fontsize=9); cv.grid(alpha=.3)
fig.suptitle(f'Insertion/Deletion (PIXEL form) - {labels[tgt][:24]} (KLIG-Adaptive)  '
             f'delAUC={_trapz(dlg, dfx):.3f}  insAUC={_trapz(ilg, ifx):.3f}',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(r'C:\Users\saame\KLIG_V1\infocube-main\_ins_del_pixel_animal.png', dpi=150, bbox_inches='tight')
print('OK saved _ins_del_pixel_animal.png', flush=True)
