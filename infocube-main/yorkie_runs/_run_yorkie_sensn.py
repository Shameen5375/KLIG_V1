import sys
sys.path.insert(0, r'C:\Users\saame\KLIG_V1\infocube-main')
from datasets import load_dataset          # MUST be first (segfaults if after klig/captum)
import numpy as np, torch, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from klig.image.attribution import ImageAttributor
from klig.image.stopping import find_sigma_stop

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_STEPS, N_SAMPLES = 50, 10
MEAN = torch.tensor([0.485,0.456,0.406]); STD = torch.tensor([0.229,0.224,0.225])
w = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=w).to(DEVICE).eval(); labels = w.meta['categories']
tfm = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                 T.Normalize(MEAN.tolist(), STD.tolist())])

def denorm(x): return (x.cpu()*STD.view(-1,1,1)+MEAN.view(-1,1,1)).clamp(0,1)
def absmax(a):
    if a.dim()==4: a=a.squeeze(0)
    i=a.abs().argmax(0,keepdim=True); return a.gather(0,i).squeeze(0)
def raw_klig(x,t,sf): return ImageAttributor(model,n_steps=N_STEPS,n_samples=N_SAMPLES,sigma_final=sf).attribute(x,target=t).attr

# locate Yorkshire terrier class
YORK = [i for i,l in enumerate(labels) if 'yorkshire' in l.lower()]
print('Yorkshire terrier class:', YORK, [labels[i] for i in YORK], flush=True)
YCLS = YORK[0]

# HF label == standard ImageNet index, so label 187 = Yorkshire terrier.
# val is class-ordered; filter on label (cheap - image not decoded unless it matches),
# pick the most confident Yorkie. Cache the chosen image so ins/del reuses the SAME one.
import os
_XPT = r'C:\Users\saame\KLIG_V1\infocube-main\_yorkie_x.pt'
tgt = YCLS
if os.path.exists(_XPT):
    x = torch.load(_XPT).to(DEVICE)
    with torch.no_grad():
        pr = model(x).softmax(-1)[0]; tgt = int(pr.argmax()); cf = float(pr[tgt])
    print(f'[cache] loaded _yorkie_x.pt  pred={labels[tgt][:24]} conf={cf:.2f}', flush=True)
else:
    ds = load_dataset('evanarlian/imagenet_1k_resized_256', split='val', streaming=True)
    best = (-1.0, None); seen = 0
    for it in ds:
        if it['label'] != YCLS:
            continue
        seen += 1
        im = it['image']
        if im.mode != 'RGB': im = im.convert('RGB')
        xx = tfm(im).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            p = model(xx).softmax(-1)[0]; c = int(p.argmax()); pcf = float(p[c])
        if c == YCLS and pcf > best[0]:
            best = (pcf, xx)
        if best[0] > 0.7 or seen >= 50:   # 50 Yorkies in val; stop early if confident
            break
    if best[1] is None:
        raise RuntimeError('No Yorkshire terrier found')
    x, cf = best[1], best[0]
    torch.save(x.cpu(), _XPT)
    print(f'found Yorkie (conf={cf:.2f}, from {seen} val Yorkies); cached -> _yorkie_x.pt', flush=True)
print(f'image: {labels[tgt][:30]}  cls={tgt}  conf={cf:.2f}', flush=True)

x_raw = x.squeeze(0)
sf = min(max(find_sigma_stop(model, x_raw, target=tgt, tau=0.95), 1/256), 1.0)
attr = absmax(raw_klig(x_raw, tgt, sf)).detach().cpu().numpy()
H, W = attr.shape; C = x_raw.shape[0]; Npix = H*W
attr_flat = attr.ravel()
# BLACK masked pixels: true black (0,0,0) in normalized space = (0 - mean)/std
baseline = ((torch.zeros(3,1,1) - MEAN.view(-1,1,1)) / STD.view(-1,1,1)).to(DEVICE).expand(C, H, W).contiguous()
with torch.no_grad(): F0 = float(model(x).softmax(-1)[0, tgt])

T_SUB, FRAC, SHOW = 200, 0.10, 4
n = int(FRAC * Npix)
rng = np.random.default_rng(0)
xf, bf = x_raw.view(C,-1), baseline.view(C,-1)
masks = [rng.choice(Npix, n, replace=False) for _ in range(T_SUB)]
asum, dF, ex = [], [], []
BS = 50
for i in range(0, T_SUB, BS):
    chunk = masks[i:i+BS]; batch = []
    for pix in chunk:
        img = xf.clone(); pt = torch.tensor(pix, device=DEVICE)
        img[:, pt] = bf[:, pt]; batch.append(img.view(C, H, W))
    with torch.no_grad():
        fm = model(torch.stack(batch)).softmax(-1)[:, tgt].cpu().numpy()
    for j, pix in enumerate(chunk):
        asum.append(float(attr_flat[pix].sum())); dF.append(F0 - float(fm[j]))
        t = i + j
        if t < SHOW:
            ov = denorm(x_raw).permute(1,2,0).numpy().copy()
            m2 = np.zeros(Npix, bool); m2[pix] = True; ov[m2.reshape(H,W)] = 0.0  # black
            ex.append((ov, asum[-1], dF[-1]))
asum = np.array(asum); dF = np.array(dF)
rho, _ = stats.pearsonr(asum, dF)

fig = plt.figure(figsize=(3*SHOW, 7), facecolor='white')
gs = fig.add_gridspec(2, SHOW, height_ratios=[1, 1.3])
for c,(ov,a_,d_) in enumerate(ex):
    ax = fig.add_subplot(gs[0, c]); ax.imshow(ov); ax.axis('off')
    ax.set_title(f'Sattr={a_:.2f}\ndF={d_:.3f}', fontsize=9)
sc = fig.add_subplot(gs[1, :])
sc.scatter(asum, dF, color='teal', alpha=0.5, s=20)
z = np.polyfit(asum, dF, 1); xs = np.linspace(asum.min(), asum.max(), 50)
sc.plot(xs, np.poly1d(z)(xs), '--', color='black')
sc.set_xlabel('attribution sum over masked pixels'); sc.set_ylabel('logit drop dF'); sc.grid(alpha=.3)
sc.annotate(f'Pearson rho = {rho:.2f}', xy=(0.03,0.92), xycoords='axes fraction', fontsize=12, fontweight='bold')
fig.suptitle(f'Sensitivity-n - {labels[tgt][:24]} (KLIG-Adaptive)  n={n} px ({int(FRAC*100)}%), T={T_SUB}',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(r'C:\Users\saame\KLIG_V1\infocube-main\_sensn_yorkie.png', dpi=150, bbox_inches='tight')
print(f'rho={rho:.3f}  OK saved _sensn_yorkie.png', flush=True)
