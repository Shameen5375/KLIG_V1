import sys, pickle
sys.path.insert(0, r'C:\Users\saame\KLIG_V1\infocube-main')
import numpy as np, torch, torch.nn.functional as F, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from transformers import CLIPModel, CLIPTokenizerFast
from torchvision.models import resnet50, ResNet50_Weights
from klig.image.attribution import ImageAttributor
from klig.image.stopping import find_sigma_stop

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_STEPS, N_SAMPLES = 50, 10
MEAN = torch.tensor([0.485,0.456,0.406]); STD = torch.tensor([0.229,0.224,0.225])
w = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=w).to(DEVICE).eval(); labels = w.meta['categories']
def denorm(x): return (x.detach().cpu()*STD.view(-1,1,1)+MEAN.view(-1,1,1)).clamp(0,1)
def raw_klig(x,t,sf): return ImageAttributor(model,n_steps=N_STEPS,n_samples=N_SAMPLES,sigma_final=sf).attribute(x,target=t).attr
def pos_flat(raw):
    if raw.dim()==4: raw=raw.squeeze(0)
    return raw.clamp(min=0).sum(0).detach().cpu().numpy().ravel()
def cos_dist(a,b):
    d = np.linalg.norm(a)*np.linalg.norm(b)
    return 1.0 if d < 1e-12 else float(1.0 - (a@b)/d)

# CLIP text embeddings for all classes
print('CLIP text embeddings...', flush=True)
cm = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
tok = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
txt = [f"a photo of a {l.split(',')[0].strip()}" for l in labels]
parts = []
with torch.no_grad():
    for i in range(0, len(txt), 64):
        inp = tok(txt[i:i+64], return_tensors="pt", padding=True, truncation=True, max_length=77).to(DEVICE)
        parts.append(cm.text_model(input_ids=inp['input_ids'], attention_mask=inp['attention_mask']).pooler_output.float().cpu())
clip_emb = F.normalize(torch.cat(parts), dim=-1)
def clip_dist(i,j): return float(1.0 - (clip_emb[i]*clip_emb[j]).sum())

ds = pickle.load(open('klig2_dist_cache/dataset.pkl','rb'))
print(f'scanning {len(ds)} images for Top-1/Top-2 KLIG map divergence...', flush=True)
pairs = []   # (d_sem, d_attr, idx, c1, c2, p1, p2, A1_2d, A2_2d)
for k, row in enumerate(ds):
    x = row['x'].to(DEVICE)
    with torch.no_grad():
        p = model(x).softmax(-1)[0]
    top2 = torch.topk(p, 2).indices.tolist(); c1, c2 = int(top2[0]), int(top2[1])
    p1, p2 = float(p[c1]), float(p[c2])
    sf = min(max(find_sigma_stop(model, x.squeeze(0), target=c1, tau=0.95), 1/256), 1.0)
    A1 = raw_klig(x.squeeze(0), c1, sf); A2 = raw_klig(x.squeeze(0), c2, sf)
    d_attr = cos_dist(pos_flat(A1), pos_flat(A2))
    d_sem = clip_dist(c1, c2)
    A1c = A1.squeeze(0).clamp(min=0).sum(0).detach().cpu().numpy()
    A2c = A2.squeeze(0).clamp(min=0).sum(0).detach().cpu().numpy()
    pairs.append((d_sem, d_attr, k, c1, c2, p1, p2, A1c, A2c))
    if k % 20 == 0: print(f'  {k}/{len(ds)}', flush=True)

best = max(pairs, key=lambda r: r[1])      # highest d_attr = clearest class-sensitive win
d_sem, d_attr, k, c1, c2, p1, p2, A1c, A2c = best
print(f'WIN idx={k}  {labels[c1][:20]}(p={p1:.2f}) vs {labels[c2][:20]}(p={p2:.2f})  d_attr={d_attr:.2f} d_sem={d_sem:.2f}', flush=True)

# ---- 4-panel figure ----
fig, ax = plt.subplots(1, 4, figsize=(16, 4.2), facecolor='white')
img = np.clip(denorm(ds[k]['x'][0]).permute(1,2,0).numpy(), 0, 1)
ax[0].imshow(img); ax[0].axis('off')
ax[0].set_title(f'Original\nTop-1: {labels[c1].split(",")[0][:16]} (p={p1:.2f})', fontsize=10, fontweight='bold')
for a, A, c, ttl in [(ax[1], A1c, c1, f'Why {labels[c1].split(",")[0][:14]}?'),
                     (ax[2], A2c, c2, f'Why {labels[c2].split(",")[0][:14]}?')]:
    disp = gaussian_filter(A, 2); vmax = max(np.percentile(disp,99), 1e-9)
    a.imshow(np.clip(disp/vmax,0,1), cmap='cividis'); a.axis('off')
    a.set_title(ttl, fontsize=10, fontweight='bold')
ax[1].text(1.02, 0.5, f'd_attr={d_attr:.2f}\nmaps differ\n-> class-sensitive',
           transform=ax[1].transAxes, ha='center', va='center', fontsize=8,
           bbox=dict(boxstyle='round', fc='#fff3cd', ec='#cc9'))

alls = np.array([r[0] for r in pairs]); alld = np.array([r[1] for r in pairs])
ax[3].scatter(alls, alld, color='lightgray', alpha=0.6, s=22)
z = np.polyfit(alls, alld, 1); xs = np.linspace(alls.min(), alls.max(), 50)
ax[3].plot(xs, np.poly1d(z)(xs), '--', color='gray')
ax[3].scatter([d_sem], [d_attr], color='red', s=260, marker='*', zorder=5, edgecolor='black')
ax[3].annotate('this example', (d_sem, d_attr), textcoords='offset points', xytext=(-10,-16),
               fontsize=9, color='red', fontweight='bold')
ax[3].set_xlabel('CLIP semantic distance  d_sem'); ax[3].set_ylabel('attribution cosine distance  d_attr')
ax[3].set_title(f'All {len(pairs)} Top1/Top2 pairs', fontsize=10, fontweight='bold'); ax[3].grid(alpha=.3)
fig.suptitle('Class Sensitivity - concrete win example (KLIG-Adaptive)', fontsize=13, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(r'C:\Users\saame\KLIG_V1\infocube-main\_cs_concrete.png', dpi=160, bbox_inches='tight')
print('OK saved _cs_concrete.png', flush=True)
