import pyarrow.parquet as pq        # MUST be before torch (native-lib segfault otherwise)
from huggingface_hub import hf_hub_download
import io
import numpy as np, torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image

REPO = 'evanarlian/imagenet_1k_resized_256'
SHARD = 'data/val-00000-of-00002-b5248be478d25e41.parquet'
# purely-animal classes: dogs (151-268) + domestic cats (281-285)
DOGCAT = set(range(151, 269)) | set(range(281, 286))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]
tfm = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])

print('locating cached val shard...', flush=True)
path = hf_hub_download(REPO, SHARD, repo_type='dataset')
print('reading parquet...', flush=True)
tbl = pq.read_table(path)
labels = tbl['label'].to_pylist()
idxs = [i for i, l in enumerate(labels) if l in DOGCAT]
print(f'{len(idxs)} dog/cat rows in shard 0', flush=True)

w = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=w).to(DEVICE).eval(); names = w.meta['categories']
imgcol = tbl['image']
best = (-1.0, None, None)
rng = np.random.default_rng(0); rng.shuffle(idxs)      # spread across dog/cat classes
for n, i in enumerate(idxs):
    if n > 1500:
        break
    d = imgcol[i].as_py()
    raw = d['bytes'] if isinstance(d, dict) else d
    im = Image.open(io.BytesIO(raw))
    if im.mode != 'RGB':
        im = im.convert('RGB')
    xx = tfm(im).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p = model(xx).softmax(-1)[0]; c = int(p.argmax()); cf = float(p[c])
    # prefer very confident dog/cat (clean close-ups, single subject, no clutter)
    if c in DOGCAT and cf > best[0]:
        best = (cf, xx.cpu(), c)
    if best[0] > 0.97:
        break
cf, xt, c = best
if xt is None:
    raise RuntimeError('no confident dog/cat found')
torch.save(xt, r'C:\Users\saame\KLIG_V1\infocube-main\_yorkie_x.pt')
print(f'animal: {names[c]}  cls={c}  conf={cf:.2f}  saved -> _yorkie_x.pt', flush=True)
