"""
Stream N random images from ImageNet-val (evanarlian/imagenet_1k_resized_256) → pool1000.pkl.
Shuffled stream = random sample (not the first N). Resumable: re-run after a network drop and it
continues from where it stopped. Each record: {idx, x (3,224,224 cpu), high_cls=top2, high_probs}.
Run:  .venv/Scripts/python build_pool_1000.py [N]   (default 1000)
"""
import sys, pickle, warnings
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
from pathlib import Path
import torch
from torchvision.models import resnet50, ResNet50_Weights
warnings.filterwarnings('ignore')

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
SEED = 123
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CACHE = Path('cs_viz_cache'); CACHE.mkdir(exist_ok=True); FILE = CACHE / 'pool1000.pkl'
print(f'[setup] device={DEVICE}  target={TARGET}')

w = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=w).to(DEVICE).eval(); preprocess = w.transforms()

pool = pickle.load(open(FILE, 'rb')) if FILE.exists() else []
print(f'[resume] {len(pool)} already collected' if pool else '[fresh] starting')
skip = len(pool)                                       # same seed → same order → skip what we have

from datasets import load_dataset as _hf
ds = _hf('evanarlian/imagenet_1k_resized_256', split='val', streaming=True).shuffle(seed=SEED, buffer_size=5000)
from tqdm import tqdm
seen, n = 0, 0
try:
    for item in tqdm(ds, total=TARGET, desc='streaming pool'):
        if len(pool) >= TARGET: break
        if n < skip: n += 1; continue                  # fast-forward past already-collected
        n += 1
        im = item['image']
        if im.mode != 'RGB': im = im.convert('RGB')
        x = preprocess(im).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pr = torch.softmax(model(x)[0], -1).cpu()
        top2 = pr.topk(2).indices.tolist()
        pool.append({'idx': len(pool), 'x': x.squeeze(0).cpu(),
                     'high_cls': top2, 'high_probs': [float(pr[c]) for c in top2]})
        if len(pool) % 100 == 0:                        # checkpoint
            pickle.dump(pool, open(FILE, 'wb'))
except Exception as e:
    print(f'[warn] stream interrupted ({type(e).__name__}: {e}); keeping {len(pool)} — re-run to continue')
pickle.dump(pool, open(FILE, 'wb'))
print(f'pool1000: {len(pool)}/{TARGET} images → {FILE}'
      + ('  ✓ complete' if len(pool) >= TARGET else '  (re-run to finish)'))
