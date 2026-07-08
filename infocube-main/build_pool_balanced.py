"""Build a class-BALANCED random pool: one image per ImageNet class (1000 total), spanning all
categories — cures the animal skew of pool1000 (which a small shuffle buffer biased to low classes).
The val stream is class-ordered, so we take the first image of each unseen class. Resumable
(HF caches shards to disk; re-run skips already-collected classes).
Run:  .venv/Scripts/python build_pool_balanced.py
"""
import sys, pickle, warnings
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
from pathlib import Path
import torch
from torchvision.models import resnet50, ResNet50_Weights
warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CACHE = Path('cs_viz_cache'); FILE = CACHE / 'pool1000_balanced.pkl'
w = ResNet50_Weights.IMAGENET1K_V2; model = resnet50(weights=w).to(DEVICE).eval(); preprocess = w.transforms()

pool = pickle.load(open(FILE,'rb')) if FILE.exists() else []
covered = {d['label'] for d in pool}
print(f'[setup] device={DEVICE} | resuming with {len(covered)}/1000 classes' if pool else '[fresh] one image per class')

from datasets import load_dataset
ds = load_dataset('evanarlian/imagenet_1k_resized_256', split='val', streaming=True)   # class-ordered
from tqdm import tqdm
try:
    for item in tqdm(ds, total=50000, desc='stratified stream'):
        if len(covered) >= 1000: break
        c = int(item['label'])
        if c in covered: continue                                   # already have this class
        im = item['image']
        if im.mode != 'RGB': im = im.convert('RGB')
        x = preprocess(im).unsqueeze(0).to(DEVICE)
        with torch.no_grad(): pr = torch.softmax(model(x)[0], -1).cpu()
        top2 = pr.topk(2).indices.tolist()
        pool.append({'idx': len(pool), 'label': c, 'x': x.squeeze(0).cpu(),
                     'high_cls': top2, 'high_probs': [float(pr[k]) for k in top2]})
        covered.add(c)
        if len(covered) % 25 == 0: pickle.dump(pool, open(FILE,'wb'))
except Exception as e:
    print(f'[warn] stream interrupted ({type(e).__name__}: {e}); keeping {len(covered)} — re-run to continue')
pickle.dump(pool, open(FILE,'wb'))
print(f'pool1000_balanced: {len(covered)}/1000 classes -> {FILE}'
      + ('  ✓ complete' if len(covered) >= 1000 else '  (re-run to finish)'))
