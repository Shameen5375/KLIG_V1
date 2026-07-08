"""Segment-level RISE — an upgraded discriminative-region layer over segment occlusion.

Segment occlusion measures the MARGINAL effect of removing ONE segment at a time.
Segment-RISE keeps a RANDOM SUBSET of segments per mask and weights the class score by
which segments were present -> per-segment importance that captures INTERACTIONS, not just
marginal effect.  Segments in, segments out (fits the CS_struct pipeline); no gradients
(independent of the attribution leaderboard -> no circularity).

Run:  .venv/Scripts/python segment_rise.py [n]   (default 5)  -> occlusion vs RISE regions.
"""
import sys, pickle, warnings
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
N_IMG = int(sys.argv[1]) if len(sys.argv) > 1 else 5
FZ_SCALE, FZ_SIGMA, FZ_MINSIZE, SEED, EPS, DR_FRAC = 0.6, 0.8, 100, 0, 1e-8, 0.25
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.manual_seed(SEED); np.random.seed(SEED)
from torchvision.models import resnet50, ResNet50_Weights
_w = ResNet50_Weights.IMAGENET1K_V2; model = resnet50(weights=_w).to(DEVICE).eval(); cats = _w.meta['categories']
_mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1); _std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

def get_segments(x, scale=FZ_SCALE, sigma=FZ_SIGMA, min_size=FZ_MINSIZE):
    img = x.cpu().numpy().transpose(1,2,0); img = (img-img.min())/(np.ptp(img)+EPS)
    H,W,_ = img.shape; im = gaussian_filter(img,(sigma,sigma,0)).reshape(-1,3); Npx=H*W
    idx = np.arange(Npx).reshape(H,W)
    A = np.concatenate([idx[:,:-1].ravel(), idx[:-1,:].ravel()]); B = np.concatenate([idx[:,1:].ravel(), idx[1:,:].ravel()])
    Wt = np.sqrt(((im[A]-im[B])**2).sum(1)); o=np.argsort(Wt); A,B,Wt=A[o],B[o],Wt[o]
    par=np.arange(Npx); rank=np.zeros(Npx,int); size=np.ones(Npx,int); intd=np.zeros(Npx); k=scale
    def find(z):
        r=z
        while par[r]!=r: r=par[r]
        while par[z]!=r: par[z],z=r,par[z]
        return r
    for a,b,w in zip(A.tolist(),B.tolist(),Wt.tolist()):
        ra,rb=find(a),find(b)
        if ra==rb: continue
        if w<=min(intd[ra]+k/size[ra], intd[rb]+k/size[rb]):
            if rank[ra]<rank[rb]: ra,rb=rb,ra
            par[rb]=ra; size[ra]+=size[rb]; intd[ra]=max(intd[ra],intd[rb],w)
            if rank[ra]==rank[rb]: rank[ra]+=1
    for a,b in zip(A.tolist(),B.tolist()):
        ra,rb=find(a),find(b)
        if ra!=rb and (size[ra]<min_size or size[rb]<min_size):
            if rank[ra]<rank[rb]: ra,rb=rb,ra
            par[rb]=ra; size[ra]+=size[rb]
    roots=np.array([find(i) for i in range(Npx)]); _,seg=np.unique(roots,return_inverse=True)
    return seg.reshape(H,W)

# ── Option B: SEGMENT-LEVEL RISE ─────────────────────────────────────────────────────────
@torch.no_grad()
def segment_rise_region(model, x, segments, y1, y2, N=2000, p_on=0.5, frac=DR_FRAC, batch=64, seed=SEED):
    """x: (3,H,W) normalized. segments: (H,W) int labels 0..K-1.
    Random SEGMENT masks (keep subset of segments ON); per-segment importance = avg class
    score when that segment is present. Returns disc (imp_y1-imp_y2), region mask R, imp1, imp2."""
    dev = x.device; K = int(segments.max()) + 1
    seg_t = torch.from_numpy(segments.astype(np.int64)).to(dev)                 # (H,W)
    rng = np.random.default_rng(seed)
    acc1 = np.zeros(K); acc2 = np.zeros(K); cnt = np.zeros(K); done = 0
    while done < N:
        b = min(batch, N - done)
        on = rng.random((b, K)) < p_on                                         # (b,K) which segments ON
        on_t = torch.from_numpy(on).float().to(dev)
        masks = on_t[:, seg_t]                                                  # (b,H,W) paint per-segment on/off
        masked = x.unsqueeze(0) * masks.unsqueeze(1)                            # (b,3,H,W)  (OFF segments -> 0)
        p = F.softmax(model(masked), -1)
        s1 = p[:, y1].cpu().numpy(); s2 = p[:, y2].cpu().numpy()                # (b,)
        acc1 += (s1[:, None] * on).sum(0); acc2 += (s2[:, None] * on).sum(0); cnt += on.sum(0)
        done += b
    imp1 = acc1 / np.maximum(cnt, 1); imp2 = acc2 / np.maximum(cnt, 1)
    disc = imp1 - imp2                                                          # per-segment discriminative score
    k_top = max(1, int(frac * K)); top = np.argsort(np.abs(disc))[-k_top:]
    R = np.isin(segments, top)                                                  # binary pixel region
    return disc, R, imp1, imp2

# ── standalone segment-RISE region viz ──────────────────────────────────────────────────
def boundaries(seg):
    b = np.zeros(seg.shape, bool)
    b[:-1,:] |= seg[:-1,:]!=seg[1:,:]; b[1:,:] |= seg[:-1,:]!=seg[1:,:]
    b[:,:-1] |= seg[:,:-1]!=seg[:,1:]; b[:,1:] |= seg[:,:-1]!=seg[:,1:]
    return b
def region_overlay(im, segments, disc, frac=DR_FRAC):
    K = len(disc); k_top = max(1, int(frac*K)); top = np.argsort(np.abs(disc))[-k_top:]
    rd = np.zeros(K); rd[top] = disc[top]; sgn = rd[segments]
    mag = np.abs(sgn)/(np.abs(sgn).max()+EPS); al=(0.55*mag)[...,None]
    col = np.where((sgn>0)[...,None], np.array([0.85,0.1,0.1]), np.array([0.1,0.3,0.9]))
    o = np.clip(im*(1-al)+col*al, 0, 1); o[boundaries(segments)] = [1.0,1.0,0.0]
    return o

if __name__ == '__main__':
    srcs=['cs_viz_cache/cands.pkl','klig2_dist_cache/klig2_dist_multiprob.pkl',
          'klig2_val_cache/klig2_dist_multiprob.pkl','cs_gate_cache/pool.pkl']
    CANDS,seen=[],set()
    for s in srcs:
        if not Path(s).exists(): continue
        for d in pickle.load(open(s,'rb')):
            if len(d.get('high_cls',[]))<2: continue
            x=d['x']; x=x.squeeze(0) if x.dim()==4 else x; fp=round(float(x.float().sum()),1)
            if fp in seen: continue
            seen.add(fp); CANDS.append({'x':x.cpu(),'high_cls':[int(c) for c in d['high_cls'][:2]]})
    import random as _r; _r.Random(SEED).shuffle(CANDS)
    sel,used=[],set()
    for d in CANDS:
        c=int(d['high_cls'][0])
        if c in used: continue
        used.add(c); sel.append(d)
        if len(sel)>=N_IMG: break
    print(f'[setup] {len(sel)} images | segment-RISE (N=2000 masks, interactions) discriminative region')
    fig, ax = plt.subplots(len(sel), 2, figsize=(7, 3.3*len(sel)), facecolor='white')
    if len(sel)==1: ax = ax[None,:]
    from tqdm import tqdm
    for r, d in enumerate(tqdm(sel, desc='segment-RISE')):
        x = d['x'].to(DEVICE); y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
        with torch.no_grad(): probs = F.softmax(model(x.unsqueeze(0))[0],0)
        if probs[y2] > probs[y1]: y1, y2 = y2, y1
        seg = get_segments(x)
        d_rise, R, i1, i2 = segment_rise_region(model, x, seg, y1, y2, N=2000)
        im = (x.cpu()*_std+_mean).clamp(0,1).permute(1,2,0).numpy()
        ax[r,0].imshow(im); ax[r,0].axis('off')
        ax[r,0].text(-0.07,0.5,f"top-1: {cats[y1].split(',')[0]}\ntop-2: {cats[y2].split(',')[0]}",
                     transform=ax[r,0].transAxes, rotation=90, va='center', ha='center', fontsize=9, fontweight='bold')
        ax[r,1].imshow(region_overlay(im, seg, d_rise)); ax[r,1].axis('off')
    for j,t in enumerate(['input (top-1 / top-2)','segment-RISE discriminative region']):
        ax[0,j].set_title(t, fontsize=11, fontweight='bold')
    plt.suptitle('Segment-RISE discriminative region (N=2000 random segment-subset masks) — '
                 'red = drives top-1, blue = top-2, yellow = superpixel edges', fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.97]); out='cs_viz_outputs/segment_rise.png'
    plt.savefig(out, dpi=145, bbox_inches='tight'); plt.close()
    print(f'saved {out}')
