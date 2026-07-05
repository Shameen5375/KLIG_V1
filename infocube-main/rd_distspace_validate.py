"""Validate rd_distspace against the brief's three checks:
  1. completeness-analog : total allocated rate correlates with F(input)-F(prior)
  2. sufficiency sanity  : a pure-noise patch gets LOW rate (can't buy back the logit)
  3. peer check          : R-D map correlates with KL-IG but is NOT equal (0 < rho < 1)
Run:  .venv/Scripts/python rd_distspace_validate.py [n]
"""
import sys, pickle, warnings
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import rd_distspace as RD
from klig.image.attribution import ImageAttributor
from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(DEVICE).eval()
_mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1); _std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

pool, seen = [], set()
for p in ['cs_viz_cache/cands.pkl','klig2_dist_cache/klig2_dist_multiprob.pkl','cs_gate_cache/pool.pkl']:
    if not Path(p).exists(): continue
    for d in pickle.load(open(p,'rb')):
        x = d['x']; x = x.squeeze(0) if x.dim()==4 else x; fp = round(float(x.float().sum()),1)
        if fp in seen: continue
        seen.add(fp); pool.append(x.cpu())
sel = pool[:K]
cfg = RD.RDConfig(n_mc=8, n_iter=150)
print(f'[setup] {len(sel)} images | {DEVICE}')

# ---- checks 1 & 3 across images -------------------------------------------------------
rates, gaps, rhos = [], [], []
for i, x in enumerate(sel):
    mu = x.to(DEVICE).unsqueeze(0)
    with torch.no_grad():
        tgt = int(model(mu).argmax())
        f_input = float(model(mu)[0, tgt])
        # F(prior): fully-loose probe (logvar=0 = prior variance), averaged
        eps = torch.randn(32, *mu.shape[1:], device=DEVICE)
        f_prior = float(model(mu + eps)[:, tgt].mean())
    r = RD.rd_attribution(model, mu, tgt, cfg)
    a_rd = r['attribution'].cpu().numpy()
    rates.append(r['info']['total_rate']); gaps.append(f_input - f_prior)
    # KL-IG peer map (same geometry, path-integral estimator)
    klig = ImageAttributor(model, n_steps=25, n_samples=8).attribute(x.to(DEVICE), target=tgt)
    a_kl = klig.attr_map('sumabs').detach().cpu().numpy()
    rho = spearmanr(a_rd.ravel(), np.abs(a_kl).ravel()).correlation
    rhos.append(rho)
    inf = r['info']
    print(f'  img{i}: tgt={tgt:3d}  rate={rates[-1]:8.1f}  F_in-F_prior={gaps[-1]:6.2f}  '
          f'D:{inf["D0"]:.2f}->{inf["D_final"]:.2f}(tau={inf["tau_abs"]:.2f})  '
          f'lam={inf["lam_final"]:.3f}  rho(R-D,KL-IG)={rho:+.3f}')

c1 = spearmanr(rates, gaps).correlation if len(rates) > 2 else float('nan')
print(f'\n[check 1] completeness-analog: spearman(total_rate, F_in-F_prior) = {c1:+.3f}  '
      f'({"PASS >0" if c1 > 0 else "FAIL"})')
mrho = float(np.nanmean(rhos))
print(f'[check 3] peer: mean rho(R-D, KL-IG) = {mrho:+.3f}  '
      f'({"PASS (0<rho<1: correlated, not identical)" if 0.0 < mrho < 0.98 else "SUSPECT"})')

# ---- check 2: pure-noise patch must get LOW rate --------------------------------------
x = sel[0].clone(); mu = x.to(DEVICE).unsqueeze(0)
with torch.no_grad(): tgt = int(model(mu).argmax())
patch = np.zeros((224,224), bool); patch[20:70, 20:70] = True   # inject noise here
xn = x.clone()
g = torch.Generator().manual_seed(0)
xn[:, 20:70, 20:70] = torch.randn(3,50,50, generator=g) * 3.0    # high-variance noise (normalized)
r = RD.rd_attribution(model, xn.to(DEVICE).unsqueeze(0), tgt, cfg)
a = r['attribution'].cpu().numpy()
in_patch = a[patch].mean(); out_patch = a[~patch].mean()
print(f'\n[check 2] noise-patch sufficiency: rate_in_patch={in_patch:.4f}  rate_elsewhere={out_patch:.4f}  '
      f'ratio={in_patch/(out_patch+1e-9):.2f}  ({"PASS (noise<=background)" if in_patch <= out_patch else "FAIL"})')
