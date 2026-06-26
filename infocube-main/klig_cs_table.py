"""
Class-sensitivity computation as a TABLE (offline; uses cached images + CLIP cache).

For each selected image (Top-1 = y1, Top-2 = y2):
  1. attribution maps A(y1), A(y2) per method
  2. d_attr = 1 - cos(A(y1), A(y2))           <- cosine distance of the maps   (SHOWN)
  3. d_sem  = 1 - cos(e_y1, e_y2)  over CLIP   <- semantic distance of labels   (SHOWN)
Then: Spearman rho(d_sem, d_attr) over the Top-1/Top-2 pairs per method
      rho > 0 -> class-conditional ; rho ~ 0 -> class-blind.

Outputs:  cs_table.png  (per-image table + rho summary)  and  klig_vs_klig2.png (maps).
Run:  .venv/Scripts/python.exe klig_cs_table.py
"""
import os, sys, math, pickle, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())
from klig import KLIntegratedGradients, make_phi_from_layer, KLIGSquared
from klig.image.stopping import find_sigma_stop
from klig.core.path import LinearPath
from torchvision.models import resnet50, ResNet50_Weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}")
CACHE = "klig2_dist_cache"

# config (cells 3 + 26)
SIGMA_FINAL = 0.25
LR_MU, LR_LV = 0.05, 0.10
LOSS_STOP = 1e-3
LV_FLOOR = 2 * math.log(1 / 256)
LV_CEIL = 4.0
MU_MIN, MU_MAX = -2.64, 2.64
N_STEPS, N_MC = 25, 3
T_DESC, N_MC_DESC = 25, 8
N_STEPS_HQ, N_MC_HQ = 50, 10
T_DESC_HQ, N_MC_DESC_HQ = 50, 16

ORDER = ["KL-IG (linear)", "KLIG-Adaptive", "KL-IG²", "KL-IG² (adaptive)"]
COLORS = {"KL-IG (linear)": "#333333", "KLIG-Adaptive": "#2d6a2d",
          "KL-IG²": "#e41a1c", "KL-IG² (adaptive)": "#8b0000"}

# model + helpers
weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights).to(DEVICE).eval()
imagenet_labels = weights.meta["categories"]
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
def denormalize(x): return x.cpu() * _STD + _MEAN
def absmax_collapse(a):
    if a.dim() == 4: a = a.squeeze(0)
    idx = a.abs().argmax(dim=0)
    return a.gather(0, idx.unsqueeze(0)).squeeze(0)
phi = make_phi_from_layer(model, "layer4")
print("model + phi ready")

# CLIP (offline from cache) -> semantic distance
import torch.nn.functional as _F
from transformers import CLIPModel, CLIPTokenizerFast
_clip_mdl = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to(DEVICE).eval()
_clip_tok = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
_txt = [f"a photo of a {l.split(',')[0].strip()}" for l in imagenet_labels]
_parts = []
with torch.no_grad():
    for i in range(0, len(_txt), 64):
        inp = _clip_tok(_txt[i:i + 64], return_tensors="pt", padding=True, truncation=True, max_length=77).to(DEVICE)
        f = _clip_mdl.text_model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"]).pooler_output
        _parts.append(f.float().cpu())
_clip_emb = _F.normalize(torch.cat(_parts, 0), dim=-1)
def clip_semantic_dist(ci, cj):
    return float(1.0 - (_clip_emb[ci] * _clip_emb[cj]).sum().item())
print("CLIP ready (offline)")

def cosine_dist_cs(a_i, a_j):            # d_attr = 1 - cos  (signed)
    ai = a_i.astype(np.float64).ravel(); aj = a_j.astype(np.float64).ravel()
    den = np.linalg.norm(ai) * np.linalg.norm(aj)
    return 1.0 if den < 1e-12 else float(1.0 - (ai @ aj) / den)

# images (bin-stratified, same as the scatter) + their counterfactuals
multi_imgs = pickle.load(open(f"{CACHE}/klig2_dist_multiprob.pkl", "rb"))
BINS = [(0.0, 0.20), (0.20, 0.35), (0.35, 0.50), (0.50, 0.70), (0.70, 1.01)]
MAX_PER_BIN = 5
buckets = {i: [] for i in range(len(BINS))}
for d in multi_imgs:
    if len(d.get("high_cls", [])) < 2:
        continue
    cd = clip_semantic_dist(d["high_cls"][0], d["high_cls"][1])
    b = next((i for i, (lo, hi) in enumerate(BINS) if lo <= cd < hi), None)
    if b is not None and len(buckets[b]) < MAX_PER_BIN:
        buckets[b].append(d)
imgs = [d for b in buckets.values() for d in b]
cf_pool = {c: v.to(DEVICE) for c, v in pickle.load(open(f"{CACHE}/klig2_cf_scatter_pool.pkl", "rb")).items()}
print(f"images: {len(imgs)} | cf classes: {len(cf_pool)}")

def _pick_cf(d):
    y2 = d["high_cls"][1]
    return cf_pool[y2] if y2 in cf_pool else next(iter(cf_pool.values()))

ig_linear = KLIntegratedGradients(model, n_steps=N_STEPS, n_samples=N_MC, sigma_final=SIGMA_FINAL, device=DEVICE)

def _build_klig2(x_cf, sigma_start, lv_floor, hq=False):
    return KLIGSquared(model, phi, x_cf,
        T=(T_DESC_HQ if hq else T_DESC), lr_mu=LR_MU, lr_lv=LR_LV,
        n_mc_path=(N_MC_DESC_HQ if hq else N_MC_DESC), n_mc_grad=(N_MC_HQ if hq else N_MC),
        sigma_start=sigma_start, loss_stop=LOSS_STOP, lv_floor=lv_floor, lv_ceil=LV_CEIL,
        mu_min=MU_MIN, mu_max=MU_MAX, clamp_samples=True, device=DEVICE)

def _gradpath(k2, x1):
    x1d = x1.to(DEVICE)
    if x1d.dim() > 1 and x1d.shape[0] == 1: x1d = x1d.squeeze(0)
    tm, tl, _ = k2._build_gradpath(x1d, x1d.shape)
    return tm, tl

def _integrate(k2, x1, target, tm, tl):
    k2.model.eval()
    x1d = x1.to(DEVICE)
    if x1d.dim() > 1 and x1d.shape[0] == 1: x1d = x1d.squeeze(0)
    _, obj = k2._resolve_target(x1d, int(target))
    saved = [p.requires_grad for p in k2.model.parameters()]
    for p in k2.model.parameters(): p.requires_grad_(False)
    acc = torch.zeros_like(x1d)
    try:
        for k in range(len(tm) - 1):
            g, _ = k2._eval_gradients(tm[k], tl[k], x1d.shape, obj)
            with torch.no_grad(): acc.add_(g * (tm[k] - tm[k + 1]))
    finally:
        for p, s in zip(k2.model.parameters(), saved): p.requires_grad_(s)
    return acc

def _maps_for_image(d, hq=False):
    """Return {method: {cls: (H,W) map}} for c1,c2 across all 4 methods."""
    x1 = d["x"].squeeze(0).to(DEVICE)
    H, W = x1.shape[1], x1.shape[2]
    c1, c2 = d["high_cls"][0], d["high_cls"][1]
    x_cf = _pick_cf(d)
    if x_cf.dim() == 4: x_cf = x_cf.squeeze(0)
    x_cf = x_cf.to(DEVICE)

    k2_fixed = _build_klig2(x_cf, SIGMA_FINAL, LV_FLOOR, hq=hq)
    p_fixed = _gradpath(k2_fixed, x1)
    sigc, k2a, pa = {}, {}, {}
    def _sig(c):
        if c not in sigc:
            sigc[c] = find_sigma_stop(model, x1, int(c), tau=0.95, n_samples=32, n_iter=12)
        return sigc[c]
    def _adapt(c):
        if c not in k2a:
            s = _sig(c); k = _build_klig2(x_cf, s, 2 * math.log(s), hq=hq)
            k2a[c] = k; pa[c] = _gradpath(k, x1)
        return k2a[c], pa[c]

    out = {m: {} for m in ORDER}
    n_steps = N_STEPS_HQ if hq else N_STEPS
    n_mc = N_MC_HQ if hq else N_MC
    for c in (c1, c2):
        # KL-IG (linear)
        out["KL-IG (linear)"][c] = absmax_collapse(
            ig_linear.attribute(x1, target=int(c)).attr).cpu().numpy().reshape(H, W)
        # KLIG-Adaptive
        r = KLIntegratedGradients(model, n_steps=n_steps, n_samples=n_mc,
            sigma_final=_sig(c), path=LinearPath(), device=DEVICE).attribute(x1, target=int(c))
        out["KLIG-Adaptive"][c] = absmax_collapse(r.attr).cpu().numpy().reshape(H, W)
        # KL-IG² (fixed CF baseline, fixed path)
        out["KL-IG²"][c] = absmax_collapse(
            _integrate(k2_fixed, x1, c, *p_fixed)).cpu().numpy().reshape(H, W)
        # KL-IG² (adaptive)
        k, pth = _adapt(c)
        out["KL-IG² (adaptive)"][c] = absmax_collapse(
            _integrate(k, x1, c, *pth)).cpu().numpy().reshape(H, W)
    return x1, c1, c2, out

# ── compute per-image rows ───────────────────────────────────────────────────
rows = []                    # one per image
series = {m: {"dsem": [], "dattr": []} for m in ORDER}
for d in tqdm(imgs, desc="per-image d_attr / d_sem"):
    _, c1, c2, mp = _maps_for_image(d, hq=False)
    dsem = clip_semantic_dist(c1, c2)
    row = {"c1": c1, "c2": c2, "dsem": dsem, "dattr": {}}
    for m in ORDER:
        da = cosine_dist_cs(mp[m][c1], mp[m][c2])
        row["dattr"][m] = da
        series[m]["dsem"].append(dsem)
        series[m]["dattr"].append(da)
    rows.append(row)

rows.sort(key=lambda r: r["dsem"])
N = len(rows)

# ── Spearman rho per method over the Top-1/Top-2 pairs ───────────────────────
summary = {}
for m in ORDER:
    rho, pv = spearmanr(series[m]["dsem"], series[m]["dattr"])
    verdict = "class-conditional" if (rho > 0 and pv < 0.05) else "class-blind"
    summary[m] = (rho, pv, verdict)

print("\n=== per-image (sorted by d_sem) ===")
hdr = f"{'Top-1':<16}{'Top-2':<16}{'d_sem':>7}" + "".join(f"{m.split(' ')[0][:9]:>11}" for m in ORDER)
print(hdr)
for r in rows:
    line = f"{imagenet_labels[r['c1']].split(',')[0][:15]:<16}{imagenet_labels[r['c2']].split(',')[0][:15]:<16}{r['dsem']:>7.2f}"
    line += "".join(f"{r['dattr'][m]:>11.2f}" for m in ORDER)
    print(line)
print(f"\n=== Spearman rho(d_sem, d_attr)  over Top-1/Top-2 pairs  (n={N}) ===")
for m in ORDER:
    rho, pv, v = summary[m]
    print(f"  {m:<20} rho={rho:+.3f}  p={pv:.3f}  -> {v}")

# ── render the table figure ──────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 0.42 * N + 3.0), facecolor="white")
gs = fig.add_gridspec(2, 1, height_ratios=[0.42 * N + 0.6, 1.6], hspace=0.18)

# top: per-image table
ax = fig.add_subplot(gs[0]); ax.axis("off")
col = ["Top-1 (y₁)", "Top-2 (y₂)", "d_sem\n1−cos(e)"] + [f"d_attr\n{m}" for m in ORDER]
cell = []
for r in rows:
    cell.append([imagenet_labels[r["c1"]].split(",")[0][:16],
                 imagenet_labels[r["c2"]].split(",")[0][:16],
                 f"{r['dsem']:.2f}"] + [f"{r['dattr'][m]:.2f}" for m in ORDER])
t = ax.table(cellText=cell, colLabels=col, loc="center", cellLoc="center")
t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 1.35)
for j, m in enumerate(ORDER):
    t[(0, 3 + j)].get_text().set_color(COLORS[m])
for (rr, cc), c in t.get_celld().items():
    if rr == 0: c.set_facecolor("#f0f0f0"); c.get_text().set_fontweight("bold")
ax.set_title("Class-sensitivity computation per image  (d_attr = 1 − cos(A_y₁, A_y₂);  d_sem = 1 − cos(e_y₁, e_y₂))",
             fontsize=11, fontweight="bold", pad=10)

# bottom: rho summary
ax2 = fig.add_subplot(gs[1]); ax2.axis("off")
scol = ["Method", "Spearman ρ(d_sem, d_attr)", "p", "n", "class-sensitivity"]
scell = [[m, f"{summary[m][0]:+.3f}", f"{summary[m][1]:.3f}", str(N), summary[m][2]] for m in ORDER]
t2 = ax2.table(cellText=scell, colLabels=scol, loc="center", cellLoc="center")
t2.auto_set_font_size(False); t2.set_fontsize(9.5); t2.scale(1, 1.5)
for i, m in enumerate(ORDER, start=1):
    t2[(i, 0)].get_text().set_color(COLORS[m]); t2[(i, 0)].get_text().set_fontweight("bold")
for (rr, cc), c in t2.get_celld().items():
    if rr == 0: c.set_facecolor("#e8e8e8"); c.get_text().set_fontweight("bold")
ax2.set_title("Class-sensitivity score = Spearman ρ(d_sem, d_attr).   ρ>0 → class-conditional ;  ρ≈0 → class-blind",
              fontsize=11, fontweight="bold", pad=8)

plt.savefig("cs_table.png", dpi=170, bbox_inches="tight")
print(f"\nSaved -> {os.path.abspath('cs_table.png')}")
