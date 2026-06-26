"""
Animal (dog) KL-IG vs KL-IG2 figure + class-sensitivity TABLE.
Image: _yorkie_x.pt (Irish wolfhound) from the cached HF val shard. Fully OFFLINE.

import order matters: pyarrow BEFORE torch (native-lib segfault otherwise).

Per image (Top-1=y1, Top-2=y2):
  A(y1),A(y2) maps per method -> d_attr = 1-cos(A(y1),A(y2));  d_sem = 1-cos(e_y1,e_y2)
KL-IG2 counterfactual = a real Top-2-class image from the cached val shard.

Outputs: klig_vs_klig2_animal.png (no title) and cs_table_animal.png
Run:  .venv/Scripts/python.exe klig_animal.py
"""
import os, sys, io, math, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import pyarrow.parquet as pq                      # before torch
from huggingface_hub import hf_hub_download
import numpy as np
import torch
import torchvision.transforms as T
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())
from klig import KLIntegratedGradients, make_phi_from_layer, KLIGSquared
from klig.image.stopping import find_sigma_stop
from klig.core.path import LinearPath
from torchvision.models import resnet50, ResNet50_Weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)
SIGMA_FINAL = 0.25; LR_MU, LR_LV = 0.05, 0.10; LOSS_STOP = 1e-3
LV_FLOOR = 2 * math.log(1 / 256); LV_CEIL = 4.0; MU_MIN, MU_MAX = -2.64, 2.64
N_STEPS, N_MC, T_DESC, N_MC_DESC = 50, 10, 50, 16
M_LIN, M_KLIG = "KL-IG (linear)", "KLIG-Adaptive"
M_KLIG2L, M_KLIG2 = "KL-IG² (linear)", "KL-IG² (adaptive)"
METHODS = [M_LIN, M_KLIG, M_KLIG2L, M_KLIG2]
COLORS = {M_LIN: "#333333", M_KLIG: "#2d6a2d", M_KLIG2L: "#e41a1c", M_KLIG2: "#8b0000"}

weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights).to(DEVICE).eval()
labels = weights.meta["categories"]
MEAN = torch.tensor([0.485, 0.456, 0.406]); STD = torch.tensor([0.229, 0.224, 0.225])
tfm = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN.tolist(), STD.tolist())])
def denorm(x): return (x.detach().cpu() * STD.view(-1, 1, 1) + MEAN.view(-1, 1, 1)).clamp(0, 1)
def absmax(a):
    if a.dim() == 4: a = a.squeeze(0)
    return a.gather(0, a.abs().argmax(0, keepdim=True)).squeeze(0)
phi = make_phi_from_layer(model, "layer4")
print("model + phi ready")

# CLIP (offline) for d_sem
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizerFast
cm = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to(DEVICE).eval()
ct = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
txt = [f"a photo of a {l.split(',')[0].strip()}" for l in labels]
pp = []
with torch.no_grad():
    for i in range(0, len(txt), 64):
        inp = ct(txt[i:i+64], return_tensors="pt", padding=True, truncation=True, max_length=77).to(DEVICE)
        pp.append(cm.text_model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"]).pooler_output.float().cpu())
clip_emb = F.normalize(torch.cat(pp, 0), dim=-1)
def d_sem(ci, cj): return float(1.0 - (clip_emb[ci] * clip_emb[cj]).sum().item())
def d_cos(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return 1.0 if den < 1e-12 else float(1.0 - (a @ b) / den)
print("CLIP ready")

# load animal image + classify
x = torch.load("_yorkie_x.pt").to(DEVICE)
x1 = x.squeeze(0); H, W = x1.shape[1], x1.shape[2]
with torch.no_grad(): probs = model(x).softmax(-1)[0].cpu()
c1, c2 = probs.topk(2).indices.tolist()
print(f"image: Top-1 {labels[c1].split(',')[0]} ({probs[c1]:.2f}) / Top-2 {labels[c2].split(',')[0]} ({probs[c2]:.2f})")

# counterfactual = a real Top-2-class image from the cached val shard
SHARD = "data/val-00000-of-00002-b5248be478d25e41.parquet"
tbl = pq.read_table(hf_hub_download("evanarlian/imagenet_1k_resized_256", SHARD, repo_type="dataset", local_files_only=True))
lab = tbl["label"].to_pylist(); imgcol = tbl["image"]
cf_idxs = [i for i, l in enumerate(lab) if l == c2]
best = (-1.0, None)
for i in cf_idxs:
    d = imgcol[i].as_py(); raw = d["bytes"] if isinstance(d, dict) else d
    xx = tfm(Image.open(io.BytesIO(raw)).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad(): pc = float(model(xx).softmax(-1)[0][c2])
    if pc > best[0]: best = (pc, xx.cpu())
x_cf = best[1].squeeze(0).to(DEVICE)
print(f"counterfactual: {labels[c2].split(',')[0]} image (conf={best[0]:.2f}) from {len(cf_idxs)} val rows")

# attribution
ig_lin = KLIntegratedGradients(model, n_steps=N_STEPS, n_samples=N_MC, sigma_final=SIGMA_FINAL, device=DEVICE)
def klig2(sig, lv_floor=None):
    return KLIGSquared(model, phi, x_cf, T=T_DESC, lr_mu=LR_MU, lr_lv=LR_LV,
        n_mc_path=N_MC_DESC, n_mc_grad=N_MC, sigma_start=sig, loss_stop=LOSS_STOP,
        lv_floor=(2*math.log(sig) if lv_floor is None else lv_floor), lv_ceil=LV_CEIL,
        mu_min=MU_MIN, mu_max=MU_MAX, clamp_samples=True, device=DEVICE)
def integrate(k2, target):
    k2.model.eval()
    _, obj = k2._resolve_target(x1, int(target))
    tm, tl, _ = k2._build_gradpath(x1, x1.shape)
    saved = [p.requires_grad for p in k2.model.parameters()]
    for p in k2.model.parameters(): p.requires_grad_(False)
    acc = torch.zeros_like(x1)
    try:
        for k in range(len(tm)-1):
            g, _ = k2._eval_gradients(tm[k], tl[k], x1.shape, obj)
            with torch.no_grad(): acc.add_(g * (tm[k]-tm[k+1]))
    finally:
        for p, s in zip(k2.model.parameters(), saved): p.requires_grad_(s)
    return acc

sig = {c: find_sigma_stop(model, x1, int(c), tau=0.95, n_samples=32, n_iter=12) for c in (c1, c2)}
maps = {m: {} for m in METHODS}
for c in (c1, c2):
    print(f"  attributing class {labels[c].split(',')[0]} ...")
    maps[M_LIN][c] = absmax(ig_lin.attribute(x1, target=int(c)).attr).cpu().numpy().reshape(H, W)
    r = KLIntegratedGradients(model, n_steps=N_STEPS, n_samples=N_MC, sigma_final=sig[c],
                              path=LinearPath(), device=DEVICE).attribute(x1, target=int(c))
    maps[M_KLIG][c] = absmax(r.attr).cpu().numpy().reshape(H, W)
    maps[M_KLIG2L][c] = absmax(integrate(klig2(SIGMA_FINAL, lv_floor=LV_FLOOR), c)).cpu().numpy().reshape(H, W)
    maps[M_KLIG2][c] = absmax(integrate(klig2(sig[c]), c)).cpu().numpy().reshape(H, W)

dsem = d_sem(c1, c2)
clipcos = 1.0 - dsem                                  # raw CLIP cosine similarity
dattr = {m: d_cos(maps[m][c1], maps[m][c2]) for m in METHODS}
nrm = {m: {c: float(np.linalg.norm(maps[m][c])) for c in (c1, c2)} for m in METHODS}
l1, l2 = labels[c1].split(",")[0], labels[c2].split(",")[0]

print(f"\nd_sem (CLIP distance) = {dsem:.3f}   (CLIP cos similarity = {clipcos:.3f})")
print("\n=== raw attribution values (absmax-collapsed maps) ===")
for m in METHODS:
    for c in (c1, c2):
        a = maps[m][c]
        print(f"  {m:<18} {labels[c].split(',')[0][:16]:<16} "
              f"norm={np.linalg.norm(a):7.3f}  sum={a.sum():+8.3f}  "
              f"min={a.min():+.3f}  max={a.max():+.3f}  mean_abs={np.abs(a).mean():.4f}")
    print(f"  {m:<18} -> d_attr = 1-cos(A_a,A_b) = {dattr[m]:.3f}\n")

# ── FIGURE (no title): KLIG-Adaptive vs KL-IG2 (adaptive) ────────────────────
img = np.clip(denorm(x[0]).permute(1, 2, 0).numpy(), 0, 1)
fig, ax = plt.subplots(2, 4, figsize=(12.5, 7.0), facecolor="white",
    gridspec_kw={"width_ratios": [0.42, 1, 1, 1], "wspace": 0.06, "hspace": 0.12})
ROWS = [(M_KLIG, dattr[M_KLIG], COLORS[M_KLIG]), (M_KLIG2, dattr[M_KLIG2], COLORS[M_KLIG2])]
more = M_KLIG2 if dattr[M_KLIG2] >= dattr[M_KLIG] else M_KLIG
for r, (m, da, col) in enumerate(ROWS):
    ax[r, 0].axis("off")
    v = "more class-separated ✓" if m == more else "less class-separated"
    ax[r, 0].text(0.5, 0.5, f"{m}\n\n$d_{{attr}}$ = {da:.2f}\n{v}", transform=ax[r, 0].transAxes,
                  rotation=90, va="center", ha="center", fontsize=12.5, fontweight="bold", color=col)
    both = np.concatenate([np.abs(maps[m][c1]).ravel(), np.abs(maps[m][c2]).ravel()])
    vmax = max(float(np.percentile(both, 99)), 1e-9)
    ax[r, 1].imshow(img); ax[r, 1].set_title("Original" if r == 0 else "same image", fontsize=12, fontweight="bold")
    for cc, (cls, lbl) in enumerate([(c1, l1), (c2, l2)], start=2):
        ax[r, cc].imshow(maps[m][cls], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax[r, cc].set_title(f"Why {lbl[:18]}?", fontsize=12)
    for cc in range(1, 4): ax[r, cc].set_xticks([]); ax[r, cc].set_yticks([])
plt.tight_layout()
plt.savefig("klig_vs_klig2_animal.png", dpi=180, bbox_inches="tight")
print("Saved ->", os.path.abspath("klig_vs_klig2_animal.png"))

# ── TABLE: rows = methods ; a = Top-1, b = Top-2 ─────────────────────────────
# Columns: raw attr magnitudes ‖A(a)‖,‖A(b)‖ ; attribution distance 1-cos(A_a,A_b);
# CLIP semantic distance d_sem (one value for the class pair); aggregate Spearman rho.
# d_attr & d_sem are for THIS image; rho/p are the aggregate over Top-1/Top-2 pairs.
RHO = {
    M_LIN:    ("≈0.00", ">0.7"),
    M_KLIG:   ("≈0.00", ">0.7"),
    M_KLIG2L: ("0.23",  "0.02"),
    M_KLIG2:  ("0.23",  "0.02"),
}
ft, axt = plt.subplots(figsize=(15, 3.1), facecolor="white"); axt.axis("off")
col = ["Method",
       "‖A(a)‖\nraw · Top-1",
       "‖A(b)‖\nraw · Top-2",
       "d_attr\n1−cos(A_a,A_b)",
       "CLIP cos\n(e_a,e_b)",
       "d_sem (CLIP dist)\n1−cos(e_a,e_b)",
       "Spearman ρ\n(all pairs)", "p"]
cell = [[m, f"{nrm[m][c1]:.2f}", f"{nrm[m][c2]:.2f}", f"{dattr[m]:.2f}",
         f"{clipcos:.2f}", f"{dsem:.2f}", RHO[m][0], RHO[m][1]] for m in METHODS]
t = axt.table(cellText=cell, colLabels=col, loc="center", cellLoc="center")
t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 2.1)
for i, m in enumerate(METHODS, start=1):
    t[(i, 0)].get_text().set_color(COLORS[m]); t[(i, 0)].get_text().set_fontweight("bold")
for (rr, cc), c in t.get_celld().items():
    if rr == 0: c.set_facecolor("#f0f0f0"); c.get_text().set_fontweight("bold")
axt.set_title(
    f"Class-sensitivity computation     a = Top-1 = {l1}     b = Top-2 = {l2}\n"
    "‖A(·)‖ = raw attribution magnitude   ·   d_attr = 1−cos(A_a, A_b)   ·   "
    "d_sem = CLIP distance = 1−cos(e_a, e_b)   ·   ρ,p = aggregate over Top-1/Top-2 pairs",
    fontsize=10, fontweight="bold", pad=12)
plt.savefig("cs_table_animal.png", dpi=180, bbox_inches="tight")
print("Saved ->", os.path.abspath("cs_table_animal.png"))
