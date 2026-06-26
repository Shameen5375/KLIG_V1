"""
Standalone: KL-IG vs KL-IG2 single-image class-sensitivity contrast figure.
ANIMAL variant -- streams ImageNet for an animal image (Top-1 in 0..399) with a
plain/empty background and a clear Top-2 class, builds a counterfactual for the
Top-2 class on the fly, and renders the chosen winner at full quality.

Honest framing -- METHOD comparison, NOT a baseline-only ablation:
  Row 0  KLIG-Adaptive      -- linear path from the N(0,1) prior (no counterfactual)
  Row 1  KL-IG2 (adaptive)  -- rep-descent path anchored to the Top-2 counterfactual

Run:  .venv/Scripts/python.exe klig_vs_klig2_fig.py
"""
import os, sys, math, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # avoid OpenMP dup-runtime segfault on Windows
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())

from klig import KLIntegratedGradients, make_phi_from_layer, KLIGSquared
from klig.image.stopping import find_sigma_stop
from klig.core.path import LinearPath
from torchvision.models import resnet50, ResNet50_Weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}")

# ── config ───────────────────────────────────────────────────────────────────
SIGMA_FINAL = 0.25
LR_MU, LR_LV = 0.05, 0.10
LOSS_STOP = 1e-3
LV_CEIL = 4.0
MU_MIN, MU_MAX = -2.64, 2.64
N_STEPS_SCATTER, N_MC_SCATTER = 25, 3            # cheap: scan candidates
T_DESCENT_SCATTER, N_MC_DESC_SCATTER = 25, 8
N_STEPS_HQ, N_MC_HQ = 50, 10                     # full quality: render winner
T_DESCENT_HQ, N_MC_DESC_HQ = 50, 16

M_KLIG = "KLIG-Adaptive"
M_KLIG2 = "KL-IG² (adaptive)"
COLORS = {M_KLIG: "#2d6a2d", M_KLIG2: "#8b0000"}

ANIMAL_CLS = set(range(0, 400))   # ImageNet 0..399 ~ animals
STREAM_MAX = 1500                 # images to scan
CS_PROB_THRESH = 0.08             # min prob to count a class as "present"
CF_ACCEPT = 0.25                  # min prob for a usable counterfactual image
D_SEM_MIN = 0.25                  # skip near-synonym Top-1/Top-2 pairs
K_EVAL = 6                        # plainest-background candidates to actually score

# ── model + helpers ───────────────────────────────────────────────────────────
weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights).to(DEVICE).eval()
preprocess = weights.transforms()
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

# ── CLIP semantic distance ─────────────────────────────────────────────────────
import torch.nn.functional as _F
from transformers import CLIPModel, CLIPTokenizerFast
_clip_mdl = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
_clip_tok = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
_labels_text = [f"a photo of a {lbl.split(',')[0].strip()}" for lbl in imagenet_labels]
_parts = []
with torch.no_grad():
    for _i in range(0, len(_labels_text), 64):
        _inp = _clip_tok(_labels_text[_i:_i + 64], return_tensors="pt",
                         padding=True, truncation=True, max_length=77).to(DEVICE)
        _feat = _clip_mdl.text_model(input_ids=_inp["input_ids"],
                                     attention_mask=_inp["attention_mask"]).pooler_output
        _parts.append(_feat.float().cpu())
_clip_emb = _F.normalize(torch.cat(_parts, dim=0), dim=-1)
def clip_semantic_dist(ci, cj):
    return float(1.0 - (_clip_emb[ci] * _clip_emb[cj]).sum().item())

def cosine_dist_cs(a_i, a_j):   # SIGNED (metric behind the CS-rho scatter)
    ai = a_i.astype(np.float64).ravel(); aj = a_j.astype(np.float64).ravel()
    denom = np.linalg.norm(ai) * np.linalg.norm(aj)
    return 1.0 if denom < 1e-12 else float(1.0 - (ai @ aj) / denom)

def _bg_plainness(x1):
    """Lower = plainer/emptier background. Std of the outer 12px frame, denormalised."""
    im = denormalize(x1).clamp(0, 1).numpy()        # (3,H,W)
    b = 12
    frame = np.concatenate([
        im[:, :b, :].reshape(3, -1), im[:, -b:, :].reshape(3, -1),
        im[:, :, :b].reshape(3, -1), im[:, :, -b:].reshape(3, -1)], axis=1)
    return float(frame.std(axis=1).mean())

# ── stream ImageNet: gather animal candidates + a best-image-per-class CF bank ─
from datasets import load_dataset as _hf
_ds = _hf("evanarlian/imagenet_1k_resized_256", split="train", streaming=True)
_ds = _ds.shuffle(seed=42, buffer_size=5000)

cf_bank = {}          # class -> (prob, x_cpu(1,3,H,W))  best confident example seen
candidates = []       # animal images with a clear Top-2
scanned = 0
for item in tqdm(_ds.take(STREAM_MAX), total=STREAM_MAX, desc="scanning ImageNet"):
    scanned += 1
    img = item["image"]
    if img.mode != "RGB":
        img = img.convert("RGB")
    x = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = model(x).softmax(-1)[0].cpu()
    x_cpu = x.cpu()
    # update CF bank for any confidently-predicted class
    top = probs.argmax().item()
    if probs[top] >= CF_ACCEPT and probs[top] > cf_bank.get(top, (-1,))[0]:
        cf_bank[top] = (float(probs[top]), x_cpu)
    # candidate?
    high = (probs > CS_PROB_THRESH).nonzero(as_tuple=True)[0].tolist()
    if len(high) < 2:
        continue
    high = sorted(high, key=lambda c: probs[c].item(), reverse=True)
    if high[0] not in ANIMAL_CLS:
        continue
    candidates.append({
        "x": x_cpu, "high_cls": high,
        "high_probs": [float(probs[c]) for c in high],
        "bg": _bg_plainness(x_cpu.squeeze(0)),
    })
print(f"scanned {scanned} | animal candidates: {len(candidates)} | CF bank: {len(cf_bank)} classes")

# keep candidates that have a usable CF for their Top-2 and pass the d_sem gate
def _cf_for(cls):
    if cls in cf_bank:
        return cf_bank[cls][1].squeeze(0).to(DEVICE)
    return None

viable = []
for d in candidates:
    c1, c2 = d["high_cls"][0], d["high_cls"][1]
    cf = _cf_for(c2)
    if cf is None:
        continue
    if clip_semantic_dist(c1, c2) < D_SEM_MIN:
        continue
    d["cf"] = cf
    d["d_sem"] = clip_semantic_dist(c1, c2)
    viable.append(d)
print(f"viable (CF available + d_sem>={D_SEM_MIN}): {len(viable)}")
if not viable:
    raise SystemExit("No viable animal candidate found — try raising STREAM_MAX or lowering D_SEM_MIN.")

# prefer the plainest backgrounds, then score that short-list
viable.sort(key=lambda d: d["bg"])
short = viable[:K_EVAL]
print("plainest-background short-list (bg std, lower=plainer):")
for d in short:
    print(f"  bg={d['bg']:.3f}  d_sem={d['d_sem']:.2f}  "
          f"{imagenet_labels[d['high_cls'][0]].split(',')[0]} / "
          f"{imagenet_labels[d['high_cls'][1]].split(',')[0]}")

# ── attribution helpers ────────────────────────────────────────────────────────
def _build_klig2(x_cf_img, sigma_start, lv_floor, hq=False):
    return KLIGSquared(
        model, phi, x_cf_img,
        T=(T_DESCENT_HQ if hq else T_DESCENT_SCATTER), lr_mu=LR_MU, lr_lv=LR_LV,
        n_mc_path=(N_MC_DESC_HQ if hq else N_MC_DESC_SCATTER),
        n_mc_grad=(N_MC_HQ if hq else N_MC_SCATTER),
        sigma_start=sigma_start, loss_stop=LOSS_STOP,
        lv_floor=lv_floor, lv_ceil=LV_CEIL,
        mu_min=MU_MIN, mu_max=MU_MAX, clamp_samples=True, device=DEVICE)

def _build_gradpath_once(klig2_obj, x1):
    x1d = x1.to(DEVICE)
    if x1d.dim() > 1 and x1d.shape[0] == 1:
        x1d = x1d.squeeze(0)
    tm, tl, _ = klig2_obj._build_gradpath(x1d, x1d.shape)
    return tm, tl

def _klig2_integrate_only(klig2_obj, x1, target, traj_mu, traj_lv):
    klig2_obj.model.eval()
    x1d = x1.to(DEVICE)
    if x1d.dim() > 1 and x1d.shape[0] == 1:
        x1d = x1d.squeeze(0)
    _, objective_fn = klig2_obj._resolve_target(x1d, int(target))
    saved = [p.requires_grad for p in klig2_obj.model.parameters()]
    for p in klig2_obj.model.parameters():
        p.requires_grad_(False)
    attr_mu_sum = torch.zeros_like(x1d)
    try:
        for k in range(len(traj_mu) - 1):
            dmu_k = traj_mu[k] - traj_mu[k + 1]
            g_mu, _ = klig2_obj._eval_gradients(traj_mu[k], traj_lv[k], x1d.shape, objective_fn)
            with torch.no_grad():
                attr_mu_sum.add_(g_mu * dmu_k)
    finally:
        for p, s in zip(klig2_obj.model.parameters(), saved):
            p.requires_grad_(s)
    return attr_mu_sum

def _attr(m_name, x1, tgt_cls, klig2=None, path=None, sig=None, hq=False):
    if m_name == M_KLIG:
        r = KLIntegratedGradients(
            model, n_steps=(N_STEPS_HQ if hq else N_STEPS_SCATTER),
            n_samples=(N_MC_HQ if hq else N_MC_SCATTER),
            sigma_final=(sig or SIGMA_FINAL), path=LinearPath(), device=DEVICE
        ).attribute(x1, target=int(tgt_cls))
        attr = r.attr
    else:
        attr = _klig2_integrate_only(klig2, x1, int(tgt_cls), path[0], path[1])
    return absmax_collapse(attr).cpu().numpy().ravel()

def _four_maps(d, hq=False):
    x1 = d["x"].squeeze(0).to(DEVICE)
    H, W = x1.shape[1], x1.shape[2]
    c1, c2 = d["high_cls"][0], d["high_cls"][1]
    x_cf = d["cf"]
    if x_cf.dim() == 4:
        x_cf = x_cf.squeeze(0)
    x_cf = x_cf.to(DEVICE)
    sig_cache = {}
    def _sig(cls):
        if cls not in sig_cache:
            sig_cache[cls] = find_sigma_stop(model, x1, int(cls), tau=0.95, n_samples=32, n_iter=12)
        return sig_cache[cls]
    maps = {}
    for cls in (c1, c2):
        sc = _sig(cls)
        k2 = _build_klig2(x_cf, sc, 2 * math.log(sc), hq=hq)
        pth = _build_gradpath_once(k2, x1)
        maps[(M_KLIG2, cls)] = _attr(M_KLIG2, x1, cls, klig2=k2, path=pth, sig=sc, hq=hq).reshape(H, W)
        maps[(M_KLIG, cls)] = _attr(M_KLIG, x1, cls, sig=sc, hq=hq).reshape(H, W)
    return x1, c1, c2, maps

# ── pick the image with the largest class-sensitivity gap (cheap scan) ────────
best = None
for d in tqdm(short, desc="KL-IG vs KL-IG2 contrast"):
    _, c1, c2, maps = _four_maps(d, hq=False)
    d_klig = cosine_dist_cs(maps[(M_KLIG, c1)], maps[(M_KLIG, c2)])
    d_klig2 = cosine_dist_cs(maps[(M_KLIG2, c1)], maps[(M_KLIG2, c2)])
    contrast = d_klig2 - d_klig
    if best is None or contrast > best["contrast"]:
        best = dict(_d=d, x=d["x"], c1=c1, c2=c2, contrast=contrast, d_sem=d["d_sem"])

c1, c2 = best["c1"], best["c2"]
lbl1 = imagenet_labels[c1].split(",")[0]
lbl2 = imagenet_labels[c2].split(",")[0]
print(f"\nChosen: Top-1 {lbl1!r} / Top-2 {lbl2!r}   d_sem={best['d_sem']:.2f}  bg={best['_d']['bg']:.3f}")

# ── re-render the winner at full quality ──────────────────────────────────────
print("re-rendering winner at full quality (n_samples=10, T=50) ...")
_, _, _, maps_hq = _four_maps(best["_d"], hq=True)
best["maps"] = maps_hq
best["d_klig"] = cosine_dist_cs(maps_hq[(M_KLIG, c1)], maps_hq[(M_KLIG, c2)])
best["d_klig2"] = cosine_dist_cs(maps_hq[(M_KLIG2, c1)], maps_hq[(M_KLIG2, c2)])
print(f"  d_attr {M_KLIG:<18s} = {best['d_klig']:.3f}")
print(f"  d_attr {M_KLIG2:<18s} = {best['d_klig2']:.3f}")
print(f"  contrast = {best['d_klig2'] - best['d_klig']:.3f}")

# ── draw: 2 rows x [label | original | why c1 | why c2]  (no suptitle) ────────
img = np.clip(denormalize(best["x"][0]).permute(1, 2, 0).cpu().numpy(), 0, 1)
fig, axes = plt.subplots(
    2, 4, figsize=(12.5, 7.0), facecolor="white",
    gridspec_kw={"width_ratios": [0.42, 1, 1, 1], "wspace": 0.06, "hspace": 0.12},
)
ROWS = [(M_KLIG, best["d_klig"], COLORS[M_KLIG]),
        (M_KLIG2, best["d_klig2"], COLORS[M_KLIG2])]
_more = M_KLIG2 if best["d_klig2"] >= best["d_klig"] else M_KLIG

for r, (method, d_attr, color) in enumerate(ROWS):
    lab = axes[r, 0]; lab.axis("off")
    verdict = "more class-separated ✓" if method == _more else "less class-separated"
    lab.text(0.5, 0.5, f"{method}\n\n$d_{{attr}}$ = {d_attr:.2f}\n{verdict}",
             transform=lab.transAxes, rotation=90, va="center", ha="center",
             fontsize=12.5, fontweight="bold", color=color)
    both = np.concatenate([np.abs(best["maps"][(method, c1)]).ravel(),
                           np.abs(best["maps"][(method, c2)]).ravel()])
    vmax = max(float(np.percentile(both, 99)), 1e-9)
    axes[r, 1].imshow(img)
    axes[r, 1].set_title("Original" if r == 0 else "same image", fontsize=12, fontweight="bold")
    for cc, (cls, lbl) in enumerate([(c1, lbl1), (c2, lbl2)], start=2):
        axes[r, cc].imshow(best["maps"][(method, cls)], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[r, cc].set_title(f"Why {lbl[:18]}?", fontsize=12)
    for cc in range(1, 4):
        axes[r, cc].set_xticks([]); axes[r, cc].set_yticks([])

plt.tight_layout()
OUT = "klig_vs_klig2_animal.png"
plt.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"\nSaved -> {os.path.abspath(OUT)}")
