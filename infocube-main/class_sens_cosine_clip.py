"""
Class-sensitivity: attribution cosine distance (d_attr) vs CLIP semantic distance (d_sem).

Per image / class pair (a,b):
  d_attr = 1 - cos(A(a), A(b))           signed cosine distance of the two saliency maps
  d_sem  = 1 - cos(e_a, e_b)             CLIP text-embedding distance of the labels
Class-sensitivity score = Spearman ρ(d_sem, d_attr) over all pairs.
  ρ > 0 -> attribution diverges as classes diverge (class-conditional).

Extracted from kl_ig2__eval.ipynb cells 24-27; generalised over any method via klig_methods.attr_map.

    from class_sens_cosine_clip import build_clip, run_cs_cosine_clip, plot_cs_scatter
    clip = build_clip(imagenet_labels, device)          # -> clip.dist(ci,cj)
    scat = run_cs_cosine_clip(images, model, METHODS, clip)   # {method:[(d_sem,d_attr)...]}
    plot_cs_scatter(scat, COLORS, out_png="cs_cosine_clip.png")
"""
from __future__ import annotations
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from klig_methods import attr_map, needs_cf, make_phi, COLORS as _COLORS


def cosine_dist_cs(a, b):
    """Signed cosine distance d_attr = 1 - cos(a, b)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return 1.0 if den < 1e-12 else float(1.0 - (a @ b) / den)


class _Clip:
    """Holds normalised CLIP text embeddings; .dist(ci,cj) = 1 - cos."""
    def __init__(self, emb):
        self.emb = emb
    def dist(self, ci, cj):
        return float(1.0 - (self.emb[int(ci)] * self.emb[int(cj)]).sum().item())


def build_clip(labels, device, model_name="openai/clip-vit-base-patch32"):
    """CLIP text embeddings for all class labels (offline: local_files_only)."""
    from transformers import CLIPModel, CLIPTokenizerFast
    cm = CLIPModel.from_pretrained(model_name, local_files_only=True).to(device).eval()
    ct = CLIPTokenizerFast.from_pretrained(model_name, local_files_only=True)
    txt = [f"a photo of a {l.split(',')[0].strip()}" for l in labels]
    parts = []
    with torch.no_grad():
        for i in range(0, len(txt), 64):
            inp = ct(txt[i:i + 64], return_tensors="pt", padding=True,
                     truncation=True, max_length=77).to(device)
            f = cm.text_model(input_ids=inp["input_ids"],
                              attention_mask=inp["attention_mask"]).pooler_output
            parts.append(f.float().cpu())
    return _Clip(F.normalize(torch.cat(parts, 0), dim=-1))


def run_cs_cosine_clip(images, model, methods, clip, attr_fn=attr_map, *, phi=None,
                       all_pairs=False, progress=True):
    """
    images : [{'x':(1,C,H,W), 'high_cls':[t1,t2,...], 'cf':(C,H,W) optional}]
    all_pairs=False -> Top-1/Top-2 only ; True -> every combination of high_cls.
    Returns {method: [(d_sem, d_attr), ...]}.
    """
    if phi is None:
        phi = make_phi(model)
    scatter = {m: [] for m in methods}
    it = images
    if progress:
        try:
            from tqdm import tqdm; it = tqdm(images, desc="CS cosine-vs-CLIP")
        except Exception:
            pass
    for d in it:
        x = d["x"]; hc = [int(c) for c in d["high_cls"]]
        if len(hc) < 2:
            continue
        pairs = list(itertools.combinations(hc, 2)) if all_pairs else [(hc[0], hc[1])]
        x_cf = d.get("cf")
        # cache maps per (method, class) within this image
        cache = {}
        for m in methods:
            cf = x_cf if needs_cf(m) else None
            for c in {c for pr in pairs for c in pr}:
                cache[(m, c)] = attr_fn(m, model, x, c, x_cf=cf, phi=phi).numpy()
        for ci, cj in pairs:
            ds = clip.dist(ci, cj)
            for m in methods:
                da = cosine_dist_cs(cache[(m, ci)], cache[(m, cj)])
                scatter[m].append((ds, da))
    return scatter


def cs_rho(scatter):
    """Spearman ρ(d_sem, d_attr) per method (positive = class-conditional)."""
    out = {}
    for m, pts in scatter.items():
        if len(pts) < 3:
            out[m] = (float("nan"), float("nan"), len(pts)); continue
        ds = np.array([p[0] for p in pts]); da = np.array([p[1] for p in pts])
        rho, p = spearmanr(ds, da)
        out[m] = (float(rho), float(p), len(pts))
    return out


def plot_cs_scatter(scatter, colors=None, out_png="cs_cosine_clip.png", n_bins=10):
    """One panel per method: d_attr vs d_sem with regression + Spearman ρ (cell-27 style)."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    colors = colors or _COLORS
    methods = [m for m in scatter if scatter[m]]
    fig, axes = plt.subplots(1, len(methods), figsize=(4.6 * len(methods), 4.4),
                             facecolor="white", squeeze=False)
    axes = axes[0]
    rho_tbl = cs_rho(scatter)
    for ax, m in zip(axes, methods):
        pts = scatter[m]
        ds = np.array([p[0] for p in pts]); da = np.array([p[1] for p in pts])
        ax.scatter(ds, da, alpha=0.3, s=14, c=ds, cmap="plasma", edgecolors="none")
        if len(ds) >= 2:
            z = np.polyfit(ds, da, 1); xs = np.linspace(ds.min(), ds.max(), 100)
            ax.plot(xs, np.poly1d(z)(xs), color=colors.get(m, "#444"), lw=2, ls="--")
        rho, p, n = rho_tbl[m]
        ax.set_title(f"{m}\nρ={rho:.3f}  p={p:.3g}  n={n}", fontsize=9,
                     color=colors.get(m, "#444"), fontweight="bold")
        ax.set_xlabel("CLIP semantic distance", fontsize=9)
        ax.set_ylabel("d_attr = 1−cos(A_a,A_b)", fontsize=9)
        ax.set_ylim(-0.05, 1.15); ax.grid(True, ls="--", alpha=0.3)
    plt.suptitle("Class sensitivity: attribution distance vs CLIP semantic distance "
                 "(ρ>0 = class-conditional)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"saved -> {out_png}")
    return rho_tbl
