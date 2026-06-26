"""
Li & Shi Class Sensitivity (CS) — pixel-space metric (arXiv 2012.15616).

Per image: CS = corr(A(c_a), A(c_b)) between two class saliency maps.
  LOW/negative corr -> maps differ -> class-SENSITIVE (good).
  HIGH corr (~1)    -> maps identical -> class-BLIND (bad).  Necessity check only.

LOCKED DECISIONS:
  * class pair : BOTH high_low (argmax vs argmin) AND top1_top2 (closest competitor)
  * maps       : SIGNED, absmax-collapsed (klig_methods.attr_map)
  * corr       : Pearson (primary) + Spearman (robustness)
  * degenerate : std(v) < 1e-8 on either map -> flag, exclude from mean, count

Importable:
    from klig_cs_lishi import run_cs_lishi, write_outputs
    rows, summary = run_cs_lishi(images, model, methods)   # images: [{x, cf?, image_id?}]
    write_outputs(rows, summary)

CLI (offline, cached val shard, KL-IG2 CF = Top-2-class image):
    .venv/Scripts/python.exe klig_cs_lishi.py [N] [--methods klig]
"""
from __future__ import annotations
import os, sys, io, csv, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
from scipy.stats import spearmanr

EPS = 1e-8
PAIRINGS = ("high_low", "top1_top2")


def corr(va, vb, eps=EPS):
    """(pearson, spearman, degenerate_flag)."""
    va = np.asarray(va).ravel(); vb = np.asarray(vb).ravel()
    if va.std() < eps or vb.std() < eps:
        return float("nan"), float("nan"), True
    pear = float(np.corrcoef(va, vb)[0, 1])
    sp, _ = spearmanr(va, vb)
    return pear, float(sp), False


def summarize(rows, methods, pairings=PAIRINGS):
    out = []
    for ptype in pairings:
        for m in methods:
            sel = [r for r in rows if r["method"] == m and r["class_pair_type"] == ptype]
            nd = [r["cs_pearson"] for r in sel if not r["degenerate_flag"]]
            nds = [r["cs_spearman"] for r in sel if not r["degenerate_flag"]]
            out.append(dict(method=m, class_pair_type=ptype,
                mean_cs=float(np.mean(nd)) if nd else float("nan"),
                median_cs=float(np.median(nd)) if nd else float("nan"),
                mean_cs_spearman=float(np.mean(nds)) if nds else float("nan"),
                n=len(sel), n_degenerate=sum(r["degenerate_flag"] for r in sel)))
    return out


def run_cs_lishi(images, model, methods=None, attr_fn=None, *, phi=None,
                 eps=EPS, pairings=PAIRINGS, progress=True):
    """
    images : [{'x':(1,C,H,W), 'cf':(C,H,W) optional, 'image_id': optional}]
    Returns (rows, summary). c_high/c_low/top1/top2 are derived from the model logits.
    """
    import torch
    import klig_methods as KM
    if methods is None: methods = KM.METHODS
    if attr_fn is None: attr_fn = KM.attr_map
    if phi is None: phi = KM.make_phi(model)
    rows = []
    it = list(enumerate(images))
    if progress:
        try:
            from tqdm import tqdm; it = tqdm(it, total=len(images), desc="Li&Shi CS")
        except Exception:
            pass
    for idx, d in it:
        x = d["x"]; iid = d.get("image_id", idx)
        with torch.no_grad():
            logits = model(x)[0]
        c_high = int(logits.argmax()); c_low = int(logits.argmin())
        top1, top2 = [int(c) for c in logits.topk(2).indices]
        x_cf = d.get("cf")
        targets = sorted({c_high, c_low, top1, top2})
        mp = {}
        for m in methods:
            cf = x_cf if KM.needs_cf(m) else None
            for c in targets:
                mp[(m, c)] = attr_fn(m, model, x, c, x_cf=cf, phi=phi).numpy().ravel()
        pair_map = {"high_low": (c_high, c_low), "top1_top2": (top1, top2)}
        for m in methods:
            for pt in pairings:
                ca, cb = pair_map[pt]
                pe, sp, deg = corr(mp[(m, ca)], mp[(m, cb)], eps)
                rows.append(dict(image_id=int(iid), method=m, class_pair_type=pt,
                                 cs_pearson=pe, cs_spearman=sp, degenerate_flag=int(deg)))
    return rows, summarize(rows, methods, pairings)


def write_outputs(rows, summary, csv_path="cs_lishi.csv",
                  summary_csv="cs_lishi_summary.csv", png="cs_lishi_summary.png"):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "method", "class_pair_type",
                                          "cs_pearson", "cs_spearman", "degenerate_flag"])
        w.writeheader(); [w.writerow(r) for r in rows]
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "class_pair_type", "mean_cs",
            "median_cs", "mean_cs_spearman", "n", "n_degenerate"])
        w.writeheader(); [w.writerow(s) for s in summary]
    pairings = sorted({s["class_pair_type"] for s in summary}, key=lambda p: ("a" if p == "high_low" else "b"))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(pairings), 1, figsize=(11, 2.8 * len(pairings)), facecolor="white")
        if len(pairings) == 1: axes = [axes]
        for ax, ptype in zip(axes, pairings):
            ax.axis("off")
            block = sorted([s for s in summary if s["class_pair_type"] == ptype],
                           key=lambda s: (np.inf if np.isnan(s["mean_cs"]) else s["mean_cs"]))
            col = ["rank", "method", "mean CS\n(Pearson)", "median CS", "mean CS\n(Spearman)", "n_deg", "n"]
            cell = [[i + 1, s["method"], f"{s['mean_cs']:.3f}", f"{s['median_cs']:.3f}",
                     f"{s['mean_cs_spearman']:.3f}", s["n_degenerate"], s["n"]] for i, s in enumerate(block)]
            t = ax.table(cellText=cell, colLabels=col, loc="center", cellLoc="center")
            t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1, 1.5)
            for (rr, cc), c in t.get_celld().items():
                if rr == 0: c.set_facecolor("#eaeaea"); c.get_text().set_fontweight("bold")
            ax.set_title(f"Li & Shi CS — pair: {ptype}  (lower mean CS = more class-sensitive · signed · Pearson)",
                         fontsize=10.5, fontweight="bold", pad=8)
        plt.tight_layout(); plt.savefig(png, dpi=180, bbox_inches="tight")
        print(f"saved -> {png}")
    except Exception as e:
        print(f"[write_outputs] render skipped: {e}")
    print(f"saved -> {csv_path} , {summary_csv}")


# ── CLI: offline val-shard runner ────────────────────────────────────────────
def _cli(n_images=25, methods_key="all"):
    import io, math
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    import torch, torchvision.transforms as T
    from PIL import Image
    from collections import defaultdict
    from torchvision.models import resnet50, ResNet50_Weights
    import klig_methods as KM

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev} | N={n_images} | methods={methods_key}")
    w = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=w).to(dev).eval()
    tfm = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    SHARD = "data/val-00000-of-00002-b5248be478d25e41.parquet"
    tbl = pq.read_table(hf_hub_download("evanarlian/imagenet_1k_resized_256", SHARD,
                                        repo_type="dataset", local_files_only=True))
    lab = tbl["label"].to_pylist(); imgcol = tbl["image"]
    idx_by_label = defaultdict(list)
    for i, l in enumerate(lab): idx_by_label[l].append(i)

    def dec(i):
        d = imgcol[i].as_py(); raw = d["bytes"] if isinstance(d, dict) else d
        return tfm(Image.open(io.BytesIO(raw)).convert("RGB")).unsqueeze(0).to(dev)

    cf_cache = {}
    def cf_for(cls):
        if cls not in cf_cache:
            best = (-1.0, None)
            for j in idx_by_label.get(cls, [])[:6]:
                xx = dec(j)
                with torch.no_grad(): p = float(model(xx).softmax(-1)[0][cls])
                if p > best[0]: best = (p, xx.squeeze(0))
            cf_cache[cls] = best[1]
        return cf_cache[cls]

    methods = (["KLIG-Adaptive", "KL-IG (linear)", "KL-IG²", "KL-IG² (adaptive)"]
               if methods_key == "klig" else KM.METHODS)
    rng = np.random.default_rng(0)
    images = []
    for ii in rng.permutation(len(lab))[: n_images]:
        x = dec(int(ii))
        with torch.no_grad(): top2 = int(model(x).softmax(-1)[0].topk(2).indices[1])
        images.append({"x": x, "cf": cf_for(top2), "image_id": int(ii)})
    rows, summary = run_cs_lishi(images, model, methods)
    write_outputs(rows, summary)
    print("\n=== Li & Shi CS (lower mean_cs = more class-sensitive) ===")
    for pt in PAIRINGS:
        print(f"\n[{pt}] rank ascending by mean_cs:")
        block = sorted([s for s in summary if s["class_pair_type"] == pt],
                       key=lambda s: (np.inf if np.isnan(s["mean_cs"]) else s["mean_cs"]))
        for i, s in enumerate(block):
            print(f"  {i+1:>2}. {s['method']:<20} mean={s['mean_cs']:+.3f} "
                  f"median={s['median_cs']:+.3f} n_deg={s['n_degenerate']} n={s['n']}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    mk = "klig" if "--methods" in sys.argv and "klig" in sys.argv else "all"
    _cli(n, mk)
