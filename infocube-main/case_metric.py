"""
CASE class-sensitivity metric (Feature Agreement, Williamson et al. 2025; arXiv:2506.07327).

For each image: attribution maps for Top-1 and Top-2 classes; top-k% feature agreement
  F(E,E',k) = |top-k(E) ∩ top-k(E')| / k
One-sided Wilcoxon, H0: median(F) >= 0.50.  Reject (p<0.05) -> class-distinct explanations.

Extracted from kl_ig2__eval.ipynb cell 33; generalised over any method via klig_methods.attr_map.

    from case_metric import run_case, case_summary_table
    res = run_case(images, model, METHODS)          # images: [{x, high_cls:[t1,t2], cf?}]
    case_summary_table(res, out_png="case_summary.png")
"""
from __future__ import annotations
import numpy as np
from scipy.stats import wilcoxon
from klig_methods import attr_map, needs_cf, make_phi

TOP_K_PCT = 0.05


def feature_agreement(e1: np.ndarray, e2: np.ndarray, k_pct: float = TOP_K_PCT) -> float:
    e1 = np.abs(np.asarray(e1).ravel()).astype(np.float64)
    e2 = np.abs(np.asarray(e2).ravel()).astype(np.float64)
    k = max(1, int(len(e1) * k_pct))
    top1 = set(np.argpartition(e1, -k)[-k:].tolist())
    top2 = set(np.argpartition(e2, -k)[-k:].tolist())
    return len(top1 & top2) / k


def run_case(images, model, methods, attr_fn=attr_map, *, phi=None,
             k_pct: float = TOP_K_PCT, progress=True):
    """
    images : list of dicts {'x':(1,C,H,W), 'high_cls':[top1,top2,...], 'cf':(C,H,W) optional}
    Returns {method: {'fa_scores':[...], 'median':float, 'wilcoxon_p':float, 'reject_H0':bool}}.
    """
    if phi is None:
        phi = make_phi(model)
    fa = {m: [] for m in methods}
    it = images
    if progress:
        try:
            from tqdm import tqdm; it = tqdm(images, desc="CASE feature-agreement")
        except Exception:
            pass
    for d in it:
        x = d["x"]; hc = d["high_cls"]
        if len(hc) < 2:
            continue
        c1, c2 = int(hc[0]), int(hc[1])
        x_cf = d.get("cf")
        for m in methods:
            cf = x_cf if needs_cf(m) else None
            a1 = attr_fn(m, model, x, c1, x_cf=cf, phi=phi).numpy()
            a2 = attr_fn(m, model, x, c2, x_cf=cf, phi=phi).numpy()
            fa[m].append(feature_agreement(a1, a2, k_pct))

    results = {}
    for m in methods:
        scores = np.asarray(fa[m], dtype=np.float64)
        if len(scores) >= 1 and np.any(scores != 0.5):
            try:
                _, p = wilcoxon(scores - 0.5, alternative="less")
            except ValueError:
                p = float("nan")
        else:
            p = float("nan")
        results[m] = dict(
            fa_scores=scores.tolist(),
            median=float(np.median(scores)) if len(scores) else float("nan"),
            mean=float(np.mean(scores)) if len(scores) else float("nan"),
            wilcoxon_p=float(p),
            reject_H0=bool(np.isfinite(p) and p < 0.05),
            n=len(scores),
        )
    return results


def case_summary_table(results, out_png="case_summary.png", title=None):
    """Render + return the CASE summary, ranked ascending by median FA (lower = more class-distinct)."""
    rows = sorted(results.items(), key=lambda kv: (np.nan_to_num(kv[1]["median"], nan=1.0)))
    table = [dict(method=m, median_FA=r["median"], mean_FA=r["mean"],
                  wilcoxon_p=r["wilcoxon_p"], class_distinct=r["reject_H0"], n=r["n"])
             for m, r in rows]
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 0.5 * len(table) + 1.6), facecolor="white"); ax.axis("off")
        col = ["rank", "method", "median FA", "mean FA", "Wilcoxon p", "class-distinct (p<.05)", "n"]
        cell = [[i + 1, t["method"], f"{t['median_FA']:.3f}", f"{t['mean_FA']:.3f}",
                 (f"{t['wilcoxon_p']:.3g}" if np.isfinite(t["wilcoxon_p"]) else "n/a"),
                 "yes" if t["class_distinct"] else "no", t["n"]] for i, t in enumerate(table)]
        tb = ax.table(cellText=cell, colLabels=col, loc="center", cellLoc="center")
        tb.auto_set_font_size(False); tb.set_fontsize(10); tb.scale(1, 1.6)
        for (rr, cc), c in tb.get_celld().items():
            if rr == 0: c.set_facecolor("#eaeaea"); c.get_text().set_fontweight("bold")
        ax.set_title(title or "CASE — Feature Agreement (lower median FA = more class-distinct; "
                     "reject H0: median≥0.5)", fontsize=10.5, fontweight="bold", pad=10)
        plt.savefig(out_png, dpi=170, bbox_inches="tight")
        print(f"saved -> {out_png}")
    except Exception as e:
        print(f"[case_summary_table] render skipped: {e}")
    return table
