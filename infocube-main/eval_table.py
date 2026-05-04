"""
eval_table.py — Fixed reproduction of the student notebook's evaluation table.

Fixes applied vs the notebook:
  1. OFR/DC: object mask from ensemble of ALL attribution methods (not KLIG-only)
  2. ExpGrad: real ImageNet train images as background (not random Gaussian)
  3. I/D: blur baseline for BOTH insertion and deletion (standard RISE protocol)
  4. Channel collapse: sum-collapse consistently across all methods and metrics

Methods evaluated:
  KL-IG (adaptive), KL-IG (σ=0.25), IDG, ExpGrad, IG-zero,
  SmoothGrad, Vanilla Grad, Blur-IG

Metrics:
  Insertion AUC ↑, Deletion AUC ↓, Ins−Del ↑, OFR ↑, DC ↑, Sensitivity-n PCC ↑,
  Gini (sparsity) ↑, Infidelity ↓, Occlusion Correlation ↑

Usage:
  # ResNet50 (default)
  python eval/eval_table.py \\
    --val-dir ~/DATA/imagenet/ILSVRC/Data/CLS-LOC/val \\
    --train-dir ~/DATA/imagenet/ILSVRC/Data/CLS-LOC/train \\
    --gt-file ~/DATA/imagenet/ILSVRC/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt \\
    --meta ~/DATA/imagenet/ILSVRC/ILSVRC2012_devkit_t12/data/meta.mat \\
    --outdir eval/results_fixed/ --n-images 1000 --n-background 100

  # ViT-B/16
  python eval/eval_table.py [same args] --model vit --outdir eval/results_vit/
"""

from __future__ import annotations

import argparse
import pickle
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from scipy import stats
from scipy.ndimage import gaussian_filter as sp_gf
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights, resnet50, vit_b_16
from tqdm import tqdm

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from eval.insertion_deletion import both_auc_signed
from infocube.image.attribution import ImageAttributor
from infocube.image.stopping import find_sigma_stop

# ---------------------------------------------------------------------------
# Constants (matching the student notebook where possible)
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
# Same spatial ops as TRANSFORM but nearest-neighbour — for segmentation masks
MASK_TRANSFORM = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.NEAREST),
    T.CenterCrop(224),
])

METHODS = [
    "KL-IG (adaptive)",
    "KL-IG (σ=0.25)",
    "IDG",
    "ExpGrad",
    "IG-zero",
    "SmoothGrad",
    "Vanilla Grad",
    "Blur-IG",
    "Guided IG",
]

# Sensitivity-n settings
SENS_FRACTIONS  = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8]
N_SENS_SUBSETS  = 30

# I/D settings
N_ID_STEPS = 50   # matches student notebook

# Blur-IG settings
BLURIG_SIGMA_MAX = 10.0
BLURIG_STEPS     = 50

# Occlusion settings
OCCLUSION_PATCH  = 14
OCCLUSION_STRIDE = 7
OCCLUSION_RATIOS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

# Infidelity settings
INFID_N_PERT  = 30
INFID_SIGMA   = 0.1

# ---------------------------------------------------------------------------
# Model and data helpers
# ---------------------------------------------------------------------------

def load_model(device: torch.device, arch: str = "resnet50"):
    if arch == "vit":
        weights = ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
        model   = vit_b_16(weights=weights).to(device).eval()
    else:
        weights = ResNet50_Weights.IMAGENET1K_V2
        model   = resnet50(weights=weights).to(device).eval()
    return model, weights.meta["categories"]


def build_ilsvrc_to_tv(meta_mat: Path, train_dir: Path) -> dict[int, int]:
    """Map ILSVRC 1-based label → torchvision 0-based class index."""
    meta  = scipy.io.loadmat(str(meta_mat))
    rows  = meta["synsets"][:, 0]
    id_to_wnid: dict[int, str] = {}
    for row in rows:
        ilsvrc_id = int(row["ILSVRC2012_ID"][0, 0])
        wnid      = str(row["WNID"][0])
        n_train   = int(row["num_train_images"][0, 0])
        if n_train > 0:
            id_to_wnid[ilsvrc_id] = wnid
    synsets_sorted = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    wnid_to_tv     = {s: i for i, s in enumerate(synsets_sorted)}
    return {k: wnid_to_tv[v] for k, v in id_to_wnid.items() if v in wnid_to_tv}


def load_val_dataset(
    val_dir: Path,
    gt_file: Path,
    meta_mat: Path,
    train_dir: Path,
    n_images: int,
    model: nn.Module,
    device: torch.device,
    seed: int = 42,
    require_mask: dict[str, Path] | None = None,
) -> list[dict]:
    """
    Load n_images from ImageNet val, one per class where possible.
    Target = ground-truth if model has >5% confidence, else top-1.

    If require_mask is provided (stem -> mask_path dict from load_imagenet_s_masks),
    only images that have an ImageNet-S ground-truth mask are included.
    This ensures OFR/DC is computed on the full N rather than ~25% of it.
    Returns list of dicts: {idx, path, x, target, label_str}.
    """
    rng = random.Random(seed)
    gt_labels = [int(l) for l in gt_file.read_text().splitlines() if l.strip()]
    ilsvrc_to_tv = build_ilsvrc_to_tv(meta_mat, train_dir)
    categories   = ResNet50_Weights.IMAGENET1K_V2.meta["categories"]

    # Group val image paths by TV class index
    all_paths = sorted(val_dir.glob("*.JPEG")) + sorted(val_dir.glob("*.jpg"))
    by_class: dict[int, list[tuple[Path, int]]] = {}
    for path in all_paths:
        if require_mask is not None and path.stem not in require_mask:
            continue  # skip images without GT masks when require_mask is active
        img_num = int(path.stem.split("_")[-1])  # 1-indexed
        ilsvrc_label = gt_labels[img_num - 1]
        tv_cls = ilsvrc_to_tv.get(ilsvrc_label)
        if tv_cls is None:
            continue
        by_class.setdefault(tv_cls, []).append((path, tv_cls))

    if require_mask is not None:
        print(f"[data] Mask-filtered pool: {sum(len(v) for v in by_class.values())} images "
              f"across {len(by_class)} classes")

    # Shuffle within each class and pick one per class
    samples: list[tuple[Path, int]] = []
    for cls_idx in sorted(by_class):
        choices = by_class[cls_idx]
        rng.shuffle(choices)
        samples.append(choices[0])

    rng.shuffle(samples)
    samples = samples[:n_images]

    dataset = []
    print(f"[data] Preprocessing {len(samples)} images...")
    for i, (path, gt_tv) in enumerate(tqdm(samples, desc="loading")):
        try:
            img = Image.open(path).convert("RGB")
            x   = TRANSFORM(img).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = model(x).softmax(-1)[0]
            top1 = int(probs.argmax())
            target = gt_tv if float(probs[gt_tv]) > 0.05 else top1
            dataset.append({
                "idx":       i,
                "path":      str(path),
                "stem":      path.stem,
                "x":         x,
                "target":    target,
                "label_str": categories[target],
            })
        except Exception as e:
            print(f"[warn] Skipping {path}: {e}")

    print(f"[data] Loaded {len(dataset)} images")
    return dataset


def load_background(train_dir: Path, n: int, device: torch.device, seed: int = 42) -> torch.Tensor:
    """Load n random ImageNet train images for ExpGrad background."""
    rng    = random.Random(seed)
    paths  = [f for synset in train_dir.iterdir() if synset.is_dir()
              for f in synset.iterdir() if f.suffix.lower() in {".jpeg", ".jpg", ".png"}]
    chosen = rng.sample(paths, min(n, len(paths)))
    tensors = []
    for p in tqdm(chosen, desc="background"):
        try:
            img = Image.open(p).convert("RGB")
            tensors.append(TRANSFORM(img))
        except Exception:
            pass
    bg = torch.stack(tensors).to(device)
    print(f"[data] Background: {bg.shape}")
    return bg


# ---------------------------------------------------------------------------
# ImageNet-S ground-truth segmentation masks
# ---------------------------------------------------------------------------

def load_imagenet_s_masks(mask_dir: Path) -> dict[str, Path]:
    """
    Build stem -> mask_path lookup for all ImageNet-S validation masks.
    mask_dir should be  .../ImageNetS919/validation-segmentation/
    """
    lookup: dict[str, Path] = {}
    for wnid_dir in mask_dir.iterdir():
        if wnid_dir.is_dir():
            for f in wnid_dir.iterdir():
                lookup[f.stem] = f
    print(f"[masks] Loaded {len(lookup)} ImageNet-S mask paths")
    return lookup


def load_imagenet_s_mask(mask_path: Path) -> np.ndarray:
    """
    Load an ImageNet-S mask, apply the same spatial transform as TRANSFORM
    (shorter-edge resize to 256, center-crop to 224), return binary (H,W) uint8.

    Mask encoding: R-channel > 0 = foreground object, 0 = background.
    Uses NEAREST interpolation so no fractional values are introduced.
    """
    m = np.array(Image.open(mask_path))        # (H, W, 3), values in {0, 1}
    fg = Image.fromarray((m[:, :, 0] > 0).astype(np.uint8) * 255, mode="L")
    fg = MASK_TRANSFORM(fg)                    # PIL -> PIL, 224×224
    return (np.array(fg) > 0).astype(np.uint8) # (224, 224) binary


# ---------------------------------------------------------------------------
# Attribution methods — all return (C, H, W) signed attribution
# ---------------------------------------------------------------------------

def attr_klig(
    model: nn.Module,
    x: torch.Tensor,
    target: int,
    sigma_final: float,
    n_steps: int = 50,
    n_samples: int = 10,
    device: torch.device | None = None,
) -> torch.Tensor:
    attributor = ImageAttributor(
        model=model, n_steps=n_steps, n_samples=n_samples,
        sigma_final=sigma_final, device=device,
    )
    result = attributor.attribute(x, target=target, show_progress=False)
    return result.attr  # (C, H, W)


def attr_ig_zero(model: nn.Module, x: torch.Tensor, target: int, n_steps: int = 50) -> torch.Tensor:
    from captum.attr import IntegratedGradients
    xb = x if x.dim() == 4 else x.unsqueeze(0)
    xb = xb.to(next(model.parameters()).device)
    ig = IntegratedGradients(model)
    attr = ig.attribute(
        xb, baselines=torch.zeros_like(xb), target=target,
        n_steps=n_steps, method="gausslegendre",
        internal_batch_size=n_steps,
    )
    return attr.detach().squeeze(0)  # (C, H, W)


def attr_expgrad(
    model: nn.Module,
    x: torch.Tensor,
    target: int,
    background: torch.Tensor,
    n_samples: int = 50,
) -> torch.Tensor:
    """Expected Gradients with real ImageNet train images as background."""
    xb  = (x if x.dim() == 4 else x.unsqueeze(0)).to(next(model.parameters()).device)
    idx = torch.randint(len(background), (n_samples,))
    bg  = background[idx].to(xb.device)                         # (n, C, H, W)
    alpha   = torch.rand(n_samples, 1, 1, 1, device=xb.device)
    interp  = (bg + alpha * (xb - bg)).requires_grad_(True)
    grads   = torch.autograd.grad(model(interp)[:, target].sum(), interp)[0]
    return (grads.detach() * (xb - bg)).mean(dim=0)             # (C, H, W)


def attr_smoothgrad(
    model: nn.Module, x: torch.Tensor, target: int, n_samples: int = 50,
) -> torch.Tensor:
    xb  = (x if x.dim() == 4 else x.unsqueeze(0)).to(next(model.parameters()).device)
    std = 0.15 * float((xb.max() - xb.min()).item())
    noisy = (xb + torch.randn(n_samples, *xb.shape[1:], device=xb.device) * std).requires_grad_(True)
    grads = torch.autograd.grad(model(noisy)[:, target].sum(), noisy)[0]
    return grads.detach().mean(dim=0)                            # (C, H, W)


def attr_vanilla(model: nn.Module, x: torch.Tensor, target: int) -> torch.Tensor:
    from captum.attr import Saliency
    xb = (x if x.dim() == 4 else x.unsqueeze(0)).to(next(model.parameters()).device)
    return Saliency(model).attribute(xb, target=target, abs=False).detach().squeeze(0)


def _idg_redistribute_alphas(
    slopes: torch.Tensor,
    n_steps: int,
    step_size: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Redistribute n_steps alpha values according to pre-characterisation slopes.
    Faithful port of getAlphaParameters() from the official IDG repo
    (github.com/chasewalker26/Integrated-Decision-Gradients).
    Returns (new_alphas, substep_sizes), both shape (n_steps,).
    """
    dev = slopes.device
    s_min, s_max = slopes.min(), slopes.max()
    slopes_01 = (slopes - s_min) / (s_max - s_min + 1e-12)
    slopes_01[0] = 0.0
    slopes_norm = slopes_01 / (slopes_01.sum() + 1e-12)

    sample_f = slopes_norm * n_steps
    sample_i = sample_f.int().clone()
    remaining = int(n_steps - sample_i.sum().item())

    sample_f_fill = sample_f.clone()
    sample_f_fill[sample_i != 0] = -1.0
    desc_order = torch.argsort(sample_f_fill, descending=True)
    if remaining > 0:
        sample_i[desc_order[:remaining]] = 1   # matches reference (sets, not +=)

    new_alphas    = torch.zeros(n_steps, device=dev)
    substep_sizes = torch.zeros(n_steps, device=dev)
    start_idx, start_val = 0, 0.0

    for n_sub_t in sample_i:
        n_sub = int(n_sub_t.item())
        start_val_next = start_val + step_size
        if n_sub > 0:
            region_alphas = torch.linspace(
                start_val, start_val_next, n_sub + 1, device=dev
            )[:n_sub]
            new_alphas[start_idx: start_idx + n_sub]    = region_alphas
            substep_sizes[start_idx: start_idx + n_sub] = step_size / n_sub
            start_idx += n_sub
        start_val = start_val_next

    return new_alphas, substep_sizes


def attr_idg(
    model: nn.Module,
    x: torch.Tensor,
    target: int,
    n_steps: int = 50,
) -> torch.Tensor:
    """
    Integrated Decision Gradients (Sattarzadeh et al., AAAI 2024; arXiv 2305.20052).
    Faithful port of the official implementation at
    github.com/chasewalker26/Integrated-Decision-Gradients.

    Phase 1 — pre-characterise with n_steps uniform forward passes to estimate
               dF/dalpha at each alpha; redistribute sampling density accordingly.
    Phase 2 — compute gradients at the new (non-uniform) alpha positions, weight
               each by the local slope and substep size (non-uniform correction),
               average, and multiply by (x - baseline).  Baseline: zeros.
    """
    dev = next(model.parameters()).device
    xb  = (x if x.dim() == 4 else x.unsqueeze(0)).to(dev).detach()
    dx  = xb                                # baseline = zeros, dx = xb - 0
    N   = n_steps
    step_size = 1.0 / N

    # Phase 1: pre-characterisation (batched, no grad)
    uniform_alphas = torch.linspace(0, 1, N, device=dev)
    interp_uniform = uniform_alphas.view(N, 1, 1, 1) * dx   # (N, C, H, W)
    with torch.no_grad():
        logits_uniform = model(interp_uniform)[:, target]    # (N,)

    slopes_uniform = torch.zeros(N, device=dev)
    slopes_uniform[1:] = (logits_uniform[1:] - logits_uniform[:-1]) / step_size

    # Phase 2a: redistribute alphas
    new_alphas, substep_sizes = _idg_redistribute_alphas(slopes_uniform, N, step_size)

    # Phase 2b: batched gradient computation at new alpha positions
    interp_new = new_alphas.view(N, 1, 1, 1) * dx           # (N, C, H, W)
    interp_new.requires_grad_(True)
    logits2   = model(interp_new)[:, target]                 # (N,)
    gradients = torch.autograd.grad(
        logits2, interp_new,
        grad_outputs=torch.ones_like(logits2),
    )[0].detach()                                             # (N, C, H, W)
    logits2 = logits2.detach()

    # Re-compute slopes at non-uniform positions via finite differences
    new_slopes = torch.zeros(N, device=dev)
    dalphas    = new_alphas[1:] - new_alphas[:-1]
    valid      = dalphas.abs() > 1e-7
    new_slopes[1:][valid] = (logits2[1:] - logits2[:-1])[valid] / dalphas[valid]

    # Weight by slope x substep, mean over steps, scale by (x - baseline)
    weighted = gradients * new_slopes.view(N, 1, 1, 1) * substep_sizes.view(N, 1, 1, 1)
    return (weighted.mean(dim=0) * dx.squeeze(0)).detach()   # (C, H, W)


def attr_blur_ig(
    model: nn.Module,
    x: torch.Tensor,
    target: int,
    n_steps: int = BLURIG_STEPS,
    sigma_max: float = BLURIG_SIGMA_MAX,
) -> torch.Tensor:
    dev  = next(model.parameters()).device
    xb   = (x if x.dim() == 4 else x.unsqueeze(0)).to(dev)
    x_np = xb[0].detach().cpu().numpy()                         # (C, H, W)
    integrated = torch.zeros(x_np.shape, device=dev)

    for k in range(n_steps):
        alpha   = (k + 0.5) / n_steps
        sigma_k = sigma_max * (1.0 - alpha)
        if sigma_k > 0.01:
            blurred = np.stack([sp_gf(x_np[c], sigma=sigma_k) for c in range(x_np.shape[0])])
        else:
            blurred = x_np.copy()
        xt = torch.tensor(blurred, dtype=xb.dtype, device=dev).unsqueeze(0).requires_grad_(True)
        model(xt)[0, target].backward()
        if xt.grad is not None:
            integrated += xt.grad[0].detach()

    baseline = np.stack([sp_gf(x_np[c], sigma=sigma_max) for c in range(x_np.shape[0])])
    bl_t = torch.tensor(baseline, dtype=xb.dtype, device=dev)
    return ((xb[0] - bl_t) * integrated / n_steps).detach()    # (C, H, W)


def attr_guided_ig(
    model: nn.Module,
    x: torch.Tensor,
    target: int,
    steps: int = 200,
    fraction: float = 0.25,
    max_dist_frac: float = 0.1,
) -> torch.Tensor:
    """
    Guided Integrated Gradients (Kapishnikov et al., CVPR 2021).

    Integrates gradients along an adaptive path from zeros to x, selecting
    features greedily by |gradient × remaining distance| at each step.
    max_dist_frac is the fraction of remaining per-feature distance moved per step
    (scale-invariant; 0.1 → ~99% coverage over ~50 updates per feature).
    """
    dev = next(model.parameters()).device
    x0  = (x if x.dim() == 4 else x.unsqueeze(0)).to(dev).detach()[0]  # (C,H,W)

    x_curr  = torch.zeros_like(x0)
    attr    = torch.zeros_like(x0)
    n_total = x0.numel()
    n_up    = max(1, int(fraction * n_total))

    model.eval()
    for _ in range(steps):
        dx = x0 - x_curr
        if dx.abs().max() < 1e-6:
            break
        x_t  = x_curr.unsqueeze(0).requires_grad_(True)
        grad = torch.autograd.grad(model(x_t)[0, target], x_t)[0][0].detach()

        gain   = (grad * dx).abs().view(-1)
        thresh = gain.kthvalue(max(1, n_total - n_up + 1)).values
        mask   = (gain >= thresh).view_as(x0).float()

        step   = dx * max_dist_frac * mask
        attr   = attr + grad * step
        x_curr = x_curr + step

    return attr.detach()


def sum_collapse(attr: torch.Tensor) -> torch.Tensor:
    """Sum over channels → (H, W)."""
    if attr.dim() == 4:
        attr = attr.squeeze(0)
    return attr.sum(dim=0)


def run_one_attr(
    method: str,
    model: nn.Module,
    x: torch.Tensor,
    target: int,
    background: torch.Tensor | None,
    n_steps: int = 50,
    n_samples: int = 10,
    eg_samples: int = 50,
    sg_samples: int = 50,
    sf_adaptive: float | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Run a single attribution method by name. Used for incremental cache updates."""
    if method == "KL-IG (adaptive)":
        assert sf_adaptive is not None
        return attr_klig(model, x, target, sf_adaptive, n_steps=n_steps, n_samples=n_samples, device=device)
    if method == "KL-IG (σ=0.25)":
        return attr_klig(model, x, target, 0.25, n_steps=n_steps, n_samples=n_samples, device=device)
    if method == "IDG":
        return attr_idg(model, x, target, n_steps=n_steps)
    if method == "ExpGrad":
        assert background is not None
        return attr_expgrad(model, x, target, background, n_samples=eg_samples)
    if method == "IG-zero":
        return attr_ig_zero(model, x, target, n_steps=50)
    if method == "SmoothGrad":
        return attr_smoothgrad(model, x, target, n_samples=sg_samples)
    if method == "Vanilla Grad":
        return attr_vanilla(model, x, target)
    if method == "Blur-IG":
        return attr_blur_ig(model, x, target)
    if method == "Guided IG":
        return attr_guided_ig(model, x, target)
    raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Supplemental metrics
# ---------------------------------------------------------------------------

def gini_coefficient(attr_map: torch.Tensor) -> float:
    """Gini coefficient of |attribution| values — higher = more sparse."""
    a = attr_map.abs().detach().cpu().numpy().ravel()
    a = a[a > 0]
    if len(a) == 0:
        return 0.0
    a = np.sort(a)
    n = len(a)
    return float((2 * np.sum(np.arange(1, n + 1) * a)) / (n * np.sum(a)) - (n + 1) / n)


def infidelity(
    model: nn.Module,
    x: torch.Tensor,
    attr_map: torch.Tensor,
    target: int,
    n_pert: int = INFID_N_PERT,
    sigma: float = INFID_SIGMA,
) -> float:
    """Expected squared error between attribution dot-product and prediction change."""
    dev = next(model.parameters()).device
    x   = x.to(dev)
    with torch.no_grad():
        f_x  = model(x)[0, target].item()
        I    = torch.randn(n_pert, *x.shape[1:], device=dev) * sigma
        f_xp = model(x - I)[:, target].cpu().numpy()
        dot  = (I * attr_map.unsqueeze(0)).view(n_pert, -1).sum(dim=1).cpu().numpy()
    return float(np.mean((dot - (f_x - f_xp)) ** 2))


def compute_occlusion_map(
    model: nn.Module,
    x: torch.Tensor,
    target: int,
    patch_size: int = OCCLUSION_PATCH,
    stride: int = OCCLUSION_STRIDE,
    batch_size: int = 64,
) -> tuple[np.ndarray, int, int]:
    """
    Pre-compute the occlusion importance map for a single image.
    Returns (occ_map, n_h, n_w) — call once per image, reuse for all methods.
    """
    dev = next(model.parameters()).device
    x   = x.to(dev)
    _, C, H, W = x.shape

    positions = [(h, w)
                 for h in range(0, H - patch_size + 1, stride)
                 for w in range(0, W - patch_size + 1, stride)]
    n_h = len(range(0, H - patch_size + 1, stride))
    n_w = len(range(0, W - patch_size + 1, stride))

    with torch.no_grad():
        f0 = model(x)[0, target].item()
        occ_scores = []
        for i in range(0, len(positions), batch_size):
            bpos  = positions[i : i + batch_size]
            batch = x.expand(len(bpos), -1, -1, -1).clone()
            for k, (h, w) in enumerate(bpos):
                batch[k, :, h : h + patch_size, w : w + patch_size] = 0.0
            occ_scores.append(model(batch)[:, target].cpu().numpy())

    occ_map = (f0 - np.concatenate(occ_scores)).reshape(n_h, n_w)
    return occ_map, n_h, n_w


def occlusion_correlation(
    attr_map: torch.Tensor,
    occ_map: np.ndarray,
    n_h: int,
    n_w: int,
    patch_size: int = OCCLUSION_PATCH,
    stride: int = OCCLUSION_STRIDE,
) -> float:
    """Spearman ρ between attribution patch means and a pre-computed occlusion map."""
    a = attr_map.abs().float().unsqueeze(0).unsqueeze(0)
    pooled    = F.avg_pool2d(a, kernel_size=patch_size, stride=stride, padding=0)
    attr_grid = pooled[0, 0, :n_h, :n_w].detach().cpu().numpy()
    rho, _    = stats.spearmanr(occ_map.ravel(), attr_grid.ravel())
    return 0.0 if np.isnan(rho) else float(rho)


# ---------------------------------------------------------------------------
# Object mask — ensemble-based (method-agnostic)
# ---------------------------------------------------------------------------

def ensemble_object_mask(x: torch.Tensor, attr_maps: dict[str, torch.Tensor]) -> np.ndarray:
    """
    Derive a GrabCut object mask seeded by the ensemble mean of all attribution
    methods' absolute maps. No single method has an advantage.

    attr_maps: {method_name: (H, W) sum-collapsed attribution tensor}
    """
    stacked = torch.stack([a.abs() for a in attr_maps.values()])  # (M, H, W)
    ensemble = stacked.mean(dim=0).detach().cpu().numpy()          # (H, W)
    H, W = ensemble.shape

    seed = (ensemble >= np.percentile(ensemble, 80)).astype(np.uint8)
    img_rgb = _denorm(x[0]).permute(1, 2, 0).detach().cpu().numpy()
    img_bgr = (np.clip(img_rgb, 0, 1) * 255).astype(np.uint8)[:, :, ::-1].copy()

    gc_mask = np.where(seed, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    gc_mask[(ensemble >= np.percentile(ensemble, 95))] = cv2.GC_FGD
    border = max(H, W) // 10
    edge = np.zeros((H, W), dtype=np.uint8)
    edge[:border, :] = edge[-border:, :] = edge[:, :border] = edge[:, -border:] = 1
    gc_mask[(edge == 1) & (ensemble < np.percentile(ensemble, 10))] = cv2.GC_BGD

    try:
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(img_bgr, gc_mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
        return np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0
        ).astype(np.uint8)
    except Exception:
        return seed


def _denorm(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(-1, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=x.device).view(-1, 1, 1)
    return x * std + mean


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def object_focus_ratio(attr_map: torch.Tensor, mask: np.ndarray) -> float:
    a = attr_map.abs().detach().cpu().numpy()
    total = a.sum()
    return float(a[mask == 1].sum() / total) if total > 1e-12 else 0.0


def discriminative_concentration(attr_map: torch.Tensor, mask: np.ndarray) -> float:
    a = attr_map.abs().detach().cpu().numpy()
    total = a.sum()
    if total < 1e-12:
        return 0.0
    in_obj   = a[mask == 1].sum() / total
    obj_area = mask.sum() / mask.size
    return float(in_obj / max(obj_area, 1e-12))


def sensitivity_n(
    model: nn.Module,
    x: torch.Tensor,
    attr_map: torch.Tensor,
    target: int,
    fractions: list[float] = SENS_FRACTIONS,
    n_subsets: int = N_SENS_SUBSETS,
) -> np.ndarray:
    """Pearson correlation between attribution subset sums and prediction drops."""
    C, H, W  = x.shape[1], x.shape[2], x.shape[3]
    n_pix    = H * W
    attr_flat = attr_map.detach().cpu().numpy().ravel()
    with torch.no_grad():
        f_orig = model(x).softmax(-1)[0, target].item()

    x0   = x.view(C, n_pix)
    pccs = []
    for frac in fractions:
        n = max(1, int(frac * n_pix))
        subsets   = np.stack([np.random.choice(n_pix, n, replace=False)
                              for _ in range(n_subsets)])
        attr_sums = attr_flat[subsets].sum(axis=1)

        mask = torch.zeros(n_subsets, n_pix, dtype=torch.bool, device=x.device)
        rows = torch.arange(n_subsets, device=x.device).repeat_interleave(n)
        cols = torch.from_numpy(subsets.ravel()).to(x.device)
        mask[rows, cols] = True

        x_batch = x0.unsqueeze(0).expand(n_subsets, -1, -1).clone()
        x_batch[mask.unsqueeze(1).expand(-1, C, -1)] = 0.0
        with torch.no_grad():
            f_masked = model(x_batch.view(n_subsets, C, H, W)).softmax(-1)[:, target].cpu().numpy()

        r, _ = stats.pearsonr(attr_sums, f_orig - f_masked)
        pccs.append(r if not np.isnan(r) else 0.0)
    return np.array(pccs)


# ---------------------------------------------------------------------------
# Per-image evaluation
# ---------------------------------------------------------------------------

def compute_metrics_from_raw(
    model: nn.Module,
    x: torch.Tensor,
    target: int,
    raw_numpy: dict[str, np.ndarray],
    stem: str,
    mask_lookup: dict[str, Path],
    device: torch.device,
    only_metrics: set[str] | None = None,
) -> tuple[dict[str, dict], float, bool]:
    """
    Compute per-method metrics from cached (C,H,W) attribution arrays.

    only_metrics: if given, compute only the named metrics (e.g. {"ofr","dc"}).
                  Valid names: id, ofr, dc, sens, gini, infid, occ.
                  None = compute all.

    Returns (per_method_metrics, obj_frac, has_gt_mask).
    """
    def want(*keys: str) -> bool:
        return only_metrics is None or any(k in only_metrics for k in keys)

    # Restore tensors and sum-collapse
    raw   = {m: torch.from_numpy(a).to(device) for m, a in raw_numpy.items()}
    amaps = {m: sum_collapse(r) for m, r in raw.items()}

    # Object mask (ImageNet-S ground truth)
    mask_path = mask_lookup.get(stem) if mask_lookup else None
    if mask_path is not None:
        obj_mask    = load_imagenet_s_mask(mask_path)
        obj_frac    = float(obj_mask.mean())
        has_gt_mask = True
    else:
        obj_mask    = None
        obj_frac    = float("nan")
        has_gt_mask = False

    # Occlusion map — one batch of forward passes, shared across all methods
    if want("occ"):
        occ_map, n_h, n_w = compute_occlusion_map(model, x, target)
    else:
        occ_map, n_h, n_w = None, 0, 0

    metrics: dict[str, dict] = {}
    for m in METHODS:
        if m not in amaps:
            continue  # method not yet computed for this image (incremental add)
        amap = amaps[m]

        if want("id"):
            (del_auc,       ins_auc,       del_auc_relu,       ins_auc_relu,
             del_auc_zeros, ins_auc_zeros, del_auc_relu_zeros, ins_auc_relu_zeros) = both_auc_signed(
                model, x.squeeze(0), amap, target,
                n_steps=N_ID_STEPS, blur_sigma=10.0, batch_size=64,
            )
        else:
            del_auc = ins_auc = del_auc_relu = ins_auc_relu = float("nan")
            del_auc_zeros = ins_auc_zeros = del_auc_relu_zeros = ins_auc_relu_zeros = float("nan")

        if want("ofr", "dc") and has_gt_mask and obj_frac > 0.02:
            ofr = object_focus_ratio(amap, obj_mask)
            dc  = discriminative_concentration(amap, obj_mask)
        else:
            ofr = float("nan")
            dc  = float("nan")

        sens_mean = float(sensitivity_n(model, x, amap, target).mean()) if want("sens") else float("nan")
        gini      = gini_coefficient(amap) if want("gini") else float("nan")
        infid     = infidelity(model, x, amap, target) if want("infid") else float("nan")
        occ       = occlusion_correlation(amap, occ_map, n_h, n_w) if want("occ") else float("nan")

        metrics[m] = {
            "del_auc":            del_auc,
            "ins_auc":            ins_auc,
            "del_auc_relu":       del_auc_relu,
            "ins_auc_relu":       ins_auc_relu,
            "del_auc_zeros":      del_auc_zeros,
            "ins_auc_zeros":      ins_auc_zeros,
            "del_auc_relu_zeros": del_auc_relu_zeros,
            "ins_auc_relu_zeros": ins_auc_relu_zeros,
            "ofr":                ofr,
            "dc":                 dc,
            "sens_mean":          sens_mean,
            "gini":               gini,
            "infid":              infid,
            "occ":                occ,
        }

    return metrics, obj_frac, has_gt_mask


def evaluate_image(
    model: nn.Module,
    row: dict,
    background: torch.Tensor,
    device: torch.device,
    mask_lookup: dict[str, Path] | None = None,
    n_steps: int = 50,
    n_samples: int = 10,
    eg_samples: int = 50,
    sg_samples: int = 50,
    only_metrics: set[str] | None = None,
) -> dict:
    """Compute attributions + metrics for one image. Caches raw attributions."""
    x      = row["x"].to(device)
    target = row["target"]

    # Adaptive sigma for KLIG
    sf_adaptive = float(np.clip(
        find_sigma_stop(model, x, target=target, tau=0.95), 1.0 / 256.0, 1.0
    ))

    # All (C, H, W) attribution tensors
    raw: dict[str, torch.Tensor] = {}
    raw["KL-IG (adaptive)"] = attr_klig(model, x, target, sf_adaptive,
                                         n_steps=n_steps, n_samples=n_samples, device=device)
    raw["KL-IG (σ=0.25)"]   = attr_klig(model, x, target, 0.25,
                                         n_steps=n_steps, n_samples=n_samples, device=device)
    raw["IDG"]          = attr_idg(model, x, target)
    raw["ExpGrad"]      = attr_expgrad(model, x, target, background, n_samples=eg_samples)
    raw["IG-zero"]      = attr_ig_zero(model, x, target, n_steps=50)
    raw["SmoothGrad"]   = attr_smoothgrad(model, x, target, n_samples=sg_samples)
    raw["Vanilla Grad"] = attr_vanilla(model, x, target)
    raw["Blur-IG"]      = attr_blur_ig(model, x, target)
    raw["Guided IG"]    = attr_guided_ig(model, x, target)

    # Store as numpy so the cache is self-contained and portable
    raw_numpy = {m: raw[m].detach().cpu().numpy() for m in METHODS if m in raw}

    # Compute all metrics from the raw arrays
    metrics, obj_frac, has_gt_mask = compute_metrics_from_raw(
        model, x, target, raw_numpy,
        stem=row.get("stem", ""),
        mask_lookup=mask_lookup or {},
        device=device,
        only_metrics=only_metrics,
    )

    result: dict = {
        "idx":         row["idx"],
        "target":      target,
        "label_str":   row["label_str"],
        "sigma_final": sf_adaptive,
        "obj_frac":    obj_frac,
        "has_gt_mask": has_gt_mask,
        "raw_attr":    raw_numpy,   # (C,H,W) float32 arrays — enables metric recompute
    }
    result.update(metrics)
    return result


# ---------------------------------------------------------------------------
# Summary table helpers
# ---------------------------------------------------------------------------

def ci95(vals: list[float]) -> float:
    arr = np.array([v for v in vals if not np.isnan(v)], dtype=np.float64)
    return float(1.96 * arr.std() / np.sqrt(max(len(arr), 1)))


def mean_clean(vals: list[float]) -> float:
    arr = np.array([v for v in vals if not np.isnan(v)], dtype=np.float64)
    return float(arr.mean()) if len(arr) > 0 else float("nan")


def print_table(summary: dict[str, dict]) -> None:
    # I/D section header
    print("\n--- Insertion / Deletion (blur substrate) ---")
    h1 = f"{'Method':<24} {'Ins|abs':>8} {'Del|abs':>8} {'I-D|abs':>8} {'Ins|relu':>9} {'Del|relu':>9} {'I-D|relu':>9}"
    print(h1); print("-" * len(h1))
    for m in METHODS:
        s = summary[m]
        print(f"{m:<24}"
              f" {s['ins_mean']:>7.4f}±{s['ins_ci']:>5.4f}"
              f" {s['del_mean']:>7.4f}±{s['del_ci']:>5.4f}"
              f" {s['ins_mean']-s['del_mean']:>8.4f}"
              f" {s['ins_relu_mean']:>8.4f}±{s['ins_relu_ci']:>5.4f}"
              f" {s['del_relu_mean']:>8.4f}±{s['del_relu_ci']:>5.4f}"
              f" {s['ins_relu_mean']-s['del_relu_mean']:>9.4f}")

    print("\n--- Insertion / Deletion (zeros substrate) ---")
    h2 = f"{'Method':<24} {'Ins|abs':>8} {'Del|abs':>8} {'I-D|abs':>8} {'Ins|relu':>9} {'Del|relu':>9} {'I-D|relu':>9}"
    print(h2); print("-" * len(h2))
    for m in METHODS:
        s = summary[m]
        print(f"{m:<24}"
              f" {s['ins_zeros_mean']:>7.4f}±{s['ins_zeros_ci']:>5.4f}"
              f" {s['del_zeros_mean']:>7.4f}±{s['del_zeros_ci']:>5.4f}"
              f" {s['ins_zeros_mean']-s['del_zeros_mean']:>8.4f}"
              f" {s['ins_relu_zeros_mean']:>8.4f}±{s['ins_relu_zeros_ci']:>5.4f}"
              f" {s['del_relu_zeros_mean']:>8.4f}±{s['del_relu_zeros_ci']:>5.4f}"
              f" {s['ins_relu_zeros_mean']-s['del_relu_zeros_mean']:>9.4f}")

    print("\n--- Other metrics ---")
    h3 = f"{'Method':<24} {'OFR':>7} {'DC':>7} {'Sens-n':>7} {'Gini':>6} {'Infid':>9} {'OccCorr':>8}"
    print(h3); print("-" * len(h3))
    for m in METHODS:
        s = summary[m]
        print(f"{m:<24}"
              f" {s['ofr_mean']:>6.4f} {s['dc_mean']:>6.4f}"
              f" {s['sens_mean']:>6.4f} {s['gini_mean']:>5.4f}"
              f" {s['infid_mean']:>8.5f} {s['occ_mean']:>7.4f}")


def save_latex(summary: dict[str, dict], outdir: Path) -> None:
    # Table 1: abs ranking (conventional)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Fixed evaluation --- $|$attr$|$ pixel ranking (magnitude saliency).}",
        r"\label{tab:core_metrics_abs}",
        r"\renewcommand{\arraystretch}{1.18}",
        r"\setlength{\tabcolsep}{5.5pt}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Ins.\ AUC $\uparrow$} & \textbf{Del.\ AUC $\downarrow$} "
        r"& \textbf{Ins$-$Del $\uparrow$} & \textbf{OFR $\uparrow$} "
        r"& \textbf{DC $\uparrow$} & \textbf{Sens.-$n$ PCC $\uparrow$} \\",
        r"\midrule",
    ]
    for m in METHODS:
        s  = summary[m]
        id_abs = s["ins_mean"] - s["del_mean"]
        lines.append(
            rf"{m} & {s['ins_mean']:.4f}_{{{s['ins_ci']:.4f}}} & "
            rf"{s['del_mean']:.4f}_{{{s['del_ci']:.4f}}} & ${id_abs:.4f}$ & "
            rf"{s['ofr_mean']:.4f}_{{{s['ofr_ci']:.4f}}} & "
            rf"{s['dc_mean']:.4f}_{{{s['dc_ci']:.4f}}} & "
            rf"{s['sens_mean']:.4f}_{{{s['sens_ci']:.4f}}} \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    path = outdir / "results_table_abs.tex"
    path.write_text("\n".join(lines))
    print(f"[out] LaTeX table (abs) → {path}")

    # Table 2: relu ranking (positive target evidence)
    lines2 = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Fixed evaluation --- relu(attr) pixel ranking (positive target evidence).}",
        r"\label{tab:core_metrics_relu}",
        r"\renewcommand{\arraystretch}{1.18}",
        r"\setlength{\tabcolsep}{5.5pt}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Ins.\ AUC $\uparrow$} & \textbf{Del.\ AUC $\downarrow$} "
        r"& \textbf{Ins$-$Del $\uparrow$} & \textbf{OFR $\uparrow$} "
        r"& \textbf{DC $\uparrow$} & \textbf{Sens.-$n$ PCC $\uparrow$} \\",
        r"\midrule",
    ]
    for m in METHODS:
        s  = summary[m]
        id_relu = s["ins_relu_mean"] - s["del_relu_mean"]
        lines2.append(
            rf"{m} & {s['ins_relu_mean']:.4f}_{{{s['ins_relu_ci']:.4f}}} & "
            rf"{s['del_relu_mean']:.4f}_{{{s['del_relu_ci']:.4f}}} & ${id_relu:.4f}$ & "
            rf"{s['ofr_mean']:.4f}_{{{s['ofr_ci']:.4f}}} & "
            rf"{s['dc_mean']:.4f}_{{{s['dc_ci']:.4f}}} & "
            rf"{s['sens_mean']:.4f}_{{{s['sens_ci']:.4f}}} \\"
        )
    lines2 += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    path2 = outdir / "results_table_relu.tex"
    path2.write_text("\n".join(lines2))
    print(f"[out] LaTeX table (relu) → {path2}")

    # Table 3: zeros substrate (abs ranking) — control for blur-substrate bias
    lines3 = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Fixed evaluation --- zeros baseline, $|$attr$|$ ranking (substrate-neutral control).}",
        r"\label{tab:core_metrics_zeros}",
        r"\renewcommand{\arraystretch}{1.18}",
        r"\setlength{\tabcolsep}{5.5pt}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Ins.\ AUC $\uparrow$} & \textbf{Del.\ AUC $\downarrow$} "
        r"& \textbf{Ins$-$Del $\uparrow$} & \textbf{OFR $\uparrow$} "
        r"& \textbf{DC $\uparrow$} & \textbf{Sens.-$n$ PCC $\uparrow$} \\",
        r"\midrule",
    ]
    for m in METHODS:
        s  = summary[m]
        id_z = s["ins_zeros_mean"] - s["del_zeros_mean"]
        lines3.append(
            rf"{m} & {s['ins_zeros_mean']:.4f}_{{{s['ins_zeros_ci']:.4f}}} & "
            rf"{s['del_zeros_mean']:.4f}_{{{s['del_zeros_ci']:.4f}}} & ${id_z:.4f}$ & "
            rf"{s['ofr_mean']:.4f}_{{{s['ofr_ci']:.4f}}} & "
            rf"{s['dc_mean']:.4f}_{{{s['dc_ci']:.4f}}} & "
            rf"{s['sens_mean']:.4f}_{{{s['sens_ci']:.4f}}} \\"
        )
    lines3 += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    path3 = outdir / "results_table_zeros.tex"
    path3.write_text("\n".join(lines3))
    print(f"[out] LaTeX table (zeros) → {path3}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--val-dir",   required=True, type=Path)
    p.add_argument("--train-dir", required=True, type=Path)
    p.add_argument("--gt-file",   required=True, type=Path)
    p.add_argument("--meta",      required=True, type=Path)
    p.add_argument("--outdir",    default=Path("eval/results_fixed"), type=Path)
    p.add_argument("--n-images",  default=1000, type=int)
    p.add_argument("--n-background", default=100, type=int)
    p.add_argument("--n-steps",   default=50,  type=int, help="KLIG integration steps")
    p.add_argument("--n-samples", default=10,  type=int, help="KLIG MC samples per step")
    p.add_argument("--model",     default="resnet50", choices=["resnet50", "vit"],
                   help="Backbone architecture")
    p.add_argument(
        "--imagenet-s-dir", default=None, type=Path,
        help="Path to ImageNetS919/validation-segmentation/ for ground-truth OFR/DC masks",
    )
    p.add_argument(
        "--recompute-metrics", action="store_true",
        help="Re-run metric computation from cached raw_attr without re-running attributions. "
             "Useful for adding/changing metrics without a full re-run.",
    )
    p.add_argument(
        "--add-new-methods", action="store_true",
        help="For each cached image, run attribution only for methods that are in METHODS but "
             "missing from the cache's raw_attr, then recompute all metrics. "
             "Much faster than a full rerun when adding one new method.",
    )
    p.add_argument(
        "--recompute-methods", type=str, default=None,
        help="Comma-separated method names to force-recompute even if already in cache. "
             "Strips those methods from cached raw_attr before running (use with "
             "--add-new-methods to rerun attribution + metrics for corrected implementations).",
    )
    p.add_argument(
        "--require-mask", action="store_true",
        help="Only evaluate images that have an ImageNet-S ground-truth mask "
             "(requires --imagenet-s-dir).  Ensures OFR/DC is computed on the full N "
             "instead of the ~25%% of val images with mask coverage.",
    )
    p.add_argument(
        "--only-metrics", type=str, default=None,
        help="Comma-separated subset of metrics to compute: id,ofr,dc,sens,gini,infid,occ. "
             "Skips the corresponding expensive steps (e.g. 'ofr,dc' skips I/D, sens-n, "
             "infidelity, occlusion). Useful with --require-mask to get tight OFR/DC CIs "
             "without rerunning full I/D on 1k masked images. Default: compute all.",
    )
    p.add_argument("--device",    default=None, type=str)
    p.add_argument("--seed",      default=42,  type=int)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[eval] Device: {device}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.outdir / "cache"
    cache_dir.mkdir(exist_ok=True)

    # ── Model ──────────────────────────────────────────────────────────────
    print(f"[eval] Architecture: {args.model}")
    model, _ = load_model(device, arch=args.model)

    # ── Dataset ────────────────────────────────────────────────────────────
    # ── ImageNet-S masks (loaded early if --require-mask so dataset filtering works) ──
    mask_lookup: dict[str, Path] = {}
    if args.imagenet_s_dir is not None:
        mask_lookup = load_imagenet_s_masks(args.imagenet_s_dir)
    else:
        if args.require_mask:
            raise ValueError("--require-mask needs --imagenet-s-dir")
        print("[masks] --imagenet-s-dir not provided; OFR/DC will be NaN for all images")

    dataset = load_val_dataset(
        args.val_dir, args.gt_file, args.meta, args.train_dir,
        args.n_images, model, device, seed=args.seed,
        require_mask=mask_lookup if args.require_mask else None,
    )

    # ── Background for ExpGrad ─────────────────────────────────────────────
    print(f"[eval] Loading {args.n_background} background images for ExpGrad...")
    background = load_background(args.train_dir, args.n_background, device, seed=args.seed)

    # ── Per-image evaluation (with cache/resume) ────────────────────────────
    all_results: list[dict] = []
    skipped = recomputed = 0

    # Parse --only-metrics into a set (None = compute all)
    only_metrics: set[str] | None = None
    if args.only_metrics:
        only_metrics = {m.strip() for m in args.only_metrics.split(",")}
        print(f"[eval] Computing only metrics: {only_metrics}")

    def want_any(*keys: str) -> bool:
        return only_metrics is None or any(k in only_metrics for k in keys)

    # Methods to force-recompute (strip from cache before processing)
    force_recompute: list[str] = (
        [m.strip() for m in args.recompute_methods.split(",")]
        if args.recompute_methods else []
    )
    if force_recompute:
        print(f"[eval] Will force-recompute attributions for: {force_recompute}")

    for row in tqdm(dataset, desc="evaluating"):
        cache_path = cache_dir / f"img_{row['idx']:06d}.pkl"

        # If --recompute-methods is set, strip those keys from the cached entry
        # so the --add-new-methods path treats them as missing and reruns them.
        if force_recompute and cache_path.exists():
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            changed = False
            for m in force_recompute:
                if m in cached.get("raw_attr", {}):
                    del cached["raw_attr"][m]
                    changed = True
                if m in cached:
                    del cached[m]
                    changed = True
            if changed:
                with open(cache_path, "wb") as f:
                    pickle.dump(cached, f)

        if args.recompute_metrics and cache_path.exists():
            # Fast path: recompute metrics from cached raw attributions only.
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if "raw_attr" not in cached:
                tqdm.write(f"  [SKIP recompute] idx={row['idx']} — no raw_attr in cache, needs full rerun")
                all_results.append(cached)
                continue
            t0 = time.time()
            try:
                x = row["x"].to(device)
                metrics, obj_frac, has_gt_mask = compute_metrics_from_raw(
                    model, x, cached["target"], cached["raw_attr"],
                    stem=row.get("stem", ""),
                    mask_lookup=mask_lookup,
                    device=device,
                    only_metrics=only_metrics,
                )
                cached["obj_frac"]    = obj_frac
                cached["has_gt_mask"] = has_gt_mask
                cached.update(metrics)
                with open(cache_path, "wb") as f:
                    pickle.dump(cached, f)
                elapsed = time.time() - t0
                tqdm.write(f"  [{row['idx']:4d}] recomputed metrics  {elapsed:.1f}s")
                recomputed += 1
                result = cached
            except Exception as exc:
                import traceback
                tqdm.write(f"  [ERROR recompute] idx={row['idx']}: {exc}")
                traceback.print_exc()
                continue

        elif cache_path.exists():
            with open(cache_path, "rb") as f:
                result = pickle.load(f)

            # Incremental method addition: compute attribution + metrics only for
            # methods that are in METHODS but missing from this cache entry.
            # Existing method entries in the cache are NOT touched.
            existing_raw = result.get("raw_attr", {})
            missing = [m for m in METHODS if m not in existing_raw]
            if missing:
                if not args.add_new_methods:
                    tqdm.write(
                        f"  [{row['idx']:4d}] missing {missing} — "
                        "rerun with --add-new-methods to fill them in"
                    )
                else:
                    x = row["x"].to(device)
                    sf_adaptive = result.get("sigma_final") or float(np.clip(
                        find_sigma_stop(model, x, target=result["target"], tau=0.95),
                        1.0 / 256.0, 1.0,
                    ))
                    t0 = time.time()

                    # 1. Run attribution only for missing methods
                    for m in missing:
                        try:
                            new_attr = run_one_attr(
                                m, model, x, result["target"], background,
                                n_steps=args.n_steps, n_samples=args.n_samples,
                                eg_samples=50, sg_samples=50,
                                sf_adaptive=sf_adaptive, device=device,
                            )
                            existing_raw[m] = new_attr.detach().cpu().numpy()
                        except Exception as exc:
                            tqdm.write(f"  [ERROR] attr {m}: {exc}")
                    result["raw_attr"] = existing_raw

                    # 2. Compute metrics only for the newly added methods
                    to_compute = [m for m in missing if m in existing_raw]
                    if to_compute:
                        raw_t  = {m: torch.from_numpy(existing_raw[m]).to(device) for m in to_compute}
                        amaps  = {m: sum_collapse(r) for m, r in raw_t.items()}

                        stem = row.get("stem", "")
                        mask_path = mask_lookup.get(stem) if mask_lookup else None
                        if mask_path is not None:
                            obj_mask   = load_imagenet_s_mask(mask_path)
                            obj_frac   = float(obj_mask.mean())
                            has_gt_mask = True
                        else:
                            obj_mask   = None
                            obj_frac   = float("nan")
                            has_gt_mask = False

                        if want_any("occ"):
                            occ_map, n_h, n_w = compute_occlusion_map(model, x, result["target"])
                        else:
                            occ_map, n_h, n_w = None, 0, 0

                        for m in to_compute:
                            amap = amaps[m]
                            if want_any("id"):
                                (del_auc, ins_auc, del_relu, ins_relu,
                                 del_zeros, ins_zeros, del_relu_z, ins_relu_z) = both_auc_signed(
                                    model, x.squeeze(0), amap, result["target"],
                                    n_steps=N_ID_STEPS, blur_sigma=10.0, batch_size=64,
                                )
                            else:
                                del_auc = ins_auc = del_relu = ins_relu = float("nan")
                                del_zeros = ins_zeros = del_relu_z = ins_relu_z = float("nan")
                            if want_any("ofr", "dc") and has_gt_mask and obj_frac > 0.02:
                                ofr = object_focus_ratio(amap, obj_mask)
                                dc  = discriminative_concentration(amap, obj_mask)
                            else:
                                ofr = float("nan")
                                dc  = float("nan")
                            result[m] = {
                                "del_auc":            del_auc,
                                "ins_auc":            ins_auc,
                                "del_auc_relu":       del_relu,
                                "ins_auc_relu":       ins_relu,
                                "del_auc_zeros":      del_zeros,
                                "ins_auc_zeros":      ins_zeros,
                                "del_auc_relu_zeros": del_relu_z,
                                "ins_auc_relu_zeros": ins_relu_z,
                                "ofr":       ofr,
                                "dc":        dc,
                                "sens_mean": float(sensitivity_n(model, x, amap, result["target"]).mean()) if want_any("sens") else float("nan"),
                                "gini":      gini_coefficient(amap) if want_any("gini") else float("nan"),
                                "infid":     infidelity(model, x, amap, result["target"]) if want_any("infid") else float("nan"),
                                "occ":       occlusion_correlation(amap, occ_map, n_h, n_w) if want_any("occ") else float("nan"),
                            }

                    with open(cache_path, "wb") as f:
                        pickle.dump(result, f)
                    tqdm.write(f"  [{row['idx']:4d}] added {missing}  {time.time()-t0:.1f}s")
                    recomputed += 1

            skipped += 1

        else:
            t0 = time.time()
            try:
                result = evaluate_image(
                    model, row, background, device,
                    mask_lookup=mask_lookup,
                    n_steps=args.n_steps, n_samples=args.n_samples,
                    only_metrics=only_metrics,
                )
                with open(cache_path, "wb") as f:
                    pickle.dump(result, f)
                elapsed = time.time() - t0
                mask_tag = "M" if result["has_gt_mask"] else "-"
                tqdm.write(
                    f"  [{row['idx']:4d}] {row['label_str'][:20]:<20} "
                    f"σ={result['sigma_final']:.3f}  [{mask_tag}]  {elapsed:.1f}s"
                )
            except Exception as exc:
                import traceback
                tqdm.write(f"  [ERROR] idx={row['idx']} {row['label_str']}: {exc}")
                traceback.print_exc()
                continue

        all_results.append(result)

    print(f"\n[eval] Done: {len(all_results)} images "
          f"({skipped} from cache, {recomputed} metrics recomputed)")

    # ── Aggregate ─────────────────────────────────────────────────────────
    agg: dict[str, dict[str, list[float]]] = {m: {
        "ins": [], "del": [], "ins_relu": [], "del_relu": [],
        "ins_zeros": [], "del_zeros": [], "ins_relu_zeros": [], "del_relu_zeros": [],
        "ofr": [], "dc": [], "sens": [],
        "gini": [], "infid": [], "occ": [],
    } for m in METHODS}

    for result in all_results:
        for m in METHODS:
            if m not in result:
                continue
            r = result[m]
            agg[m]["ins"].append(r["ins_auc"])
            agg[m]["del"].append(r["del_auc"])
            agg[m]["ins_relu"].append(r.get("ins_auc_relu", float("nan")))
            agg[m]["del_relu"].append(r.get("del_auc_relu", float("nan")))
            agg[m]["ins_zeros"].append(r.get("ins_auc_zeros", float("nan")))
            agg[m]["del_zeros"].append(r.get("del_auc_zeros", float("nan")))
            agg[m]["ins_relu_zeros"].append(r.get("ins_auc_relu_zeros", float("nan")))
            agg[m]["del_relu_zeros"].append(r.get("del_auc_relu_zeros", float("nan")))
            agg[m]["ofr"].append(r["ofr"])
            agg[m]["dc"].append(r["dc"])
            agg[m]["sens"].append(r["sens_mean"])
            agg[m]["gini"].append(r.get("gini", float("nan")))
            agg[m]["infid"].append(r.get("infid", float("nan")))
            agg[m]["occ"].append(r.get("occ", float("nan")))

    summary: dict[str, dict] = {}
    for m in METHODS:
        summary[m] = {
            "ins_mean":            mean_clean(agg[m]["ins"]),
            "ins_ci":              ci95(agg[m]["ins"]),
            "del_mean":            mean_clean(agg[m]["del"]),
            "del_ci":              ci95(agg[m]["del"]),
            "ins_relu_mean":       mean_clean(agg[m]["ins_relu"]),
            "ins_relu_ci":         ci95(agg[m]["ins_relu"]),
            "del_relu_mean":       mean_clean(agg[m]["del_relu"]),
            "del_relu_ci":         ci95(agg[m]["del_relu"]),
            "ins_zeros_mean":      mean_clean(agg[m]["ins_zeros"]),
            "ins_zeros_ci":        ci95(agg[m]["ins_zeros"]),
            "del_zeros_mean":      mean_clean(agg[m]["del_zeros"]),
            "del_zeros_ci":        ci95(agg[m]["del_zeros"]),
            "ins_relu_zeros_mean": mean_clean(agg[m]["ins_relu_zeros"]),
            "ins_relu_zeros_ci":   ci95(agg[m]["ins_relu_zeros"]),
            "del_relu_zeros_mean": mean_clean(agg[m]["del_relu_zeros"]),
            "del_relu_zeros_ci":   ci95(agg[m]["del_relu_zeros"]),
            "ofr_mean":       mean_clean(agg[m]["ofr"]),
            "ofr_ci":         ci95(agg[m]["ofr"]),
            "dc_mean":        mean_clean(agg[m]["dc"]),
            "dc_ci":          ci95(agg[m]["dc"]),
            "sens_mean":      mean_clean(agg[m]["sens"]),
            "sens_ci":        ci95(agg[m]["sens"]),
            "gini_mean":      mean_clean(agg[m]["gini"]),
            "gini_ci":        ci95(agg[m]["gini"]),
            "infid_mean":     mean_clean(agg[m]["infid"]),
            "infid_ci":       ci95(agg[m]["infid"]),
            "occ_mean":       mean_clean(agg[m]["occ"]),
            "occ_ci":         ci95(agg[m]["occ"]),
            "n":              len([v for v in agg[m]["ins"] if not np.isnan(v)]),
        }

    print_table(summary)
    save_latex(summary, args.outdir)

    # Save full summary as pickle
    with open(args.outdir / "summary.pkl", "wb") as f:
        pickle.dump(summary, f)
    print(f"[out] Summary pickle → {args.outdir / 'summary.pkl'}")


if __name__ == "__main__":
    main()
