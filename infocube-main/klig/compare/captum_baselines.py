"""
Thin wrappers around Captum methods for comparison.

All functions return a (H, W) attribution tensor (abs-max collapsed across
channels) so that comparisons with ImageAttributionResult.attr_map() are
apples-to-apples.

Methods provided:
  - run_ig:                 Integrated Gradients (Captum), zero baseline
  - run_smoothgrad:         SmoothGrad (Captum NoiseTunnel wrapping Saliency)
  - run_expected_gradients: Expected Gradients (Captum GradientShap),
                            baseline = random samples from a background set.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from captum.attr import GradientShap, IntegratedGradients, NoiseTunnel, Saliency


def _absmax_collapse(attr: torch.Tensor) -> torch.Tensor:
    """(1, C, H, W) or (C, H, W) -> (H, W) via abs-max across channels."""
    if attr.dim() == 4:
        attr = attr.squeeze(0)
    abs_a = attr.abs()
    idx = abs_a.argmax(dim=0, keepdim=True)
    return attr.gather(0, idx).squeeze(0)


def run_ig(
    model: nn.Module,
    x: torch.Tensor,
    target: int,
    n_steps: int = 50,
    baseline: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Captum Integrated Gradients.

    Returns:
        (H, W) attribution map (abs-max collapsed).
    """
    model.eval()
    if x.dim() == 3:
        x = x.unsqueeze(0)

    x = x.to(next(model.parameters()).device)
    if baseline is None:
        baseline = torch.zeros_like(x)
    else:
        baseline = baseline.to(x.device)
        if baseline.dim() == 3:
            baseline = baseline.unsqueeze(0)

    ig = IntegratedGradients(model)
    attr = ig.attribute(
        x,
        baselines=baseline,
        target=target,
        n_steps=n_steps,
        method="gausslegendre",
    )

    return _absmax_collapse(attr.detach())


def run_smoothgrad(
    model: nn.Module,
    x: torch.Tensor,
    target: int,
    n_samples: int = 50,
    stdev_spread: float = 0.15,
) -> torch.Tensor:
    """
    Captum SmoothGrad (NoiseTunnel wrapping Saliency).

    Returns:
        (H, W) attribution map (abs-max collapsed).
    """
    model.eval()
    if x.dim() == 3:
        x = x.unsqueeze(0)

    x = x.to(next(model.parameters()).device)
    x = x.requires_grad_(True)

    saliency = Saliency(model)
    nt = NoiseTunnel(saliency)

    input_range = float((x.max() - x.min()).item())
    stdevs = stdev_spread * input_range

    attr = nt.attribute(
        x,
        nt_type="smoothgrad",
        nt_samples=n_samples,
        stdevs=stdevs,
        target=target,
        abs=False,
    )

    return _absmax_collapse(attr.detach())


def run_expected_gradients(
    model: nn.Module,
    x: torch.Tensor,
    target: int,
    background: torch.Tensor,
    n_samples: int = 50,
) -> torch.Tensor:
    """
    Expected Gradients (Erion et al. 2021).

    Uses Captum's GradientShap with stdevs=0.0 (pure EG, no SmoothGrad noise)
    and a background set of training images as baselines.

    Returns:
        (H, W) attribution map (abs-max collapsed).
    """
    model.eval()
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.to(next(model.parameters()).device)
    background = background.to(x.device)

    gs = GradientShap(model)
    attr = gs.attribute(
        x,
        baselines=background,
        target=target,
        n_samples=n_samples,
        stdevs=0.0,
    )

    return _absmax_collapse(attr.detach())


def run_all(
    model: nn.Module,
    x: torch.Tensor,
    target: int,
    ig_steps: int = 50,
    sg_samples: int = 50,
    background: torch.Tensor | None = None,
    eg_samples: int = 50,
) -> dict[str, torch.Tensor]:
    """
    Convenience: run all Captum baselines and return a dict of (H, W) maps.

    If background is provided, Expected Gradients is also included.
    """
    results = {
        "IG (zero)":   run_ig(model, x, target=target, n_steps=ig_steps),
        "SmoothGrad":  run_smoothgrad(model, x, target=target, n_samples=sg_samples),
    }
    if background is not None:
        results["ExpectedGradients"] = run_expected_gradients(
            model, x, target=target, background=background, n_samples=eg_samples,
        )
    return results
