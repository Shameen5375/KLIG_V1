# KLIG: KL-path Integrated Gradients

KLIG is a feature attribution method for image classifiers that integrates gradients along a path through **distribution space** (mu, logvar) rather than pixel space. It uses KL divergence as the natural measure of information added per feature.

Instead of interpolating from a black image to the input (as standard Integrated Gradients does), KLIG interpolates from a pure-noise distribution N(0, I) to a near-deterministic distribution centered at the input image. At each step along this path, the model receives noisy samples drawn from the current distribution, and the gradient of the model's output with respect to the distribution parameters (mu and logvar) is computed via the reparameterization trick.

Key advantages over standard IG:

- **Adaptive stopping** (`sigma_stop`): A binary search finds the maximum noise level the model can tolerate while retaining confidence, so the integration path ends at a meaningful point rather than an arbitrary sigma=1/256.
- **Completeness in KL space**: The sum of attributions approximates E[f(x_final)] - E[f(x_noise)], giving a principled information-theoretic decomposition.
- **mu/logvar decomposition**: Attribution splits into a mean-shift component (which features matter) and a variance-reduction component (which features need to be precise).

## Installation

Requires Python >= 3.10.

```bash
pip install -e .
```

This installs PyTorch, torchvision, Captum, matplotlib, numpy, tqdm, and Pillow.

## Usage

### Quick start: compare KLIG against baselines on a single image

```bash
python compare.py --images path/to/image.jpg --outdir results/
```

This loads a pretrained ResNet50, runs KLIG alongside Captum IG and SmoothGrad, and saves a PNG grid to `results/`.

### Recommended: use adaptive sigma_stop

The `--adaptive-sigma` flag enables per-image adaptive stopping, which is responsible for most of KLIG's advantage over baselines:

```bash
python compare.py --images path/to/image.jpg --outdir results/ --adaptive-sigma
```

### Include Expected Gradients baseline

To also compare against Expected Gradients (which approximates SHAP values), provide a directory of training images as background:

```bash
python compare.py \
    --images path/to/image.jpg \
    --outdir results/ \
    --adaptive-sigma \
    --background-dir /path/to/imagenet/train/ \
    --n-background 100
```

### Process a directory of images

```bash
python compare.py --images path/to/image_dir/ --outdir results/ --adaptive-sigma
```

### All options

```
--images          Image file(s) or directories to process (required)
--outdir          Output directory (default: results/)
--target          ImageNet class index to attribute (default: argmax)
--n-steps         KLIG integration steps (default: 50)
--n-samples       KLIG MC samples per step (default: 10)
--ig-steps        Captum IG steps (default: 50)
--sg-samples      SmoothGrad samples (default: 50)
--sigma-final     KLIG final sigma (default: 1/256, ignored with --adaptive-sigma)
--adaptive-sigma  Enable adaptive sigma_stop (recommended)
--clip-pct        Attribution colour scale clip percentile (default: 99)
--background-dir  ImageNet train dir for Expected Gradients
--n-background    Number of background images (default: 100)
--device          Torch device, e.g. 'cuda' or 'cpu' (default: auto)
```

## Output

The script produces a PNG grid per image with the following layout:

- **Left column**: Top-15 predicted class probabilities (bar chart) and the original image.
- **Attribution columns** (one row per top-3 predicted class):
  - **KLIG combined**: Full KL-path attribution (mu + logvar components).
  - **KLIG mu-only**: Attribution from the mean-shift component -- which features need to be in the right place.
  - **KLIG logvar-only**: Attribution from the variance-reduction component -- which features need to be precise (low noise).
  - **IG (zero)**: Standard Integrated Gradients with a zero (black) baseline.
  - **SmoothGrad**: Gradient averaged over noisy copies of the input.
  - **ExpectedGradients** (if background provided): IG averaged over training-image baselines (approximates SHAP).

Attribution maps use a diverging red-blue colourmap: blue = positive attribution (evidence for the class), red = negative (evidence against).

## Project structure

```
klig/
  core/
    path.py         Path parameterizations: LinearPath, PowerPath, DecoupledPath
    integrator.py   KLIntegratedGradients engine (model-agnostic)
    kl.py           KL divergence utilities
  image/
    attribution.py  ImageAttributor wrapper with channel-collapse utilities
    stopping.py     Adaptive sigma_stop via binary search
    viz.py          Attribution grid rendering
  compare/
    captum_baselines.py   Wrappers for Captum IG, SmoothGrad, Expected Gradients
compare.py          Main comparison script (CLI entry point)
pyproject.toml      Package metadata and dependencies
```
