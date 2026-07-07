# Augmentation-consistency of gated CS_struct — paper text

## Methods

**Augmentation consistency.** Following the consistency-under-augmentation principle of
COSE, we test whether gated CS_struct is stable under image changes that do not alter the
model's decision, yet responsive to changes that do. For each of *n* = 60 confused-pair
images we apply five label-preserving transforms — horizontal flip, a ±10° rotation, a
brightness scaling (×1.2), a contrast scaling (×1.3), and a centre crop-and-zoom (0.85) —
and recompute the full pipeline on each transformed image: Felzenszwalb superpixels, the
per-segment occlusion deltas that define the discriminative region *R* (top-25% of segments
by |Δ_y1 − Δ_y2|), the per-class attributions, and the gated coherence score. Because
CS_struct is a bounded coherence ratio, we quantify stability with the absolute drift
|CS_transformed − CS_original|. As these images lie near the y1/y2 decision boundary by
construction, a transform occasionally changes the top-1 prediction; we therefore report the
fraction of prediction-preserving cases per transform and restrict the pooled stability
statistic to that subset. To establish that the metric is not merely insensitive, we add a
label-changing control: on the same image we replace y2 with a random class and recompute
CS_struct, which should drift substantially more. Significance is assessed with a paired
per-image Wilcoxon signed-rank test comparing each image's mean preserving drift against its
class-swap drift. Finally, under the near-exact horizontal flip we recompute CS_struct for all
eleven attribution methods and correlate the pre- and post-flip method rankings with Spearman's ρ.

## Results

Gated CS_struct is stable under label-preserving transforms and sensitive to class change.
The mean pooled drift under label-preserving transforms is 0.063, against a baseline CS_struct
of 0.104 ± 0.015, whereas the label-changing class-swap control drifts 0.111 — 1.8× larger
(paired Wilcoxon p = 0.027). The per-transform breakdown shows the smallest drift for the
photometric transforms (brightness 0.041, contrast 0.051) and the largest for the ±10°
rotation (0.088), which is also the transform that most often changes the prediction
(prediction preserved in 57% of cases, versus 75–88% for the others). Under horizontal flip
the eleven-method ranking is exactly preserved (Spearman ρ = +1.00), with KL-IG²-adaptive
remaining the top-scoring method (0.099 → 0.123). A per-image visual example confirms the
mechanism: the discriminative region *R* and the class-difference coherence map translate,
flip, and rotate rigidly with the object under geometric transforms. Together with our
synthetic ground-truth validation, this provides a real-image robustness result: CS_struct is
neither an artefact of the synthetic construction nor fragile to trivial, label-preserving
image changes, and the method ordering it induces is invariant to image orientation.

## Reproduce

- `augment_consistency.py [n]` — runs the check (checkpointed to `cs_viz_cache/`), emits the
  2-panel figure `cs_viz_outputs/augment_consistency.png` and summary pickle.
- `augment_table.py` — detailed tables (`cs_viz_outputs/augment_table.png`): per-transform
  consistency + class-swap control, and the base-vs-flip method leaderboard with ranks.
- `augment_example.py` — representative per-image visualization
  (`cs_viz_outputs/augment_example.png`): region *R* and class-diff coherence across transforms.
