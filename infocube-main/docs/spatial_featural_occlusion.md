# Spatial vs Featural regime analysis (segment occlusion) — paper text

## Methods

**Discriminative regions by segment occlusion.** For each confused-pair image we ask
*where* the model's evidence for its top-1 and top-2 classes lives, using a purely
model-based (gradient-free) probe. We first partition the image into superpixels with
Felzenszwalb segmentation (scale 0.6, σ 0.8, min-size 100), yielding on the order of 60
segments per image. We then occlude each segment in turn (setting its pixels to zero) and
measure the resulting drop in the model's softmax probability for the top-1 class,
d₁(k) = p_{y1}(x) − p_{y1}(x∖k), and for the top-2 class, d₂(k), where k indexes segments.
A large d₁(k) means segment k is important for the top-1 decision; d₂(k) likewise for top-2.

**Spatial vs featural regime.** To decide whether the two classes are separated by *where*
the model looks (different regions) or by *what* it reads from a shared region (same region,
different features), we define a per-image regime ratio

  ρ = Σ_k (d₁(k) − d₂(k))² / Σ_k (d₁(k) + d₂(k))².

When the two classes rely on the *same* segments (d₁ ≈ d₂), the numerator collapses and ρ is
small — the **featural** regime. When they rely on *different* segments (segments important
for one class are unimportant for the other), d₁ − d₂ is large and ρ is high — the **spatial**
regime. The discriminative region R used elsewhere in the pipeline is the top-25% of segments
by |d₁ − d₂|.

**Null control and threshold.** ρ is scale-dependent, so we calibrate it against a random-pair
control: for each image we draw two unrelated classes and recompute ρ from their occlusion
deltas. We classify an image as spatial if its confusable-pair ρ exceeds the control
threshold τ = mean(ρ_random) + std(ρ_random), and featural otherwise, and we test the overall
shift with a Wilcoxon signed-rank test of confusable vs random ρ.

**Category breakdown.** To see whether the regime depends on image content, we map each
image's top-1 class to a high-level semantic category (dog, bird, snake, reptile, fish,
arthropod, food, vehicle, …) via the WordNet hypernym hierarchy, and report the spatial/
featural split per category.

## Results

Confused pairs are overwhelmingly separated by *region*, not by shared-region features. Over
n = 1000 images, 935 (93.5%) fall in the spatial regime and 65 (6.5%) in the featural regime.
The confusable regime ratio (mean 1.934, median 1.076) is far above the random-pair control
(mean 0.354, median 0.237; threshold τ = 0.694), and the shift is overwhelmingly significant
(Wilcoxon p ≈ 4·10⁻¹⁵²). The confusable ρ distribution sits almost entirely to the right of
the control, confirming that the discriminative region the metric gates to is a real,
class-separating structure rather than a random selection.

The regime is largely content-independent but shows an interpretable gradient. Across the 17
WordNet super-categories present, most are ≥90% spatial — e.g. bird 98% (291/298), amphibian
96%, other-mammal 95%, insect/arthropod 92%, fish 91%. The most featural categories are the
fine-grained species groups, reptile and snake, both at 87% spatial (13% featural): pairs such
as *horned viper / sidewinder* occupy the same region of the image and differ only in surface
pattern, exactly the featural regime the ratio is designed to detect. We note that this pool
is animal-skewed (986/1000 top-1 classes are animals), so the small object categories carry
few images; the per-category table (139 distinct classes; 17 semantic buckets) is the
informative cut rather than the coarse animal/object split.

## Reproduce

- `occlusion_1k.py [n] [pool.pkl]` — segment + occlusion over `pool1000.pkl`; emits the overall
  table (+animal/object split), per-class table, stats figure, and examples. Lightweight
  checkpoint (`cs_viz_cache/occlusion_1k.pkl`).
- `occlusion_1k_category.py` — per-semantic-category table via WordNet + boundary-overlay
  examples (proof the highlight is per-superpixel).
- `occlusion_spatial_featural.py [n]` — same analysis on the smaller cached pool (n=100).
