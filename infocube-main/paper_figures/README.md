# Paper figures — current versions (2026-07-09)

## Gameability (problem → solution)
- `diff_metric_blindness.png` — difference-based class sensitivity is gameable by noise (flat under constant-energy structure→noise sweep)
- `coherence_difference_payoff.png` — coherence difference collapses on noise where difference metrics stay flat

## Synthetic validation of CS_struct
- `cs_v2_rank_recovery.png` — recovers planted coherence ranking, Spearman ρ = 1.000 ± 0.000 (300 draws)
- `cs_v3_agreement_divergence.png` — agrees with region-hit (r=0.99); diverges on noise-in-R where hit is fooled
- `cs_synthetic_validation.png` — combined 3-panel version

## Experiment 1 — augmentation consistency (n=100)
- `augment_table.png` — Table 1 per-transform drift + class-swap control (1.6×, p=8.4e-3); Table 2 leaderboard base-vs-flip (ρ=+1.000)
- `augment_consistency.png` — 2-panel stability/sensitivity figure
- `augment_example.png` — quilt vs handkerchief across transforms, top-1/top-2 preds per row

## Experiment 2 — cue ablation (texture/edge/shape; smooth-warp shape, no tile-shuffle)
- `cue_class_sens_table.png` (+ `.csv`) — R-specific collapse of class differential, ResNet vs ViT (n=100; n=500 pending)
- `cue_class_sens.png` — grouped bar version
- `feature_type_probe.png` — margin-drop probe inside vs outside R (n=50, fixed shape op)
- `feature_type_viz.png` — cue-removal examples (SHAPE = smooth warp)

## R validation
- `r_equivariance.png` — R moves with the image under flip/rotate, above unrelated-image chance (n=30)

Pending (from scheduled run): CS_struct leaderboard n=1000 (`cs_leaderboard_balanced_table.png`/`_bar.png`), cue table at n=500.
