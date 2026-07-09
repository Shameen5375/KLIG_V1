# Paper figures — clear names (updated 2026-07-09)

Naming: `exp1_*` = augmentation consistency · `exp2_*` = cue ablation (texture/edge/shape) ·
`gameability_*` = difference-metric problem/solution · `validation_*` = CS_struct / R validation.
`*_TABLE` = the table; `*_bars`/`*_panels` = the chart version.

## exp1 — Augmentation consistency (CS_struct drift; n=100)
| file | contents | source script |
|---|---|---|
| `exp1_augment_consistency_TABLE.png` | Table 1 per-transform drift + class-swap control (1.6×, p=8.4e-3); Table 2 leaderboard base-vs-flip (ρ=+1.000) | `augment_table.py` |
| `exp1_augment_stability_panels.png` | 2-panel stability scatter + drift boxplot | `augment_consistency.py` |
| `exp1_augment_example_image.png` | one image across transforms, top-1/top-2 preds per row | `augment_example.py` |

## exp2 — Cue ablation (class-differential collapse; smooth warp, no tile-shuffle)
| file | contents | source script |
|---|---|---|
| `exp2_cue_ablation_TABLE.png` / `.csv` | R-specific collapse per cue × arch (n=100; n=500 pending) | `cue_class_sens.py` |
| `exp2_cue_ablation_bars.png` | bar version | `cue_class_sens.py` |
| `exp2_cue_probe_TABLE.png` | margin-drop table, inside R vs outside-R control (n=50) | `feature_type_probe.py` |
| `exp2_cue_probe_bars.png` | bar version of the probe | `feature_type_probe.py` |
| `exp2_cue_removal_examples.png` | what each cue removal looks like (4 images) | `feature_type_viz.py` |
| `exp2_cue_removal_grid.png` | cue removals × images grid | `feature_type_grid.py` |

## gameability — problem → solution
| file | contents | source script |
|---|---|---|
| `gameability_diff_metrics_flat.png` | difference metrics flat under constant-energy structure→noise (gameable by noise) | `diff_metric_blindness.py` |
| `gameability_coherence_payoff.png` | coherence difference collapses on noise (not gameable) | `diff_metric_blindness.py` |

## validation — CS_struct and R
| file | contents | source script |
|---|---|---|
| `validation_V2_rank_recovery.png` | recovers planted coherence ranking, ρ = 1.000 ± 0.000 | `cs_synthetic_validation.py` |
| `validation_V3_agreement_divergence.png` | agrees with region-hit (r=0.99); diverges on noise-in-R | `cs_synthetic_validation.py` |
| `validation_V2V3_combined.png` | combined 3-panel version | `cs_synthetic_validation.py` |
| `validation_R_equivariance.png` | R moves with the image under flip/rotate, above chance (n=30) | `r_equivariance.py` |

## Pending (scheduled run)
- `leaderboard_csstruct_TABLE.png` / `leaderboard_csstruct_bars.png` — all-11-method gated CS_struct ranking, n=1000 (`cs_struct_leaderboard.py`)
- `exp2_cue_ablation_TABLE.png` refresh at n=500
