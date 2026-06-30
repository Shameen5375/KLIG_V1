"""Consolidated texture-vs-shape summary table (Approach 1 + Approach 2), n=50, ResNet50 vs ViT-B/16."""
import sys
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
OUT = Path('cs_viz_outputs')

# ── Approach 1: cue ablation inside discriminative region R (absolute y1-vs-y2 margin drop, n=50) ──
# (inside-R mean ± SE, outside-R control mean, R-specific = inside - outside)
A1 = [
    # cue,        RN_in, RN_se, RN_out,  VT_in, VT_se, VT_out
    ('Texture',   0.00, 0.03, 0.00,   0.08, 0.04, 0.00),
    ('Edge',      0.09, 0.03, 0.08,   0.22, 0.06, 0.05),
    ('Shape',     0.08, 0.04, -0.00,  0.28, 0.04, 0.04),
]
# ── Approach 2: cue-conflict shape-bias = #shape/(#shape+texture), valid regime ──
A2 = [
    # sigma, RN_sb, RN_counts,   VT_sb, VT_counts
    (2, 0.62, '8/5',   1.00, '16/0'),
    (3, 0.12, '1/7',   1.00, '11/0'),
    (4, 0.00, '0/16',  0.50, '5/5'),
]
SAN1 = {'Texture': '43%', 'Edge': '5%', 'Shape': '218%'}     # grad-energy retained (cue-removal sanity)
SAN2 = {'ResNet50': '25/50', 'ViT-B/16': '30/50'}           # lowpass-shape recovered (validity)

# CSV
rows = []
for c, ri, rse, ro, vi, vse, vo in A1:
    rows.append(dict(approach='A1_cue_ablation', metric=f'{c}_margin_drop',
                     ResNet50_inR=ri, ResNet50_outR=ro, ResNet50_Rspecific=round(ri-ro,2),
                     ViT_inR=vi, ViT_outR=vo, ViT_Rspecific=round(vi-vo,2)))
for s, rsb, rc, vsb, vc in A2:
    rows.append(dict(approach='A2_cue_conflict', metric=f'shape_bias_sigma{s}',
                     ResNet50_inR=rsb, ResNet50_outR='', ResNet50_Rspecific='',
                     ViT_inR=vsb, ViT_outR='', ViT_Rspecific=''))
pd.DataFrame(rows).to_csv(OUT/'texture_shape_summary.csv', index=False)

# ── figure: two stacked tables ──
fig, axes = plt.subplots(2, 1, figsize=(12, 8.6), facecolor='white',
                         gridspec_kw={'height_ratios': [3.4, 3.0]})
def style(tb, ncol, bold_cells):
    tb.auto_set_font_size(False); tb.set_fontsize(10); tb.scale(1, 1.6)
    for j in range(ncol):
        tb[0, j].set_facecolor('#34495e'); tb[0, j].set_text_props(color='white', fontweight='bold')
    for (r, c) in bold_cells:
        tb[r, c].set_text_props(fontweight='bold'); tb[r, c].set_facecolor('#cfe3f5')

# Table 1 — Approach 1
ax = axes[0]; ax.axis('off')
cols1 = ['Cue (removed in R)', 'ResNet50  inR±SE', 'ResNet50  outR', 'ResNet50  R-specific',
         'ViT-B/16  inR±SE', 'ViT-B/16  outR', 'ViT-B/16  R-specific', 'sanity\n(edge-E kept)']
cells1, bold1 = [], []
for r, (c, ri, rse, ro, vi, vse, vo) in enumerate(A1, start=1):
    rsp, vsp = ri-ro, vi-vo
    cells1.append([c, f'{ri:.2f} ± {rse:.2f}', f'{ro:+.2f}', f'{rsp:+.2f}',
                   f'{vi:.2f} ± {vse:.2f}', f'{vo:+.2f}', f'{vsp:+.2f}', SAN1[c]])
    bold1.append((r, 6 if vsp > rsp else 3))           # bold the larger R-specific reliance
tb1 = ax.table(cellText=cells1, colLabels=cols1, cellLoc='center', loc='center')
style(tb1, len(cols1), bold1)
ax.set_title('APPROACH 1 — cue ablation inside the discriminative region R  (absolute y1-vs-y2 margin drop, n=50)\n'
             'R-specific = inside−outside (controls for global fragility).  bold = architecture relying MORE on that cue',
             fontsize=10, fontweight='bold', pad=10)

# Table 2 — Approach 2
ax = axes[1]; ax.axis('off')
cols2 = ['low/high split σ', 'ResNet50 shape-bias', 'ResNet50 (shape/texture)',
         'ViT-B/16 shape-bias', 'ViT-B/16 (shape/texture)', 'validity\n(lowpass→shape)']
cells2, bold2 = [], []
for r, (s, rsb, rc, vsb, vc) in enumerate(A2, start=1):
    val = f"RN {SAN2['ResNet50']} / ViT {SAN2['ViT-B/16']}" if r == 1 else ''
    cells2.append([f'σ = {s}', f'{rsb:.2f}', rc, f'{vsb:.2f}', vc, val])
    bold2.append((r, 3 if vsb > rsb else 1))
tb2 = ax.table(cellText=cells2, colLabels=cols2, cellLoc='center', loc='center')
style(tb2, len(cols2), bold2)
ax.set_title('APPROACH 2 — cue conflict (frequency-hybrid Geirhos)  shape-bias = #shape/(#shape+texture), n=50\n'
             '1 = pure shape, 0 = pure texture.  Valid window σ=2–3 (lowpass keeps shape recognizable).  bold = more shape-biased',
             fontsize=10, fontweight='bold', pad=10)

fig.suptitle('Texture vs Shape — ResNet50 vs ViT-B/16   (TL;DR: ViT relies on shape; ResNet flips to texture / is texture-blind in R)',
             fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT/'texture_shape_summary.png', dpi=190, bbox_inches='tight'); plt.close()

# console print
print('\n=== APPROACH 1: cue ablation in R (margin drop, n=50) ===')
print(f'{"cue":8s} | {"ResNet R-spec":>13s} | {"ViT R-spec":>11s}')
for c, ri, rse, ro, vi, vse, vo in A1:
    print(f'{c:8s} | {ri-ro:>+13.2f} | {vi-vo:>+11.2f}')
print('\n=== APPROACH 2: cue-conflict shape-bias (n=50) ===')
print(f'{"sigma":6s} | {"ResNet":>7s} | {"ViT":>5s}')
for s, rsb, rc, vsb, vc in A2:
    print(f'  {s:3d}  | {rsb:>7.2f} | {vsb:>5.2f}')
print(f'\nsaved {OUT}/texture_shape_summary.png  +  .csv')
