"""Annotated explainer: the texture-vs-shape numbers WITH their meaning on the figure (n=50)."""
import sys
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
OUT = Path('cs_viz_outputs')
cols = {'ResNet50': '#2c7fb8', 'ViT-B/16': '#d95f0e'}
CUES = ['texture', 'edge', 'shape']
# Approach-1 means/SE (inR, outR)
A1 = {'ResNet50': {'in':[0.00,0.09,0.08],'se':[0.03,0.03,0.04],'out':[0.00,0.08,-0.00]},
      'ViT-B/16': {'in':[0.08,0.22,0.28],'se':[0.04,0.06,0.04],'out':[0.00,0.05,0.04]}}
SAN1 = ['43%','5%','218%']
# Approach-2 shape-bias
SIG = [2,3,4]; SB = {'ResNet50':[0.62,0.12,0.00],'ViT-B/16':[1.00,1.00,0.50]}
CNT = {'ResNet50':['8/5','1/7','0/16'],'ViT-B/16':['16/0','11/0','5/5']}

fig = plt.figure(figsize=(17, 8.4), facecolor='white')
gsP = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.22, left=0.05, right=0.98, top=0.84, bottom=0.10)

# ── Panel A: Approach 1 ──
ax = fig.add_subplot(gsP[0, 0]); xc = np.arange(3); w = 0.2
for j, mn in enumerate(A1):
    ax.bar(xc+(2*j-1.5)*w, A1[mn]['in'], w, yerr=A1[mn]['se'], color=cols[mn], capsize=3, label=f'{mn} (inside R)')
    ax.bar(xc+(2*j-0.5)*w, A1[mn]['out'], w, color=cols[mn], alpha=0.32, hatch='///', label=f'{mn} (outside R, control)')
ax.set_xticks(xc); ax.set_xticklabels([c.upper() for c in CUES], fontsize=11)
ax.set_ylabel('margin drop  Δ[p(y1)−p(y2)]  when cue removed'); ax.axhline(0, color='k', lw=0.6)
ax.set_ylim(-0.05, 0.40); ax.legend(fontsize=8, loc='upper left')
ax.set_title('APPROACH 1 — cue ablation inside region R  (n=50)', fontsize=11, fontweight='bold')
# annotations explaining the numbers
ax.annotate('ViT shape: 0.28 inside R vs 0.04 outside\n→ 0.24 R-specific (control passes:\nsolid ≫ hatched)',
            xy=(2-0.25, 0.28), xytext=(0.95, 0.345), fontsize=8.5,
            bbox=dict(boxstyle='round', fc='#fff3e0', ec='#d95f0e'),
            arrowprops=dict(arrowstyle='->', color='#d95f0e'))
ax.annotate('ResNet edge: inside ≈ outside (0.09 ≈ 0.08)\n→ R-specific ~0 = GLOBAL blur fragility,\nNOT what R carries (control catches it)',
            xy=(1-0.15, 0.085), xytext=(0.15, 0.20), fontsize=8.5,
            bbox=dict(boxstyle='round', fc='#e3f0fa', ec='#2c7fb8'),
            arrowprops=dict(arrowstyle='->', color='#2c7fb8'))
ax.annotate('ResNet texture ≈ 0\n(texture-blind in R)', xy=(0-0.15, 0.0), xytext=(-0.45, 0.13), fontsize=8.5,
            bbox=dict(boxstyle='round', fc='#e3f0fa', ec='#2c7fb8'), arrowprops=dict(arrowstyle='->', color='#2c7fb8'))
ax.text(0.5, -0.115, 'bar = avg drop in the y1-vs-y2 probability gap when that cue is removed.  '
        'solid = inside R, hatched = equal-area control outside R.  R-specific = solid − hatched.',
        transform=ax.transAxes, ha='center', fontsize=8, style='italic')
ax.text(0.5, -0.155, '⚠ magnitudes INFLATED by cue leakage — shape-shuffle ADDS edges (sanity: edge-energy kept '
        f'tex={SAN1[0]}, edge={SAN1[1]}, shape={SAN1[2]}); suggestive, not clean.',
        transform=ax.transAxes, ha='center', fontsize=8, color='#b00020')

# ── Panel B: Approach 2 ──
ax2 = fig.add_subplot(gsP[0, 1])
for mn in SB:
    ax2.plot(SIG, SB[mn], 'o-', color=cols[mn], lw=2.4, ms=8, label=mn)
    for s, v, c in zip(SIG, SB[mn], CNT[mn]): ax2.annotate(f'{v:.2f}\n({c})', (s, v), textcoords='offset points',
        xytext=(0, 10 if mn=='ViT-B/16' else -22), ha='center', fontsize=8, color=cols[mn], fontweight='bold')
ax2.axhline(0.5, ls=':', color='gray'); ax2.text(4.02, 0.5, 'chance', fontsize=8, color='gray', va='center')
ax2.set_ylim(-0.05, 1.12); ax2.set_xticks(SIG); ax2.set_xlabel('low/high frequency split  σ (px)')
ax2.set_ylabel('shape-bias  =  #shape / (#shape + #texture)')
ax2.set_title('APPROACH 2 — cue conflict (shape vs texture)  (n=50)', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9, loc='center left')
ax2.annotate('ViT = 1.00: of all decided trials,\nit ALWAYS picked the shape class\n(16/0, 11/0) — never texture',
             xy=(3, 1.0), xytext=(2.25, 0.66), fontsize=8.5,
             bbox=dict(boxstyle='round', fc='#fff3e0', ec='#d95f0e'), arrowprops=dict(arrowstyle='->', color='#d95f0e'))
ax2.annotate('ResNet decays 0.62→0.12→0.00:\nflips to TEXTURE as high-freq grows\n(0/16 = all texture)',
             xy=(4, 0.0), xytext=(2.3, 0.16), fontsize=8.5,
             bbox=dict(boxstyle='round', fc='#e3f0fa', ec='#2c7fb8'), arrowprops=dict(arrowstyle='->', color='#2c7fb8'))
ax2.text(0.5, -0.135, '1 = always shape, 0 = always texture, 0.5 = chance.  (shape/texture) = #decisions each.  '
         'valid only σ≤3 (shape stays recognizable).',
         transform=ax2.transAxes, ha='center', fontsize=8, style='italic')
ax2.text(0.5, -0.175, '✓ decision-readout (no ablation leakage), patch-agnostic (fair to ViT).  '
         '⚠ small decided-n (8–16 of 50); frequency ≈ shape/texture is a proxy.',
         transform=ax2.transAxes, ha='center', fontsize=8, color='#444')

fig.suptitle('Texture vs Shape — numbers explained   (ViT weights SHAPE · ResNet leans TEXTURE / texture-blind in R)',
             fontsize=13, fontweight='bold')
plt.savefig(OUT/'feature_type_explained.png', dpi=170, bbox_inches='tight'); plt.close()
print(f'saved {OUT}/feature_type_explained.png')
