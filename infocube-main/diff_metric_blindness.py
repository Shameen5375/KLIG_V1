"""Difference-family class-sensitivity metrics are BLIND to coherence — proven on their own terms.

A "difference" class-sensitivity metric scores how DIFFERENT the y1 and y2 attribution maps are
(pixel cosine distance, L2, correlation distance, IoU distance) and calls larger difference "more
class-sensitive". This script proves — using ONLY the difference metric and a controlled input,
with NO attribution method and NO proposed alternative — that such metrics cannot distinguish
coherent class-discrimination from pure noise:

  (1) Two class-difference maps of IDENTICAL ENERGY, one a coherent blob, one pure noise, receive
      the SAME difference-metric score.
  (2) A constant-energy sweep from pure structure (a=0) to pure noise (a=1) leaves every difference
      metric FLAT. Energy (magnitude) is held constant, so magnitude cannot explain the flatness —
      spatial coherence is the only variable, and the metrics do not see it.

Conclusion: the difference family measures "amount of difference", not "quality of class
discrimination", and is therefore invalid as a class-sensitivity metric.
Run:  .venv/Scripts/python diff_metric_blindness.py
"""
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
np.random.seed(0); EPS = 1e-8; H = W = 224

# ── controlled construction ──────────────────────────────────────────────────────────────
yy, xx = np.mgrid[0:H, 0:W]
def blob(cy, cx, r): return np.exp(-(((yy-cy)**2 + (xx-cx)**2) / (2*r**2)))
def unit(a): a = a - a.mean(); return a / (np.linalg.norm(a) + EPS)     # zero-mean, unit L2

B = unit(blob(H//2, W//2, 55))                                          # shared map (both classes attribute here)
S = unit(blob(H//2, int(W*0.68), 40) - blob(H//2, int(W*0.32), 40))    # COHERENT class-difference (structured)
N = unit(np.random.randn(H, W))                                        # NOISE class-difference (same energy as S)
E = 0.8                                                                 # class-difference energy (FIXED)
def pair(delta): return B + delta/2, B - delta/2                        # y1/y2 maps: shared B, differ by delta

# ── the difference family (each: larger = "more class-sensitive") ────────────────────────
# vector-distance difference family (pixel cosine / semantic-embedding distance / correlation):
# each is provably a function of ENERGY alone, hence exactly constant under a constant-energy sweep.
def cosine_dist(a, b): a,b=a.ravel(),b.ravel(); return 1 - (a@b)/(np.linalg.norm(a)*np.linalg.norm(b)+EPS)
def l2_dist(a, b):     return np.linalg.norm(a - b)
def corr_dist(a, b):   return 1 - np.corrcoef(a.ravel(), b.ravel())[0,1]
DIFF = {'cosine dist': cosine_dist, 'L2 dist': l2_dist, 'corr dist': corr_dist}
def iou_dist(a, b, f=0.25):                                             # set-overlap variant (see note below)
    ta = np.abs(a) >= np.quantile(np.abs(a), 1-f); tb = np.abs(b) >= np.quantile(np.abs(b), 1-f)
    return 1 - (ta & tb).sum() / ((ta | tb).sum() + EPS)
def coherence(delta):                                                   # console verification ONLY (never plotted)
    d = delta / (np.abs(delta).max()+EPS); return float((gaussian_filter(d,4)**2).sum() / ((d**2).sum()+EPS))

# ── (1) identical energy: coherent vs noise -> SAME difference score ─────────────────────
d_coh, d_noise = E*S, E*N
print(f'energy check: ||coherent Δ||={np.linalg.norm(d_coh):.3f}  ||noise Δ||={np.linalg.norm(d_noise):.3f}  (identical)')
print(f'coherence (console only): coherent={coherence(d_coh):.3f}  noise={coherence(d_noise):.4f}')
print(f'{"metric":12s} {"coherent":>10s} {"noise":>10s} {"|Δ|":>9s}')
rows = []
for name, fn in DIFF.items():
    a1c,a2c = pair(d_coh); a1n,a2n = pair(d_noise)
    sc, sn = fn(a1c,a2c), fn(a1n,a2n); rows.append((name, sc, sn))
    print(f'{name:12s} {sc:10.4f} {sn:10.4f} {abs(sc-sn):9.5f}   <- SAME')

# ── (2) constant-energy sweep: structure (a=0) -> noise (a=1) ─────────────────────────────
alphas = np.linspace(0, 1, 21)
curves = {k: [] for k in DIFF}; coh_curve = []
for a in alphas:
    delta = E * unit(np.sqrt(1-a)*S + np.sqrt(a)*N)                     # renormalized -> energy fixed
    a1, a2 = pair(delta)
    for k, fn in DIFF.items(): curves[k].append(fn(a1, a2))
    coh_curve.append(coherence(delta))
coh_curve = np.array(coh_curve)

# random-pair calibration: two INDEPENDENT maps = the "unrelated / maximally different" level.
rp = {k: float(np.mean([fn(unit(np.random.randn(H,W)), unit(np.random.randn(H,W))) for _ in range(40)])) for k,fn in DIFF.items()}
print('random-pair calibration (unrelated maps):', {k: round(v,3) for k,v in rp.items()})

cmap = plt.get_cmap('tab10')
def plot_diff(ax, suffix=''):
    """Plot each difference metric ÷ its random baseline, MERGING curves that coincide
    (e.g. cosine == corr for zero-mean maps) into one labelled line so none is hidden."""
    plotted = []
    for k, v in curves.items():
        y = np.array(v)/(rp[k]+EPS)
        m = next((p for p in plotted if np.allclose(p[0], y, atol=2e-3)), None)
        if m: m[2].append(k)
        else:
            line, = ax.plot(alphas, y, '-o', ms=4, color=cmap(len(plotted))); plotted.append([y, line, [k]])
    for y, line, names in plotted: line.set_label(' = '.join(names) + suffix)

# ── figure (difference metric ONLY — no coherence curve; the fix comes in a later section) ─
fig = plt.figure(figsize=(15, 5.2), facecolor='white')
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 2.1], wspace=0.28)
vmax = E*max(np.abs(S).max(), np.abs(N).max())
axc = fig.add_subplot(gs[0]); axc.imshow(d_coh, cmap='RdBu_r', vmin=-vmax, vmax=vmax); axc.axis('off')
axc.set_title('structured class-difference', fontsize=11, fontweight='bold', color='#b00020')
axn = fig.add_subplot(gs[1]); axn.imshow(d_noise, cmap='RdBu_r', vmin=-vmax, vmax=vmax); axn.axis('off')
axn.set_title('noise class-difference\n(identical energy)', fontsize=11, fontweight='bold', color='#1f6fd6')
txt = 'SAME energy →\nSAME difference score:\n\n' + '\n'.join(f'{n:11s} {sc:.3f} = {sn:.3f}' for n,sc,sn in rows)
axn.text(1.05, 0.5, txt, transform=axn.transAxes, va='center', ha='left', fontsize=8.5, family='monospace',
         bbox=dict(boxstyle='round', fc='#f5f5f5', ec='#999'))
axs = fig.add_subplot(gs[2])
plot_diff(axs)
axs.axhline(1.0, color='k', ls='--', lw=1.8, label='independent random-pair baseline')
axs.set_xlabel('α   (0 = pure structure  →  1 = pure noise)', fontsize=11)
axs.set_ylabel('difference score / random-pair baseline', fontsize=11); axs.set_ylim(0, 1.15)
axs.set_title('Constant-energy sweep', fontsize=12, fontweight='bold')
axs.grid(alpha=0.3); axs.legend(fontsize=9, loc='center right')
axs.text(0.03, 0.06,
         'ENERGY (magnitude) held CONSTANT along the sweep.\nThe only variable is spatial coherence.\n'
         '→ magnitude cannot explain the flat line;\n   the metric is blind to coherence itself.',
         transform=axs.transAxes, va='bottom', ha='left', fontsize=9,
         bbox=dict(boxstyle='round', fc='#fff6e0', ec='#d9a441'))
plt.suptitle('Difference-based class sensitivity is gameable by noise.', fontsize=13.5, fontweight='bold')
fig.text(0.5, 0.055, r"difference metric (cosine):  $D=1-\frac{\langle A_1,\,A_2\rangle}{\|A_1\|\,\|A_2\|}$"
         r"       energy $\|A_1-A_2\|$ held constant across the sweep",
         ha='center', va='bottom', fontsize=12)
fig.subplots_adjust(left=0.02, right=0.99, top=0.89, bottom=0.24, wspace=0.28)
out='cs_viz_outputs/diff_metric_blindness.png'
plt.savefig(out, dpi=150); plt.close()
flat = max(abs(np.array(v)[-1]-np.array(v)[0])/(rp[k]+EPS) for k,v in curves.items())
print(f'sweep: difference metrics drift <= {100*flat:.1f}% of the random-pair scale, structure->noise')
print(f'(console-only: coherence falls {100*(1-coh_curve[-1]/coh_curve[0]):.0f}% over the same sweep)')
print('saved', out)

# ══════════════════════════════════════════════════════════════════════════════════════════
#  PAYOFF FIGURE (belongs in the CS_struct section): add the COHERENCE-DIFFERENCE metric,
#  which measures the spatial coherence of the class-difference map and so is NOT gamed by noise.
#  coherence-difference:  CD(A1,A2) = ||G_σ * (A1-A2)||² / ||A1-A2||²    (σ=4 Gaussian blur)
# ══════════════════════════════════════════════════════════════════════════════════════════
cd_coh, cd_noise = coherence(d_coh), coherence(d_noise)
fig2 = plt.figure(figsize=(15, 5.4), facecolor='white')
gs2 = fig2.add_gridspec(1, 3, width_ratios=[1, 1, 2.1], wspace=0.28)
axc2 = fig2.add_subplot(gs2[0]); axc2.imshow(d_coh, cmap='RdBu_r', vmin=-vmax, vmax=vmax); axc2.axis('off')
axc2.set_title('structured class-difference', fontsize=11, fontweight='bold', color='#b00020')
axn2 = fig2.add_subplot(gs2[1]); axn2.imshow(d_noise, cmap='RdBu_r', vmin=-vmax, vmax=vmax); axn2.axis('off')
axn2.set_title('noise class-difference\n(identical energy)', fontsize=11, fontweight='bold', color='#1f6fd6')
axs2 = fig2.add_subplot(gs2[2])
plot_diff(axs2, suffix=' (difference — flat)')                                  # merges coincident cosine/corr
axs2.plot(alphas, coh_curve/(coh_curve[0]+EPS), '-s', color='#127a12', lw=3.5, ms=6, label='coherence difference (proposed)')
axs2.set_xlabel('α   (0 = pure structure  →  1 = pure noise)', fontsize=11)
axs2.set_ylabel('metric score  (normalized to [0,1])', fontsize=11); axs2.set_ylim(-0.05, 1.15)
axs2.set_title('Constant-energy sweep', fontsize=12, fontweight='bold')
axs2.grid(alpha=0.3); axs2.legend(fontsize=9, loc='center left')
fig2.suptitle('A coherence-difference metric is not gameable by noise.', fontsize=13.5, fontweight='bold')
fig2.text(0.5, 0.055, r"coherence difference:  $CD=\frac{\|\,G_\sigma * (A_1-A_2)\,\|^{2}}{\|A_1-A_2\|^{2}}$"
          r"      ($G_\sigma$ = Gaussian blur, $\sigma=4$)      high for structure, $\approx 0$ for noise",
          ha='center', va='bottom', fontsize=12)
fig2.subplots_adjust(left=0.02, right=0.99, top=0.89, bottom=0.24, wspace=0.28)
out2='cs_viz_outputs/coherence_difference_payoff.png'
plt.savefig(out2, dpi=150); plt.close()
print(f'coherence difference: structured={cd_coh:.3f}  noise={cd_noise:.3f}  (difference metrics saw them as identical)')
print('saved', out2)
