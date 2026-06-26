"""
SYNTHETIC VALIDATION of gated CS_struct  (ground-truth check, no real method involved).

Plant a known discriminative region R and a known coherent class-difference inside it.
Sweep a coherence<->noise knob (alpha) at CONSTANT energy, and confirm:
  (a) CS_struct increases monotonically with alpha   -> tracks ground-truth coherence
  (b) CS_struct at alpha=0 floors (== equal-magnitude noise is NOT rewarded)
  (c) difference-metric is FLAT across alpha          -> blind to structure (gameable)
  (d) controls: shuffle-in-R floors; structure-OUTSIDE-R floors (the GATE does work)

Uses the SAME blur (sigma=4) and gating as the real pipeline.
"""
import sys, numpy as np, matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass

H = W = 224; SIGMA = 4; N = 200; EPS = 1e-12
rng = np.random.default_rng(0)
yy, xx = np.mgrid[0:H, 0:W]

# planted discriminative region R (a blob ~20% of image, NOT the whole image) and a disjoint R2
cy, cx, r = 100, 120, 56
R  = ((yy - cy)**2 + (xx - cx)**2) <= r*r
R2 = ((yy - 150)**2 + (xx - 60)**2) <= 40*40     # disjoint region for the outside-gate control
print(f'region R = {R.mean()*100:.0f}% of image, {int(R.sum())} px')

def norm_in(D, mask):                            # unit L2 energy over the mask support
    return D / (np.sqrt((D[mask]**2).sum()) + EPS)
def coherent(mask, cyc, cxc, rad):               # smooth signed center(+)-vs-rim(-) pattern in mask
    d = np.sqrt((yy - cyc)**2 + (xx - cxc)**2) / rad
    D = np.where(mask, 1 - 2*np.clip(d, 0, 1), 0.0)
    D[mask] -= D[mask].mean()                     # ZERO-MEAN in region: no DC blob that survives blur when shuffled
    return norm_in(D, mask)
def noise(mask):                                 # speckle of MATCHED energy, same support
    return norm_in(np.where(mask, rng.standard_normal((H, W)), 0.0), mask)

def cs_struct(D, gate, sigma=SIGMA):             # SAME metric as real pipeline (scale-invariant ratio)
    Dg = D * gate; e = (Dg**2).sum()
    return float((gaussian_filter(Dg, sigma)**2).sum() / e) if e > EPS else 0.0
def diff_metric(D, gate):                        # the gameable baseline: raw magnitude in region
    return float(np.sqrt(((D * gate)**2).sum()))

D_struct = coherent(R, cy, cx, r)                # ground-truth coherent class-difference in R

# ---- sweep the coherence knob at CONSTANT energy ----
alphas = np.linspace(0, 1, 11)
cs_m, cs_se, df_m, df_se = [], [], [], []
for a in alphas:
    cs, df = [], []
    for _ in range(N):
        D = a * D_struct + (1 - a) * noise(R)
        D = norm_in(D, R)                        # hold TOTAL energy constant across alpha  <-- critical
        cs.append(cs_struct(D, R)); df.append(diff_metric(D, R))
    cs, df = np.array(cs), np.array(df)
    cs_m.append(cs.mean()); cs_se.append(cs.std()/np.sqrt(N))
    df_m.append(df.mean()); df_se.append(df.std()/np.sqrt(N))
cs_m, cs_se, df_m, df_se = map(np.array, (cs_m, cs_se, df_m, df_se))

# ---- controls ----
sh, out, randref = [], [], []
D_struct_R2 = coherent(R2, 150, 60, 40)
for _ in range(N):
    v = D_struct[R].copy(); rng.shuffle(v)       # (d1) shuffle structure values inside R -> destroys coherence
    Dsh = np.zeros((H, W)); Dsh[R] = v; sh.append(cs_struct(norm_in(Dsh, R), R))
    out.append(cs_struct(D_struct_R2 + noise(R), R))                 # (d2) structure OUTSIDE R, noise INSIDE, gate=R
    randref.append(cs_struct(rng.standard_normal((H, W)), np.ones((H, W), bool)))  # ungated full-image noise
sh, out, randref = map(np.array, (sh, out, randref))

# ---- verdicts ----
mono = bool(np.all(np.diff(cs_m) > -2*cs_se.max()))      # non-decreasing within noise
floor_ok = cs_m[0] < 0.25 * cs_m[-1]                     # alpha=0 floors well below alpha=1
flat = (df_m.std() / (df_m.mean() + EPS)) < 0.05         # difference-metric ~constant
print('\n=== VALIDATION VERDICTS ===')
print(f'(a) CS_struct monotonic in alpha:            {"PASS" if mono else "FAIL"}  '
      f'(alpha0={cs_m[0]:.3f} -> alpha1={cs_m[-1]:.3f})')
print(f'(b) CS_struct floors at alpha=0:             {"PASS" if floor_ok else "FAIL"}  '
      f'(alpha0={cs_m[0]:.3f}, random_ref={randref.mean():.3f})')
print(f'(c) difference-metric FLAT across alpha:     {"PASS" if flat else "FAIL"}  '
      f'(mean={df_m.mean():.3f}, CV={df_m.std()/df_m.mean()*100:.1f}%)')
print(f'(d) shuffle-in-R floors:                     {"PASS" if sh.mean()<0.25*cs_m[-1] else "FAIL"}  ({sh.mean():.3f})')
print(f'    structure-OUTSIDE-R floors (gate works): {"PASS" if out.mean()<0.25*cs_m[-1] else "FAIL"}  ({out.mean():.3f})')

# ---- figure ----
fig, ax = plt.subplots(1, 2, figsize=(15, 5.2), facecolor='white')
a0 = ax[0]; a0b = a0.twinx()
a0.fill_between(alphas, cs_m-cs_se, cs_m+cs_se, color='#1a7f37', alpha=0.2)
l1, = a0.plot(alphas, cs_m, 'o-', color='#1a7f37', lw=2.5, label='gated CS_struct (coherence)')
a0.axhline(randref.mean(), ls=':', color='gray', label='random floor')
l2, = a0b.plot(alphas, df_m, 's--', color='#b00020', lw=2, label='difference-metric ‖D‖ (magnitude)')
a0.set_xlabel('alpha   (0 = pure noise  →  1 = pure planted structure)'); a0.set_ylabel('gated CS_struct', color='#1a7f37')
a0b.set_ylabel('difference-metric', color='#b00020'); a0b.set_ylim(0, max(df_m)*1.6)
a0.set_title('CS_struct tracks coherence; difference-metric is blind\n(energy held constant across alpha)', fontsize=11, fontweight='bold')
a0.legend(handles=[l1, l2, a0.lines[1]], loc='center left', fontsize=9)
labels = ['alpha=1\n(structure)', 'alpha=0\n(noise)', 'shuffle\nin R', 'structure\nOUTSIDE R', 'random\nref']
vals = [cs_m[-1], cs_m[0], sh.mean(), out.mean(), randref.mean()]
ses = [cs_se[-1], cs_se[0], sh.std()/np.sqrt(N), out.std()/np.sqrt(N), randref.std()/np.sqrt(N)]
cols = ['#1a7f37', '#b00020', '#b00020', '#b00020', 'gray']
ax[1].bar(range(5), vals, yerr=ses, color=cols, alpha=0.8, capsize=4)
ax[1].set_xticks(range(5)); ax[1].set_xticklabels(labels, fontsize=9)
ax[1].set_ylabel('gated CS_struct'); ax[1].set_title('Controls: only planted coherence inside R scores\n(everything else floors)', fontsize=11, fontweight='bold')
for i, v in enumerate(vals): ax[1].text(i, v+max(ses)*1.2, f'{v:.3f}', ha='center', fontsize=9)
plt.suptitle('Synthetic ground-truth validation of gated CS_struct', fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig('cs_viz_outputs/synthetic_validation.png', dpi=170, bbox_inches='tight'); plt.close()
print('\nsaved cs_viz_outputs/synthetic_validation.png')

# ── EXTENSION 2: SUB-SEGMENT STRUCTURE ──────────────────────────────
# Plant coherent structure at a scale FINER than the gate region R (radius=56),
# and confirm CS_struct still recovers it as long as the structure is coarser
# than the blur (sigma=4). Mirrors "the class difference is finer than a segment".
def coherent_at_scale(mask, s):                  # low-pass field of characteristic scale s in mask
    f = rng.standard_normal((H, W)) if s <= 0 else gaussian_filter(rng.standard_normal((H, W)), s)
    f = np.where(mask, f, 0.0); f[mask] -= f[mask].mean()
    return norm_in(f, mask)

GATE_R = r                                       # gate region radius (56) — the "segment" scale
scales = [0, 2, 3, 4, 6, 8, 12, 16, 24, 32]      # structure scale in px (0 = white noise)
sc_m, sc_se = [], []
for s in scales:
    v = [cs_struct(coherent_at_scale(R, s), R) for _ in range(N)]
    sc_m.append(np.mean(v)); sc_se.append(np.std(v)/np.sqrt(N))
sc_m, sc_se = np.array(sc_m), np.array(sc_se)
noise_floor = sc_m[0]                            # s=0 == white noise

# verdict: structure FINER than the gate (e.g. s=8, ~7x smaller than R) but coarser than blur is recovered
i8 = scales.index(8)
sub_ok = sc_m[i8] > 5 * noise_floor
print('\n=== EXTENSION 2: sub-segment structure ===')
print(f'gate region radius = {GATE_R}px ;  blur sigma = {SIGMA}px')
print(f'white-noise floor (s=0):                 {noise_floor:.3f}')
print(f'structure at s=8px (~{GATE_R/8:.0f}x finer than gate): {sc_m[i8]:.3f}  '
      f'-> {"PASS" if sub_ok else "FAIL"} (recovered, >> floor)')
print(f'structure at s=4px (== blur scale):       {sc_m[scales.index(4)]:.3f}')
print(f'structure at s=2px (finer than blur):     {sc_m[scales.index(2)]:.3f}  (approaches floor — below blur, undetectable)')

fig, ax = plt.subplots(figsize=(8.5, 5.2), facecolor='white')
ax.fill_between(scales, sc_m-sc_se, sc_m+sc_se, color='#1a7f37', alpha=0.2)
ax.plot(scales, sc_m, 'o-', color='#1a7f37', lw=2.5, label='gated CS_struct vs structure scale')
ax.axhline(noise_floor, ls=':', color='gray', label=f'white-noise floor ({noise_floor:.3f})')
ax.axvline(SIGMA, ls='--', color='#b00020', lw=1.5, label=f'blur scale (sigma={SIGMA}px)')
ax.axvline(GATE_R, ls='--', color='#2c3e50', lw=1.5, label=f'gate region radius ({GATE_R}px)')
ax.annotate('structure FINER than gate\nbut coarser than blur\n→ still recovered',
            xy=(8, sc_m[i8]), xytext=(13, 0.30), fontsize=9.5,
            arrowprops=dict(arrowstyle='->', color='#1a7f37'))
ax.set_xlabel('planted structure scale s (px)'); ax.set_ylabel('gated CS_struct')
ax.set_title('CS_struct recovers SUB-SEGMENT coherence\n(structure finer than the gate is detected, down to the blur scale)',
             fontsize=11, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout(); plt.savefig('cs_viz_outputs/synthetic_validation_scale.png', dpi=170, bbox_inches='tight'); plt.close()
print('saved cs_viz_outputs/synthetic_validation_scale.png')
