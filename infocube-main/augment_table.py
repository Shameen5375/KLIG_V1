"""Detailed tables for the augmentation-consistency check (from the cached 60-image run):
  Table 1 — per-transform consistency (mean CS, abs drift ±SE, pred-kept%) + class-swap control
  Table 2 — method leaderboard, CS_struct baseline vs flipped, with ranks (ordering preserved)
Run:  .venv/Scripts/python augment_table.py
"""
import sys, pickle, warnings
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon
warnings.filterwarnings('ignore')
EPS = 1e-8; HEAD = 'KL-IG² (adaptive)'
rows = pickle.load(open('cs_viz_cache/augment_consistency.pkl','rb'))
summ = pickle.load(open('cs_viz_outputs/augment_consistency_summary.pkl','rb'))
methods = summ['methods']; mb, mf = np.array(summ['mb']), np.array(summ['mf'])
n = len(rows); TR = list(rows[0]['transforms'])
def se(v): v=np.asarray(v,float); return v.std()/np.sqrt(len(v))

# ── Table 1: per-transform ───────────────────────────────────────────────────────────────
cs0 = np.array([r['cs0'] for r in rows])
t1 = []
for name in TR:
    cst = np.array([r['transforms'][name]['cs'] for r in rows])
    ad  = np.array([abs(r['transforms'][name]['cs']-r['cs0']) for r in rows])
    pk  = np.mean([r['transforms'][name]['pred_preserved'] for r in rows])
    t1.append([name, f'{cst.mean():.3f} ± {se(cst):.3f}', f'{ad.mean():.3f} ± {se(ad):.3f}', f'{100*pk:.0f}%'])
ctrl = np.array([abs(r['control_swap']-r['cs0']) for r in rows])
pres_pool = np.array([np.mean([abs(r['transforms'][k]['cs']-r['cs0'])
                     for k in TR if r['transforms'][k]['pred_preserved']] or
                    [abs(r['transforms'][k]['cs']-r['cs0']) for k in TR]) for r in rows])
pv = wilcoxon(pres_pool, ctrl).pvalue
t1.append(['— baseline —', f'{cs0.mean():.3f} ± {se(cs0):.3f}', '—', '—'])
t1.append(['CTRL: class-swap', '—', f'{ctrl.mean():.3f} ± {se(ctrl):.3f}', 'label-CHANGING'])

# ── Table 2: method leaderboard base vs flip ─────────────────────────────────────────────
rb = (-mb).argsort().argsort()+1; rf = (-mf).argsort().argsort()+1        # 1-based ranks
order = (-mb).argsort()
rho = spearmanr(mb, mf).correlation
t2 = [[methods[i], f'{mb[i]:.3f}', f'{mf[i]:.3f}', f'{rb[i]}', f'{rf[i]}'] for i in order]

# ── render ───────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 0.42*(len(t2)+len(t1))+2.4), facecolor='white')
gs = fig.add_gridspec(2, 1, height_ratios=[len(t1)+1.5, len(t2)+1.5], hspace=0.18)
ax1 = fig.add_subplot(gs[0]); ax1.axis('off')
tb1 = ax1.table(cellText=t1, colLabels=['transform','mean CS_struct','abs. drift |ΔCS|','pred kept'],
                cellLoc='center', loc='center')
tb1.auto_set_font_size(False); tb1.set_fontsize(10); tb1.scale(1,1.55)
for j in range(4): tb1[0,j].set_facecolor('#34495e'); tb1[0,j].set_text_props(color='white',fontweight='bold')
for i in range(len(t1)):
    lab=t1[i][0]
    if lab.startswith('CTRL'):
        for j in range(4): tb1[i+1,j].set_facecolor('#f6d5d5')
    elif lab.startswith('—'):
        for j in range(4): tb1[i+1,j].set_facecolor('#eeeeee')
ax1.set_title(f'Table 1 — per-transform consistency of gated CS_struct  (n={n}, HEAD={HEAD})\n'
              f'label-preserving drift {pres_pool.mean():.3f}  vs  class-swap {ctrl.mean():.3f}  '
              f'({ctrl.mean()/(pres_pool.mean()+EPS):.1f}×, paired Wilcoxon p={pv:.1e})',
              fontsize=10.5, fontweight='bold', pad=10)

ax2 = fig.add_subplot(gs[1]); ax2.axis('off')
tb2 = ax2.table(cellText=t2, colLabels=['method','CS base','CS flip','rank base','rank flip'],
                cellLoc='center', loc='center')
tb2.auto_set_font_size(False); tb2.set_fontsize(9.5); tb2.scale(1,1.5)
for j in range(5): tb2[0,j].set_facecolor('#34495e'); tb2[0,j].set_text_props(color='white',fontweight='bold')
for i,mi in enumerate(order):
    if methods[mi]==HEAD:
        for j in range(5): tb2[i+1,j].set_facecolor('#cfe3f5'); tb2[i+1,j].set_text_props(fontweight='bold')
ax2.set_title(f'Table 2 — method leaderboard under horizontal flip  (Spearman rank ρ = {rho:+.3f}, ordering preserved)\n'
              'blue = headline method (stays #1)', fontsize=10.5, fontweight='bold', pad=10)
out='cs_viz_outputs/augment_table.png'
plt.savefig(out, dpi=170, bbox_inches='tight'); plt.close()
print(f'saved {out}  | preserving {pres_pool.mean():.3f} vs ctrl {ctrl.mean():.3f} p={pv:.2e} | rho={rho:+.3f}')
