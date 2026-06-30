"""Emit the ViT gated-CS_struct table from segment_store_vit.pkl (no torch/CUDA -> no segfault)."""
import sys, pickle
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
OUT = Path('cs_viz_outputs')
store = pickle.load(open(OUT/'segment_store_vit.pkl', 'rb'))
PO = ['Vanilla Grad','SmoothGrad','IG-zero','Blur-IG','IDG','Guided IG','ExpGrad',
      'KLIG-Adaptive','KL-IG (linear)','KL-IG²','KL-IG² (adaptive)']
methods = list(store[0]['gated_cs'].keys())
tmethods = [m for m in PO if m in methods] + [m for m in methods if m not in PO]
def _mse(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    return (v.mean(), v.std()/np.sqrt(len(v))) if len(v) else (np.nan, np.nan)
trows = [dict(method=m, gated_CSstruct=_mse([R['gated_cs'][m] for R in store])[0],
             gc_se=_mse([R['gated_cs'][m] for R in store])[1]) for m in tmethods]
tdf = pd.DataFrame(trows).sort_values('gated_CSstruct', ascending=False).reset_index(drop=True)
tdf.to_csv(OUT/'segment_metric_table_vit.csv', index=False)
print(f'\n=== ViT-B/16 gated CS_struct (n={len(store)}) ===')
for r in tdf.itertuples(): print(f'{r.method:20s} {r.gated_CSstruct:.3f} ± {r.gc_se:.3f}')

o = np.argsort(-tdf.gated_CSstruct.values); i1, i2 = int(o[0]), int(o[1])
fig, ax = plt.subplots(figsize=(7, 0.5*len(tdf)+1.1), facecolor='white'); ax.axis('off')
cells = [[r.method, f'{r.gated_CSstruct:.3f} ± {r.gc_se:.3f}'] for r in tdf.itertuples()]
tb = ax.table(cellText=cells, colLabels=['method', 'gated CS_struct (ViT-B/16)'], cellLoc='center', loc='center')
tb.auto_set_font_size(False); tb.set_fontsize(10); tb.scale(1, 1.5)
for j in range(2): tb[0, j].set_facecolor('#34495e'); tb[0, j].set_text_props(color='white', fontweight='bold')
tb[(i1+1, 1)].set_text_props(fontweight='bold'); tb[(i1+1, 1)].set_facecolor('#cfe3f5')
tb[(i2+1, 1)].set_text_props(fontstyle='italic'); tb[(i2+1, 1)].set_facecolor('#eaf3fb')
plt.title(f'ViT-B/16 — gated CS_struct (n={len(store)}, ±SE)\nbold=highest  italic=2nd', fontsize=10, fontweight='bold', pad=10)
plt.tight_layout(); plt.savefig(OUT/'segment_metric_table_vit.png', dpi=200, bbox_inches='tight'); plt.close()
print(f'saved {OUT}/segment_metric_table_vit.png  +  .csv')
