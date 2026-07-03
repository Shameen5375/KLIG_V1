"""Builds rd_attribution_viz.ipynb — visualization for R-D Path Attribution (logic lives in rd_attribution.py)."""
import nbformat as nbf
nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def code(s): C.append(nbf.v4.new_code_cell(s))

md("""# R-D Path Attribution — visualization
Attribution as **rate allocation**: score each region by how many bits it must keep for the model to
still predict class *c*. Regions you can noise into oblivion without moving the logit are unimportant;
regions that demand precision are the explanation. Logic in `rd_attribution.py`; this notebook only visualizes.

Flip the knobs in the setup cell (`GRID`, `OPERATOR`, `TAU`, `RANDOM`) and re-run.""")

code("""# ── setup (run once) ──
import importlib, pickle, numpy as np, torch, matplotlib.pyplot as plt
import rd_attribution as RD; importlib.reload(RD)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# knobs
GRID     = 14          # region grid (grid×grid patches)
OPERATOR = 'noise'     # 'noise' (amplitude rate) or 'blur' (spatial rate)
N_MC     = 4           # Monte-Carlo noise draws
TAU      = 0.30        # sufficiency tolerance (fraction of max single-region drop)
RANDOM   = False       # True -> a different image each run

if 'MODEL' not in globals():
    from torchvision.models import resnet50, ResNet50_Weights
    MODEL = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(DEVICE).eval()
    CATS  = ResNet50_Weights.IMAGENET1K_V2.meta['categories']
    _mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1); _std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    _pool = pickle.load(open('cs_viz_outputs/segment_store_vit.pkl','rb'))
    IMGS = [ (( (d['x'].squeeze(0) if d['x'].dim()==4 else d['x']) * _std + _mean).clamp(0,1)) for d in _pool ]
    print(f'loaded {len(IMGS)} images')

idx = int(np.random.default_rng().integers(len(IMGS))) if RANDOM else 0
img01 = IMGS[idx]
cfg = RD.RDConfig(grid=GRID, operator=OPERATOR, n_mc=N_MC, tau=TAU)
res = RD.run_rd_attribution(MODEL, img01, cfg, DEVICE)
print(f'image #{idx}  target = {CATS[res["target_class"]].split(",")[0]}  L0={res["L0"]:.2f}  '
      f'nonzero regions {int((res[\"suff\"]>0).sum())}/{res[\"n_reg\"]}')""")

md("## 1. Score maps — sufficiency (primary, noise-robust) vs sensitivity (baseline)")
code("""im = img01.permute(1,2,0).cpu().numpy()
fig, ax = plt.subplots(1, 3, figsize=(14, 4.8), facecolor='white')
ax[0].imshow(im); ax[0].set_title(f'image: {CATS[res["target_class"]].split(",")[0]}'); ax[0].axis('off')
for a, key, t in [(ax[1],'suff_map','SUFFICIENCY  (required rate = bits a region must keep)'),
                  (ax[2],'sens_map','SENSITIVITY  (area under logit-drop curve — more gameable)')]:
    a.imshow(im); h = a.imshow(res[key], cmap='inferno', alpha=0.6); a.set_title(t, fontsize=10); a.axis('off')
    plt.colorbar(h, ax=a, fraction=0.046)
plt.suptitle('R-D Path Attribution — hot = the model needs this region to stay faithful', fontweight='bold')
plt.tight_layout(); plt.show()""")

md("## 2. Global R(D) curve — do the per-region scores compose into a sensible information budget?\n"
   "Protect high-score regions, dump max noise on the rest, sweep the budget. The **importance** ordering "
   "should keep the logit up with far less rate than a **random** ordering.")
code("""rd = res['rd_curve']; b = rd['budget']
fig, ax = plt.subplots(figsize=(7.5, 5), facecolor='white')
ax.plot(b, rd['dist_importance'], 'o-', color='#1a7f37', lw=2.4, label='protect by importance (ours)')
ax.plot(b, rd['dist_random'], 's--', color='#b00020', lw=2, label='protect random regions')
half = 0.5*rd['Dmax']; d50 = b[np.argmax(rd['dist_importance']<=half)] if (rd['dist_importance']<=half).any() else 1.0
ax.axhline(half, ls=':', color='gray'); ax.annotate(f'D_50%  (budget≈{d50:.2f})', (d50, half), textcoords='offset points', xytext=(8,8))
ax.set_xlabel('rate budget  (fraction of regions protected)'); ax.set_ylabel('distortion  Δ logit')
ax.set_title(f'Global R(D)  ·  importance AUC {np.trapezoid(rd["dist_importance"],b):.2f} '
             f'< random {np.trapezoid(rd["dist_random"],b):.2f}  = scores compose', fontweight='bold', fontsize=10)
ax.legend(); plt.tight_layout(); plt.show()""")

md("## 3. Per-region R-D curves — the raw distortion-vs-noise curves the scores are read from")
code("""order = np.argsort(-res['suff']); hi, lo = order[0], order[-1]
lv = res['levels']
fig, ax = plt.subplots(figsize=(7.5, 5), facecolor='white')
ax.plot(lv, res['drops'][hi], 'o-', color='#1a7f37', lw=2.4, label=f'most important region (rate={res["suff"][hi]:.2f})')
ax.plot(lv, res['drops'][lo], 's--', color='#2c7fb8', lw=2, label=f'least important region (rate={res["suff"][lo]:.2f})')
thr = res['cfg'].tau * res['drops'].max(); ax.axhline(thr, ls=':', color='gray'); ax.annotate('tolerance τ·Dmax',(lv[0],thr))
ax.set_xlabel('noise level σ  (more noise → less rate)'); ax.set_ylabel('logit drop  L0 − L')
ax.set_title('important region breaks under little noise (steep); unimportant tolerates it (flat)', fontsize=10, fontweight='bold')
ax.legend(); plt.tight_layout(); plt.show()""")

md("## 4. Spatial variant (§7) — swap the noise operator for **blur**. Rate becomes spatial resolution.\n"
   "Regions where coarsening destroys the class are spatially-discriminative (tests the spatial-regime hypothesis).")
code("""res_blur = RD.run_rd_attribution(MODEL, img01, RD.RDConfig(grid=GRID, operator='blur', n_mc=N_MC, tau=TAU), DEVICE)
fig, ax = plt.subplots(1, 3, figsize=(14, 4.8), facecolor='white')
ax[0].imshow(im); ax[0].set_title('image'); ax[0].axis('off')
for a, r, t in [(ax[1], res, 'AMPLITUDE rate (noise)'), (ax[2], res_blur, 'SPATIAL rate (blur)')]:
    a.imshow(im); a.imshow(r['suff_map'], cmap='inferno', alpha=0.6); a.set_title(t); a.axis('off')
plt.suptitle('same machinery, swapped operator — amplitude vs spatial information demand', fontweight='bold')
plt.tight_layout(); plt.show()""")

md("## 5. Validation (§9) — circularity-free anchors")
code("""v = res['validation']
print('sanity floor  (corner ≤ median):     ', 'PASS' if v['sanity_floor_pass'] else 'FAIL',
      f\"(corner {v['corner_score']:.2f} vs median {v['median_score']:.2f})\")
print('noise-gameability (noise patch low): ', 'PASS' if v['noise_patch_low'] else 'FAIL',
      f\"(pure-noise patch scores {v['noise_patch_score']:.2f})\")
# synthetic anchor: gray out all but the object centre -> map must concentrate there
gray = torch.full_like(img01, 0.5); H = img01.shape[-1]; c0, c1 = H//4, 3*H//4
syn = gray.clone(); syn[:, c0:c1, c0:c1] = img01[:, c0:c1, c0:c1]
rs = RD.run_rd_attribution(MODEL, syn, RD.RDConfig(grid=GRID, operator='noise', n_mc=N_MC, tau=TAU), DEVICE)
box = np.zeros((H,H), bool); box[c0:c1, c0:c1] = True
topk = rs['suff_map'] >= np.quantile(rs['suff_map'], 0.9)
inbox = (topk & box).sum() / max(topk.sum(), 1)
print(f'synthetic anchor (top-10% score inside the kept centre box): {inbox*100:.0f}%  -> {"PASS" if inbox>0.6 else "CHECK"}')
fig, ax = plt.subplots(1, 2, figsize=(9, 4.6), facecolor='white')
ax[0].imshow(syn.permute(1,2,0).cpu().numpy()); ax[0].set_title('synthetic: only centre kept'); ax[0].axis('off')
ax[1].imshow(syn.permute(1,2,0).cpu().numpy()); ax[1].imshow(rs['suff_map'],cmap='inferno',alpha=0.6)
ax[1].set_title(f'score concentrates in box: {inbox*100:.0f}%'); ax[1].axis('off'); plt.tight_layout(); plt.show()""")

nb['cells'] = C
nb.metadata['kernelspec'] = {'display_name': 'KLIG venv', 'language': 'python', 'name': 'klig-venv'}
nbf.write(nb, 'rd_attribution_viz.ipynb')
print('wrote rd_attribution_viz.ipynb  (%d cells)' % len(C))
