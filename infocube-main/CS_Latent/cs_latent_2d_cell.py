# ════════════════════════════════════════════════════════════════════════════
# CS_latent — 2D latent-space visualization (paired phi(T_y1) / phi(T_y2), ViT)
# Two panels: KL-IG² (adaptive) vs KL-IG (linear). One shared projection (fit once).
# filled circle = y1, open circle = y2, line = the pair. Longer line => more
# class-separable in latent space.
# PREREQ: run the CS_latent cell first (needs _pool, cf_csl, _attr_map, enc_vit,
#         _build_klig2, _build_gradpath_once, find_sigma_stop).
# ════════════════════════════════════════════════════════════════════════════
PROJ_METHOD = 'UMAP'   # 'UMAP' = nice clusters | 'PCA' = line length ~ true CS scale
from sklearn.decomposition import PCA
if PROJ_METHOD == 'UMAP':
    try:
        import umap
    except Exception:
        print('umap unavailable -> falling back to PCA'); PROJ_METHOD = 'PCA'
_PROJ = PROJ_METHOD

VIZ2D_METHODS = ['KL-IG² (adaptive)', 'KLIG-Adaptive']   # both adaptive

recs = []   # (method, z_y1, z_y2, cs_cosine)
for d in tqdm(_pool, desc='2D latent viz'):
    x1   = d['x'].squeeze(0).to(DEVICE)
    H, W = x1.shape[1], x1.shape[2]
    y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
    x_cf = cf_csl.get(y2)
    if x_cf is None:
        continue
    x_cf = (x_cf.squeeze(0) if x_cf.dim() == 4 else x_cf).to(DEVICE)

    k2f = _build_klig2(x_cf, SIGMA_FINAL, LV_FLOOR)
    pf  = _build_gradpath_once(k2f, x1)
    _sc, _ac = {}, {}
    def _sig(cls):
        if cls not in _sc:
            _sc[cls] = find_sigma_stop(model, x1, int(cls), tau=0.95,
                                       n_samples=32, n_iter=12)
        return _sc[cls]
    def _adapt(cls):
        if cls not in _ac:
            s = _sig(cls); k2 = _build_klig2(x_cf, s, 2*math.log(s))
            _ac[cls] = (k2, _build_gradpath_once(k2, x1), s)
        return _ac[cls]
    ctx = {'k2f': k2f, 'pf': pf, 'sig': _sig, 'adapt': _adapt}

    for m in VIZ2D_METHODS:
        a1 = _attr_map(m, x1, y1, H, W, ctx)
        a2 = _attr_map(m, x1, y2, H, W, ctx)
        m1 = torch.from_numpy(np.abs(a1)).float().to(DEVICE); m1 = m1/(m1.max()+1e-8)
        m2 = torch.from_numpy(np.abs(a2)).float().to(DEVICE); m2 = m2/(m2.max()+1e-8)
        z1 = enc_vit(x1 * m1.unsqueeze(0)).cpu().numpy()
        z2 = enc_vit(x1 * m2.unsqueeze(0)).cpu().numpy()
        cs = 1.0 - float(z1 @ z2 / (np.linalg.norm(z1)*np.linalg.norm(z2) + 1e-9))
        recs.append((m, z1, z2, cs))

# ── one shared projection over ALL z (both methods, both y1/y2) ───────────────
n  = len(recs)
Z  = np.stack([r[1] for r in recs] + [r[2] for r in recs])    # (2n, d)
if _PROJ == 'UMAP':
    P = umap.UMAP(n_components=2, random_state=0,
                  n_neighbors=15, min_dist=0.1).fit_transform(Z)
else:
    P = PCA(n_components=2, random_state=0).fit_transform(Z)
P1, P2 = P[:n], P[n:]                                          # y1 pts, y2 pts

# ── plot: two panels, shared axes ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
xlim = (P[:,0].min()-1, P[:,0].max()+1)
ylim = (P[:,1].min()-1, P[:,1].max()+1)
for ax, m in zip(axes, VIZ2D_METHODS):
    col = COLORS.get(m, 'gray')
    idx = [i for i, r in enumerate(recs) if r[0] == m]
    for i in idx:
        ax.plot([P1[i,0], P2[i,0]], [P1[i,1], P2[i,1]],
                '-', color=col, alpha=0.45, lw=1, zorder=1)
        ax.scatter(P1[i,0], P1[i,1], facecolor=col, edgecolor='black',
                   s=45, lw=0.6, zorder=3)                      # y1 filled
        ax.scatter(P2[i,0], P2[i,1], facecolor='white', edgecolor=col,
                   s=45, lw=1.2, zorder=3)                      # y2 open
    sep = np.mean([np.linalg.norm(P1[i]-P2[i]) for i in idx]) if idx else 0.0
    cs_mean = np.mean([recs[i][3] for i in idx]) if idx else 0.0
    ax.set_title(f'{m}\nmean 2D separation = {sep:.2f}   |   CS_latent = {cs_mean:.3f}',
                 fontsize=11, fontweight='bold', color=col)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xticks([]); ax.set_yticks([])

from matplotlib.lines import Line2D
leg = [Line2D([0],[0], marker='o', color='gray', markerfacecolor='gray',
              markeredgecolor='black', lw=0, label='y1 (target)'),
       Line2D([0],[0], marker='o', color='gray', markerfacecolor='white',
              markeredgecolor='gray', lw=0, label='y2 (counterfactual)')]
axes[0].legend(handles=leg, loc='upper left', fontsize=9, framealpha=0.9)

fig.suptitle(f'CS_latent — {_PROJ} of ViT phi(T_y1) vs phi(T_y2)   '
             f'(longer line = more class-separable)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('cs_latent_2d.png', dpi=160, bbox_inches='tight')
plt.show()
print(f'saved cs_latent_2d.png  ({_PROJ}, {n} pairs/method)')
