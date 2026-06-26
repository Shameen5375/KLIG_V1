# ════════════════════════════════════════════════════════════════════════════
# CS_latent — Option B: PCA of DELTA vectors  delta = phi(T_y1) - phi(T_y2)  (ViT)
# One delta per (image, method). PCA fit once over ALL deltas. Color by method.
# KL-IG² deltas spread far from origin (class-specific directions);
# IG / Random deltas cluster near 0.
# PREREQ: run the CS_latent cell first (_pool, cf_csl, _attr_map, enc_vit, ...).
# ════════════════════════════════════════════════════════════════════════════
from sklearn.decomposition import PCA

DELTA_METHODS = ['KL-IG² (adaptive)', 'KLIG-Adaptive', 'IG', 'Random']

deltas, dmeth = [], []
for d in tqdm(_pool, desc='delta vectors'):
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
    def _sig(c):
        if c not in _sc:
            _sc[c] = find_sigma_stop(model, x1, int(c), tau=0.95, n_samples=32, n_iter=12)
        return _sc[c]
    def _adapt(c):
        if c not in _ac:
            s = _sig(c); k2 = _build_klig2(x_cf, s, 2*math.log(s))
            _ac[c] = (k2, _build_gradpath_once(k2, x1), s)
        return _ac[c]
    ctx = {'k2f': k2f, 'pf': pf, 'sig': _sig, 'adapt': _adapt}
    for m in DELTA_METHODS:
        a1 = _attr_map(m, x1, y1, H, W, ctx)
        a2 = _attr_map(m, x1, y2, H, W, ctx)
        m1 = torch.from_numpy(np.abs(a1)).float().to(DEVICE); m1 = m1/(m1.max()+1e-8)
        m2 = torch.from_numpy(np.abs(a2)).float().to(DEVICE); m2 = m2/(m2.max()+1e-8)
        z1 = enc_vit(x1 * m1.unsqueeze(0)).cpu().numpy()
        z2 = enc_vit(x1 * m2.unsqueeze(0)).cpu().numpy()
        deltas.append(z1 - z2); dmeth.append(m)

D     = np.stack(deltas)
dmeth = np.array(dmeth)
P     = PCA(n_components=2, random_state=0).fit_transform(D)   # fit once on all deltas

# ── scatter ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7), facecolor='white')
ax.axhline(0, color='gray', lw=0.6, zorder=0)
ax.axvline(0, color='gray', lw=0.6, zorder=0)
for m in DELTA_METHODS:
    mask = dmeth == m
    ax.scatter(P[mask, 0], P[mask, 1], s=38, alpha=0.7,
               color=COLORS.get(m, 'gray'), edgecolor='white', lw=0.4, label=m)
ax.scatter([0], [0], marker='+', s=220, color='black', lw=2, zorder=5)
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.set_title('CS_latent delta vectors  phi(T_y1) - phi(T_y2)   (ViT, PCA)\n'
             'far from origin = stronger class-specific direction',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9, framealpha=0.9)
plt.tight_layout()
plt.savefig('cs_latent_delta_pca.png', dpi=160, bbox_inches='tight')
plt.show()

# ── spread stats (the quantitative story) ─────────────────────────────────────
print(f"\n{'method':22s} {'mean||delta||':>13s} {'std||delta||':>12s} {'mean PCA radius':>16s}")
print('-' * 66)
for m in DELTA_METHODS:
    mask = dmeth == m
    norms = np.linalg.norm(D[mask], axis=1)
    rad   = np.linalg.norm(P[mask], axis=1)
    print(f'{m:22s} {norms.mean():13.3f} {norms.std():12.3f} {rad.mean():16.3f}')
print('\n(mean||delta|| in full ViT space is the faithful spread measure; '
      'PCA radius is its 2D shadow)')
print('saved cs_latent_delta_pca.png')
