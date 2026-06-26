# ════════════════════════════════════════════════════════════════════════════
# CS_latent — class-sensitivity in latent space
# T_yk = x * |A_yk|/max|A_yk|  ;  CS = distance(phi(T_y1), phi(T_y2))
# Higher = the two class attributions produce more separable representations.
# Reuses scatter helpers (_attr_for_class_fast, _build_klig2, _build_gradpath_once,
# _pick_cf_scatter) — RUN THE CS-SCATTER CELL FIRST so those exist.
# ════════════════════════════════════════════════════════════════════════════
import timm, pandas as pd
import torch.nn.functional as F
from captum.attr import IntegratedGradients

N_CS_LATENT = 50          # bump to >=200 for the final number
_rng_csl    = np.random.default_rng(0)

# ── Encoders ────────────────────────────────────────────────────────────────
# 1) ResNet-50 layer4 (avgpool output, 2048) — task-model oracle
_res_feats = {}
_res_hook  = model.avgpool.register_forward_hook(
    lambda m, i, o: _res_feats.__setitem__('z', o.detach()))
def enc_resnet(t):
    with torch.no_grad():
        model(t.unsqueeze(0).to(DEVICE))
    return _res_feats['z'].flatten()                      # (2048,)

# 2) ViT-B/16 patch tokens (mean-pooled, 768)
_vit = timm.create_model('vit_base_patch16_224', pretrained=True).to(DEVICE).eval()
def enc_vit(t):
    with torch.no_grad():
        f = _vit.forward_features(t.unsqueeze(0).to(DEVICE))   # (1, 1+N, 768)
    return f[0, 1:].mean(0)                                # exclude CLS -> (768,)

# 3) CLIP image encoder (512) — baseline oracle (what CASE uses implicitly)
def enc_clip(t):
    with torch.no_grad():
        vo = _clip_mdl.vision_model(pixel_values=t.unsqueeze(0).to(DEVICE))
        z  = _clip_mdl.visual_projection(vo.pooler_output)    # (1,512)
    return z[0]                                            # (512,)

ENCODERS = {'ResNet': enc_resnet, 'ViT': enc_vit, 'CLIP': enc_clip}

# ── IG baseline (captum, zero baseline) ──────────────────────────────────────
_ig = IntegratedGradients(model)
def _ig_attr(x1, cls):
    a = _ig.attribute(x1.unsqueeze(0), target=int(cls),
                      baselines=torch.zeros_like(x1).unsqueeze(0), n_steps=32)
    return absmax_collapse(a.squeeze(0)).detach().cpu().numpy()   # (H,W) signed

# ── CS_latent core ───────────────────────────────────────────────────────────
def cs_latent_score(x1, attr_y1, attr_y2, enc):
    # signed maps in, abs only for the mask magnitude; per-image normalize to [0,1]
    m1 = torch.from_numpy(np.abs(attr_y1)).float().to(DEVICE); m1 = m1/(m1.max()+1e-8)
    m2 = torch.from_numpy(np.abs(attr_y2)).float().to(DEVICE); m2 = m2/(m2.max()+1e-8)
    T1 = x1 * m1.unsqueeze(0)                              # (3,H,W) broadcast
    T2 = x1 * m2.unsqueeze(0)
    z1, z2 = enc(T1), enc(T2)
    cos = 1.0 - F.cosine_similarity(z1.unsqueeze(0), z2.unsqueeze(0)).item()
    l2  = torch.norm(z1 - z2).item()
    return cos, l2

# ── Methods → per-class attribution (signed (H,W)) ───────────────────────────
CSL_METHODS = ['KL-IG² (adaptive)', 'KL-IG²', 'KL-IG (linear)',
               'KLIG-Adaptive', 'IG', 'Random']

def _attr_map(method, x1, cls, H, W, ctx):
    if method == 'KLIG-Adaptive':
        return _attr_for_class_fast('KLIG-Adaptive', x1, cls,
                                    sig_adapt=ctx['sig'](cls)).reshape(H, W)
    if method == 'KL-IG (linear)':
        return _attr_for_class_fast('KL-IG (linear)', x1, cls).reshape(H, W)
    if method == 'KL-IG²':                                  # fixed-sigma
        return _attr_for_class_fast('KL-IG²', x1, cls,
                                    klig2_fixed=ctx['k2f'],
                                    path_fixed=ctx['pf']).reshape(H, W)
    if method == 'KL-IG² (adaptive)':
        k2, pth, sig = ctx['adapt'](cls)
        return _attr_for_class_fast('KL-IG² (adaptive)', x1, cls,
                                    klig2_adapt=k2, path_adapt=pth,
                                    sig_adapt=sig).reshape(H, W)
    if method == 'IG':
        return _ig_attr(x1, cls)
    if method == 'Random':
        return _rng_csl.uniform(-1, 1, size=(H, W)).astype('float32')
    raise ValueError(method)

# ── Run ──────────────────────────────────────────────────────────────────────
Y2_MIN_PROB = 0.10   # CF = next predicted class, only if its prob exceeds this
import random
_cands = [d for d in multi_imgs
          if d.get('high_cls') and len(d['high_cls']) >= 2
          and d['high_probs'][1] > Y2_MIN_PROB]
random.seed(0)                              # pinned -> reproducible sample
random.shuffle(_cands)
# prefer unique top-1 classes for maximum variety, then pad to N
_pool, _seen_cls = [], set()
for d in _cands:
    if d['high_cls'][0] not in _seen_cls:
        _pool.append(d); _seen_cls.add(d['high_cls'][0])
    if len(_pool) >= N_CS_LATENT: break
if len(_pool) < N_CS_LATENT:
    _ids = {id(d) for d in _pool}
    for d in _cands:
        if len(_pool) >= N_CS_LATENT: break
        if id(d) not in _ids: _pool.append(d)
print(f'CS_latent over {len(_pool)} images (y2 prob > {Y2_MIN_PROB}, '
      f'{len(set(d["high_cls"][0] for d in _pool))} distinct top-1 classes) '
      f'x {len(CSL_METHODS)} methods x {len(ENCODERS)} encoders')

# ── CF pool for THIS set's y2 classes (best-by-probability, guaranteed real CF) ─
_csl_cache = CACHE_DIR / 'klig2_cf_csl_pool.pkl'
_need = {int(d['high_cls'][1]) for d in _pool}
if not FORCE_RECOMPUTE and _csl_cache.exists():
    _cf_cpu = pickle.load(open(_csl_cache, 'rb'))
    if len(set(_cf_cpu) & _need) < len(_need): _csl_cache.unlink()
if FORCE_RECOMPUTE or not _csl_cache.exists():
    _cf_cpu = {}
    # seed from already-loaded multi_imgs (top-1==c -> real image of class c, no download)
    for _d2 in multi_imgs:
        _c0 = int(_d2['high_cls'][0])
        if _c0 in _need and _c0 not in _cf_cpu:
            _xx = _d2['x']; _xx = _xx.squeeze(0) if _xx.dim() == 4 else _xx
            _cf_cpu[_c0] = _xx.cpu()
    # then seed from cf_pool (already built)
    for _c in _need:
        if _c not in _cf_cpu and 'cf_pool' in globals() and _c in cf_pool:
            _cf_cpu[_c] = cf_pool[_c].cpu()
    _still = _need - set(_cf_cpu)
    print(f'  seeded {len(_cf_cpu)}/{len(_need)} from memory; streaming for {len(_still)}')
    if _still:
        from datasets import load_dataset as _hf
        _s = _hf('evanarlian/imagenet_1k_resized_256', split='train',
                 streaming=True).shuffle(seed=13, buffer_size=500)
        _best, _lock, _sc = {c: (-1.0, None) for c in _still}, set(), 0
        _pb = tqdm(total=len(_still), desc='CS_latent CF pool')
        for item in _s:
            _sc += 1
            if len(_lock) >= len(_still) or _sc >= 8000: break
            im = item['image']
            if im.mode != 'RGB': im = im.convert('RGB')
            xx = preprocess(im).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pr = model(xx).softmax(-1)[0].cpu()
            xc = xx.cpu()
            for c in _still:
                if c in _lock: continue
                if float(pr[c]) > _best[c][0]:
                    _best[c] = (float(pr[c]), xc)
                    if pr[c] >= 0.30: _lock.add(c); _pb.update(1)
        _pb.close()
        for c in _still:
            if _best[c][1] is not None: _cf_cpu[c] = _best[c][1]
    pickle.dump(_cf_cpu, open(_csl_cache, 'wb'))
cf_csl = {c: v.to(DEVICE) for c, v in _cf_cpu.items()}
print(f'CS_latent CF pool: {len(cf_csl)}/{len(_need)} y2 classes')

rows = []
for d in tqdm(_pool, desc='CS_latent'):
    x1   = d['x'].squeeze(0).to(DEVICE)
    H, W = x1.shape[1], x1.shape[2]
    y1, y2 = int(d['high_cls'][0]), int(d['high_cls'][1])
    img_id = d['idx']

    # per-image KL-IG² context (CF + fixed path + per-class adaptive path)
    x_cf = cf_csl.get(y2)
    if x_cf is None:        # no real CF for this y2 -> skip image
        continue
    if x_cf.dim() == 4: x_cf = x_cf.squeeze(0)
    x_cf = x_cf.to(DEVICE)
    k2f = _build_klig2(x_cf, SIGMA_FINAL, LV_FLOOR)
    pf  = _build_gradpath_once(k2f, x1)
    _sig_cache, _ad_cache = {}, {}
    def _sig(cls):
        if cls not in _sig_cache:
            _sig_cache[cls] = find_sigma_stop(model, x1, int(cls), tau=0.95,
                                              n_samples=32, n_iter=12)
        return _sig_cache[cls]
    def _adapt(cls):
        if cls not in _ad_cache:
            s = _sig(cls); k2 = _build_klig2(x_cf, s, 2*math.log(s))
            _ad_cache[cls] = (k2, _build_gradpath_once(k2, x1), s)
        return _ad_cache[cls]
    ctx = {'k2f': k2f, 'pf': pf, 'sig': _sig, 'adapt': _adapt}

    for m in CSL_METHODS:
        a1 = _attr_map(m, x1, y1, H, W, ctx)
        a2 = _attr_map(m, x1, y2, H, W, ctx)
        for enc_name, enc in ENCODERS.items():
            cos, l2 = cs_latent_score(x1, a1, a2, enc)
            rows.append({'image_id': img_id, 'method': m, 'encoder': enc_name,
                         'cs_cosine': cos, 'cs_l2': l2, 'y1': y1, 'y2': y2})

_res_hook.remove()
df_csl = pd.DataFrame(rows)
df_csl.to_csv('cs_latent_results.csv', index=False)
print('saved cs_latent_results.csv  (', len(df_csl), 'rows )')

# ── Summary table: mean +/- std cosine per method x encoder ──────────────────
print('\nCS_latent (cosine) — mean +/- std  [higher = more class-sensitive]')
piv = (df_csl.groupby(['method', 'encoder'])['cs_cosine']
       .agg(['mean', 'std']).reset_index())
tab = piv.pivot(index='method', columns='encoder', values='mean').reindex(CSL_METHODS)
print(tab[['ResNet', 'ViT', 'CLIP']].round(4).to_string())
print('\n(full per-image data in cs_latent_results.csv; cs_l2 also stored)')
