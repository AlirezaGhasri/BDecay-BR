"""JAX DeepSets GNN implementation over a star graph for B-Meson decays."""
import numpy as np
import config as cfg
from src.phase_space import kin, expand

try:
    import jax
    import jax.numpy as jnp
    HAVE_JAX = True
except ImportError:
    HAVE_JAX = False

K_SLOTS = 6
S_SLOTS = K_SLOTS + 2

def raw_feats(r, prefix):
    g = lambda k: r.get(f"{prefix} {k}", np.nan)
    w = g("Width")
    w = 0 if not np.isfinite(w) else w
    mm = g("Mass")
    mm = 0.0 if not np.isfinite(mm) else mm
    q = np.nan_to_num(np.array([r.get(f"{prefix} Number of {x} Quarks", 0.0) for x in cfg.Q5])).tolist()
    ch = np.nan_to_num(r.get(prefix + " Charge", 0.0))
    pt = r.get(prefix + " Type", 0.0)
    f = [
        np.log10(max(mm, 1e-4)),
        np.log10(max(w, 1e-25)),
        ch,
        r.get(prefix + " Quantum I", np.nan),
        r.get(prefix + " Quantum J", np.nan),
        r.get(prefix + " Quantum P", np.nan),
        r.get(prefix + " Quantum G", np.nan),
        r.get(prefix + " Quantum C", np.nan)
    ] + q
    f += [sum(q) / 3, float(pt == 1), float(pt == 2), float(pt == 3), float(pt == 5)]
    return np.array(f, float)

def _particle_tables(d):
    TAB, MASS = {}, {}
    for _, r in d.iterrows():
        for prefix in ["Initial Particle", "Intermidiate Particle"] + cfg.P:
            n = r.get(prefix + " Name")
            if isinstance(n, str) and n not in TAB:
                TAB[n] = raw_feats(r, prefix)
                m = r.get(prefix + " Mass", 0.0)
                MASS[n] = 0.0 if not np.isfinite(m) else m
    return TAB, MASS

def fit_norm(d, F5, n_topo, n_qt, core_cols):
    TAB, _ = _particle_tables(d)
    A = np.stack(list(TAB.values()))
    ex_cols = [c for c in F5.columns if c not in core_cols]
    ex = np.nan_to_num(F5[ex_cols].values.astype(float))
    return dict(
        mu=np.nanmean(A[:, [0, 1]], 0),
        sd=np.nanstd(A[:, [0, 1]], 0) + 1e-9,
        ex_scale=np.maximum(np.abs(ex).max(0), 1e-9),
        ex_cols=ex_cols,
        n_topo=n_topo,
        n_qt=n_qt
    )

def build_graph(d, F5, norm):
    TAB, MASS = _particle_tables(d)
    mu, sd = norm["mu"], norm["sd"]

    def enc(name):
        f = TAB[name].copy()
        f[[0, 1]] = (f[[0, 1]] - mu) / sd
        miss = np.isnan(f[3:8]).astype(float)
        f[3:8] = np.nan_to_num(f[3:8])
        return np.concatenate([f, miss])

    FDIM = len(enc(next(iter(TAB))))
    n = len(d)
    F_NODE = FDIM + 3
    nodes = np.zeros((n, S_SLOTS, F_NODE))
    mask = np.zeros((n, S_SLOTS))
    n_topo, n_qt = norm["n_topo"], norm["n_qt"]
    lab = np.zeros((n, n_topo + 4 + n_qt + 2))
    hasi = np.zeros((n, 1))
    d_r = d.reset_index(drop=True)
    for i, (_, r) in enumerate(d_r.iterrows()):
        fin = expand(r)[:K_SLOTS]
        nodes[i, 0, :FDIM] = enc(r["Initial Particle Name"])
        nodes[i, 0, FDIM] = 1
        mask[i, 0] = 1
        for j, k_ in enumerate(fin):
            nodes[i, 1 + j, :FDIM] = enc(k_)
            nodes[i, 1 + j, FDIM + 1] = 1
            mask[i, 1 + j] = 1
        Rn = r.get("Intermidiate Particle Name")
        if isinstance(Rn, str):
            nodes[i, K_SLOTS + 1, :FDIM] = enc(Rn)
            nodes[i, K_SLOTS + 1, FDIM + 2] = 1
            mask[i, K_SLOTS + 1] = 1
            hasi[i] = 1
        top = r.get("Decay Topology")
        ty = r.get("Decay Type")
        qt = r.get("Decay Quark Transition")
        ck = r.get("CKM Matrix Element")
        if pd.notna(top):
            lab[i, int(top) - 1] = 1
        if pd.notna(ty):
            lab[i, n_topo + int(ty)] = 1
        if pd.notna(qt):
            lab[i, n_topo + 4 + int(qt)] = 1
        if pd.notna(ck) and ck > 0:
            lab[i, -2] = 2 * np.log10(ck)
            lab[i, -1] = 1
    KINmat = np.array([kin(MASS[r["Initial Particle Name"]], [MASS[k_] for k_ in expand(r)[:K_SLOTS]]) for _, r in d_r.iterrows()])
    ex = np.nan_to_num(F5.reset_index(drop=True)[norm["ex_cols"]].values.astype(float)) / norm["ex_scale"]
    GLOB = np.concatenate([lab, hasi, KINmat, ex], 1)
    return (jnp.asarray(nodes, jnp.float32), jnp.asarray(mask, jnp.float32), jnp.asarray(GLOB, jnp.float32)), F_NODE

if HAVE_JAX:
    def mlp(p, x):
        for W, b in p[:-1]:
            x = jax.nn.gelu(x @ W + b)
        W, b = p[-1]
        return x @ W + b

    def init_mlp(key, sizes):
        ps = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            key, k = jax.random.split(key)
            ps.append((jax.random.normal(k, (a, b)) * np.sqrt(1.0 / a), jnp.zeros(b)))
        return ps

    def init_params(key, gdim, fnode, H=32):
        ks = jax.random.split(key, 3)
        return dict(
            enc=init_mlp(ks[0], [fnode, H, H]),
            gl=init_mlp(ks[1], [gdim, H, 16]),
            out=init_mlp(ks[2], [3 * H + 16, 64, 1])
        )

    def forward(p, nd, mk, gl):
        h = mlp(p["enc"], nd) * mk[..., None]
        par, hl, lm = h[:, 0], h[:, 1:], mk[:, 1:, None]
        s = hl.sum(1)
        mx = jnp.where(lm > 0, hl, -1e9).max(1)
        mx = jnp.where(mk[:, 1:].sum(1, keepdims=True) > 0, mx, 0.0)
        hg = mlp(p["gl"], gl)
        o = mlp(p["out"], jnp.concatenate([par, s, mx, hg], -1))[:, 0]
        return -jax.nn.softplus(o + 3.0)

    def loss_fn(p, nd, mk, gl, yy, l2):
        r = forward(p, nd, mk, gl) - yy
        huber = jnp.mean(jnp.where(jnp.abs(r) < 1.5, 0.5 * r**2, 1.5 * (jnp.abs(r) - 0.75)))
        reg = sum(jnp.sum(W**2) for grp in p.values() for W, _ in grp)
        return huber + l2 * reg

    @jax.jit
    def adam_step(p, mom, vel, t, nd, mk, gl, yy, lr, l2):
        g = jax.grad(loss_fn)(p, nd, mk, gl, yy, l2)
        mom = jax.tree_util.tree_map(lambda m, gg: 0.9 * m + 0.1 * gg, mom, g)
        vel = jax.tree_util.tree_map(lambda v, gg: 0.999 * v + 0.001 * gg**2, vel, g)
        upd = jax.tree_util.tree_map(lambda m, v: (m / (1.0 - 0.9**t)) / (jnp.sqrt(v / (1.0 - 0.999**t)) + 1e-8), mom, vel)
        return jax.tree_util.tree_map(lambda a, u: a - lr * u, p, upd), mom, vel

    def fit(graph, y, idx, seed, fnode, steps, lr, l2, H=32):
        nodes, mask, glob = graph
        nd, mk, gl = nodes[idx], mask[idx], glob[idx]
        yy = jnp.asarray(y[idx], jnp.float32)
        p = init_params(jax.random.PRNGKey(seed), glob.shape[1], fnode, H)
        mom = jax.tree_util.tree_map(jnp.zeros_like, p)
        vel = jax.tree_util.tree_map(jnp.zeros_like, p)
        for t in range(1, steps + 1):
            cur = lr * (0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * t / steps)))
            p, mom, vel = adam_step(p, mom, vel, jnp.float32(t), nd, mk, gl, yy, jnp.float32(cur), jnp.float32(l2))
        return p

    def predict(graph, p, idx):
        nodes, mask, glob = graph
        return np.asarray(forward(p, nodes[idx], mask[idx], glob[idx]))
