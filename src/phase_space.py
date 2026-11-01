"""Deterministic n-body phase-space feature (logPhi)."""
import hashlib
import numpy as np
import config as cfg


def _row_seed(Mp, ms):
    s = repr((round(float(Mp), 6), tuple(np.round(ms, 6).tolist())))
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def kin(Mp, ms):
    ms = np.array(ms, float)
    n = len(ms)
    Q = Mp - ms.sum()
    if Q <= 0:
        return np.array([Q / Mp, -8.0, 0.0, n / 8, 1.0])
    if n == 1:
        return np.array([Q / Mp, 0.0, 0.0, n / 8, 0.0])
    if n == 2:
        p = np.sqrt(
            max((Mp**2 - (ms[0] + ms[1])**2) * (Mp**2 - (ms[0] - ms[1])**2), 1e-30)
        ) / (2 * Mp)
        return np.array([Q / Mp, np.log10(p), p / Mp, n / 8, 0.0])
    rng = np.random.default_rng(_row_seed(Mp, ms))
    N_ = 2000
    r = np.sort(rng.random((N_, n - 2)), 1)
    r = np.hstack([np.zeros((N_, 1)), r, np.ones((N_, 1))])
    ms = ms[rng.permutation(n)]
    Mk = np.cumsum(ms)[None, :] + r * Q
    w = np.ones(N_)
    for k_ in range(1, n):
        a, b, c = Mk[:, k_], Mk[:, k_ - 1], ms[k_]
        w *= np.sqrt(np.clip((a**2 - (b + c)**2) * (a**2 - (b - c)**2), 0, None)) / (2 * a)
    return np.array([Q / Mp, np.log10(w.mean() * Q ** (n - 2) + 1e-300), 0.0, n / 8, 0.0])


def flat_masses(r):
    out = []
    for i in range(1, 5):
        name = r.get(f"Product Particle {i} Name")
        if isinstance(name, str):
            m = r[f"Product Particle {i} Mass"]
            k = int(r.get(f"Product Particle {i} Multiplicity") or 1)
            out += [m] * k
    return np.nan_to_num(np.array(out, float))


def expand(r):
    out = []
    for p in cfg.P:
        if isinstance(r.get(p + " Name"), str):
            out += [r[p + " Name"]] * int(r.get(p + " Multiplicity") or 1)
    return out
