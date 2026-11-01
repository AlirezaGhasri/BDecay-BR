"""Tabular physics features for each decay (no particle-name features)."""
import numpy as np
import pandas as pd
import config as cfg
from src.phase_space import kin, flat_masses


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _nz(x):
    return -99.0 if (x is None or not np.isfinite(x)) else float(x)


def slots(r):
    out = []
    for p in cfg.P:
        if not isinstance(r.get(p + " Name"), str):
            continue
        k = _f(r.get(p + " Multiplicity"))
        out.append(
            dict(
                k=int(k) if np.isfinite(k) else 1,
                ptype=_f(r.get(p + " Type")),
                charge=np.nan_to_num(_f(r.get(p + " Charge"))),
                mass=_f(r.get(p + " Mass")),
                I=_f(r.get(p + " Quantum I")),
                J=_f(r.get(p + " Quantum J")),
                Pp=_f(r.get(p + " Quantum P")),
                G=_f(r.get(p + " Quantum G")),
                C=_f(r.get(p + " Quantum C")),
                q=np.nan_to_num(
                    np.array([_f(r.get(f"{p} Number of {x} Quarks")) for x in cfg.Q5])
                ),
            )
        )
    return out


STRUCT_COLS = [
    "n_charged_leptons",
    "n_neutrinos",
    "n_hidden_charm",
    "n_isoscalar_pseudoscalar",
    "n_spin_ge1",
    "n_baryon_products",
    "has_baryon_pair",
    "has_4plus_baryons",
    "is_loop_fcnc",
]


def structure_flags(r):
    n_cl = n_nu = n_hc = n_isops = n_spin = n_bar = 0
    for s in slots(r):
        k, flavourless = s["k"], np.abs(s["q"]).sum() == 0
        if s["ptype"] == 3:
            if abs(s["charge"]) > 0.5:
                n_cl += k
            else:
                n_nu += k
        neutral = abs(s["charge"]) < 0.5
        if flavourless and neutral and s["ptype"] == 1 and s["mass"] > cfg.HIDDEN_CHARM_MIN_MASS:
            n_hc += k
        if (
            flavourless
            and neutral
            and s["I"] == 0
            and s["J"] == 0
            and s["Pp"] == -1
            and s["mass"] < cfg.LIGHT_ISOSCALAR_MAX_MASS
        ):
            n_isops += k
        if s["J"] >= 1:
            n_spin += k
        if abs(s["q"].sum()) >= 3:
            n_bar += k
    dt = r["Decay Type"]
    loop = int((n_cl >= 2 and n_nu == 0) or (dt == 2 and n_nu == 0) or (dt == 0))
    return [n_cl, n_nu, n_hc, n_isops, n_spin, n_bar, int(n_bar >= 2), int(n_bar >= 4), loop]


def feats(d):
    X = pd.DataFrame(index=d.index)
    num = lambda c: pd.to_numeric(d[c], errors="coerce")
    mult = lambda i: d[f"{cfg.P[i]} Multiplicity"].fillna(0)
    X["Q"] = num("Decay Q Value")
    X["Pstar"] = num("Decay P*")
    X["M0"] = num("Initial Particle Mass")
    X["Q_over_M"] = X["Q"] / X["M0"]
    X["logW0"] = np.log10(num("Initial Particle Width").clip(lower=1e-30))
    X["type"] = num("Decay Type")
    X["topo"] = num("Decay Topology")
    X["qt"] = num("Decay Quark Transition")
    ckm = num("CKM Matrix Element")
    X["logCKM2"] = 2 * np.log10(ckm.clip(lower=1e-30))
    X["nprod"] = sum(mult(i) for i in range(4))
    X["nrows"] = d[[p + " Name" for p in cfg.P]].notna().sum(axis=1)
    X["has_inter"] = d["Intermidiate Particle Name"].notna().astype(int)
    for q in cfg.Q5:
        X["prod_n" + q] = sum(
            num(f"{cfg.P[i]} Number of {q} Quarks").fillna(0) * mult(i) for i in range(4)
        )
        X["init_n" + q] = num(f"Initial Particle Number of {q} Quarks")
    X["init_Q"] = num("Initial Particle Charge")
    X["init_I"] = num("Initial Particle Quantum I")
    X["init_J"] = num("Initial Particle Quantum J")
    X["init_P"] = num("Initial Particle Quantum P")
    ms = np.stack([num(f"{cfg.P[i]} Mass").values for i in range(4)], 1)
    X["mmax"] = np.nanmax(ms, 1)
    X["mmin"] = np.nanmin(ms, 1)
    ws = np.stack([num(f"{cfg.P[i]} Width").fillna(0).values for i in range(4)], 1)
    X["logWprod"] = np.log10(ws.sum(1) + 1e-25)
    X["Delta_S"] = X["prod_ns"] - X["init_ns"]
    X["Delta_C"] = X["prod_nc"] - X["init_nc"]
    init_flavourless = X[[f"init_n{q}" for q in cfg.Q5]].abs().sum(axis=1) == 0
    X["init_hidden_charm"] = (init_flavourless & (X["M0"] > cfg.HIDDEN_CHARM_MIN_MASS)).astype(int)
    X["CKM_order"] = ckm.map(
        lambda x: 3 if pd.isna(x) else (0 if x > 0.1 else 1 if x > 0.01 else 2 if x > 0.001 else 3)
    )
    X["n_diagrams"] = num("n_diagrams") if "n_diagrams" in d.columns else 1
    X["log_ckm2_diag2"] = num("log_ckm2_diag2") if "log_ckm2_diag2" in d.columns else 0.0
    S = pd.DataFrame(
        [structure_flags(r) for _, r in d.iterrows()], index=d.index, columns=STRUCT_COLS
    )
    X = pd.concat([X, S], axis=1)
    X["Is_Charmless"] = (
        (X["Delta_C"] == 0) & (X["n_hidden_charm"] == 0) & (X["init_hidden_charm"] == 0)
    ).astype(int)
    return X


def flow(r):
    init = np.array([_f(r[f"Initial Particle Number of {q} Quarks"]) for q in cfg.Q5], float)
    spect = init.copy()
    spect[4] = 0
    ncar = nheavy = nlc = 0
    for s in slots(r):
        k, v, ch = s["k"], s["q"], s["charge"]
        if np.any(
            (spect != 0) & (np.sign(v) == np.sign(spect)) & (np.abs(v) > 0) & (np.arange(5) < 3)
        ):
            ncar += k
        if abs(v[3]) + abs(v[4]) > 0:
            nheavy += k
        if abs(ch) == 1 and np.abs(v).sum() == 2 and abs(v[3]) + abs(v[4]) == 0:
            nlc += k
    return [ncar, nheavy, nlc, int(nheavy == 0), abs(np.nan_to_num(init[:3])).sum()]


def _particle_key(prefix, r):
    ptype = _f(r.get(prefix + " Type"))
    if ptype not in (1.0, 2.0):
        return None
    q = np.nan_to_num(np.array([_f(r.get(f"{prefix} Number of {x} Quarks")) for x in cfg.Q5]))
    if np.abs(q).sum() > 0:
        return ("q",) + tuple(q.tolist())
    mass = _f(r.get(prefix + " Mass"))
    return (
        "n",
        _nz(_f(r.get(prefix + " Quantum I"))),
        _nz(_f(r.get(prefix + " Quantum G"))),
        _nz(_f(r.get(prefix + " Quantum C"))),
        bool(mass > cfg.HIDDEN_CHARM_MIN_MASS),
    )


def build_ground_state_table(d):
    ground = {}
    for _, r in d.iterrows():
        for prefix in ["Initial Particle", "Intermidiate Particle"] + cfg.P:
            if not isinstance(r.get(prefix + " Name"), str):
                continue
            mass = _f(r.get(prefix + " Mass"))
            key = _particle_key(prefix, r)
            if key is None or not np.isfinite(mass) or mass <= 0:
                continue
            ground[key] = min(mass, ground.get(key, mass))
    return ground


def excitation_height(r, ground):
    best = 0.0
    for prefix in ["Initial Particle", "Intermidiate Particle"] + cfg.P:
        if not isinstance(r.get(prefix + " Name"), str):
            continue
        mass = _f(r.get(prefix + " Mass"))
        key = _particle_key(prefix, r)
        if key is None or not np.isfinite(mass) or mass <= 0:
            continue
        ref = ground.get(key)
        if ref and ref > 0:
            best = max(best, np.log10(mass / ref))
    return best


def build_F5(d, ground):
    X = feats(d)
    X["logPhi"] = [
        kin(r["Initial Particle Mass"], flat_masses(r))[1] for _, r in d.iterrows()
    ]
    X["excitation_height"] = [excitation_height(r, ground) for _, r in d.iterrows()]
    XFL = pd.DataFrame(
        [flow(r) for _, r in d.iterrows()],
        index=d.index,
        columns=[
            "n_spect_carriers",
            "n_heavyflav_prod",
            "n_Wcur_cand",
            "no_heavy_prod",
            "init_light_content",
        ],
    )
    return pd.concat([X, XFL], axis=1)
