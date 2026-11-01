"""Load PDG CSV, adapt schema, clean, keep B-meson rows only."""
import re
import numpy as np
import pandas as pd
import config as cfg

TYPE_MAP = {"Meson": 1, "Baryon": 2, "Lepton": 3, "Quark": 4, "Boson": 5}
DECAY_TYPE_MAP = {"Hadronic": 3, "Semi-Leptonic": 1, "Leptonic": 2}


def parse_struct(s):
    """Structure As Int -> net (u,d,s,c,b) = quarks - antiquarks; X -> 0."""
    if not isinstance(s, str):
        return np.zeros(5)
    toks = re.findall(r"-?\d|X", s.strip("'\""))
    if len(toks) != 10:
        return np.zeros(5)
    v = np.array([0.0 if t == "X" else float(t) for t in toks])
    return v[:5] - v[5:]


def norm_topology(s):
    if not isinstance(s, str) or s.startswith("Unresolvable"):
        return None
    return re.sub(r"\s*->.*$", "", s).strip()


def norm_qt(s):
    return None if not isinstance(s, str) else s.split(";")[0].strip()


def adapt_schema(raw):
    d = raw.copy()
    for prefix in ["Initial Particle", "Intermidiate Particle"] + cfg.P:
        struct_col = prefix + " Structure As Int"
        if struct_col not in d.columns:
            continue
        V = np.array([parse_struct(s) for s in d[struct_col]])
        has = d[prefix + " Name"].notna().values
        for j, q in enumerate(cfg.Q5):
            d[f"{prefix} Number of {q} Quarks"] = np.where(has, V[:, j], np.nan)
        if prefix + " Type" in d.columns:
            d[prefix + " Type"] = d[prefix + " Type"].map(TYPE_MAP)
    for i in range(1, 5):
        col = f"Product Particle {i} Multiplicity"
        if col not in d.columns:
            d[col] = np.where(d[f"Product Particle {i} Name"].notna(), 1.0, np.nan)
    if d["Decay Type"].dtype == object:
        d["Decay Type"] = d["Decay Type"].map(DECAY_TYPE_MAP).fillna(0)
    if "Diagram 1 Topology" in d.columns:
        d["_topo_str"] = d["Diagram 1 Topology"].map(norm_topology)
        d["_qt_str"] = d["Diagram 1 Quark Transition"].map(norm_qt)
        d["CKM Matrix Element"] = pd.to_numeric(
            d["Diagram 1 Effective CKM Matrix Element"], errors="coerce"
        )
        d["n_diagrams"] = pd.to_numeric(d["Number of Diagrams"], errors="coerce")
        c2 = pd.to_numeric(d["Diagram 2 Effective CKM Matrix Element"], errors="coerce")
        d["log_ckm2_diag2"] = (2 * np.log10(c2.clip(lower=1e-30))).fillna(0.0)
    return d


def build_vocab(d):
    if "_topo_str" not in d.columns:
        return d, [], []
    topo = sorted(d["_topo_str"].dropna().unique())
    qt = sorted(d["_qt_str"].dropna().unique())
    d["Decay Topology"] = d["_topo_str"].map({t: i + 1 for i, t in enumerate(topo)})
    d["Decay Quark Transition"] = d["_qt_str"].map({t: i for i, t in enumerate(qt)})
    return d, topo, qt


def clean_rows(d):
    d = d.copy()
    d["BR"] = pd.to_numeric(d["Decay Value"], errors="coerce")
    if "Decay Catagories" in d.columns:
        cats = d["Decay Catagories"].astype(str).map(
            lambda s: set(re.findall(r"Category (\d+)", s))
        )
    else:
        cats = pd.Series([set()] * len(d), index=d.index)
    mult_total = sum(d[f"{p} Multiplicity"].fillna(0) for p in cfg.P)
    miss_mass = np.zeros(len(d), dtype=bool)
    for p in ["Initial Particle"] + cfg.P:
        if p + " Mass" in d.columns:
            miss_mass |= (d[p + " Name"].notna() & ~np.isfinite(d[p + " Mass"])).values
    if "Intermidiate Particle Name" in d.columns and "Intermidiate Particle Mass" in d.columns:
        miss_mass |= (
            d["Intermidiate Particle Name"].notna()
            & ~np.isfinite(d["Intermidiate Particle Mass"])
        ).values
    rules = {
        "BR missing or outside (0,1]": ~(d["BR"] > 0) | ~(d["BR"] <= 1),
        "category in DROP_CATEGORIES": cats.map(lambda s: bool(s & cfg.DROP_CATEGORIES)),
        "bad description": d["Decay Description"].astype(str).str.contains(
            cfg.BAD_DESC_REGEX, regex=True
        ),
        "unresolved topology": (
            d["Decay Topology"].isna()
            if "Decay Topology" in d.columns
            else pd.Series(False, index=d.index)
        ),
        "missing mass": pd.Series(miss_mass, index=d.index),
        "fewer than 2 products": mult_total < 2,
    }
    keep = pd.Series(True, index=d.index)
    for name, bad in rules.items():
        n = int((bad & keep).sum())
        keep &= ~bad
        print(f"  filter '{name}': removed {n}")
    d = d[keep].copy()
    r = d["BR"] > cfg.TARGET_BR_MAX
    print(f"  filter 'B ratio-row BR>{cfg.TARGET_BR_MAX}': removed {int(r.sum())}")
    return d[~r].copy()


def load_b_only(csv_path=None):
    path = csv_path or cfg.CSV
    raw = pd.read_csv(path)
    print(f"raw rows: {len(raw)}")
    d, topo, qt = build_vocab(adapt_schema(raw))
    d = clean_rows(d)
    d = d[d["Initial Particle Name"].isin(cfg.TARGET_SPECIES)].reset_index(drop=True)
    print(f"B-only rows: {len(d)}")
    print(d["Initial Particle Name"].value_counts().to_dict())
    return d, topo, qt
