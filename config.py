"""
CONFIG — B-only BR training including XGBoost, GNN, and Blend config.
"""
import glob
import os
import time

_cands = (
    glob.glob("/content/*All*PDG*Training*Decays*2026*.csv")
    + glob.glob("./*All*PDG*Training*Decays*2026*.csv")
    + glob.glob("/content/*B*Particles*Cleaned*.csv")
    + glob.glob("./*B*Particles*Cleaned*.csv")
)
CSV = _cands[0] if _cands else "/content/All_PDG_Training_Decays_2026.csv"

TARGET_SPECIES = ["B+", "B0", "B_s()0", "Bbar_s()0"]

QUICK_TEST = False
SEEDS = (0, 1, 2, 3, 4)
SPLIT_STRATEGY = "family" # or "similar_particles"

# XGBoost settings
XGB_PARAMS = dict(
    learning_rate=0.03,
    colsample_bytree=0.5,
    max_depth=4,
    subsample=0.8,
    min_child_weight=2,
    n_estimators=600,
    n_jobs=1,
    verbosity=0,
)

# GNN settings (JAX)
RUN_GNN = True
GNN_SEEDS = 2
GNN_SETTINGS = dict(steps=600, l2=3e-3, H=32, lr=2e-3)

DROP_CATEGORIES = {"3", "4", "5", "6", "7", "9", "10", "12"}
BAD_DESC_REGEX = r"\[|S-wave|P-wave|D-wave|semileptonic| via |\(ex\."
TARGET_BR_MAX = 0.1

HIDDEN_CHARM_MIN_MASS = 2.8
LIGHT_ISOSCALAR_MAX_MASS = 2.0
P = [f"Product Particle {i}" for i in range(1, 5)]
Q5 = "udscb"

OUT_DIR = "./BDecay_BR_Outputs"
os.makedirs(OUT_DIR, exist_ok=True)
t_start = time.time()

if QUICK_TEST:
    SEEDS = (0,)
    RUN_GNN = False
    XGB_PARAMS = {**XGB_PARAMS, "n_estimators": 200}
