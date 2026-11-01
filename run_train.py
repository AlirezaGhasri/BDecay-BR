"""Train XGBoost + GNN Models on B-meson exclusive weak decays & evaluate Blends."""
import json
import time
import numpy as np
import pandas as pd
import config as cfg
from src.data import load_b_only
from src.features import build_ground_state_table, build_F5
from src.cv import family, make_folds, check_no_group_leakage, rmse
from src import models_xgb
from src import models_gnn

CORE_COLS = [
    "Q", "Pstar", "M0", "Q_over_M", "logW0", "type", "topo", "qt", "logCKM2", "nprod", "nrows", "has_inter",
    "init_Q", "init_I", "init_J", "init_P", "mmax", "mmin", "logWprod", "logPhi"
] + [f"prod_n{q}" for q in cfg.Q5] + [f"init_n{q}" for q in cfg.Q5]

def main():
    t0 = time.time()
    d, topo, qt = load_b_only()
    y = np.log10(d["BR"].values)
    ground = build_ground_state_table(d)
    F5 = build_F5(d, ground)
    X = np.nan_to_num(F5.values.astype(float), nan=0.0)
    
    print(f"Tabular features: {F5.shape[1]} | rows: {len(d)}")
    groups = d.apply(family, axis=1).values
    print(f"CV groups: {len(np.unique(groups))}")

    run_gnn = cfg.RUN_GNN and models_gnn.HAVE_JAX
    if run_gnn:
        norm = models_gnn.fit_norm(d, F5, len(topo), len(qt), CORE_COLS)
        graph, fnode = models_gnn.build_graph(d, F5, norm)
        print("JAX GNN star graphs built successfully.")
    else:
        print("JAX not available or RUN_GNN disabled -> skipping GNN and Blend.")

    oof_xgb_all = np.zeros(len(d))
    oof_gnn_all = np.zeros(len(d))
    
    for sd in cfg.SEEDS:
        fo = make_folds(groups, sd)
        check_no_group_leakage(groups, fo)
        
        oof_xgb_sd = np.full(len(d), np.nan)
        oof_gnn_sd = np.full(len(d), np.nan)
        
        for f in range(5):
            te = np.where(fo == f)[0]
            tr = np.where(fo != f)[0]
            
            # XGBoost Model
            m_xgb = models_xgb.fit(X[tr], y[tr], sd)
            oof_xgb_sd[te] = models_xgb.predict(m_xgb, X[te])
            
            # JAX GNN Model Ensemble
            if run_gnn:
                models_ens = []
                for b in range(cfg.GNN_SEEDS):
                    g_seed = 1000 * sd + b
                    m_gnn = models_gnn.fit(
                        graph, y, tr, g_seed, fnode,
                        cfg.GNN_SETTINGS["steps"], cfg.GNN_SETTINGS["lr"], cfg.GNN_SETTINGS["l2"]
                    )
                    models_ens.append(m_gnn)
                oof_gnn_sd[te] = np.mean([models_gnn.predict(graph, p, te) for p in models_ens], axis=0)
                
        oof_xgb_all += oof_xgb_sd
        if run_gnn:
            oof_gnn_all += oof_gnn_sd
            print(f"Seed {sd} -> XGB RMSE: {rmse(y, oof_xgb_sd):.4f} dex | GNN RMSE: {rmse(y, oof_gnn_sd):.4f} dex")
        else:
            print(f"Seed {sd} -> XGB RMSE: {rmse(y, oof_xgb_sd):.4f} dex")

    final_xgb = oof_xgb_all / len(cfg.SEEDS)
    print(f"\n[Final Performance - XGBoost Only]")
    print(f"Mean-OOF RMSE: {rmse(y, final_xgb):.4f} dex")
    print(f"Mean-OOF MAE:  {np.mean(np.abs(final_xgb - y)):.4f} dex")

    final_pred = final_xgb
    if run_gnn:
        final_gnn = oof_gnn_all / len(cfg.SEEDS)
        final_blend = (final_xgb + final_gnn) / 2.0
        final_pred = final_blend
        print(f"\n[Final Performance - JAX GNN Only]")
        print(f"Mean-OOF RMSE: {rmse(y, final_gnn):.4f} dex")
        print(f"\n[Final Performance - Model Blend (XGBoost + GNN)]")
        print(f"Mean-OOF RMSE: {rmse(y, final_blend):.4f} dex")
        print(f"Mean-OOF MAE:  {np.mean(np.abs(final_blend - y)):.4f} dex")

    # Output creation
    out = d[["Decay Description", "Initial Particle Name"]].copy()
    out["true_BR"] = d["BR"].values
    out["true_log10BR"] = y
    out["pred_log10BR"] = final_pred
    out["pred_BR"] = 10 ** final_pred
    out["abs_err_dex"] = np.abs(final_pred - y)
    out = out.sort_values("abs_err_dex", ascending=False)
    out.to_csv(f"{cfg.OUT_DIR}/oof_predictions.csv", index=False)

    metrics = {
        "n_rows": int(len(d)),
        "rmse_dex": rmse(y, final_pred),
        "mae_dex": float(np.mean(np.abs(final_pred - y))),
        "seeds": list(cfg.SEEDS),
    }
    with open(f"{cfg.OUT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Saved results to → {cfg.OUT_DIR} | Total time: {time.time() - t0:.0f}s")

if __name__ == "__main__":
    main()
