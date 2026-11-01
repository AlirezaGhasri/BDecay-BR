# B-Meson Exclusive Weak Branching Ratio Predictor (BDecay-BR)

This repository provides a production-ready, modular pipeline for training machine learning models on exclusive weak branching ratios of B-mesons ($B^+$, $B^0$, $B_s^0$, and $\bar{B}_s^0$).

## Highlights
- **JAX-powered GNN:** Incorporates a deep sets graph model for predicting decays directly.
- **XGBoost & Blend:** Fits high-capacity decision trees alongside the GNN, offering optional OOF (Out-Of-Fold) ensemble blends.

## Repository Directory Structure
```text
BDecay-BR/
├── config.py                  # Globals, hyperparams, and dataset path selection
├── run_train.py               # Main execution and evaluation pipeline
└── src/
    ├── data.py                # Adapts raw schemas and filters rows
    ├── phase_space.py         # Monte-Carlo multi-body phase-space calculator
    ├── features.py            # Feature engineering (flavour, structures, kinematics)
    ├── cv.py                  # Grouped leakage-free cross-validation partitioning
    ├── models_xgb.py          # XGBoost training wrappers
    └── models_gnn.py          # JAX Star-Graph GNN DeepSets implementation
```

## Quick Start
```bash
pip install -r requirements.txt
python run_train.py
