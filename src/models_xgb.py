"""XGBoost fit / predict."""
import xgboost as xgb
import config as cfg


def fit(X, y, seed):
    m = xgb.XGBRegressor(**cfg.XGB_PARAMS, random_state=seed)
    return m.fit(X, y)


def predict(model, X):
    return model.predict(X)
