"""Feature engineering and the scikit-learn model pipeline.

The whole pipeline (feature engineering + imputation + one-hot + model) is one
object, so the API only has to hand it the raw request fields.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder, PolynomialFeatures,
                                   StandardScaler)

NUMERIC = ["battery_start_pct", "session_minutes", "charger_kw", "temp_c",
           "vehicle_age_yr", "queue_before"]
CATEGORICAL = ["vehicle", "station_type"]
FEATURES = NUMERIC + CATEGORICAL
ENGINEERED = ["max_energy_kwh", "battery_room_pct"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Physics-style helpers known at plug-in time.

    max_energy_kwh  = charger power x planned hours (an upper bound)
    battery_room_pct = how empty the battery is
    """
    out = df.copy()
    out["max_energy_kwh"] = out["charger_kw"] * out["session_minutes"] / 60.0
    out["battery_room_pct"] = 100.0 - out["battery_start_pct"]
    return out


def make_model(kind: str, params: dict, random_state: int = 42):
    if kind in ("ridge", "ridge_interactions"):
        return Ridge(**params)
    if kind == "random_forest":
        return RandomForestRegressor(random_state=random_state, n_jobs=-1, **params)
    if kind == "gradient_boosting":
        return GradientBoostingRegressor(random_state=random_state, loss="absolute_error", **params)
    raise ValueError(f"unknown model kind {kind!r}")


def build_pipeline(kind: str, params: dict, random_state: int = 42) -> Pipeline:
    numeric = NUMERIC + ENGINEERED
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    linear = kind in ("ridge", "ridge_interactions")
    if linear:
        num_steps.append(("scale", StandardScaler()))
    pre = ColumnTransformer([
        ("num", Pipeline(num_steps), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    steps = [("features", FunctionTransformer(add_features, validate=False)), ("pre", pre)]
    if kind == "ridge_interactions":
        # every pair of inputs, e.g. station_type x session_minutes, so the
        # energy-per-minute slope can differ between stations
        steps.append(("interact", PolynomialFeatures(2, interaction_only=True, include_bias=False)))
    steps.append(("model", make_model(kind, params, random_state)))
    return Pipeline(steps)


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))
