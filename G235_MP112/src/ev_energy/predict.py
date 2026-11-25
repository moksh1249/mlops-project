"""Predict holdout_features.csv with the @champion model and export it.

Also writes models/model.joblib (the champion pipeline the API loads), so
the container does not need MLflow at run time.

Run:  python -m ev_energy.predict
"""

from __future__ import annotations

import joblib
import mlflow.sklearn
import pandas as pd

from .config import load_params, resolve
from .features import FEATURES
from .train import setup_mlflow


def predict_holdout(params: dict | None = None) -> pd.DataFrame:
    params = params or load_params()
    setup_mlflow(params)
    name = params["mlflow"]["registered_model_name"]
    champion = mlflow.sklearn.load_model(f"models:/{name}@champion")

    model_path = resolve(params["outputs"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion, model_path)

    holdout = pd.read_csv(resolve(params["data"]["holdout"]))
    preds = pd.DataFrame({
        "record_id": holdout[params["data"]["id_column"]],
        "prediction": champion.predict(holdout[FEATURES]).round(3),
    })
    out = resolve(params["outputs"]["predictions"])
    out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out, index=False)
    print(f"wrote {len(preds)} predictions -> {out}")
    print(f"saved champion pipeline -> {model_path}")
    return preds


if __name__ == "__main__":
    predict_holdout()
