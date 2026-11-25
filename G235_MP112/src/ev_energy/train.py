"""Train v1: try every candidate in params.yaml, one MLflow run each.

The best by 5-fold CV MAE on cleaned train.csv is registered as version 1 of
the model and gets the alias `champion`.

Run:  python -m ev_energy.train
"""

from __future__ import annotations

import json
import os

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import KFold, cross_val_score

from .cleaning import load_clean_train, load_later
from .config import PROJECT_ROOT, file_sha256, load_params, resolve
from .data_checks import run_all_checks
from .features import FEATURES, build_pipeline, mae

STATE_FILE = "run_state.json"


def setup_mlflow(params: dict) -> None:
    os.chdir(PROJECT_ROOT)  # sqlite:///mlflow.db and mlartifacts/ live in the repo root
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])


def log_sklearn_model(pipe, X_example):
    return mlflow.sklearn.log_model(
        pipe, name="model", input_example=X_example.head(3),
        serialization_format="cloudpickle",
    )


def save_state(params: dict, update: dict) -> dict:
    path = resolve(params["outputs"]["reports_dir"]) / STATE_FILE
    state = json.loads(path.read_text()) if path.exists() else {}
    state.update(update)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))
    return state


def load_state(params: dict) -> dict:
    return json.loads((resolve(params["outputs"]["reports_dir"]) / STATE_FILE).read_text())


def train_v1(params: dict | None = None) -> dict:
    params = params or load_params()
    setup_mlflow(params)
    tp = params["training"]
    target = params["data"]["target"]
    data_hash = file_sha256(resolve(params["data"]["train"]))

    issues = run_all_checks(params)
    train = load_clean_train(params, issues)
    later = load_later(params)
    X, y = train[FEATURES], train[target]

    cv = KFold(n_splits=tp["cv_folds"], shuffle=True, random_state=tp["random_state"])
    results = []
    for cand in tp["candidates"]:
        with mlflow.start_run(run_name=f"v1_{cand['name']}") as run:
            pipe = build_pipeline(cand["model"], cand["params"], tp["random_state"])
            cv_mae = -cross_val_score(pipe, X, y, cv=cv, scoring="neg_mean_absolute_error").mean()
            pipe.fit(X, y)
            mlflow.log_params({"data_sha256": data_hash, "model_kind": cand["model"],
                               "stage": "v1", "train_rows": len(train),
                               **{f"hp_{k}": v for k, v in cand["params"].items()}})
            mlflow.log_metrics({
                "cv_mae": float(cv_mae),
                "train_mae": mae(y, pipe.predict(X)),
                "mae_later": mae(later[target], pipe.predict(later[FEATURES])),
            })
            info = log_sklearn_model(pipe, X)
            results.append({"name": cand["name"], "run_id": run.info.run_id,
                            "cv_mae": float(cv_mae), "model_uri": info.model_uri,
                            "candidate": cand})
            print(f"{cand['name']:<14} cv_mae={cv_mae:.3f}  run={run.info.run_id}")

    best = min(results, key=lambda r: r["cv_mae"])
    name = params["mlflow"]["registered_model_name"]
    mv = mlflow.register_model(best["model_uri"], name)
    client = MlflowClient()
    client.set_registered_model_alias(name, "champion", mv.version)
    print(f"best = {best['name']} -> {name} v{mv.version} @champion")

    return save_state(params, {
        "issues": issues,
        "v1": {"run_id": best["run_id"], "version": int(mv.version),
               "candidate": best["candidate"], "cv_mae": best["cv_mae"]},
        "champion_version": int(mv.version),
        "all_v1_runs": [{k: r[k] for k in ("name", "run_id", "cv_mae")} for r in results],
    })


if __name__ == "__main__":
    train_v1()
