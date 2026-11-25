"""Two months later: drift (PSI), per-group MAE, retrain v2, promotion rule.

Run:  python -m ev_energy.monitoring
"""

from __future__ import annotations

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from .cleaning import load_clean_train, load_later
from .config import file_sha256, load_params, resolve
from .features import FEATURES, NUMERIC, build_pipeline, mae
from .train import load_state, log_sklearn_model, save_state, setup_mlflow


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index with decile bins taken from the reference
    (cleaned train) data. Outer edges are open so every later value lands in
    a bin. PSI = sum((a - e) * ln(a / e))."""
    e = expected.dropna().to_numpy(dtype=float)
    a = actual.dropna().to_numpy(dtype=float)
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e_pct = np.histogram(e, edges)[0] / len(e)
    a_pct = np.histogram(a, edges)[0] / len(a)
    e_pct = np.clip(e_pct, eps, None)
    a_pct = np.clip(a_pct, eps, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def psi_table(train: pd.DataFrame, later: pd.DataFrame, bins: int) -> dict:
    return {c: round(psi(train[c], later[c], bins), 4) for c in NUMERIC}


def mae_by_group(model, df: pd.DataFrame, group_col: str, target: str) -> dict:
    out = {}
    for g, part in df.groupby(group_col):
        out[str(g)] = round(mae(part[target], model.predict(part[FEATURES])), 4)
    return out


def choose_later_weight(train: pd.DataFrame, later_fit: pd.DataFrame, cand: dict,
                        params: dict) -> tuple[float, dict]:
    """Pick how much more a later.csv row counts than an old train row.

    Uses only `later_fit` (the first 70%): its oldest part is added to train
    for fitting and its newest part is the validation set. The real check
    rows (last 30%) are never looked at here.
    """
    mp, target = params["monitoring"], params["data"]["target"]
    cut = int(len(later_fit) * (1 - mp["v2_inner_val_fraction"]))
    inner_fit, inner_val = later_fit.iloc[:cut], later_fit.iloc[cut:]
    data = pd.concat([train, inner_fit], ignore_index=True)
    scores = {}
    for w in mp["v2_later_weights"]:
        sw = np.r_[np.ones(len(train)), np.full(len(inner_fit), w)]
        m = build_pipeline(cand["model"], cand["params"], params["training"]["random_state"])
        m.fit(data[FEATURES], data[target], model__sample_weight=sw)
        scores[w] = mae(inner_val[target], m.predict(inner_val[FEATURES]))
    best = min(scores, key=scores.get)
    return float(best), {str(k): round(v, 4) for k, v in scores.items()}


def run_monitoring(params: dict | None = None) -> dict:
    params = params or load_params()
    setup_mlflow(params)
    mp, d = params["monitoring"], params["data"]
    target, group_col = d["target"], d["group_column"]
    name = params["mlflow"]["registered_model_name"]
    state = load_state(params)
    client = MlflowClient()

    train = load_clean_train(params, state["issues"])
    later = load_later(params)  # sorted by date

    # 1. drift
    psi_values = psi_table(train, later, mp["psi_bins"])
    most_drifted = max(psi_values, key=psi_values.get)

    # 2. v1 per group on later.csv
    v1 = mlflow.sklearn.load_model(f"models:/{name}/{state['v1']['version']}")
    by_group = mae_by_group(v1, later, group_col, target)
    worst_group = max(by_group, key=by_group.get)

    # 3. retrain v2 on train + first 70% of later, check on the last 30%
    cut = int(len(later) * mp["later_train_fraction"])
    later_fit, check = later.iloc[:cut], later.iloc[cut:]
    v2_data = pd.concat([train, later_fit], ignore_index=True)
    cand = state["v1"]["candidate"]
    rs = params["training"]["random_state"]
    later_weight, weight_scores = choose_later_weight(train, later_fit, cand, params)
    weights = np.r_[np.ones(len(train)), np.full(len(later_fit), later_weight)]
    v2 = build_pipeline(cand["model"], cand["params"], rs)
    v2.fit(v2_data[FEATURES], v2_data[target], model__sample_weight=weights)

    mae_v1 = mae(check[target], v1.predict(check[FEATURES]))
    mae_v2 = mae(check[target], v2.predict(check[FEATURES]))

    # 4. promotion rule: v2 must be at least 8% better
    improvement = (mae_v1 - mae_v2) / mae_v1
    decision = "promote" if improvement >= mp["promotion_min_improvement"] else "keep"

    with mlflow.start_run(run_name="v2_retrain") as run:
        mlflow.log_params({"data_sha256": file_sha256(resolve(d["train"])), "stage": "v2",
                           "model_kind": cand["model"], "train_rows": len(v2_data),
                           "later_rows_used": cut, "check_rows": len(check),
                           "later_row_weight": later_weight, "decision": decision})
        mlflow.log_metrics({"mae_v1_on_check": mae_v1, "mae_v2_on_check": mae_v2,
                            "mae_improvement_frac": improvement,
                            **{f"psi_{k}": v for k, v in psi_values.items()},
                            **{f"mae_later_v1_{k}": v for k, v in by_group.items()}})
        info = log_sklearn_model(v2, v2_data[FEATURES])
        v2_run_id = run.info.run_id

    mv = mlflow.register_model(info.model_uri, name)  # becomes version 2
    champion = int(mv.version) if decision == "promote" else state["v1"]["version"]
    client.set_registered_model_alias(name, "champion", champion)

    print("PSI:", psi_values, "-> most drifted:", most_drifted)
    print("v1 MAE by group:", by_group, "-> worst:", worst_group)
    print(f"v2 later-row weight = {later_weight} (inner-val MAE {weight_scores})")
    print(f"check MAE v1={mae_v1:.4f} v2={mae_v2:.4f} improvement={improvement:.1%} -> {decision}")
    print(f"@champion = version {champion}")

    return save_state(params, {
        "monitoring": {"psi": psi_values, "most_drifted_feature": most_drifted,
                       "mae_by_group": by_group, "worst_served_group": worst_group,
                       "mae_v1_on_check": round(mae_v1, 4), "mae_v2_on_check": round(mae_v2, 4),
                       "improvement": round(improvement, 4), "decision": decision},
        "v2": {"run_id": v2_run_id, "version": int(mv.version),
               "later_row_weight": later_weight, "inner_weight_scores": weight_scores},
        "champion_version": champion,
    })


if __name__ == "__main__":
    run_monitoring()
