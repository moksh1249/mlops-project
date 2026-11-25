"""Fill reports/findings.yaml from the numbers the pipeline produced.

Run:  python -m ev_energy.findings
"""

from __future__ import annotations

import yaml

from .config import load_params, resolve
from .train import load_state

ISSUE_KEYS = {
    "leaking_column": ["column"],
    "unit_mixup": ["column", "group_value", "factor"],
    "sentinel_value": ["column", "value", "count"],
    "duplicate_rows": ["count"],
    "stuck_sensor": ["column", "start_date", "end_date"],
    "target_typos": ["count"],
    "spelling_variants": ["column", "true_levels"],
}


def write_findings(params: dict | None = None) -> dict:
    params = params or load_params()
    state = load_state(params)
    path = resolve(params["outputs"]["findings"])
    form = yaml.safe_load(path.read_text(encoding="utf-8"))

    for kind, keys in ISSUE_KEYS.items():
        found = state["issues"][kind]
        entry = {"present": bool(found["present"])}
        for k in keys:
            entry[k] = found.get(k) if found["present"] else None
        form["data_issues"][kind] = entry

    form["tracking"] = {
        "experiment_name": params["mlflow"]["experiment_name"],
        "best_run_id": state["v1"]["run_id"],
        "registered_model_name": params["mlflow"]["registered_model_name"],
        "champion_version": state["champion_version"],
    }

    m = state["monitoring"]
    form["monitoring"] = {
        "psi": m["psi"],
        "most_drifted_feature": m["most_drifted_feature"],
        "mae_by_group": m["mae_by_group"],
        "worst_served_group": m["worst_served_group"],
        "mae_v1_on_check": m["mae_v1_on_check"],
        "mae_v2_on_check": m["mae_v2_on_check"],
        "decision": m["decision"],
    }

    header = ("# MP112 findings --- filled in by `python -m ev_energy.findings`.\n"
              "# Every value below was computed by code in src/ev_energy/.\n\n")
    path.write_text(header + yaml.safe_dump(form, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")
    print(f"wrote {path}")
    return form


if __name__ == "__main__":
    write_findings()
