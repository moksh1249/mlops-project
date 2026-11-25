"""Clean train.csv using what data_checks found.

Order matters:
1. drop exact duplicate rows
2. sentinel value -> NaN (imputed later inside the model pipeline)
3. undo the unit mix-up for the affected group (divide by the factor)
4. undo target typos (divide by 10)
5. map spelling variants to the clean levels
6. drop a leaking column if there is one
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_checks import map_to_canonical, run_all_checks
from .config import load_params, resolve


def clean_frame(df: pd.DataFrame, issues: dict, params: dict) -> pd.DataFrame:
    d = params["data"]
    out = df.copy()

    if issues["duplicate_rows"]["present"]:
        out = out.drop_duplicates().reset_index(drop=True)

    s = issues["sentinel_value"]
    if s["present"]:
        out[s["column"]] = out[s["column"]].astype(float)
        out.loc[out[s["column"]] == s["value"], s["column"]] = np.nan

    u = issues["unit_mixup"]
    if u["present"]:
        mask = out[d["group_column"]] == u["group_value"]
        out.loc[mask, u["column"]] = out.loc[mask, u["column"]] / u["factor"]

    t = issues["target_typos"]
    if t["present"]:
        mask = out[d["target"]] > t["threshold"]
        out.loc[mask, d["target"]] = out.loc[mask, d["target"]] / params["cleaning"]["target_typo_factor"]

    sp = issues["spelling_variants"]
    if sp["present"]:
        canonical = sorted(set(sp["mapping"].values()))
        out[sp["column"]] = out[sp["column"]].map(lambda x: map_to_canonical(x, canonical))

    lk = issues["leaking_column"]
    if lk["present"]:
        out = out.drop(columns=[c for c in lk.get("all_candidates", [lk["column"]]) if c in out])

    return out


def load_clean_train(params: dict | None = None, issues: dict | None = None) -> pd.DataFrame:
    params = params or load_params()
    issues = issues or run_all_checks(params)
    raw = pd.read_csv(resolve(params["data"]["train"]))
    return clean_frame(raw, issues, params)


def load_later(params: dict | None = None) -> pd.DataFrame:
    """later.csv is clean; sort it by date as the problem asks."""
    params = params or load_params()
    later = pd.read_csv(resolve(params["data"]["later"]))
    return later.sort_values(params["data"]["date_column"], kind="stable").reset_index(drop=True)


if __name__ == "__main__":
    df = load_clean_train()
    print(df.describe(include="all").T)
