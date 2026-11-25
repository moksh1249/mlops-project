"""Find the seven kinds of data problem in train.csv.

Every check takes the raw DataFrame (as given, before any cleaning) and
returns a small dict with the same keys as reports/findings.yaml.

Run:  python -m ev_energy.data_checks
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from .config import load_params, resolve


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def normalise_label(value: str) -> str:
    """'  Three Wheeler.' -> 'three_wheeler' (case, spaces, dots, dashes)."""
    s = str(value).strip().lower().rstrip(".")
    s = re.sub(r"[\s\-]+", "_", s)
    return s


def map_to_canonical(value: str, canonical: list[str]) -> str:
    """Map one messy label to one of the clean levels.

    Exact match after normalising first; otherwise a unique prefix match
    handles abbreviations such as 'thr.' -> 'three_wheeler'.
    """
    s = normalise_label(value)
    if s in canonical:
        return s
    hits = [c for c in canonical if c.startswith(s) or c.replace("_", "").startswith(s)]
    if len(hits) == 1:
        return hits[0]
    raise ValueError(f"cannot map label {value!r} to one of {canonical}")


def replace_sentinels(df: pd.DataFrame, numeric: list[str], sentinel: dict) -> pd.DataFrame:
    out = df.copy()
    if sentinel.get("present"):
        col, val = sentinel["column"], sentinel["value"]
        out.loc[out[col] == val, col] = np.nan
    return out


# --------------------------------------------------------------------------- #
# the seven checks
# --------------------------------------------------------------------------- #
def check_leaking_column(train: pd.DataFrame, holdout: pd.DataFrame, target: str,
                         numeric: list[str], corr_limit: float = 0.98) -> dict:
    """A leak is a column that is in train but not available at prediction
    time (missing from the holdout file), or one that agrees almost exactly
    with the target."""
    not_in_holdout = [c for c in train.columns if c not in holdout.columns and c != target]
    suspicious = list(not_in_holdout)
    for col in numeric:
        if col in train.columns:
            r = train[[col, target]].corr(method="spearman").iloc[0, 1]
            if abs(r) >= corr_limit:
                suspicious.append(col)
    if suspicious:
        return {"present": True, "column": suspicious[0], "all_candidates": suspicious}
    return {"present": False, "column": None}


def check_sentinel_value(train: pd.DataFrame, numeric: list[str], candidates: list,
                         min_repeats: int) -> dict:
    found = []
    for col in numeric:
        counts = train[col].value_counts()
        for val in candidates:
            n = int(counts.get(val, 0))
            if n >= min_repeats:
                found.append({"column": col, "value": int(val), "count": n})
    if not found:
        return {"present": False, "column": None, "value": None, "count": None}
    best = max(found, key=lambda d: d["count"])
    return {"present": True, **best}


def check_unit_mixup(train: pd.DataFrame, numeric: list[str], group_col: str,
                     factors: list, tolerance: float) -> dict:
    """For each numeric column compare every group's mean with the mean of the
    other groups. If one group is ~factor times bigger, it used another unit."""
    best = None
    for col in numeric:
        for g in train[group_col].dropna().unique():
            inside = train.loc[train[group_col] == g, col].dropna()
            outside = train.loc[train[group_col] != g, col].dropna()
            if inside.empty or outside.empty or outside.mean() == 0:
                continue
            ratio = inside.mean() / outside.mean()
            if ratio <= 0:
                continue
            for f in factors:
                rel_err = abs(np.log(ratio / f))  # symmetric relative error
                if rel_err <= tolerance and (best is None or rel_err < best["err"]):
                    best = {"column": col, "group_value": str(g), "factor": int(f),
                            "err": rel_err, "ratio": float(ratio),
                            "max_ratio": float(inside.max() / outside.max())}
    if best is None:
        return {"present": False, "column": None, "group_value": None, "factor": None}
    return {"present": True, "column": best["column"], "group_value": best["group_value"],
            "factor": best["factor"], "observed_mean_ratio": round(best["ratio"], 3),
            "observed_max_ratio": round(best["max_ratio"], 3)}


def check_duplicate_rows(train: pd.DataFrame) -> dict:
    n = int(train.duplicated().sum())
    return {"present": n > 0, "count": n if n > 0 else None}


def check_stuck_sensor(train: pd.DataFrame, numeric: list[str], date_col: str,
                       min_days: int) -> dict:
    """Sort by date and find the longest run of identical consecutive values."""
    df = train.drop_duplicates().sort_values(date_col, kind="stable")
    best = None
    for col in numeric:
        s = df[col]
        run_id = (s != s.shift()).cumsum()
        runs = df.groupby(run_id).agg(value=(col, "first"), rows=(col, "size"),
                                      start=(date_col, "min"), end=(date_col, "max"),
                                      days=(date_col, "nunique"))
        top = runs.sort_values(["days", "rows"]).iloc[-1]
        if best is None or top["days"] > best["days"]:
            best = {"column": col, **top.to_dict()}
    if best is None or best["days"] < min_days:
        return {"present": False, "column": None, "start_date": None, "end_date": None,
                "longest_run_days": None if best is None else int(best["days"])}
    return {"present": True, "column": best["column"], "start_date": str(best["start"]),
            "end_date": str(best["end"]), "value": float(best["value"]),
            "rows": int(best["rows"])}


def check_target_typos(train: pd.DataFrame, target: str, factor: float) -> dict:
    """Sort the target and look for the biggest jump between neighbours. If the
    values above that gap are a small group about `factor` times bigger than
    the rest, they are typos."""
    v = np.sort(train[target].dropna().to_numpy())
    jumps = v[1:] / np.maximum(v[:-1], 1e-9)
    i = int(np.argmax(jumps))
    threshold = float(v[i])
    above = v[v > threshold]
    is_typo_group = jumps[i] > 3 and len(above) < 0.05 * len(v)
    if not is_typo_group:
        return {"present": False, "count": None}
    count = int((train[target] > threshold).sum())  # counted on rows as given
    return {"present": True, "count": count, "threshold": threshold,
            "gap_ratio": round(float(jumps[i]), 2)}


def check_spelling_variants(train: pd.DataFrame, later: pd.DataFrame,
                            categorical: list[str]) -> dict:
    """later.csv has the clean spellings; any extra raw spelling in train is a
    variant. true_levels = number of categories after mapping."""
    best = None
    for col in categorical:
        canonical = sorted(later[col].dropna().unique().tolist())
        raw_levels = train[col].dropna().unique().tolist()
        extra = [x for x in raw_levels if x not in canonical]
        if not extra:
            continue
        mapped = {x: map_to_canonical(x, canonical) for x in raw_levels}
        info = {"column": col, "true_levels": len(set(mapped.values())),
                "raw_levels": len(raw_levels), "mapping": mapped}
        if best is None or len(extra) > best["raw_levels"]:
            best = info
    if best is None:
        return {"present": False, "column": None, "true_levels": None}
    return {"present": True, **best}


# --------------------------------------------------------------------------- #
def run_all_checks(params: dict | None = None) -> dict:
    params = params or load_params()
    d, f, c = params["data"], params["features"], params["cleaning"]
    train = pd.read_csv(resolve(d["train"]))
    later = pd.read_csv(resolve(d["later"]))
    holdout = pd.read_csv(resolve(d["holdout"]))
    numeric, categorical = f["numeric"], f["categorical"]

    sentinel = check_sentinel_value(train, numeric, c["sentinel_values"], c["sentinel_min_repeats"])
    # unit check must ignore sentinels, otherwise -999 dominates the means
    no_sentinel = replace_sentinels(train, numeric, sentinel)

    return {
        "leaking_column": check_leaking_column(train, holdout, d["target"], numeric),
        "unit_mixup": check_unit_mixup(no_sentinel, numeric, d["group_column"],
                                       c["unit_factors_to_try"], c["unit_ratio_tolerance"]),
        "sentinel_value": sentinel,
        "duplicate_rows": check_duplicate_rows(train),
        "stuck_sensor": check_stuck_sensor(train, numeric, d["date_column"], c["stuck_min_days"]),
        "target_typos": check_target_typos(train, d["target"], c["target_typo_factor"]),
        "spelling_variants": check_spelling_variants(train, later, categorical),
    }


if __name__ == "__main__":
    print(json.dumps(run_all_checks(), indent=2, default=str))
