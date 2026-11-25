import numpy as np
import pandas as pd

from ev_energy import data_checks as dc
from ev_energy.cleaning import clean_frame
from ev_energy.monitoring import psi

NUMERIC = ["battery_start_pct", "session_minutes", "charger_kw", "temp_c",
           "vehicle_age_yr", "queue_before"]


def test_duplicates_counted_as_extra_copies(synthetic_sessions):
    df = pd.concat([synthetic_sessions, synthetic_sessions.iloc[:4]], ignore_index=True)
    assert dc.check_duplicate_rows(df) == {"present": True, "count": 4}
    assert dc.check_duplicate_rows(synthetic_sessions)["present"] is False


def test_sentinel_found(synthetic_sessions):
    df = synthetic_sessions.copy()
    df.loc[:6, "queue_before"] = -999
    res = dc.check_sentinel_value(df, NUMERIC, [-1, -999, 999], min_repeats=5)
    assert res == {"present": True, "column": "queue_before", "value": -999, "count": 7}


def test_unit_mixup_finds_exact_factor(synthetic_sessions):
    df = synthetic_sessions.copy()
    mask = df["station_type"] == "Residential"
    df.loc[mask, "vehicle_age_yr"] *= 12  # years recorded as months
    res = dc.check_unit_mixup(df, NUMERIC, "station_type", [10, 12, 60, 100, 1000], 0.25)
    assert res["present"] and res["column"] == "vehicle_age_yr"
    assert res["group_value"] == "Residential" and res["factor"] == 12


def test_target_typos(synthetic_sessions):
    df = synthetic_sessions.copy()
    df.loc[:2, "energy_kwh"] *= 10
    res = dc.check_target_typos(df, "energy_kwh", 10)
    assert res["present"] and res["count"] == 3


def test_stuck_sensor_absent_on_random_data(synthetic_sessions):
    assert dc.check_stuck_sensor(synthetic_sessions, NUMERIC, "date", 7)["present"] is False


def test_spelling_variants_mapped():
    assert dc.map_to_canonical(" Three Wheeler.", ["car", "three_wheeler", "two_wheeler"]) == "three_wheeler"
    assert dc.map_to_canonical("thr.", ["car", "three_wheeler", "two_wheeler"]) == "three_wheeler"
    assert dc.map_to_canonical("CAR ", ["car", "three_wheeler", "two_wheeler"]) == "car"


def test_psi_zero_for_same_and_large_for_shift():
    rng = np.random.default_rng(1)
    a = pd.Series(rng.normal(0, 1, 2000))
    assert psi(a, a) < 1e-6
    assert psi(a, a + 3) > 1.0


def test_clean_frame_fixes_everything(synthetic_sessions):
    raw = synthetic_sessions.copy()
    raw.loc[0, "queue_before"] = -999
    raw.loc[1, "energy_kwh"] = 500.0
    raw.loc[2, "vehicle"] = " CAR"
    issues = {
        "leaking_column": {"present": False},
        "duplicate_rows": {"present": True},
        "sentinel_value": {"present": True, "column": "queue_before", "value": -999},
        "unit_mixup": {"present": False},
        "target_typos": {"present": True, "threshold": 200.0},
        "spelling_variants": {"present": True, "column": "vehicle",
                              "mapping": {"car": "car", "two_wheeler": "two_wheeler",
                                          "three_wheeler": "three_wheeler"}},
    }
    params = {"data": {"group_column": "station_type", "target": "energy_kwh"},
              "cleaning": {"target_typo_factor": 10}}
    out = clean_frame(pd.concat([raw, raw.iloc[[5]]]), issues, params)
    assert len(out) == len(raw)
    assert np.isnan(out.loc[0, "queue_before"])
    assert out.loc[1, "energy_kwh"] == 50.0
    assert out.loc[2, "vehicle"] == "car"
