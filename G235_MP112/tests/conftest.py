"""Shared test fixtures. Nothing here reads the CSVs (they are not in git):
small data is built inside the tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_sessions() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame({
        "record_id": [f"T{i:04d}" for i in range(n)],
        "date": pd.date_range("2026-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "battery_start_pct": rng.integers(5, 80, n),
        "session_minutes": rng.integers(15, 240, n),
        "charger_kw": rng.uniform(3, 60, n).round(1),
        "temp_c": rng.uniform(5, 45, n).round(1),
        "vehicle_age_yr": rng.uniform(0, 8, n).round(1),
        "queue_before": rng.integers(0, 10, n),
        "vehicle": rng.choice(["two_wheeler", "three_wheeler", "car"], n),
        "station_type": rng.choice(["Mall", "Highway", "Residential"], n),
    })
    df["energy_kwh"] = (10 + 0.15 * df["session_minutes"] + 0.2 * df["charger_kw"]
                        - 0.5 * df["vehicle_age_yr"] + rng.normal(0, 1, n)).round(2)
    return df
