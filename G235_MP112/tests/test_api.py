"""API tests with FastAPI's TestClient. A tiny model is trained on synthetic
data, so the test does not depend on the CSVs or on models/model.joblib."""

import importlib

import joblib
import pytest
from fastapi.testclient import TestClient

from ev_energy.features import FEATURES, build_pipeline

GOOD = {
    "battery_start_pct": 30, "session_minutes": 120, "charger_kw": 22.0,
    "temp_c": 28.5, "vehicle_age_yr": 3.0, "queue_before": 2,
    "vehicle": "car", "station_type": "Mall",
}


@pytest.fixture
def client(tmp_path, monkeypatch, synthetic_sessions):
    model = build_pipeline("ridge", {"alpha": 1.0})
    model.fit(synthetic_sessions[FEATURES], synthetic_sessions["energy_kwh"])
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    monkeypatch.setenv("MODEL_PATH", str(path))
    import app.main as main
    main = importlib.reload(main)
    return TestClient(main.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_ok(client):
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["energy_kwh"], float)
    assert body["energy_kwh"] > 0


def test_rule_session_minutes_above_400_rejected(client):
    assert client.post("/predict", json={**GOOD, "session_minutes": 401}).status_code == 422
    assert client.post("/predict", json={**GOOD, "session_minutes": 400}).status_code == 200


def test_rule_charger_kw_below_1_5_rejected(client):
    assert client.post("/predict", json={**GOOD, "charger_kw": 1.4}).status_code == 422
    assert client.post("/predict", json={**GOOD, "charger_kw": 1.5}).status_code == 200


def test_missing_or_extra_field_rejected(client):
    missing = {k: v for k, v in GOOD.items() if k != "temp_c"}
    assert client.post("/predict", json=missing).status_code == 422
    assert client.post("/predict", json={**GOOD, "energy_kwh": 10}).status_code == 422


def test_unknown_vehicle_rejected(client):
    assert client.post("/predict", json={**GOOD, "vehicle": "bus"}).status_code == 422


def test_prediction_is_logged_as_json(client, capsys):
    import json
    client.post("/predict", json=GOOD)
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("{")]
    rec = json.loads(lines[-1])
    assert rec["message"] == "prediction" and "energy_kwh" in rec
