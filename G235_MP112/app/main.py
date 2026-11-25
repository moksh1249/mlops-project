"""MP112 prediction service.

POST /predict  -> energy_kwh for one charging session
GET  /health   -> {"status": "ok", ...}

The client's two rules live in the pydantic schema, so a bad request gets
422 and never reaches the model:
  * session_minutes above 400 is impossible   -> Field(le=400)
  * charger_kw below 1.5 is a typing mistake  -> Field(ge=1.5)
"""

from __future__ import annotations

import os
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from ev_energy.features import FEATURES
from ev_energy.logging_utils import get_logger

MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).resolve().parents[1] / "models" / "model.joblib"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "champion")

log = get_logger("mp112.api")
app = FastAPI(title="MP112 EV charging energy", version="1.0")


class Session(BaseModel):
    model_config = ConfigDict(extra="forbid")  # exactly these fields, nothing else

    battery_start_pct: float = Field(..., ge=0, le=100, description="battery level at plug-in, %")
    session_minutes: float = Field(..., gt=0, le=400, description="planned session length, minutes")
    charger_kw: float = Field(..., ge=1.5, description="charger power, kW")
    temp_c: float = Field(..., ge=-30, le=60, description="air temperature, deg C")
    vehicle_age_yr: float = Field(..., ge=0, le=40, description="vehicle age, years")
    queue_before: int = Field(..., ge=0, description="vehicles waiting before it")
    vehicle: Literal["two_wheeler", "three_wheeler", "car"]
    station_type: Literal["Mall", "Highway", "Residential"]


class Prediction(BaseModel):
    energy_kwh: float
    model_version: str


@lru_cache(maxsize=1)
def get_model():
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=Prediction)
def predict(session: Session) -> Prediction:
    start = time.perf_counter()
    row = pd.DataFrame([session.model_dump()])[FEATURES]
    value = float(get_model().predict(row)[0])
    # no personal data in this request; we still log only model inputs/outputs
    log.info("prediction", extra={"fields": {
        "request_id": uuid.uuid4().hex,
        "model_version": MODEL_VERSION,
        "inputs": session.model_dump(),
        "energy_kwh": round(value, 3),
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
    }})
    return Prediction(energy_kwh=round(value, 3), model_version=MODEL_VERSION)
