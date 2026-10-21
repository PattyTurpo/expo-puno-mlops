"""API REST con FastAPI que sirve el modelo Champion en producción."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from app.schemas import ExportInput, PredictionOutput
from src import config

app = FastAPI(
    title="ExpoPredict Puno - FOB Valuation API",
    description="Supervised Learning API using Stacking (KNN + Lasso -> Ridge)",
    version="1.0.0",
)


def load_champion():
    if not config.CHAMPION_MODEL_PATH.exists() or not config.REGISTRY_JSON_PATH.exists():
        raise HTTPException(status_code=503, detail="Champion model not trained yet.")
    model = joblib.load(config.CHAMPION_MODEL_PATH)
    registry = json.loads(config.REGISTRY_JSON_PATH.read_text(encoding="utf-8"))
    return model, registry["current_champion"]


def classify_shipment(fob_value: float) -> str:
    if fob_value >= 200000:
        return "High-Value Export"
    if fob_value >= 20000:
        return "Medium Export"
    return "Small Export"


@app.get("/health")
def health():
    _, champ = load_champion()
    return {
        "status": "ok",
        "model_version": champ["version"],
        "r2_score": champ["metrics"]["r2"],
    }


@app.get("/model-info")
def model_info():
    _, champ = load_champion()
    return champ


@app.post("/predict", response_model=PredictionOutput)
def predict(data: ExportInput):
    model, champ = load_champion()
    df = pd.DataFrame([data.model_dump()])[config.FEATURES]
    raw_pred = float(model.predict(df)[0])
    fob_usd = max(10.0, round(raw_pred, 2))

    return PredictionOutput(
        predicted_fob_usd=fob_usd,
        shipment_category=classify_shipment(fob_usd),
        model_version=champ["version"],
        model_r2=champ["metrics"]["r2"],
    )