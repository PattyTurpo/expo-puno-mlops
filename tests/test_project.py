"""Pruebas unitarias de Datos, Modelo Supervisado y API para Integración Continua (CI)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from app.main import app
from src import config
from src.train import load_xy

client = TestClient(app)


def test_1_datasets_exist_and_have_no_nulls():
    """Verifica que los lotes de datos existan y estén limpios."""
    for path in [config.TEST_PATH, config.TRAIN_V1_PATH, config.TRAIN_V2_PATH]:
        assert path.exists(), f"Falta el archivo {path}"
        X, y = load_xy(path)
        assert len(X) > 500
        assert X.isna().sum().sum() == 0
        assert (y > 0).all()


def test_2_champion_model_meets_r2_quality_gate():
    """Verifica que el modelo Champion entrenado supere el R2 mínimo de 0.70."""
    assert config.CHAMPION_MODEL_PATH.exists(), "Falta entrenar el modelo Champion"
    res = client.get("/model-info")
    assert res.status_code == 200
    info = res.json()
    assert info["metrics"]["r2"] >= config.MIN_R2_SCORE


def test_3_api_predict_endpoint_works():
    """Verifica que el endpoint de predicción responda correctamente."""
    payload = {
        "mes": 3,
        "sector": "AGROPECUARIO",
        "mercado_destino": "Estados Unidos",
        "peso_neto_kg": 5450.0,
        "cantidad": 5450.0,
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["predicted_fob_usd"] > 0
    assert body["shipment_category"] in ["Small Export", "Medium Export", "High-Value Export"]


def test_4_api_rejects_invalid_month():
    """Verifica que la validación bloquee datos erróneos (ej. mes 15)."""
    payload = {
        "mes": 15,
        "sector": "AGROPECUARIO",
        "mercado_destino": "Estados Unidos",
        "peso_neto_kg": 1000.0,
        "cantidad": 100.0,
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 422