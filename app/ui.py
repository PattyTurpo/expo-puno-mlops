"""Frontend en inglés con Streamlit para el usuario final (Exportadores de Puno)."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import requests
import streamlit as st
from src import config

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="ExpoPredict Puno - FOB Estimator", layout="wide")
st.title("📦 ExpoPredict Puno: Export FOB Value Estimator")
st.markdown(
    "**Intelligent Commercial Valuation System for Regional Exporters in Puno, Peru.**  \n"
    "*Powered by a Supervised Learning Ensemble: K-Nearest Neighbors (KNN) + Lasso Regression + Ridge Regression.*"
)

# Cargar información del registro MLOps
registry_info = None
if config.REGISTRY_JSON_PATH.exists():
    reg_data = json.loads(config.REGISTRY_JSON_PATH.read_text(encoding="utf-8"))
    registry_info = reg_data.get("current_champion")

with st.sidebar:
    st.header("⚙️ MLOps Model Status")
    if registry_info:
        st.success(f"Active Champion: **{registry_info['version']}**")
        st.write(f"**Hold-out R² Score:** `{registry_info['metrics']['r2']}`")
        st.write(f"**5-Fold CV R² Score:** `{registry_info['cv_5fold_r2']}`")
        st.write(f"**Hold-out RMSE:** `${registry_info['metrics']['rmse']:,.2f} USD`")
        st.write(f"**Hold-out MAE:** `${registry_info['metrics']['mae']:,.2f} USD`")
        st.caption(f"Last trained: {registry_info['timestamp']}")
        with st.expander("Optimized Hyperparameters"):
            st.json(registry_info["best_hyperparameters"])
    else:
        st.warning("No model registered yet. Run `python -m src.train`.")

st.subheader("Enter Shipment Characteristics (Features)")
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    sector = col1.selectbox("Economic Sector", options=config.SECTORS, index=0)
    market = col1.selectbox("Destination Market", options=config.TOP_MARKETS, index=2)
    month = col1.slider("Shipment Month", min_value=1, max_value=12, value=3)
    weight = col2.number_input("Net Weight (kg)", min_value=1.0, max_value=500000.0, value=5450.0, step=100.0)
    quantity = col2.number_input("Quantity (Commercial Units)", min_value=1.0, max_value=500000.0, value=5450.0, step=50.0)

    submitted = st.form_submit_button("Predict FOB Value (USD)", type="primary")

if submitted:
    payload = {
        "mes": int(month),
        "sector": sector,
        "mercado_destino": market,
        "peso_neto_kg": float(weight),
        "cantidad": float(quantity),
    }

    # Intenta usar la API FastAPI; si no está prendida, usa el modelo local directamente
    try:
        res = requests.post(f"{API_URL}/predict", json=payload, timeout=3)
        res.raise_for_status()
        out = res.json()
        source = "FastAPI Backend Service"
    except Exception:
        model = joblib.load(config.CHAMPION_MODEL_PATH)
        df_in = pd.DataFrame([payload])[config.FEATURES]
        val = max(10.0, round(float(model.predict(df_in)[0]), 2))
        cat = "High-Value Export" if val >= 200000 else ("Medium Export" if val >= 20000 else "Small Export")
        out = {
            "predicted_fob_usd": val,
            "shipment_category": cat,
            "model_version": registry_info["version"] if registry_info else "v1",
        }
        source = "Embedded Champion Model"

    st.subheader("📊 Prediction Result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated FOB Value (USD)", f"${out['predicted_fob_usd']:,.2f}")
    c2.metric("Commercial Category", out["shipment_category"])
    c3.metric("Served By", f"{out['model_version']} ({source})")