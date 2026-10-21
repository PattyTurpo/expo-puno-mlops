"""Interfaz Nativa de Streamlit con Selector de Idioma (Español / English) para ExpoPredict Puno."""
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

st.set_page_config(
    page_title="ExpoPredict Puno - FOB Estimator",
    layout="wide",
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

TEXTS = {
    "Español": {
        "title": "ExpoPredict Puno: Estimador de Valor FOB de Exportación",
        "subtitle": (
            "**Sistema Inteligente de Valoración Comercial para Exportadores de Puno, Perú.**  \n"
            "*Impulsado por un Ensamble de Aprendizaje Supervisado: K-Nearest Neighbors (KNN) + Regresión Lasso + Regresión Ridge.*"
        ),
        "profile_header": "Identificación del Operador Comercial",
        "role_label": "Perfil del Interesado",
        "roles": [
            "Exportador / Productor Regional",
            "Analista de Inteligencia Comercial (PROMPERÚ)",
            "Gerente de Logística y Aduanas",
        ],
        "company_label": "Razón Social / Empresa",
        "welcome_msg": "Sesión activa: **{company}** ({role})",
        "sidebar_title": "Estado del Modelo MLOps",
        "active_champ": "Champion Activo",
        "r2_holdout": "Hold-out R² Score",
        "cv_r2": "5-Fold CV R² Score",
        "rmse_label": "Hold-out RMSE",
        "mae_label": "Hold-out MAE",
        "last_trained": "Último entrenamiento",
        "hyperparams": "Hiperparámetros Optimizados",
        "no_model": "No hay modelo registrado. Ejecute `python -m src.train`.",
        "form_title": "Características del Embarque (Features)",
        "sector_label": "Sector Económico",
        "market_label": "Mercado de Destino",
        "month_label": "Mes de Embarque",
        "weight_label": "Peso Neto (kg)",
        "qty_label": "Cantidad (Unidades Comerciales)",
        "submit_btn": "Predecir Valor FOB (USD)",
        "report_title": "Resultado de la Predicción",
        "metric_fob": "Valor FOB Estimado (USD)",
        "metric_scale": "Categoría Comercial",
        "metric_engine": "Servido Por",
        "cat_high": "Exportación de Alto Valor",
        "cat_med": "Exportación Mediana",
        "cat_small": "Exportación Pequeña",
        "insight_high": "Nota Comercial: Embarque de alto valor proyectado hacia **{market}**. Se sugiere seguro logístico internacional.",
        "insight_std": "Nota Comercial: Operación dentro del rango estándar para el sector **{sector}** hacia **{market}**.",
    },
    "English": {
        "title": "ExpoPredict Puno: Export FOB Value Estimator",
        "subtitle": (
            "**Intelligent Commercial Valuation System for Regional Exporters in Puno, Peru.**  \n"
            "*Powered by a Supervised Learning Ensemble: K-Nearest Neighbors (KNN) + Lasso Regression + Ridge Regression.*"
        ),
        "profile_header": "Stakeholder Identification",
        "role_label": "Stakeholder Profile",
        "roles": [
            "Regional Exporter / Producer",
            "Trade Intelligence Analyst (PROMPERÚ)",
            "Logistics & Customs Manager",
        ],
        "company_label": "Company Name",
        "welcome_msg": "Active session: **{company}** ({role})",
        "sidebar_title": "MLOps Model Status",
        "active_champ": "Active Champion",
        "r2_holdout": "Hold-out R² Score",
        "cv_r2": "5-Fold CV R² Score",
        "rmse_label": "Hold-out RMSE",
        "mae_label": "Hold-out MAE",
        "last_trained": "Last trained",
        "hyperparams": "Optimized Hyperparameters",
        "no_model": "No model registered yet. Run `python -m src.train`.",
        "form_title": "Enter Shipment Characteristics (Features)",
        "sector_label": "Economic Sector",
        "market_label": "Destination Market",
        "month_label": "Shipment Month",
        "weight_label": "Net Weight (kg)",
        "qty_label": "Quantity (Commercial Units)",
        "submit_btn": "Predict FOB Value (USD)",
        "report_title": "Prediction Result",
        "metric_fob": "Estimated FOB Value (USD)",
        "metric_scale": "Commercial Category",
        "metric_engine": "Served By",
        "cat_high": "High-Value Export",
        "cat_med": "Medium Export",
        "cat_small": "Small Export",
        "insight_high": "Commercial Insight: High-value shipment detected for **{market}**. Consider specialized logistics insurance.",
        "insight_std": "Commercial Insight: Standard shipment operation for **{sector}** targeting **{market}**.",
    },
}

registry_info = None
if config.REGISTRY_JSON_PATH.exists():
    reg_data = json.loads(config.REGISTRY_JSON_PATH.read_text(encoding="utf-8"))
    registry_info = reg_data.get("current_champion")

# Selector de idioma arriba a la derecha
head_col1, head_col2 = st.columns([4, 1])
with head_col2:
    lang = st.selectbox("Language / Idioma", options=["English", "Español"], index=0)

t = TEXTS[lang]

with head_col1:
    st.title(t["title"])
    st.markdown(t["subtitle"])

# Barra lateral nativa (igual que antes)
with st.sidebar:
    st.header(t["sidebar_title"])
    if registry_info:
        st.success(f"{t['active_champ']}: **{registry_info['version']}**")
        st.write(f"**{t['r2_holdout']}:** `{registry_info['metrics']['r2']}`")
        st.write(f"**{t['cv_r2']}:** `{registry_info['cv_5fold_r2']}`")
        st.write(f"**{t['rmse_label']}:** `${registry_info['metrics']['rmse']:,.2f} USD`")
        st.write(f"**{t['mae_label']}:** `${registry_info['metrics']['mae']:,.2f} USD`")
        st.caption(f"{t['last_trained']}: {registry_info['timestamp']}")
        with st.expander(t["hyperparams"]):
            st.json(registry_info["best_hyperparameters"])
    else:
        st.warning(t["no_model"])

# Interacción con el interesado
st.divider()
st.subheader(t["profile_header"])
u_col1, u_col2, u_col3 = st.columns([2, 2, 3])
with u_col1:
    user_role = st.selectbox(t["role_label"], options=t["roles"])
with u_col2:
    user_name = st.text_input(t["company_label"], value="Puno AgroExport SAC")
with u_col3:
    st.write("")
    st.info(t["welcome_msg"].format(company=user_name, role=user_role))

# Formulario de predicción nativo
st.subheader(t["form_title"])
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    sector = col1.selectbox(t["sector_label"], options=config.SECTORS, index=0)
    market = col1.selectbox(t["market_label"], options=config.TOP_MARKETS, index=2)
    month = col1.slider(t["month_label"], min_value=1, max_value=12, value=4)
    weight = col2.number_input(t["weight_label"], min_value=1.0, max_value=500000.0, value=5450.0, step=100.0)
    quantity = col2.number_input(t["qty_label"], min_value=1.0, max_value=500000.0, value=5450.0, step=50.0)

    submitted = st.form_submit_button(t["submit_btn"], type="primary")

if submitted:
    payload = {
        "mes": int(month),
        "sector": sector,
        "mercado_destino": market,
        "peso_neto_kg": float(weight),
        "cantidad": float(quantity),
    }

    try:
        res = requests.post(f"{API_URL}/predict", json=payload, timeout=3)
        res.raise_for_status()
        out = res.json()
        fob_val = float(out["predicted_fob_usd"])
        version_used = out["model_version"]
        source = "FastAPI Backend Service"
    except Exception:
        model = joblib.load(config.CHAMPION_MODEL_PATH)
        df_in = pd.DataFrame([payload])[config.FEATURES]
        fob_val = max(10.0, round(float(model.predict(df_in)[0]), 2))
        version_used = registry_info["version"] if registry_info else "v1"
        source = "Embedded Champion Model"

    if fob_val >= 200000:
        cat_label = t["cat_high"]
    elif fob_val >= 20000:
        cat_label = t["cat_med"]
    else:
        cat_label = t["cat_small"]

    st.subheader(f"{t['report_title']} — {user_name}")
    c1, c2, c3 = st.columns(3)
    c1.metric(t["metric_fob"], f"${fob_val:,.2f}")
    c2.metric(t["metric_scale"], cat_label)
    c3.metric(t["metric_engine"], f"{version_used} ({source})")

    if fob_val >= 100000:
        st.success(t["insight_high"].format(market=market))
    else:
        st.info(t["insight_std"].format(sector=sector, market=market))