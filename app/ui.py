"""Interfaz Nativa de Streamlit con Pestaña Gráfica de Arquitectura y Proceso MLOps."""
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
        "tab_calc": "Calculadora Predictiva FOB",
        "tab_process": "Arquitectura, Modelos y Flujo MLOps",
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
        "arch_title": "1. Flujo Gráfico del Sistema (ML Workflow + CI/CT/CD)",
        "arch_desc": "Diagrama en tiempo real de cómo se procesan los datos de exportación, cómo aprende el modelo supervisado y cómo se despliega automáticamente en producción:",
        "models_title": "2. Desglose Matemático del Ensamble (Modelos de Clase)",
        "m1_title": "Preprocesamiento y Escalamiento",
        "m1_desc": "Aplica **StandardScaler** (media 0, desviación 1) y **PolynomialFeatures (Grado 2)** para capturar relaciones no lineales multiplicando características entre sí, además de **OneHotEncoder** para Sector y Mercado Destino.",
        "m2_title": "Modelo Base 1: KNN Regressor",
        "m2_desc": "Algoritmo *lazy* basado en ejemplos que evalúa la vecindad de cada nuevo embarque calculando distancias euclidianas ponderadas por el inverso de la distancia (`weights='distance'`).",
        "m3_title": "Modelo Base 2: Regresión Lasso (L1)",
        "m3_desc": "Minimiza los residuos al cuadrado (OLS) añadiendo una penalización sobre el valor absoluto de los coeficientes (`alpha * sum(|a_i|)`), realizando selección automática de características al reducir el ruido a cero.",
        "m4_title": "Meta-Modelo: Regresión Ridge (L2)",
        "m4_desc": "Combina las predicciones de KNN y Lasso aplicando regularización cuadrática (`alpha * sum(a_i^2)`) para evitar el sobreajuste (*overfitting*) y entregar la predicción final en dólares FOB.",
        "eval_title": "3. Evaluación Gráfica y Comparativa de Métricas",
        "plot_caption": "Gráfico de dispersión en el conjunto de prueba (Hold-out 20%): Valor FOB Real vs. Valor FOB Predicho.",
        "comp_caption": "Comparación de R² Score: Regresión Lineal Simple (OLS) vs. Ensamble Stacking Optimizado.",
        "reg_title": "4. Historial de Versiones y Puerta de Promoción (Registro MLOps)",
        "reg_desc": "Cada vez que el pipeline de GitHub Actions se ejecuta, evalúa si el nuevo modelo supera el piso mínimo (`R² >= 0.70`) antes de promoverlo a producción:",
    },
    "English": {
        "title": "ExpoPredict Puno: Export FOB Value Estimator",
        "subtitle": (
            "**Intelligent Commercial Valuation System for Regional Exporters in Puno, Peru.**  \n"
            "*Powered by a Supervised Learning Ensemble: K-Nearest Neighbors (KNN) + Lasso Regression + Ridge Regression.*"
        ),
        "tab_calc": "FOB Predictive Calculator",
        "tab_process": "Architecture, Models & MLOps Flow",
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
        "arch_title": "1. Visual System Workflow (ML Workflow + CI/CT/CD)",
        "arch_desc": "Real-time diagram showing how export data is processed, how the supervised ensemble learns, and how it is automatically deployed to production:",
        "models_title": "2. Mathematical Ensemble Breakdown (Course Models)",
        "m1_title": "Preprocessing & Feature Engineering",
        "m1_desc": "Applies **StandardScaler** (mean 0, std 1) and **PolynomialFeatures (Degree 2)** to capture non-linear feature interactions, plus **OneHotEncoder** for Sector and Destination Market.",
        "m2_title": "Base Model 1: KNN Regressor",
        "m2_desc": "Instance-based *lazy* algorithm that explores the neighborhood of each shipment using inverse-distance weighted Euclidean distance (`weights='distance'`).",
        "m3_title": "Base Model 2: Lasso Regression (L1)",
        "m3_desc": "Minimizes Ordinary Least Squares (OLS) residuals with an L1 penalty (`alpha * sum(|a_i|)`), performing automatic feature selection by shrinking irrelevant coefficients to zero.",
        "m4_title": "Meta-Model: Ridge Regression (L2)",
        "m4_desc": "Combines KNN and Lasso predictions using L2 quadratic regularization (`alpha * sum(a_i^2)`) to prevent overfitting and output the final FOB USD valuation.",
        "eval_title": "3. Visual Evaluation & Metric Comparison",
        "plot_caption": "Hold-out Test Set (20%) Scatter Plot: Real FOB Value vs. Predicted FOB Value.",
        "comp_caption": "R² Score Comparison: Simple Linear Regression (OLS Baseline) vs. Tuned Stacking Ensemble.",
        "reg_title": "4. Model Registry History & Promotion Gate (MLOps)",
        "reg_desc": "Every time the GitHub Actions pipeline runs, it verifies whether the candidate model beats the quality floor (`R² >= 0.70`) before promoting it to production:",
    },
}

reg_data = {"current_champion": None, "history": []}
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

# Barra lateral nativa
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

# Pestañas principales: 1) Calculadora, 2) Cómo funciona todo el proceso gráficamente
tab1, tab2 = st.tabs([t["tab_calc"], t["tab_process"]])

with tab1:
    st.subheader(t["profile_header"])
    u_col1, u_col2, u_col3 = st.columns([2, 2, 3])
    with u_col1:
        user_role = st.selectbox(t["role_label"], options=t["roles"])
    with u_col2:
        user_name = st.text_input(t["company_label"], value="Puno AgroExport SAC")
    with u_col3:
        st.write("")
        st.info(t["welcome_msg"].format(company=user_name, role=user_role))

    st.subheader(t["form_title"])
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        sector = col1.selectbox(t["sector_label"], options=config.SECTORS, index=0)
        market = col1.selectbox(t["market_label"], options=config.TOP_MARKETS, index=2)
        month = col1.slider(t["month_label"], min_value=1, max_value=12, value=4)
        weight = col2.number_input(t["weight_label"], min_value=1.0, max_value=500000.0, value=5450.0, step=100.0)
        quantity = col2.number_input(t["qty_label"], min_value=1.0, max_value=500000.0, value=5450.0, step=50.0)

        submitted = st.form_submit_button(t["submit_btn"], type="primary")

    # Inicializar el historial de sesión si no existe
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

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

        # Registrar la consulta actual en la memoria de la sesión (Tiempo Real)
        st.session_state.prediction_history.append({
            "Consulta #": len(st.session_state.prediction_history) + 1,
            "Sector": sector,
            "Destino": market,
            "Peso (kg)": weight,
            "FOB Estimado (USD)": fob_val
        })

        st.subheader(f"{t['report_title']} — {user_name}")
        r1, r2, r3 = st.columns(3)
        r1.metric(t["metric_fob"], f"${fob_val:,.2f}")
        r2.metric(t["metric_scale"], cat_label)
        r3.metric(t["metric_engine"], f"{version_used} ({source})")

        if fob_val >= 100000:
            st.success(t["insight_high"].format(market=market))
        else:
            st.info(t["insight_std"].format(sector=sector, market=market))

    # --- GRÁFICO EN TIEMPO REAL DEL HISTORIAL DE SESIÓN ---
    if st.session_state.prediction_history:
        st.divider()
        st.subheader("📈 Real-Time Session Monitoring (Live Chart)")
        st.caption("Gráfico dinámico que registra la evolución de las estimaciones realizadas durante la sesión actual.")
        
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        # Gráfico de líneas interactivo de Streamlit
        st.line_chart(history_df.set_index("Consulta #")["FOB Estimado (USD)"])
        
        with st.expander("Ver tabla de consultas en tiempo real"):
            st.dataframe(history_df, use_container_width=True)

with tab2:
    # 1. Diagrama Gráfico del Flujo Completo
    st.subheader(t["arch_title"])
    st.write(t["arch_desc"])

    graphviz_dot = """
    digraph MLOpsFlow {
        rankdir=LR;
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=10];

        subgraph cluster_data {
            label = "1. Data & Preprocessing";
            style = "dashed";
            raw [label="Puno Exports Dataset\\n(7,346 Shipments)", fillcolor="#E3F2FD", color="#1E88E5"];
            split [label="Hold-out Split (src/data.py)\\n80% Train / 20% Fixed Test", fillcolor="#E3F2FD", color="#1E88E5"];
            prep [label="StandardScaler +\\nPolynomialFeatures (deg=2)\\n+ OneHotEncoder", fillcolor="#BBDEFB", color="#1976D2"];
            raw -> split -> prep;
        }

        subgraph cluster_model {
            label = "2. Supervised Ensemble (src/train.py)";
            style = "dashed";
            knn [label="Base 1: KNN Regressor\\n(Euclidean Distance, k=7)", fillcolor="#FFF3E0", color="#FB8C00"];
            lasso [label="Base 2: Lasso Regression\\n(L1 Penalty Feature Selection)", fillcolor="#FFF3E0", color="#FB8C00"];
            ridge [label="Meta-Model: Ridge (L2)\\n(Stacking Regressor + 5-Fold CV)", fillcolor="#FFE0B2", color="#F57C00"];
            prep -> knn -> ridge;
            prep -> lasso -> ridge;
        }

        subgraph cluster_mlops {
            label = "3. CI/CT/CD & Production";
            style = "dashed";
            gate [label="Promotion Gate\\nR² >= 0.70 & Pytest CI", shape=diamond, fillcolor="#E8F5E9", color="#43A047"];
            champ [label="Champion Registry\\n(champion_model.joblib)", fillcolor="#C8E6C9", color="#2E7D32"];
            prod [label="Production App\\nFastAPI + Streamlit Cloud", fillcolor="#D1C4E9", color="#5E35B1"];
            ridge -> gate;
            gate -> champ [label="Promoted"];
            champ -> prod [label="Auto-Deploy"];
        }
    }
    """
    st.graphviz_chart(graphviz_dot, use_container_width=True)

    st.divider()

    # 2. Tarjetas explicativas de los modelos de clase
    st.subheader(t["models_title"])
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.info(f"**{t['m1_title']}**\n\n{t['m1_desc']}")
    with m_col2:
        st.info(f"**{t['m2_title']}**\n\n{t['m2_desc']}")
    with m_col3:
        st.info(f"**{t['m3_title']}**\n\n{t['m3_desc']}")
    with m_col4:
        st.info(f"**{t['m4_title']}**\n\n{t['m4_desc']}")

    st.divider()

    # 3. Gráficos de Evaluación y Comparación
    st.subheader(t["eval_title"])
    g_col1, g_col2 = st.columns(2)

    with g_col1:
        if config.PLOT_PATH.exists():
            st.image(str(config.PLOT_PATH), caption=t["plot_caption"], use_container_width=True)
        else:
            st.warning("Plot not found.")

    with g_col2:
        st.write(f"**{t['comp_caption']}**")
        if registry_info:
            comp_df = pd.DataFrame(
                {
                    "Model": [
                        "1. Baseline OLS (Linear)",
                        "2. Minimum Quality Gate",
                        "3. Stacking Champion (KNN+Lasso->Ridge)",
                    ],
                    "R2 Score": [
                        registry_info.get("baseline_ols_r2", 0.2888),
                        config.MIN_R2_SCORE,
                        registry_info["metrics"]["r2"],
                    ],
                }
            ).set_index("Model")
            st.bar_chart(comp_df, height=320)

    st.divider()

    # 4. Tabla en vivo del Registro de Modelos (MLOps History)
    st.subheader(t["reg_title"])
    st.write(t["reg_desc"])
    if reg_data.get("history"):
        rows = []
        for item in reg_data["history"]:
            rows.append(
                {
                    "Version": item["version"],
                    "Timestamp": item["timestamp"],
                    "Training Batch": item["train_file"],
                    "Rows": item["train_rows"],
                    "Hold-out R²": item["metrics"]["r2"],
                    "RMSE (USD)": item["metrics"]["rmse"],
                    "MAE (USD)": item["metrics"]["mae"],
                    "Promoted": "✅ YES" if item["promoted"] else "❌ REJECTED",
                    "Reason": item["reason"],
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)