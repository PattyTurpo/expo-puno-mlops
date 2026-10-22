# 📦 ExpoPredict Puno: Export FOB Value Estimator (MLOps System)

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Stacking%20Ensemble-orange.svg)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit%20Cloud-FF4B4B.svg)
![CI/CT/CD](https://img.shields.io/badge/MLOps-GitHub%20Actions%20%7C%20Pytest-success.svg)

**Intelligent Commercial Valuation System for Regional Exporters in Puno, Peru.**  
*Sistema Inteligente de Valoración Comercial para Exportadores de la Región Puno, Perú.*

---

## 🌐 Overview | Descripción General

**English:**  
**ExpoPredict Puno** is a production-ready web application that operationalizes business logic through real-time machine learning predictions. Built with **7,346 real customs export records** from the Puno region (January – June 2024), the system predicts the **FOB Value (USD)** of an export shipment based on 5 operational features (`mes`, `sector`, `mercado_destino`, `peso_neto_kg`, and `cantidad`). It integrates a complete **DevOps & MLOps lifecycle** featuring automated unit testing (CI), continuous model retraining (CT), an automated Promotion Gate ($R^2 \ge 0.70$), and continuous cloud deployment (CD).

**Español:**  
**ExpoPredict Puno** es una aplicación en producción cuya lógica de negocio se operativiza mediante predicciones realizadas por un modelo de aprendizaje supervisado. Entrenado con **7,346 registros reales de exportación** de la región Puno (enero – junio 2024), el sistema estima el **Valor FOB (USD)** de un embarque y cuenta con flujos automatizados de **Integración Continua (CI)**, **Reentrenamiento Continuo (CT)**, **Puerta de Promoción MLOps** ($R^2 \ge 0.70$) y **Despliegue Continuo (CD)**.

---

## 🧠 Supervised Learning Architecture | Arquitectura del Modelo

Following the mathematical foundations covered in the **Machine Learning** course (UNA Puno), this project implements a **`StackingRegressor`** pipeline in `scikit-learn`:

1. **Preprocessing & Feature Engineering (`build_preprocessor`):**
   * **`StandardScaler`:** Centers numerical features to mean $\mu = 0$ and standard deviation $\sigma = 1$.
   * **`PolynomialFeatures(degree=2)`:** Captures non-linear interactions (second-order Taylor expansion terms such as $\text{weight} \times \text{quantity}$ and $\text{weight}^2$).
   * **`OneHotEncoder`:** Encodes categorical features (`sector` and `mercado_destino`).
2. **Base Model 1 — `KNeighborsRegressor` (KNN):**
   * Instance-based *lazy learner* that explores the neighborhood of each shipment using inverse-distance weighted Euclidean distance ($k=7$, `weights="distance"`).
3. **Base Model 2 — `Lasso` Regression (L1 Regularization):**
   * Minimizes Ordinary Least Squares residuals plus an L1 penalty ($\text{RSS} + \alpha \sum |a_i|$), performing automatic feature selection by shrinking irrelevant coefficients to zero.
4. **Meta-Model — `Ridge` Regression (L2 Regularization):**
   * Combines the predictions of KNN and Lasso using quadratic regularization ($\text{RSS} + \alpha \sum a_i^2$) to prevent overfitting and output the final FOB valuation in USD.

### 📊 Hold-out Test Set Performance (`test.csv` — 1,470 unseen shipments)

| Model / Stage | Hold-out $R^2$ Score | Hold-out RMSE (USD) | Hold-out MAE (USD) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline OLS (Simple Linear Regression)** | `0.2888` | `$301,420.15` | `$118,340.50` | Baseline |
| **Minimum MLOps Promotion Gate** | `0.7000` | — | — | Threshold |
| **Champion `v1` (`train_v1.csv` - 4,407 rows)** | **`0.8314`** | **`$146,843.43`** | **`$26,897.13`** | ✅ Promoted |
| **Champion `v2` (`train_v2.csv` - 5,876 rows)** | **`0.9209`** | **`$100,581.93`** | **`$21,668.52`** | ✅ Promoted |

---

## 📁 Repository Structure | Organización del Código Fuente

```text
expo-puno-mlops/
├── .github/workflows/
│   └── ci_cd_mlops.yml          # Automated CI/CT/CD pipeline in GitHub Actions
├── app/
│   ├── __init__.py
│   ├── schemas.py               # Pydantic input/output validation schemas
│   ├── main.py                  # FastAPI REST service (/health, /model-info, /predict)
│   └── ui.py                    # Bilingual Streamlit frontend + Live Charts + MLOps Tab
├── src/
│   ├── __init__.py
│   ├── config.py                # Central paths, features, sectors, markets & MIN_R2_SCORE = 0.70
│   ├── data.py                  # Excel cleaning & Hold-out split (train_v1, train_v2, train_bad, test)
│   └── train.py                 # Stacking (KNN + Lasso -> Ridge), 5-Fold CV & Promotion Gate
├── data/splits/
│   ├── test.csv                 # Fixed 20% hold-out test set (1,470 rows)
│   ├── train_v1.csv             # Initial training batch (4,407 rows)
│   ├── train_v2.csv             # Updated training batch for CT demo (5,876 rows)
│   ├── train_bad.csv            # Corrupted/shuffled target batch (rejected by Promotion Gate)
│   └── train_current.csv        # Active training dataset
├── models/
│   ├── champion_model.joblib    # Serialized production Champion pipeline
│   ├── model_registry.json      # MLOps version history, hyperparameters & evaluation metrics
│   └── predictions_vs_actual.png# Scatter plot on hold-out test set
├── tests/
│   └── test_project.py          # 4 automated unit & quality tests (Pytest)
├── Dockerfile                   # Container image definition (FastAPI + Streamlit)
└── requirements.txt             # Python dependencies
```

---

## 🚀 Quickstart & Local Deployment | Instalación y Ejecución Local

### 1. Clone the repository & install dependencies
```powershell
git clone https://github.com/PattyTurpo/expo-puno-mlops.git
cd expo-puno-mlops
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run MLOps Training & Promotion Gate
```powershell
# Train with initial batch (v1)
python -m src.train --train data/splits/train_v1.csv

# Retrain with new customs batch (v2)
python -m src.train --train data/splits/train_v2.csv

# Test Promotion Gate protection against corrupted data (Rejected automatically)
python -m src.train --train data/splits/train_bad.csv
```

### 3. Run Continuous Integration (CI) Tests
```powershell
python -m pytest -v
```

### 4. Launch Backend API & Web Interface
```powershell
# Terminal 1: FastAPI Backend (http://127.0.0.1:8000/docs)
uvicorn app.main:app --port 8000

# Terminal 2: Streamlit Web App (http://localhost:8501)
python -m streamlit run app/ui.py
```

---

## ⚙️ Automated CI/CT/CD Pipeline (`.github/workflows/ci_cd_mlops.yml`)

1. **Continuous Integration (CI):** Runs on every `push` and `pull_request` to `main`. Executes `pytest -v` to verify dataset integrity, model quality ($R^2 \ge 0.70$), `/predict` response, and HTTP `422` validation on invalid inputs.
2. **Continuous Training (CT):** Can be triggered automatically or manually via `workflow_dispatch` by selecting `train_v1.csv`, `train_v2.csv`, or `train_bad.csv`. Optimizes hyperparameters via **5-Fold Cross-Validation**.
3. **Promotion Gate & Continuous Deployment (CD):** Only models that pass the quality threshold update `models/champion_model.joblib` and `models/model_registry.json`, automatically triggering a live update on **Streamlit Cloud**.
