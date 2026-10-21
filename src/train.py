"""Paso 3 y 4: Entrenamiento, Optimización K-Fold CV, Evaluación y Registro MLOps."""
import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from src import config


def load_xy(csv_path: Path):
    df = pd.read_csv(csv_path)
    X = df[config.FEATURES].copy()
    y = df[config.TARGET].astype(float)
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Normalización (media=0, std=1) y Características Polinómicas (grado 2) vistas en clase."""
    num_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, config.NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), config.CATEGORICAL_FEATURES),
        ]
    )


def build_ensemble(params: dict) -> Pipeline:
    """Ensamble Stacking con los 3 modelos de clase: KNN + Lasso -> Ridge."""
    base_models = [
        (
            "knn",
            KNeighborsRegressor(
                n_neighbors=params["knn_k"],
                weights=params["knn_weights"],
            ),
        ),
        (
            "lasso",
            Lasso(
                alpha=params["lasso_alpha"],
                max_iter=5000,
                tol=1e-2,
                random_state=config.SEED,
            ),
        ),
    ]
    meta_model = Ridge(alpha=params["ridge_alpha"], random_state=config.SEED)

    stacking = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=3,
        n_jobs=1,
    )
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", stacking),
    ])


def evaluate_metrics(y_true, y_pred) -> dict:
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "rmse": round(float(root_mean_squared_error(y_true, y_pred)), 2),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
    }


def optimize_with_kfold(X_train, y_train) -> tuple[dict, float]:
    """Búsqueda de hiperparámetros óptimos (k en KNN, alpha en Lasso y Ridge) con 5-Fold CV."""
    kf = KFold(n_splits=5, shuffle=True, random_state=config.SEED)
    grid = [
        {"knn_k": 3, "knn_weights": "distance", "lasso_alpha": 10.0, "ridge_alpha": 1.0},
        {"knn_k": 5, "knn_weights": "distance", "lasso_alpha": 50.0, "ridge_alpha": 1.0},
        {"knn_k": 7, "knn_weights": "distance", "lasso_alpha": 100.0, "ridge_alpha": 10.0},
        {"knn_k": 5, "knn_weights": "uniform", "lasso_alpha": 20.0, "ridge_alpha": 5.0},
    ]

    best_cv_r2 = -float("inf")
    best_params = grid[0]

    for candidate in grid:
        fold_scores = []
        for tr_idx, val_idx in kf.split(X_train, y_train):
            model = build_ensemble(candidate)
            model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            preds = model.predict(X_train.iloc[val_idx])
            fold_scores.append(r2_score(y_train.iloc[val_idx], preds))
        mean_score = float(np.mean(fold_scores))
        if mean_score > best_cv_r2:
            best_cv_r2 = mean_score
            best_params = candidate

    return best_params, round(best_cv_r2, 4)


def save_plot(y_true, y_pred, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.4, color="royalblue", edgecolors="k", linewidth=0.3)
    max_val = max(float(y_true.max()), float(y_pred.max()))
    ax.plot([0, max_val], [0, max_val], "r--", lw=2, label="Ajuste Ideal (Residuo = 0)")
    ax.set_xlabel("Valor FOB Real (USD)")
    ax.set_ylabel("Valor FOB Predicho (USD)")
    ax.set_title("Predicción vs Valor Real (Conjunto de Prueba)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train_and_evaluate(train_file: str) -> dict:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, y_train = load_xy(Path(train_file))
    X_test, y_test = load_xy(config.TEST_PATH)

    # 1. Modelo Base Lineal Simple (OLS) para comparar
    ols_baseline = make_pipeline(build_preprocessor(), LinearRegression()).fit(X_train, y_train)
    ols_r2 = round(float(r2_score(y_test, ols_baseline.predict(X_test))), 4)

    # 2. Optimización de hiperparámetros con Validación Cruzada (5-Fold CV)
    best_params, cv_r2 = optimize_with_kfold(X_train, y_train)

    # 3. Entrenamiento del modelo final ensamblado
    final_model = build_ensemble(best_params)
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)
    metrics = evaluate_metrics(y_test, y_pred)

    # 4. Puerta de Promoción MLOps (Compara contra el modelo Champion actual)
    registry = {"current_champion": None, "history": []}
    if config.REGISTRY_JSON_PATH.exists():
        registry = json.loads(config.REGISTRY_JSON_PATH.read_text(encoding="utf-8"))

    champ = registry.get("current_champion")
    champ_r2 = champ["metrics"]["r2"] if champ else None

    if metrics["r2"] < config.MIN_R2_SCORE:
        promoted = False
        reason = f"RECHAZADO: R2 ({metrics['r2']}) es menor al mínimo permitido ({config.MIN_R2_SCORE})."
    elif champ_r2 is None or metrics["r2"] >= (champ_r2 - config.TOLERANCE):
        promoted = True
        reason = f"PROMOVIDO: R2 ({metrics['r2']}) supera el control de calidad (Champion previo: {champ_r2})."
    else:
        promoted = False
        reason = f"RECHAZADO: R2 ({metrics['r2']}) es inferior al Champion actual ({champ_r2})."

    new_version = len(registry["history"]) + 1
    run_record = {
        "version": f"v{new_version}",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_file": str(train_file),
        "train_rows": len(X_train),
        "architecture": "StackingRegressor (Polynomial + KNN + Lasso -> Ridge)",
        "best_hyperparameters": best_params,
        "cv_5fold_r2": cv_r2,
        "baseline_ols_r2": ols_r2,
        "metrics": metrics,
        "promoted": promoted,
        "reason": reason,
    }

    registry["history"].append(run_record)

    if promoted:
        registry["current_champion"] = run_record
        joblib.dump(final_model, config.CHAMPION_MODEL_PATH)
        save_plot(y_test, y_pred, config.PLOT_PATH)

    config.REGISTRY_JSON_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(json.dumps(run_record, indent=2))
    return run_record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(config.CURRENT_TRAIN_PATH))
    args = parser.parse_args()
    train_and_evaluate(args.train)