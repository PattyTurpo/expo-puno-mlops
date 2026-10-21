"""Configuración central del proyecto ExpoPredict Puno."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_EXCEL = DATA_DIR / "raw" / "RptExportaciones_puno.xlsx"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = ROOT / "models"

TRAIN_V1_PATH = SPLITS_DIR / "train_v1.csv"
TRAIN_V2_PATH = SPLITS_DIR / "train_v2.csv"
TRAIN_BAD_PATH = SPLITS_DIR / "train_bad.csv"
CURRENT_TRAIN_PATH = SPLITS_DIR / "train_current.csv"
TEST_PATH = SPLITS_DIR / "test.csv"

CHAMPION_MODEL_PATH = MODELS_DIR / "champion_model.joblib"
REGISTRY_JSON_PATH = MODELS_DIR / "model_registry.json"
PLOT_PATH = MODELS_DIR / "predictions_vs_actual.png"

TARGET = "monto_fob"
FEATURES = ["mes", "sector", "mercado_destino", "peso_neto_kg", "cantidad"]
NUMERIC_FEATURES = ["mes", "peso_neto_kg", "cantidad"]
CATEGORICAL_FEATURES = ["sector", "mercado_destino"]

SECTORS = [
    "AGROPECUARIO",
    "MINEROS",
    "PESQUERO",
    "TEXTIL",
    "VARIOS (inc. joyería)",
    "QUÍMICO",
    "MINERÍA NO METÁLICA",
    "METAL-MECÁNICO",
    "SIDERO-METALÚRGICO",
    "PIELES Y CUEROS",
]

TOP_MARKETS = [
    "India",
    "Emiratos Árabes Unidos",
    "Estados Unidos",
    "Canadá",
    "Chile",
    "Países Bajos",
    "Japón",
    "España",
    "Reino Unido",
    "Hong Kong",
    "Italia",
    "China",
    "Suiza",
    "Francia",
    "México",
    "Otros",
]

MIN_R2_SCORE = 0.70
TOLERANCE = 0.01
SEED = 42