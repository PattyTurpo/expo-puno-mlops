"""Paso 1 y 2 del ML Workflow: Extracción de características y partición del dataset."""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.model_selection import train_test_split
from src import config


def extract_and_clean() -> pd.DataFrame:
    if not config.RAW_EXCEL.exists():
        raise FileNotFoundError(f"Coloca el archivo Excel en: {config.RAW_EXCEL}")

    df = pd.read_excel(config.RAW_EXCEL, sheet_name="Exportaciones")
    df = df.dropna(
        subset=[
            "Fecha embarque",
            "Monto FOB (USD)",
            "Sector",
            "Mercado destino",
            "Peso neto (kg)",
            "Cantidad",
        ]
    )

    df["fecha"] = pd.to_datetime(df["Fecha embarque"], format="%d/%m/%Y", errors="coerce")
    df["mes"] = df["fecha"].dt.month.fillna(1).astype(int)
    df["sector"] = df["Sector"].astype(str).str.strip()

    valid_markets = [m for m in config.TOP_MARKETS if m != "Otros"]
    df["mercado_destino"] = (
        df["Mercado destino"]
        .astype(str)
        .str.strip()
        .apply(lambda x: x if x in valid_markets else "Otros")
    )
    df["peso_neto_kg"] = df["Peso neto (kg)"].astype(float)
    df["cantidad"] = df["Cantidad"].astype(float)
    df["monto_fob"] = df["Monto FOB (USD)"].astype(float)

    df = df[df["monto_fob"] > 0].copy()
    return df[config.FEATURES + [config.TARGET]]


def split_dataset() -> None:
    df = extract_and_clean()
    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # División Hold-out vista en clase: 80% entrenamiento total, 20% prueba fija
    train_full, test = train_test_split(
        df, test_size=0.20, random_state=config.SEED
    )

    # Lote 1 (60% de los datos) -> Para entrenar el modelo inicial v1
    train_v1, _ = train_test_split(
        train_full, train_size=0.75, random_state=config.SEED
    )

    # Lote 2 (80% de los datos) -> Simula llegada de nuevos datos para reentrenamiento (MLOps)
    train_v2 = train_full.copy()

    # Lote corrupto -> Para demostrar que el pipeline rechaza un mal modelo
    train_bad = train_v2.copy()
    train_bad[config.TARGET] = (
        train_bad[config.TARGET].sample(frac=1, random_state=config.SEED).to_numpy()
    )

    test.to_csv(config.TEST_PATH, index=False)
    train_v1.to_csv(config.TRAIN_V1_PATH, index=False)
    train_v2.to_csv(config.TRAIN_V2_PATH, index=False)
    train_bad.to_csv(config.TRAIN_BAD_PATH, index=False)
    shutil.copyfile(config.TRAIN_V1_PATH, config.CURRENT_TRAIN_PATH)

    print("=== PARTICIÓN DE DATOS COMPLETADA ===")
    print(f"Conjunto de Prueba fijo (test.csv) : {len(test)} filas")
    print(f"Lote Inicial (train_v1.csv)        : {len(train_v1)} filas")
    print(f"Lote Nuevo (train_v2.csv)          : {len(train_v2)} filas")
    print(f"Lote Corrupto (train_bad.csv)      : {len(train_bad)} filas")


if __name__ == "__main__":
    split_dataset()