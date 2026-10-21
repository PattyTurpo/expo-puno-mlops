"""Esquemas de validación de datos con Pydantic."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Literal
from pydantic import BaseModel, Field, field_validator
from src.config import SECTORS, TOP_MARKETS


class ExportInput(BaseModel):
    mes: int = Field(ge=1, le=12, description="Shipment month (1-12)")
    sector: str = Field(description="Export economic sector")
    mercado_destino: str = Field(description="Destination country/market")
    peso_neto_kg: float = Field(gt=0, le=500000, description="Net weight in kilograms")
    cantidad: float = Field(gt=0, le=500000, description="Commercial units/quantity")

    @field_validator("sector")
    @classmethod
    def check_sector(cls, v: str) -> str:
        if v not in SECTORS:
            raise ValueError(f"Invalid sector. Choose from: {SECTORS}")
        return v

    @field_validator("mercado_destino")
    @classmethod
    def check_market(cls, v: str) -> str:
        if v not in TOP_MARKETS:
            raise ValueError(f"Invalid market. Choose from: {TOP_MARKETS}")
        return v


class PredictionOutput(BaseModel):
    predicted_fob_usd: float
    shipment_category: Literal["Small Export", "Medium Export", "High-Value Export"]
    model_version: str
    model_r2: float