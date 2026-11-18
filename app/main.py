"""
Smart Irrigation System - FastAPI Application Backend
Provides RESTful APIs for dataset statistics, model metrics, and real-time pump prediction.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import joblib
import pandas as pd
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.data_loader import load_data, get_default_dataset_path
from src.models import FEATURE_COLUMNS


# -----------------------------------------------------------------------------
# FastAPI App Initialization
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Smart Irrigation System API",
    description=(
        "Production-grade RESTful API for environmental data analytics, "
        "machine learning model benchmarking, and automated actuator status prediction."
    ),
    version="2.0.0",
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Static files & templates
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# -----------------------------------------------------------------------------
# Singleton Cache for Model and Dataset Summary
# -----------------------------------------------------------------------------
_CACHED_MODEL = None
_CACHED_DATASET_SUMMARY = None
_CACHED_STATISTICS = None


def get_model():
    """
    Trained model pipeline'ını (Decision Tree Pipeline) güvenli bir şekilde yükler.
    Model her istekte yeniden eğitilmez; singleton olarak hafızada tutulur.
    """
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    model_path = PROJECT_ROOT / "models" / "best_model.joblib"
    if not model_path.exists():
        fallback_path = PROJECT_ROOT / "models" / "decision_tree.joblib"
        if fallback_path.exists():
            model_path = fallback_path
        else:
            raise FileNotFoundError(
                f"Model artifact not found at {model_path.resolve()}. "
                "Please run model training pipeline ('src/models.py') before invoking predictions."
            )

    try:
        _CACHED_MODEL = joblib.load(model_path)
        return _CACHED_MODEL
    except Exception as e:
        raise RuntimeError(f"Failed to load trained model pipeline: {e}")


def get_dataset_cache() -> Dict[str, Any]:
    """
    project_dataset.csv dosyasını merkezi load_data() ile yükler ve özet istatistikleri önbelleğe alır.
    """
    global _CACHED_DATASET_SUMMARY, _CACHED_STATISTICS
    if _CACHED_DATASET_SUMMARY is not None and _CACHED_STATISTICS is not None:
        return _CACHED_DATASET_SUMMARY, _CACHED_STATISTICS

    df = load_data()
    pump_counts = df["DC Water pump Status"].value_counts().to_dict()

    _CACHED_DATASET_SUMMARY = {
        "total_records": int(len(df)),
        "total_columns": int(len(df.columns)),
        "columns": df.columns.tolist(),
        "missing_values": int(df.isnull().sum().sum()),
        "date_range": {
            "start": str(df["date"].min()),
            "end": str(df["date"].max()),
        },
        "pump_status": {
            "off": int(pump_counts.get(0, 0)),
            "on": int(pump_counts.get(1, 0)),
            "off_percentage": round(float(pump_counts.get(0, 0) / len(df) * 100), 2),
            "on_percentage": round(float(pump_counts.get(1, 0) / len(df) * 100), 2),
        },
    }

    numeric_cols = [
        "Soil Moisture Resistance (Ohm)",
        "Ambient Temperature (deg. C)",
        "Atmospheric Humidity (%)",
    ]

    stats = {}
    for col in numeric_cols:
        series = df[col]
        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        stats[col] = {
            "mean": round(float(series.mean()), 2),
            "std": round(float(series.std()), 2),
            "min": round(float(series.min()), 2),
            "25%": round(q25, 2),
            "50%": round(float(series.median()), 2),
            "75%": round(q75, 2),
            "max": round(float(series.max()), 2),
            "iqr": round(q75 - q25, 2),
            "skewness": round(float(series.skew()), 2),
        }

    _CACHED_STATISTICS = stats
    return _CACHED_DATASET_SUMMARY, _CACHED_STATISTICS


# -----------------------------------------------------------------------------
# Pydantic Schemas for Request Validation
# -----------------------------------------------------------------------------
class SensorInput(BaseModel):
    """
    Environmental and soil sensor input schema with physical range validations.
    """
    soil_moisture_resistance: float = Field(
        ...,
        ge=0.0,
        le=5000.0,
        description="Soil Moisture Resistance in Ohms (Ω). Must be non-negative. Standard operational range: 0 - 2000 Ω.",
        example=525.0,
    )
    ambient_temperature: float = Field(
        ...,
        ge=-10.0,
        le=60.0,
        description="Ambient air temperature in Celsius (°C). Valid agricultural range: -10°C to 60°C.",
        example=31.2,
    )
    atmospheric_humidity: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Relative atmospheric humidity percentage (%). Valid physical range: 0% to 100%.",
        example=45.0,
    )


class PredictionResponse(BaseModel):
    """
    Structured model inference response.
    """
    predicted_status: int = Field(..., description="Binary actuator status: 0 for OFF, 1 for ON")
    pump_status: str = Field(..., description="Human-readable actuator state: 'OFF' or 'ON'")
    model_used: str = Field(..., description="Identifier of the ML model employed")
    input_features: Dict[str, float] = Field(..., description="Echoed input features used for inference")
    note: str = Field(..., description="Operational and agronomic disclaimer")


# -----------------------------------------------------------------------------
# Routes & Endpoints
# -----------------------------------------------------------------------------
@app.get("/", summary="Dashboard Home")
async def home(request: Request):
    """Renders the dashboard UI."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health", summary="Health Check")
def health_check():
    """Returns application health and model readiness status."""
    model_loaded = False
    try:
        get_model()
        model_loaded = True
    except Exception:
        pass

    dataset_ready = get_default_dataset_path().exists()

    return {
        "status": "healthy" if (model_loaded and dataset_ready) else "degraded",
        "model_loaded": model_loaded,
        "model_name": "Decision Tree (Pipeline)",
        "dataset_available": dataset_ready,
    }


@app.get("/api/dataset", summary="Dataset Summary")
def dataset_info():
    """
    Returns dataset dimensions, column schema, date boundaries,
    missing count, and pump class balance. Compatible with existing frontend keys.
    """
    try:
        summary, _ = get_dataset_cache()
        return summary
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unable to read dataset: {str(e)}",
        )


@app.get("/api/statistics", summary="Descriptive Statistics")
def dataset_statistics():
    """Returns parametric and non-parametric statistics for numeric attributes."""
    try:
        _, stats = get_dataset_cache()
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unable to compute statistics: {str(e)}",
        )


@app.get("/api/models", summary="Trained Model Benchmarks")
def models_comparison():
    """
    Returns performance metrics of the 6 evaluated classification models.
    Also clarifies the disabled status of regression due to the absence of water volume labels.
    """
    comparison_file = PROJECT_ROOT / "reports" / "model_comparison.csv"
    if not comparison_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model comparison report ('reports/model_comparison.csv') not found. Run 'src/models.py' first.",
        )

    try:
        df_models = pd.read_csv(comparison_file)
        models_list = df_models.to_dict(orient="records")

        return {
            "task": "Actuator State Classification (DC Water pump Status)",
            "target": "DC Water pump Status (0: OFF, 1: ON)",
            "models_count": len(models_list),
            "models": models_list,
            "best_model": {
                "name": "Decision Tree Classifier",
                "accuracy": 1.0,
                "f1_score": 1.0,
                "key_rule": "Soil Moisture Resistance >= 500 Ohm -> Pump ON",
            },
            "regression_status": "Disabled / Pending quantitative water flow meter data",
            "regression_note": (
                "Continuous water quantity (liters/m³) cannot be predicted because no volumetric "
                "flow sensor was included in the dataset. Synthetic target fabrication is strictly avoided."
            ),
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model comparison: {str(e)}",
        )


@app.post(
    "/api/predict",
    response_model=PredictionResponse,
    summary="Predict Pump Actuation Status",
)
def predict_pump_status(payload: SensorInput):
    """
    Predicts irrigation pump status (ON or OFF) based on input environmental and soil sensor data.
    Validates physical parameter boundaries before inference.
    """
    try:
        pipeline = get_model()
    except FileNotFoundError as fnf_err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(fnf_err),
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model execution error: {str(err)}",
        )

    # Construct input dataframe matching exact feature names and order
    input_data = pd.DataFrame(
        [[
            payload.soil_moisture_resistance,
            payload.ambient_temperature,
            payload.atmospheric_humidity,
        ]],
        columns=FEATURE_COLUMNS,
    )

    try:
        prediction = int(pipeline.predict(input_data)[0])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference computation failed: {str(e)}",
        )

    pump_status_str = "ON" if prediction == 1 else "OFF"

    return PredictionResponse(
        predicted_status=prediction,
        pump_status=pump_status_str,
        model_used="Decision Tree (Pipeline)",
        input_features={
            "Soil Moisture Resistance (Ohm)": payload.soil_moisture_resistance,
            "Ambient Temperature (deg. C)": payload.ambient_temperature,
            "Atmospheric Humidity (%)": payload.atmospheric_humidity,
        },
        note=(
            "Actuator state prediction only. Reflects controller behavior (threshold: ~500 Ω) "
            "and does not represent physiological crop irrigation volume or exact agronomic water demand."
        ),
    )
