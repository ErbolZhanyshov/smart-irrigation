import pandas as pd
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(
    title="Smart Irrigation System",
    description="AI-powered smart irrigation system",
    version="1.0.0"
)

# Static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# HTML templates
templates = Jinja2Templates(directory="app/templates")


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.get("/health")
def health_check():
    return {"status": "healthy"}

BASE_DIR = Path(__file__).resolve().parent
@app.get("/api/dataset")
def dataset_info():
    dataset_path = BASE_DIR.parent / "data" / "raw" / "dataset.xlsx"

    df = pd.read_excel(dataset_path)

    pump_counts = df["Pump_Status"].value_counts().to_dict()

    return {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "columns": df.columns.tolist(),
        "missing_values": int(df.isnull().sum().sum()),
        "pump_status": {
            "off": int(pump_counts.get(0, 0)),
            "on": int(pump_counts.get(1, 0))
        }
    }
