from fastapi import FastAPI

app = FastAPI(
    title="Smart Irrigation System",
    description="AI-powered smart irrigation system",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Welcome to Smart Irrigation System!",
        "status": "API is running"
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}