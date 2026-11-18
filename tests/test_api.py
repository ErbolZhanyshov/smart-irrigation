"""
Comprehensive Unit & Integration Tests for Smart Irrigation System API
"""

import unittest
from fastapi.testclient import TestClient
from app.main import app


class TestSmartIrrigationAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_health_check(self):
        """GET /health should return 200 OK and status healthy."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertTrue(data["dataset_available"])

    def test_02_dataset_info(self):
        """GET /api/dataset should return valid summary with 5857 records."""
        response = self.client.get("/api/dataset")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_records"], 5857)
        self.assertEqual(data["total_columns"], 5)
        self.assertEqual(data["missing_values"], 0)
        self.assertEqual(data["pump_status"]["off"], 2459)
        self.assertEqual(data["pump_status"]["on"], 3398)
        self.assertIn("Soil Moisture Resistance (Ohm)", data["columns"])

    def test_03_statistics(self):
        """GET /api/statistics should return summary metrics for 3 features."""
        response = self.client.get("/api/statistics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("Soil Moisture Resistance (Ohm)", data)
        self.assertIn("Ambient Temperature (deg. C)", data)
        self.assertIn("Atmospheric Humidity (%)", data)
        self.assertEqual(data["Soil Moisture Resistance (Ohm)"]["min"], 4.0)
        self.assertEqual(data["Soil Moisture Resistance (Ohm)"]["max"], 1024.0)

    def test_04_models_comparison(self):
        """GET /api/models should return benchmarks and clear regression status."""
        response = self.client.get("/api/models")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["models_count"], 6)
        self.assertEqual(data["best_model"]["name"], "Decision Tree Classifier")
        self.assertIn("Disabled", data["regression_status"])

    def test_05_predict_pump_on(self):
        """POST /api/predict with Resistance >= 500 should predict pump ON (1)."""
        payload = {
            "soil_moisture_resistance": 530.0,
            "ambient_temperature": 31.5,
            "atmospheric_humidity": 45.0
        }
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["predicted_status"], 1)
        self.assertEqual(data["pump_status"], "ON")
        self.assertEqual(data["model_used"], "Decision Tree (Pipeline)")

    def test_06_predict_pump_off(self):
        """POST /api/predict with Resistance < 500 should predict pump OFF (0)."""
        payload = {
            "soil_moisture_resistance": 460.0,
            "ambient_temperature": 30.0,
            "atmospheric_humidity": 55.0
        }
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["predicted_status"], 0)
        self.assertEqual(data["pump_status"], "OFF")

    def test_07_validation_negative_resistance(self):
        """POST /api/predict with negative resistance must fail validation (HTTP 422)."""
        payload = {
            "soil_moisture_resistance": -15.0,
            "ambient_temperature": 25.0,
            "atmospheric_humidity": 50.0
        }
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_08_validation_invalid_humidity(self):
        """POST /api/predict with humidity > 100% must fail validation (HTTP 422)."""
        payload = {
            "soil_moisture_resistance": 480.0,
            "ambient_temperature": 25.0,
            "atmospheric_humidity": 120.0
        }
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_09_validation_extreme_temperature(self):
        """POST /api/predict with extreme temperature (> 60 C) must fail validation (HTTP 422)."""
        payload = {
            "soil_moisture_resistance": 480.0,
            "ambient_temperature": 85.0,
            "atmospheric_humidity": 50.0
        }
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_10_openapi_docs(self):
        """GET /openapi.json and GET /docs should be operational."""
        response_schema = self.client.get("/openapi.json")
        self.assertEqual(response_schema.status_code, 200)
        self.assertIn("Smart Irrigation System API", response_schema.json()["info"]["title"])

        response_docs = self.client.get("/docs")
        self.assertEqual(response_docs.status_code, 200)

    def test_11_home_ui(self):
        """GET / should render the index.html template successfully with all dashboard elements."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.text
        self.assertIn("SMART", html)
        self.assertIn("pumpChart", html)
        self.assertIn("featuresChart", html)
        self.assertIn("modelsChart", html)
        self.assertIn("prediction-form", html)
        self.assertIn("pending-card", html)

    def test_12_static_assets(self):
        """Static CSS and JS files should be served properly."""
        css_resp = self.client.get("/static/css/style.css")
        self.assertEqual(css_resp.status_code, 200)
        self.assertIn(".navbar", css_resp.text)
        self.assertIn(".predict-form", css_resp.text)

        js_resp = self.client.get("/static/js/script.js")
        self.assertEqual(js_resp.status_code, 200)
        self.assertIn("loadDatasetStats", js_resp.text)
        self.assertIn("loadModelBenchmarks", js_resp.text)
        self.assertIn("setupPredictionForm", js_resp.text)


if __name__ == "__main__":
    unittest.main()

