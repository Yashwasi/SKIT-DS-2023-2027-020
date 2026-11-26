"""
tests/test_api_integration.py
-----------------------------
Integration and unit tests for FastAPI backend, ML model inference, and database layer.
Covers:
- GET /health and GET /health/database liveness/readiness probes.
- POST /predict valid requests, input boundaries, missing fields, and physical bounds.
- Model loading failure handling and fallback.
- Optional persistence mode when PostgreSQL is unavailable.
- Mocked database persistence and request_id propagation.
- OpenAPI documentation generation.
"""

from __future__ import annotations

import math
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from api.main import app, _spherical_volume


class TestFastAPIIntegration(unittest.TestCase):
    """End-to-end integration tests for FastAPI application."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    # -------------------------------------------------------------------------
    # 1. Health Endpoints
    # -------------------------------------------------------------------------

    def test_health_liveness_endpoint(self):
        """GET /health returns 200 OK and status 'ok'."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertIn("timestamp", data)

    @patch("database.connection.check_database_health")
    def test_database_health_endpoint_offline(self, mock_health):
        """
        GET /health/database returns HTTP 503 when PostgreSQL is offline,
        providing structured diagnostic details without leaking credentials.
        """
        mock_health.return_value = {
            "status": "unhealthy",
            "is_connected": False,
            "safe_url": "postgresql://postgres:***@localhost:5432/raindrop_db",
            "error": "Connection refused",
        }
        response = self.client.get("/health/database")
        self.assertEqual(response.status_code, 503)
        data = response.json()
        self.assertIn("status", data)
        self.assertFalse(data.get("is_connected"))
        self.assertIn("safe_url", data)
        self.assertNotIn("password", str(data.get("safe_url")).lower())
        self.assertIn("error", data)

    @patch("database.connection.check_database_health")
    def test_database_health_endpoint_online(self, mock_health):
        """
        GET /health/database returns HTTP 200 when PostgreSQL is healthy and connected.
        """
        mock_health.return_value = {
            "status": "healthy",
            "is_connected": True,
            "latency_ms": 1.25,
            "timescaledb_extension": False,
            "timescaledb_version": None,
            "port_listening": True,
            "safe_url": "postgresql://postgres:***@localhost:5432/raindrop_db",
            "error": None,
        }
        response = self.client.get("/health/database")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertTrue(data.get("is_connected"))
        self.assertIn("latency_ms", data)

    # -------------------------------------------------------------------------
    # 2. Valid Prediction Request & Response Structure
    # -------------------------------------------------------------------------

    def test_valid_prediction_request(self):
        """
        POST /predict with valid physical measurements returns 200 OK,
        theoretical volume, XGBoost prediction, and execution latency.
        """
        payload = {"diameter_mm": 1.20, "velocity_m_s": 3.80}
        response = self.client.post("/predict", json=payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Check required fields
        self.assertEqual(data["diameter_mm"], 1.20)
        self.assertEqual(data["velocity_m_s"], 3.80)
        self.assertEqual(data["inputs"]["diameter_mm"], 1.20)
        self.assertEqual(data["inputs"]["velocity_m_s"], 3.80)

        # Check theoretical formula V = (pi/6) * d^3
        expected_formula_vol = round((math.pi / 6.0) * (1.20 ** 3), 8)
        self.assertAlmostEqual(data["formula_volume_mm3"], expected_formula_vol, places=6)

        # Check XGBoost model prediction
        self.assertIsNotNone(data["predicted_volume_mm3"])
        self.assertGreater(data["predicted_volume_mm3"], 0.0)
        self.assertEqual(data["predicted_volume_mm3"], data["xgb_volume_mm3"])

        # Check latency and status
        self.assertGreater(data["latency_ms"], 0.0)
        self.assertEqual(data["status"], "success")

    def test_prediction_response_schema_completeness(self):
        """Ensure all required fields exist in PredictResponse."""
        payload = {"diameter_mm": 0.85, "velocity_m_s": 2.90}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        expected_keys = {
            "request_id",
            "diameter_mm",
            "velocity_m_s",
            "inputs",
            "formula_volume_mm3",
            "predicted_volume_mm3",
            "xgb_volume_mm3",
            "latency_ms",
            "status",
            "database_audit",
        }
        self.assertTrue(expected_keys.issubset(set(data.keys())))

    # -------------------------------------------------------------------------
    # 3. Input Validation & HTTP Errors
    # -------------------------------------------------------------------------

    def test_missing_input_fields(self):
        """Missing required fields must return 422 Unprocessable Entity."""
        # Empty body
        r1 = self.client.post("/predict", json={})
        self.assertEqual(r1.status_code, 422)

        # Missing velocity_m_s
        r2 = self.client.post("/predict", json={"diameter_mm": 1.5})
        self.assertEqual(r2.status_code, 422)

        # Missing diameter_mm
        r3 = self.client.post("/predict", json={"velocity_m_s": 4.0})
        self.assertEqual(r3.status_code, 422)

    def test_invalid_diameter_boundaries(self):
        """Non-positive or excessive diameters must return 422."""
        # Negative diameter
        r1 = self.client.post("/predict", json={"diameter_mm": -1.0, "velocity_m_s": 3.0})
        self.assertEqual(r1.status_code, 422)

        # Zero diameter
        r2 = self.client.post("/predict", json={"diameter_mm": 0.0, "velocity_m_s": 3.0})
        self.assertEqual(r2.status_code, 422)

        # Exceeds maximum disdrometer limit (> 15.0 mm)
        r3 = self.client.post("/predict", json={"diameter_mm": 18.5, "velocity_m_s": 3.0})
        self.assertEqual(r3.status_code, 422)

    def test_invalid_velocity_boundaries(self):
        """Non-positive or excessive velocities must return 422."""
        # Negative velocity
        r1 = self.client.post("/predict", json={"diameter_mm": 1.0, "velocity_m_s": -2.0})
        self.assertEqual(r1.status_code, 422)

        # Zero velocity
        r2 = self.client.post("/predict", json={"diameter_mm": 1.0, "velocity_m_s": 0.0})
        self.assertEqual(r2.status_code, 422)

        # Exceeds atmospheric terminal maximum (> 25.0 m/s)
        r3 = self.client.post("/predict", json={"diameter_mm": 1.0, "velocity_m_s": 35.0})
        self.assertEqual(r3.status_code, 422)

    # -------------------------------------------------------------------------
    # 4. Model Failure Handling & Safe Degradation
    # -------------------------------------------------------------------------

    @patch("api.main._predict_xgb", return_value=None)
    def test_model_loading_or_prediction_failure(self, mock_predict):
        """
        When the ML model fails to load or predict, the API must:
        - Return HTTP 200 without crashing.
        - Set status to 'model_unavailable'.
        - Set predicted_volume_mm3 to None.
        - Still calculate and return theoretical formula_volume_mm3.
        """
        payload = {"diameter_mm": 1.5, "velocity_m_s": 4.0}
        response = self.client.post("/predict", json=payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "model_unavailable")
        self.assertIsNone(data["predicted_volume_mm3"])
        self.assertIsNone(data["xgb_volume_mm3"])
        # Theoretical formula must still be provided
        self.assertAlmostEqual(data["formula_volume_mm3"], _spherical_volume(1.5), places=6)

    # -------------------------------------------------------------------------
    # 5. Database Integration & Audit Logging
    # -------------------------------------------------------------------------

    @patch("database.connection.log_prediction_audit", return_value=(None, "skipped (database unavailable)"))
    def test_database_unavailable_optional_persistence(self, mock_audit):
        """
        When PostgreSQL is offline, predictions continue seamlessly in
        optional-persistence mode. request_id must be None and database_audit
        must accurately report 'skipped (database unavailable)'.
        """
        payload = {"diameter_mm": 1.10, "velocity_m_s": 3.20}
        response = self.client.post("/predict", json=payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsNone(data["request_id"])
        self.assertIn("skipped", data["database_audit"].lower())
        mock_audit.assert_called_once()

    def test_database_live_persistence(self):
        """
        When live PostgreSQL is reachable, POST /predict persists an audit log
        and returns a valid integer request_id with database_audit 'persisted'.
        """
        from database.connection import probe_tcp_port
        from database.config import db_config
        is_up, _ = probe_tcp_port(db_config.host, db_config.port, timeout_sec=0.2)
        if not is_up:
            self.skipTest("Live database not reachable")

        payload = {"diameter_mm": 1.10, "velocity_m_s": 3.20}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsNotNone(data["request_id"])
        self.assertIsInstance(data["request_id"], int)
        self.assertEqual(data["database_audit"], "persisted")

    @patch("database.connection.log_prediction_audit", return_value=(4242, "persisted"))
    def test_database_audit_logging_mocked_success(self, mock_audit):
        """
        When database connection is available, the endpoint captures the audit log ID
        and reports status 'persisted'.
        """
        payload = {"diameter_mm": 1.45, "velocity_m_s": 4.60}
        response = self.client.post("/predict", json=payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["request_id"], 4242)
        self.assertEqual(data["database_audit"], "persisted")
        mock_audit.assert_called_once()

    # -------------------------------------------------------------------------
    # 6. Additional Endpoints (/predict/volume, /predict/batch, /validate, /)
    # -------------------------------------------------------------------------

    def test_root_endpoint_metadata(self):
        """GET / returns project metadata, team info, and running status."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "running")
        self.assertIn("project", data)
        self.assertIn("team", data)

    def test_predict_volume_endpoint(self):
        """POST /predict/volume returns formula volume and model comparison."""
        payload = {"diameter_mm": 1.0, "velocity_m_s": 3.0}
        response = self.client.post("/predict/volume", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["diameter_mm"], 1.0)
        self.assertAlmostEqual(data["volume_formula_mm3"], _spherical_volume(1.0), places=6)
        self.assertIn("formula", data["models_available"])

    def test_predict_batch_endpoint(self):
        """POST /predict/batch handles multiple valid raindrop samples."""
        payload = {
            "samples": [
                {"diameter_mm": 0.8, "velocity_m_s": 2.5},
                {"diameter_mm": 1.5, "velocity_m_s": 4.5},
            ]
        }
        response = self.client.post("/predict/batch", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["count"], 2)
        self.assertEqual(len(data["predictions"]), 2)

    def test_validate_data_row_endpoint(self):
        """POST /validate checks physical boundaries and detects non-positive values."""
        # Valid row
        r_valid = self.client.post(
            "/validate",
            json={"diameter_mm": 1.0, "velocity_m_s": 3.0, "volume_mm3": 0.523599},
        )
        self.assertEqual(r_valid.status_code, 200)
        self.assertTrue(r_valid.json()["is_valid"])

        # Invalid row (negative diameter)
        r_invalid = self.client.post(
            "/validate",
            json={"diameter_mm": -1.0, "velocity_m_s": 3.0, "volume_mm3": 0.5},
        )
        self.assertEqual(r_invalid.status_code, 200)
        self.assertFalse(r_invalid.json()["is_valid"])
        self.assertGreater(len(r_invalid.json()["issues"]), 0)

    # -------------------------------------------------------------------------
    # 7. OpenAPI Documentation
    # -------------------------------------------------------------------------

    def test_openapi_schema_generation(self):
        """Verify OpenAPI JSON schema includes /predict and /health/database."""
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        schema = response.json()

        paths = schema.get("paths", {})
        self.assertIn("/predict", paths)
        self.assertIn("/health", paths)
        self.assertIn("/health/database", paths)
        self.assertIn("post", paths["/predict"])


if __name__ == "__main__":
    unittest.main()
