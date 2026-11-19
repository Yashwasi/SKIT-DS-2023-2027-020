import math
from pathlib import Path
import unittest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_volume_predictor.json"

class TestModelInference(unittest.TestCase):
    """Integration tests verifying trained model loading and prediction behavior."""

    def test_model_file_exists(self):
        """Verify that the serialized XGBoost model file exists on disk."""
        self.assertTrue(
            MODEL_PATH.exists(),
            f"Expected model weight file at {MODEL_PATH}"
        )

    def test_xgboost_prediction_execution(self):
        """Verify that XGBoost loads and produces positive volume predictions."""
        try:
            import xgboost as xgb
        except ImportError:
            self.skipTest("xgboost is not installed in the environment.")

        model = xgb.XGBRegressor()
        model.load_model(str(MODEL_PATH))

        # Test single small drop (d=0.7mm, v=2.0m/s)
        X = np.array([[0.7, 2.0]], dtype=np.float64)
        pred_volume = float(model.predict(X)[0])

        self.assertGreater(pred_volume, 0.0, "Predicted volume must be strictly positive")
        # Spherical volume for 0.7mm is ~0.179 mm3
        formula_vol = (math.pi / 6.0) * (0.7 ** 3)
        self.assertAlmostEqual(pred_volume, formula_vol, delta=0.5)

    def test_batch_prediction_shape(self):
        """Verify model handles batch inputs correctly."""
        try:
            import xgboost as xgb
        except ImportError:
            self.skipTest("xgboost is not installed in the environment.")

        model = xgb.XGBRegressor()
        model.load_model(str(MODEL_PATH))

        X_batch = np.array([
            [0.6, 1.8],
            [1.0, 3.2],
            [2.0, 5.5],
            [3.5, 7.8],
        ], dtype=np.float64)

        preds = model.predict(X_batch)
        self.assertEqual(len(preds), 4, "Batch output count must match input count")
        self.assertTrue((preds > 0).all(), "All predicted volumes must be positive")

if __name__ == "__main__":
    unittest.main()
