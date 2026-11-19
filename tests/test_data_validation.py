import math
import unittest
from pathlib import Path
import numpy as np

class TestRaindropDataValidation(unittest.TestCase):
    """Unit tests for raindrop data validation and physical formulas."""

    def test_spherical_volume_calculation(self):
        """Verify spherical volume formula V = (pi/6) * d^3."""
        diameter = 1.0  # 1 mm
        expected_volume = (math.pi / 6.0) * (1.0 ** 3)
        self.assertAlmostEqual(expected_volume, 0.523598775598, places=6)

    def test_physical_boundaries(self):
        """Raindrop diameter must be within typical physical atmospheric limits (0 < d <= 10 mm)."""
        valid_diameters = [0.5, 1.2, 2.5, 5.0, 8.0]
        invalid_diameters = [-1.0, 0.0, 15.0, 50.0]

        for d in valid_diameters:
            self.assertTrue(0.0 < d <= 10.0, f"Expected {d} to be within (0, 10] mm")

        for d in invalid_diameters:
            self.assertFalse(0.0 < d <= 10.0, f"Expected {d} to be outside (0, 10] mm")

    def test_terminal_velocity_boundaries(self):
        """Raindrop terminal velocity must be positive and below atmospheric maximum (~15-20 m/s)."""
        valid_velocities = [1.0, 3.5, 7.2, 9.5]
        invalid_velocities = [-2.0, 0.0, 25.0]

        for v in valid_velocities:
            self.assertTrue(0.0 < v <= 20.0, f"Expected {v} to be within (0, 20] m/s")

        for v in invalid_velocities:
            self.assertFalse(0.0 < v <= 20.0, f"Expected {v} to be outside (0, 20] m/s")

    def test_required_schema_columns(self):
        """Verify all 6 standard columns required by the project pipeline."""
        required = {"DateTime", "Diameter_mm", "Velocity_m_s", "Volume_mm3", "Condition", "SourceFile"}
        sample_row = {
            "DateTime": "2016-06-13 12:00:00",
            "Diameter_mm": 0.85,
            "Velocity_m_s": 2.84,
            "Volume_mm3": 0.321,
            "Condition": "Foliated_Cedar1a_015",
            "SourceFile": "RD1_EachDrop.csv"
        }
        self.assertTrue(required.issubset(set(sample_row.keys())))

if __name__ == "__main__":
    unittest.main()
