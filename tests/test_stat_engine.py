"""Unit tests for the deterministic Stat Engine."""

import math
import unittest
import numpy as np
from stat_tutor.stat_engine import calculate_sample_variance


class TestStatEngine(unittest.TestCase):
    def test_simple_variance_ddof0_and_ddof1(self) -> None:
        res = calculate_sample_variance([2, 4, 6])
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["sample_size"], 3)
        self.assertEqual(res["degrees_of_freedom"], 2)
        self.assertAlmostEqual(res["mean"], 4.0)
        self.assertAlmostEqual(res["sum_squared_deviations"], 8.0)
        self.assertAlmostEqual(res["variance_ddof1"], 4.0)
        self.assertAlmostEqual(res["sample_variance"], 4.0)
        self.assertAlmostEqual(res["variance_ddof0"], 8.0 / 3.0)

    def test_matches_main_py_sensor_dataset_and_numpy(self) -> None:
        data = [12.5, 14.2, 13.8, 15.1, 14.0]
        res = calculate_sample_variance(data)
        np_ddof0 = float(np.var(data, ddof=0))
        np_ddof1 = float(np.var(data, ddof=1))

        self.assertAlmostEqual(res["variance_ddof0"], np_ddof0, places=9)
        self.assertAlmostEqual(res["variance_ddof1"], np_ddof1, places=9)
        self.assertAlmostEqual(res["sample_variance"], np_ddof1, places=9)
        # Note: in main.py prompt, 0.7016 is ddof=0 and 0.8770 is ddof=1
        self.assertAlmostEqual(res["variance_ddof1"], 0.8770, places=4)
        self.assertAlmostEqual(res["variance_ddof0"], 0.7016, places=4)

    def test_rejects_invalid_inputs(self) -> None:
        invalid_cases = [
            [],
            [42.0],
            "1, 2, 3",
            {"numbers": [1, 2, 3]},
            None,
            [1.0, "abc", 3.0],
            [1.0, float("nan"), 3.0],
            [1.0, float("inf"), 3.0],
            [1.0, -float("inf"), 3.0],
            [True, False, 1.0],
        ]
        for bad_input in invalid_cases:
            with self.subTest(bad_input=bad_input):
                with self.assertRaises(ValueError):
                    calculate_sample_variance(bad_input)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
