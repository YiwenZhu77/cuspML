"""Small regression witnesses for formula, precision, feature order and held-out folds.

Run: python -m unittest discover -s reproduce/current -p test_train.py
Tests using release data are skipped when the data pack has not been downloaded.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import train


class FormulaTests(unittest.TestCase):
    def test_newell_2006_southward_and_northward(self):
        frame = pd.DataFrame({"sw_v": [400., 400.], "B_T": [5., 5.],
                              "sin_clock_half": [1., 0.]})
        actual = train.newell_predictor(frame)
        np.testing.assert_allclose(actual, [2000 ** (2 / 3), 0], rtol=1e-14)
        wrong_2007_twice = (400 ** (4 / 3) * 5 ** (2 / 3)) ** (2 / 3)
        self.assertGreater(abs(actual[0] - wrong_2007_twice), 1)

    def test_metrics_keep_float64_residual_precision(self):
        y = np.array([70., 71., 72.], dtype=np.float32)
        p = np.array([70.1, 71.1, 72.1], dtype=np.float64)
        result = train.metrics(y, p)
        self.assertAlmostEqual(result["MAE"], .1, places=12)
        self.assertAlmostEqual(result["RMSE"], .1, places=12)
        with self.assertRaises(ValueError):
            train.metrics(y, [1., float("nan"), 2.])


@unittest.skipUnless((Path(__file__).parent / "data" / "manifest.json").exists(),
                     "Download the current data pack to run release-data regression checks")
class ReleaseDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Path(__file__).parent / "data"
        cls.frame, cls.features, _, cls.x, cls.training, cls.testing = train.load_inputs(cls.data)
        cls.y = cls.frame.abs_eq_mlat.to_numpy(np.float32)

    def test_baseline_refit_and_target_isolation(self):
        pred, details = train.fit_newell(self.data, self.frame, self.testing)
        self.assertAlmostEqual(train.metrics(self.y[self.testing], pred)["MAE"],
                               1.748628477148995, places=11)
        self.assertAlmostEqual(details["coefficients"][0], 78.40937950077127, places=11)
        # Changing held-out labels cannot affect fitted coefficients or predictions.
        changed = self.frame.copy()
        changed.loc[changed.index[self.testing], "abs_eq_mlat"] = 0
        alternate, altered_details = train.fit_newell(self.data, changed, self.testing)
        np.testing.assert_array_equal(pred, alternate)
        self.assertEqual(details, altered_details)

    def test_cumulative_windows_preserve_feature_order(self):
        names = {w: train.history_features(self.features, w) for w in (0, 15, 30, 60, 90, 120)}
        self.assertEqual([len(v) for v in names.values()], [16, 34, 52, 74, 92, 110])
        self.assertEqual(names[60], self.features)
        self.assertEqual(names[90][:74], self.features)
        self.assertEqual(names[120][:74], self.features)
        for width, columns in names.items():
            self.assertEqual(len(columns), len(set(columns)))
            self.assertTrue(set(columns).issubset(self.frame.columns))

    def test_grouped_folds_match_released_prediction_indices(self):
        folds, _ = train.control_folds(self.frame, self.x, self.y, self.data)
        self.assertEqual({name: len(parts) for name, parts in folds.items()},
                         {"day": 1, "storm": 5, "block": 5, "year": 23})
        for name, parts in folds.items():
            for index, (training, testing) in enumerate(parts):
                self.assertEqual(np.intersect1d(training, testing).size, 0)
                path = self.data / "predictions" / f"control_{name}_{index}.npz"
                with np.load(path) as original:
                    np.testing.assert_array_equal(testing, original["test_indices"])


if __name__ == "__main__":
    unittest.main()
