"""Spillway: tuning_rows reads the installer's tune.csv for the Engine view."""
import tempfile
import unittest
from pathlib import Path

from serve.server import tuning_rows


class TuningRows(unittest.TestCase):
    def test_rows_and_best(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tune.csv"
            p.write_text("draft length 2,61.5\n--spec 3, vram reserve 450 MiB,0\nbest,--spec 3 STRATA_PREFETCH_MAX=5,70.2\n")
            r = tuning_rows(p)
        self.assertEqual(r["runs"], [{"label": "draft length 2", "tok_s": 61.5},
                                     {"label": "--spec 3, vram reserve 450 MiB", "tok_s": 0.0}])
        self.assertEqual(r["best"], {"settings": "--spec 3 STRATA_PREFETCH_MAX=5", "tok_s": 70.2})

    def test_missing_file(self):
        self.assertIsNone(tuning_rows(Path(tempfile.gettempdir()) / "no-such-tune.csv"))


if __name__ == "__main__":
    unittest.main()
