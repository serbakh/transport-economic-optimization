"""Перевірки серії: повнота журналів, seed і відмова від перезапису."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.dissertation_series import main, run_series
from src.resource_schedule import ResourceWindow as W, ScenarioSpec


class SeriesTests(unittest.TestCase):
    def test_small_series_saves_complete_reproducible_trajectories(self):
        spec = ScenarioSpec('small', 100, 42, tuple(W(k,k,1) for k in ('truck','loader','scale','unload')))
        with tempfile.TemporaryDirectory() as tmp:
            stats = run_series(spec,[42,43],tmp)
            manifest = json.loads((Path(tmp)/'series_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['status'],'complete')
            self.assertEqual(manifest['completed_runs'],2)
            self.assertTrue((Path(tmp)/'seed_42/manifest.json').exists())
            self.assertGreater(stats['all_returns_hours']['mean'],stats['loading_hours']['mean'])
            with self.assertRaises(FileExistsError):
                run_series(spec,[42,43],tmp)

    def test_preview_never_calls_simulator(self):
        path = Path(__file__).resolve().parents[1]/'examples/current_10000_11_2_2.json'
        with patch('src.dissertation_series.simulate') as simulate, contextlib.redirect_stdout(io.StringIO()):
            main(['--scenario',str(path)])
        simulate.assert_not_called()

    def test_failure_saved_without_success_summary(self):
        spec = ScenarioSpec('blocked',100,42,(W('car','truck',1), W('loader','loader',1,0,1),
                                           W('scale','scale',1),W('unload','unload',1)))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                run_series(spec,[42,43],tmp)
            manifest = json.loads((Path(tmp)/'series_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['status'],'failed')
            self.assertFalse((Path(tmp)/'statistics.json').exists())


if __name__=='__main__':
    unittest.main()
