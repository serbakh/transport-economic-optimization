"""Контракт перевіряється за добами, а не за середнім темпом партії."""
import unittest
import json
from pathlib import Path
import tempfile

from src.daily_contract import evaluate_contract
from src.dissertation_model import simulate
from src.resource_schedule import ScenarioSpec, ResourceWindow as W
from src.dissertation_series import run_series


def trajectory(hours, deliveries, complete=True):
    start = 495
    return {'summary': dict(start_min=start, loading_finish_min=start + hours * 60 if complete else None,
                            simulation_hours=hours, status='complete' if complete else 'incomplete'),
            'trips': [dict(finish_min=start + h * 60, cargo_tons=q) for h, q in deliveries]}


class ContractTests(unittest.TestCase):
    def test_fast_total_does_not_replace_daily_norm(self):
        r = evaluate_contract(trajectory(55, [(24, 3500), (48, 4500), (55, 2000)]))
        self.assertFalse(r['eligible'])
        self.assertEqual(r['failed_days'], [1])

    def test_boundary_deliveries_and_partial_last_day(self):
        r = evaluate_contract(trajectory(60, [(24, 4000), (48, 4000), (60, 2000)]))
        self.assertTrue(r['eligible'])
        self.assertEqual([d['delivered_tons'] for d in r['days']], [4000, 4000, 2000])
        self.assertIsNone(r['days'][-1]['meets_minimum'])

    def test_slow_tail_becomes_a_full_day(self):
        r = evaluate_contract(trajectory(73, [(24, 4000), (48, 4000), (72, 1500), (73, 500)]))
        self.assertFalse(r['eligible'])
        self.assertEqual(r['failed_days'], [3])

    def test_exact_completion_boundary_is_not_exempt(self):
        r = evaluate_contract(trajectory(48, [(24, 7000), (48, 3000)]))
        self.assertFalse(r['eligible'])
        self.assertEqual(r['full_days_checked'], 2)

    def test_partial_only_and_incomplete(self):
        self.assertTrue(evaluate_contract(trajectory(20, [(20, 10000)]))['eligible'])
        self.assertFalse(evaluate_contract(trajectory(25, [(24, 4500)], False))['eligible'])
        for value in (0, -1, float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                evaluate_contract(trajectory(20, []), value)

    def test_integration_counts_cargo_before_final_returns(self):
        spec = ScenarioSpec('small_contract', 100, 42, (
            W('cars', 'truck', 2), W('loaders', 'loader', 1),
            W('scale', 'scale', 1), W('feeders', 'unload', 1)))
        result = simulate(spec, daily_min_tons=4000)
        self.assertTrue(result['contract']['eligible'])
        self.assertAlmostEqual(sum(d['delivered_tons'] for d in result['contract']['days']), 100)
        self.assertEqual(result['contract']['days'][-1]['until_min'], result['summary']['loading_finish_min'])

    def test_series_keeps_violations_instead_of_silently_dropping_runs(self):
        spec = ScenarioSpec('slow', 10000, 42, (
            W('cars', 'truck', 2), W('loaders', 'loader', 1),
            W('scale', 'scale', 1), W('feeders', 'unload', 1)))
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / 'series'
            run_series(spec, [42, 43], output, daily_min_tons=4000)
            manifest = json.loads((output / 'series_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['contract_checked_runs'], 2)
            self.assertEqual(manifest['contract_passed_runs'], 0)
            self.assertEqual(manifest['status'], 'complete')

