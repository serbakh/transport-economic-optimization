"""Повнота зовнішніх умов і правильність калькуляції за їх зміни."""
import unittest
import csv
from pathlib import Path
import tempfile
from unittest.mock import patch

from src.conditions_contract_study import CONTEXTS, FEATURES, context_id, context_worker
from src.dissertation_model import Conditions, simulate
from src.static_contract_study import specification, observation, cost_from_physical


class ConditionsStudyTests(unittest.TestCase):
    def test_original_grid_and_model_inputs(self):
        self.assertEqual(len(CONTEXTS), 12)
        self.assertEqual(set(CONTEXTS), {(p, d) for p in (0, 5, 50) for d in (0, 5, 15, 30)})
        self.assertEqual(len({context_id(*c) for c in CONTEXTS}), 12)
        self.assertEqual(FEATURES[-2:], ['external_scale_load', 'road_delay_percent'])
        self.assertNotIn('seed', FEATURES)

    def test_high_external_load_is_repeatable_and_costs_remain_consistent(self):
        conditions = Conditions(external_scale_load=50, road_delay_percent=30)
        spec = specification(11, 2, 2, 1000)
        result = simulate(spec, conditions, daily_min_tons=4000)
        repeated = simulate(spec, conditions, daily_min_tons=4000)
        self.assertEqual(result['summary'], repeated['summary'])
        self.assertEqual(result['contract'], repeated['contract'])
        row = observation(result, 11, 2, 2)
        self.assertAlmostEqual(cost_from_physical(row), result['costs']['total'])
        self.assertTrue(any(o['truck_id'] == 'external' for o in result['operations']))
        self.assertEqual(sum(d['delivered_tons'] for d in result['contract']['days']), 10000)

    def test_context_records_keep_conditions_and_independent_seeds(self):
        with tempfile.TemporaryDirectory() as root:
            with patch('src.conditions_contract_study.itertools.product', return_value=[(9, 2, 2)]):
                label, count = context_worker((root, 5, 15, [1000], [10000]))
            self.assertEqual(count, 2)
            with (Path(root) / f'{label}_runs.csv').open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({r['configuration'] for r in rows}, {'9_2_2_5_15'})
            self.assertEqual({r['resource_configuration'] for r in rows}, {'9_2_2'})
            self.assertEqual({r['phase'] for r in rows}, {'training', 'control'})
            self.assertEqual({r['seed'] for r in rows}, {'1000', '10000'})
            self.assertTrue(all(r['external_scale_load'] == '5' and r['road_delay_percent'] == '15'
                                for r in rows))
