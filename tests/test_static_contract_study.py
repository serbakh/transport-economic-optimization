"""Фізичні цілі нового ML та калькуляція залишаються узгодженими."""
from dataclasses import replace
import unittest

from src.dissertation_model import Rates, calculate_costs, simulate
from src.static_contract_study import observation, specification, cost_from_physical


class StaticStudyTests(unittest.TestCase):
    def test_cost_reconstruction_and_new_prices_without_retraining(self):
        result = simulate(specification(9, 2, 2, 1000), daily_min_tons=4000)
        row = observation(result, 9, 2, 2)
        self.assertAlmostEqual(cost_from_physical(row), result['costs']['total'])
        rates = replace(Rates(), diesel_litre=130, feeder_equipment_12h=5000)
        self.assertAlmostEqual(cost_from_physical(row, rates), calculate_costs(result, rates)['total'])
        full = [d['delivered_tons'] for d in result['contract']['days'] if d['full_day']]
        self.assertEqual(row['eligible'], all(q >= 4000 for q in full))
        self.assertEqual(row['minimum_full_day_tons'], min(full))
        self.assertEqual(row['first_day_tons'], result['contract']['days'][0]['delivered_tons'])

    def test_unfinished_run_is_not_a_training_target(self):
        result = simulate(specification(4, 1, 1, 1000), max_hours=1, daily_min_tons=4000)
        with self.assertRaises(RuntimeError):
            observation(result, 4, 1, 1)

    def test_prices_are_not_scenario_features(self):
        spec = specification(11, 2, 2, 1000)
        self.assertEqual(spec.batch_tons, 10000)
        self.assertTrue(all(w.available_until_min is None for w in spec.resources))
        self.assertEqual(spec.scheduled_counts(0), dict(truck=11, loader=2, scale=1, unload=2))
