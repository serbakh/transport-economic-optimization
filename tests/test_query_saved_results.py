"""Готові запити мають відтворювати зафіксовані рішення та змінювати лише ціни."""
import csv
from dataclasses import replace
from pathlib import Path
import unittest

from src.dissertation_model import Rates
from tools.query_saved_results import query

RESULTS = Path(__file__).resolve().parents[1] / 'results/conditions_contract_ml_2026_11_07'


class SavedResultsTests(unittest.TestCase):
    def test_recommendations_match_all_twelve_recorded_decisions(self):
        with (RESULTS / 'decisions_by_conditions.csv').open(encoding='utf-8-sig', newline='') as stream:
            decisions = list(csv.DictReader(stream))
        self.assertEqual(len(decisions), 12)
        for decision in decisions:
            _, external, _, road = decision['context'].split('_')
            report = query(RESULTS, 11, 2, 2, int(external), int(road), Rates())
            self.assertEqual(report['recommendation']['configuration'], decision['ml_configuration'])

    def test_changed_rates_preserve_physical_predictions(self):
        base = query(RESULTS, 11, 2, 2, 0, 0, Rates())['selected']
        changed = query(RESULTS, 11, 2, 2, 0, 0, replace(Rates(), diesel_litre=120))['selected']
        for name in base:
            if name not in ('cost_uah', 'unit_cost_uah'):
                self.assertEqual(base[name], changed[name])
        self.assertAlmostEqual(changed['cost_uah'] - base['cost_uah'], base['trips'] * 8 * 20)

    def test_unrepresented_conditions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'інтерполяція'):
            query(RESULTS, 11, 2, 2, 1, 0, Rates())
