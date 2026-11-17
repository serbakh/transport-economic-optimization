"""Перевірка меж доступності, груп і захисту від хибних конфігурацій."""
import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import unittest

from src.resource_schedule import ResourceWindow, ScenarioSpec, main
from src.vessel_cycle import Config, simulate


class ResourceScheduleTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(__file__).resolve().parent / 'fixtures/resource_schedule.json'
        self.value = json.loads(self.path.read_text(encoding='utf-8'))
        self.spec = ScenarioSpec.from_dict(self.value)

    def test_half_open_windows_and_additive_counts(self):
        self.assertEqual([self.spec.scheduled_counts(t)['truck']
                          for t in (0, 479.999, 480, 1199.999, 1200, 5000)],
                         [9, 9, 11, 11, 9, 9])
        self.assertEqual(self.spec.scheduled_counts(480)['scale'], 1)

    def test_invalid_windows(self):
        for changes in ({'count': 0}, {'count': True}, {'available_from_min': -1},
                        {'available_from_min': float('nan')}, {'available_until_min': float('inf')},
                        {'available_until_min': 0}, {'available_from_min': 480, 'available_until_min': 480},
                        {'resource': 'typo'}, {'group_id': ''}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ResourceWindow(**{'group_id': 'trucks', 'resource': 'truck', 'count': 2, **changes})

    def test_duplicate_group_rejected_but_overlapping_distinct_groups_allowed(self):
        with self.assertRaises(ValueError):
            replace(self.spec, resources=self.spec.resources + (self.spec.resources[0],))
        extra = ResourceWindow('another_group', 'truck', 1, 480, 600)
        self.assertEqual(replace(self.spec, resources=self.spec.resources+(extra,)).scheduled_counts(500)['truck'], 12)

    def test_unknown_fields_and_missing_resource_rejected(self):
        with self.assertRaises(ValueError):
            ScenarioSpec.from_dict({**self.value, 'bath_tons': 12000})
        with self.assertRaises(ValueError):
            ScenarioSpec.from_dict({**self.value, 'resources': self.value['resources'][:-1]})
        malformed = {**self.value, 'resources': [{**self.value['resources'][0], 'until': 10}]}
        with self.assertRaises(ValueError):
            ScenarioSpec.from_dict(malformed)

    def test_round_trip_and_invalid_seed_batch_and_version(self):
        self.assertEqual(ScenarioSpec.from_dict(json.loads(json.dumps(self.spec.to_dict()))), self.spec)
        for change in ({'seed': True}, {'seed': 1.5}, {'batch_tons': 0},
                       {'batch_tons': 0.001}, {'schema_version': 2}, {'schema_version': 1.0}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(self.spec, **change)

    def test_preview_is_explicitly_not_a_simulation(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main([str(self.path)])
        result = json.loads(output.getvalue())
        self.assertEqual(result['status'], 'configuration_only_not_simulated')
        self.assertEqual([x['elapsed_min'] for x in result['planned_changes']], [0, 480, 1200])

    def test_current_engine_already_supports_12000_tons_reproducibly(self):
        # Контроль наявної можливості: не експеримент зі зміною ресурсів.
        config = Config(batch_tons=12000, seed=42)
        result = simulate(config)
        self.assertEqual(result, simulate(config))
        self.assertEqual(sum(t['cargo_units'] for t in result['trips']), 1200000)
        self.assertEqual(result['summary']['delivered_tons'], 12000)
        self.assertGreater(result['summary']['duration_hours'], 24)


if __name__ == '__main__':
    unittest.main()
