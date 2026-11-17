"""Внутрішні перевірки правил історичної моделі; не виробнича валідація."""
import ast
import math
from pathlib import Path
import unittest

import simpy

from src.vessel_cycle import Config, START, operate, simulate


class CycleTests(unittest.TestCase):
    def test_mass_concurrency_and_last_trip(self):
        """Баланс вантажу, резервування залишку та послідовність рейсів."""
        for batch in (0.01, 27.5, 100, 10000):
            result = simulate(Config(batch_tons=batch))
            summary, trips = result['summary'], result['trips']
            self.assertEqual(sum(t['cargo_units'] for t in trips), round(batch * 100))
            self.assertEqual(summary['delivered_tons'], batch)
            self.assertEqual(summary['unreserved_tons'], 0)
            self.assertEqual(summary['finish_min'], max(t['finish_min'] for t in trips))
            self.assertEqual(len({t['reservation_id'] for t in trips}), len(trips))
            partial = [t for t in trips if t['cargo_tons'] < t['sampled_capacity_tons']]
            self.assertLessEqual(len(partial), 1)
            if partial:
                self.assertEqual(partial[0]['reservation_id'], trips[-1]['reservation_id'])
            for t in trips:
                self.assertGreater(t['cargo_units'], 0)
                self.assertLessEqual(t['cargo_tons'], t['sampled_capacity_tons'])
            for truck_id in range(1, 10):
                rows = sorted((t for t in trips if t['truck_id'] == truck_id), key=lambda t: t['trip_no'])
                if rows:
                    self.assertEqual(rows[0]['return_before_min'], 0)
                for before, after in zip(rows, rows[1:]):
                    self.assertAlmostEqual(after['start_min'] - before['finish_min'], 20)
            self.assertEqual(summary['inter_trip_return_hours'],
                             sum(max(0, r['trips'] - 1) for r in result['trucks']) / 3)

    def test_repeatability_and_seed_effect(self):
        """Відтворення однакового запуску і вплив іншого seed."""
        first = simulate()
        self.assertEqual(first, simulate())
        self.assertNotEqual(first['summary']['duration_hours'], simulate(Config(seed=43))['summary']['duration_hours'])

    def test_calendar(self):
        """Завершення операцій з урахуванням регламентованих перерв."""
        for start, duration, finish in ((719, 3, 782), (719, 2, 721),
                                        (720, 3, 783), (1439, 3, 1502),
                                        (479, 3, 497), (1199, 3, 1217)):
            env = simpy.Environment(initial_time=start)
            segments = []
            env.run(until=env.process(operate(env, duration, segments)))
            self.assertEqual(env.now, finish)
            self.assertAlmostEqual(sum(b-a for a, b in segments), duration)

    def test_resource_accounting(self):
        """Відсутність накладання робіт та баланс активного й утримуваного часу."""
        result = simulate()
        for resource in result['resources']:
            rows = [r for r in result['operations'] if
                    (r['resource'], r['unit_id']) == (resource['resource'], resource['unit_id'])]
            for before, after in zip(rows, rows[1:]):
                self.assertLessEqual(before['released_min'], after['acquired_min'] + 1e-8)
            for r in rows:
                self.assertAlmostEqual(sum(b-a for a, b in r['active_segments']), r['active_min'])
                self.assertAlmostEqual(r['held_min'], r['active_min'] + r['break_while_held_min'])
                self.assertGreaterEqual(r['queue_min'], 0)
            self.assertLessEqual(resource['active_hours'], resource['held_hours'] + 1e-8)
            self.assertLessEqual(resource['held_hours'], result['summary']['duration_hours'] + 1e-8)
        trips = result['trips']
        self.assertAlmostEqual(result['summary']['loader_active_hours'], sum(t['load_min'] for t in trips)/60)
        self.assertAlmostEqual(result['summary']['unload_active_hours'], sum(t['unload_min'] for t in trips)/60)
        self.assertEqual(len(result['operations']), 4 * len(trips))

    def test_invalid_parameters(self):
        """Відмова від некоректної маси та кількості техніки."""
        for batch in (0, -1, math.inf, math.nan, 0.001):
            with self.assertRaises(ValueError):
                simulate(Config(batch_tons=batch))
        for trucks in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                simulate(Config(trucks=trucks))

    def test_calendar_matches_baseline_functions(self):
        """Тотожність календарних правил нової та вихідної функцій."""
        # Читаємо лише визначення: імпорт baseline запустив би всю початкову сітку.
        path = Path(__file__).resolve().parents[1] / 'baseline/src/simulation/digital_twin.py'
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        names = {'is_break_time', 'time_until_break_end', 'wait_until_work_time',
                 'time_until_next_break', 'perform_operation_with_breaks'}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        self.assertEqual(len(nodes), len(names))
        scope = {'BREAK_INTERVALS': [(0, 60), (480, 495), (720, 780), (1200, 1215)],
                 'FINISH_THRESHOLD_MIN': 1}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
        for day in (0, 1, 3):
            for offset in (0, 59.5, 478, 479, 480, 494.5, 495, 718, 719, 720, 779, 1199, 1439):
                for duration in (0.5, 1, 2, 3, 9, 13.5):
                    old = simpy.Environment(initial_time=day*1440 + offset)
                    new = simpy.Environment(initial_time=day*1440 + offset)
                    old.run(until=old.process(scope['perform_operation_with_breaks'](old, duration)))
                    new.run(until=new.process(operate(new, duration, [])))
                    self.assertAlmostEqual(old.now, new.now)

    def test_other_capacities(self):
        """Повне завершення партії за різного складу ресурсів."""
        for trucks, loaders, unload in ((4, 1, 1), (20, 3, 3), (9, 1, 3)):
            result = simulate(Config(trucks=trucks, loaders=loaders, unload_points=unload))
            self.assertEqual(result['summary']['delivered_tons'], 10000)
            self.assertEqual(sum(t['cargo_units'] for t in result['trips']), 1000000)
            for row in result['resources']:
                self.assertLessEqual(row['held_hours'], result['summary']['duration_hours'] + 1e-8)


if __name__ == '__main__':
    unittest.main()
