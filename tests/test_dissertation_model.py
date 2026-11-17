"""Дослідницькі інваріанти: вантаж, завершення рейсу, витрати і зміни ресурсів."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from src.dissertation_model import Conditions, Rates, calculate_costs, simulate, write_result
from src.resource_schedule import ResourceWindow as W, ScenarioSpec


def scenario(batch=100, windows=(), seed=42, trucks=2):
    """Невелика партія перевіряє логіку без масової генерації."""
    return ScenarioSpec('test', batch, seed, (
        W('cars', 'truck', trucks), W('loaders', 'loader', 1),
        W('scale', 'scale', 1), W('feeders', 'unload', 1), *windows))


class DissertationTests(unittest.TestCase):
    def test_mass_returns_and_individual_finish(self):
        result = simulate(scenario(1000))
        s = result['summary']
        self.assertEqual(s['status'], 'complete')
        self.assertEqual(sum(t['cargo_units'] for t in result['trips']), 100000)
        self.assertEqual(s['unreserved_tons'], 0)
        self.assertEqual(s['loading_finish_min'], max(t['finish_min'] for t in result['trips']))
        self.assertEqual(s['all_returns_finish_min'], max(t['return_finish_min'] for t in result['trips']))
        self.assertAlmostEqual(s['all_returns_hours']-s['loading_hours'], 20/60)
        for row in result['resource_usage']:
            if row['resource'] == 'truck':
                expected = max(t['return_finish_min'] for t in result['trips'] if t['truck_id']==row['unit_id'])
            else:
                expected = max(o['released_min'] for o in result['operations'] if o['unit_id']==row['unit_id'])
            self.assertEqual(row['accounting_finish_min'], expected)
            self.assertEqual(row['engaged_from_min'], s['start_min'])
        for trip in result['trips']:
            self.assertEqual(trip['return_finish_min']-trip['finish_min'], 20)
        self.assertLessEqual(sum(t['cargo_tons'] < t['sampled_capacity_tons'] for t in result['trips']), 1)

    def test_repeatability_changed_seed_and_12000_tons(self):
        spec = scenario(12000)
        first = simulate(spec)
        self.assertEqual(first, simulate(spec))
        self.assertNotEqual(first['summary']['loading_hours'], simulate(replace(spec, seed=43))['summary']['loading_hours'])
        self.assertEqual(first['summary']['delivered_tons'], 12000)
        self.assertAlmostEqual(sum(d['delivered_tons'] for d in first['daily']), 12000)
        self.assertLessEqual(first['daily'][-1]['observed_hours'], 24)

    def test_delayed_addition_and_removal_finish_accepted_trips(self):
        spec = scenario(1000, (W('extra', 'truck', 1, 60, 100),))
        result = simulate(spec)
        selected = [t for t in result['trips'] if t['truck_id']=='extra:1']
        self.assertTrue(selected)
        self.assertTrue(all(t['reserved_at_min'] >= 555 for t in selected))
        self.assertTrue(all(t['reserved_at_min'] < 595 for t in selected))
        self.assertTrue(all(t['return_finish_min'] is not None for t in selected))
        usage = next(r for r in result['resource_usage'] if r['unit_id']=='extra:1')
        self.assertEqual(usage['engaged_from_min'], 555)
        self.assertEqual(usage['accounting_finish_min'], selected[-1]['return_finish_min'])

    def test_future_resources_not_mobilized_after_loading(self):
        result = simulate(scenario(1, (W('late', 'truck', 1, 10000), W('late_loader', 'loader', 1, 10000))))
        for row in result['resource_usage']:
            if row['unit_id'].startswith('late'):
                self.assertFalse(row['activated'])
                self.assertEqual(row['engaged_hours'], 0)
        self.assertLess(result['summary']['all_returns_hours'], 5)

    def test_same_resource_never_overlaps_and_expired_equipment_drains(self):
        spec = scenario(2000, (W('temporary_loader', 'loader', 1, 10, 40),))
        result = simulate(spec)
        for uid in {o['unit_id'] for o in result['operations']}:
            ops = sorted((o for o in result['operations'] if o['unit_id']==uid),key=lambda o:o['acquired_min'])
            for a,b in zip(ops, ops[1:]):
                self.assertLessEqual(a['released_min'], b['acquired_min'])
            if uid=='temporary_loader:1':
                self.assertTrue(all(o['acquired_min'] < 535 for o in ops))
            for op in ops:
                self.assertAlmostEqual(sum(b-a for a,b in op['active_segments']), op['active_min'])

    def test_impossible_schedule_and_time_limit_not_success(self):
        spec = ScenarioSpec('blocked', 1000, 42, (
            W('cars','truck',2), W('loaders','loader',1,0,1),
            W('scale','scale',1), W('feeders','unload',1)))
        result = simulate(spec,max_hours=2)
        self.assertEqual(result['summary']['status'],'incomplete')
        self.assertIsNone(result['costs'])
        self.assertEqual(result['summary']['simulation_hours'],2)
        no_cars = replace(spec, resources=(W('cars','truck',1,0,1), *spec.resources[1:]))
        self.assertEqual(simulate(no_cars,max_hours=1)['summary']['status'],'incomplete')

    def test_costs_recomputed_without_simulation_and_driver_tail_included(self):
        result=simulate(scenario(100))
        base=calculate_costs(result)
        double=calculate_costs(result,replace(Rates(),driver_12h=2500))
        self.assertAlmostEqual(double['total']-base['total'],base['components']['drivers'])
        truck_hours=sum(r['engaged_hours'] for r in result['resource_usage'] if r['resource']=='truck')
        self.assertAlmostEqual(base['components']['drivers'],truck_hours*1250/12)
        without_returns=sum((max(t['finish_min'] for t in result['trips'] if t['truck_id']==uid)-495)/60
                            for uid in {t['truck_id'] for t in result['trips']})
        self.assertGreater(truck_hours,without_returns)

    def test_start_time_and_external_conditions(self):
        spec=scenario(500)
        self.assertEqual(simulate(spec,start_min=600)['summary']['start_min'],600)
        conditions=Conditions(external_scale_load=50,road_delay_percent=30)
        result=simulate(spec,conditions,start_min=600)
        self.assertEqual(result['summary']['status'],'complete')
        self.assertTrue(any(o['stage']=='external' for o in result['operations']))
        scale=next(r for r in result['resource_usage'] if r['resource']=='scale')
        internal=sum(o['active_min'] for o in result['operations'] if o['resource']=='scale' and o['stage']!='external')
        self.assertAlmostEqual(scale['active_hours'],internal/60)

    def test_snapshot_mass_balance_and_output_protection(self):
        result=simulate(scenario(1000))
        for row in result['snapshots']:
            self.assertAlmostEqual(row['delivered_tons']+row['unreserved_tons']+row['in_transit_or_reserved_tons'],1000)
        with tempfile.TemporaryDirectory() as tmp:
            write_result(result,tmp)
            self.assertEqual(json.loads((Path(tmp)/'result.json').read_text(encoding='utf-8'))['summary'],result['summary'])
            with self.assertRaises(FileExistsError):
                write_result(result,tmp)

    def test_invalid_physical_and_money_parameters(self):
        for kwargs in ({'capacity_tons':(0,1)},{'load_min':(10,8)}, {'return_min':-1},
                       {'weigh_min':0},{'external_scale_load':100},{'road_delay_percent':float('nan')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Conditions(**kwargs)
        with self.assertRaises(ValueError):
            Rates(driver_12h=-1)
        for kwargs in ({'start_min':1440},{'max_hours':0},{'observe_every_min':0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                simulate(scenario(),**kwargs)


if __name__=='__main__':
    unittest.main()
