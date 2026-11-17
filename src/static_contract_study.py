"""Перший новий ML-експеримент: постійний склад, 10 000 т, добова норма.

Архів статті не змінюється. Без --execute показується лише план.
Зовнішні впливи нульові; результати не поширюються на змінний склад.
"""
import argparse
import ast
from dataclasses import replace
import hashlib
import itertools
import json
from pathlib import Path
import platform
import random
import statistics
import sys
import time

from src.dissertation_model import Conditions, Rates, simulate
from src.resource_schedule import ResourceWindow as W, ScenarioSpec

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['trucks', 'loaders', 'unload_points']
TARGETS = ['loading_hours', 'trips', 'truck_hours', 'loader_hours', 'loader_active_hours',
           'scale_hours', 'feeder_hours', 'feeder_active_hours', 'first_day_tons', 'minimum_full_day_tons']


def specification(n, l, b, seed):
    return ScenarioSpec(f'{n}_{l}_{b}', 10000, seed, (
        W('trucks', 'truck', n), W('loaders', 'loader', l),
        W('scale', 'scale', 1), W('feeders', 'unload', b)))


def cost_from_physical(row, rates=Rates()):
    """Ті самі ставки; ML прогнозує фізичні величини, а не ціни."""
    return (row['trips'] * rates.litres_per_trip * rates.diesel_litre
            + row['truck_hours'] * (rates.driver_12h + rates.truck_maintenance_12h) / 12
            + row['loader_active_hours'] * rates.loader_active_hour
            + row['loader_hours'] * rates.loader_maintenance_12h / 12
            + row['scale_hours'] * rates.scale_worker_12h / 12
            + row['feeder_hours'] * (rates.feeder_operator_12h + rates.feeder_equipment_12h) / 12
            + row['feeder_active_hours'] * rates.feeder_kw * rates.electricity_kwh
            + row['loading_hours'] * rates.mechanic_12h / 12)


def observation(result, n, l, b):
    s, contract = result['summary'], result['contract']
    if s['status'] != 'complete':
        raise RuntimeError('Незавершений прогін не є навчальним прикладом завершення партії')
    usage = result['resource_usage']
    hours = lambda kind, key: sum(r[key] for r in usage if r['resource'] == kind)
    full = [d['delivered_tons'] for d in contract['days'] if d['full_day']]
    row = dict(configuration=f'{n}_{l}_{b}', seed=s['seed'], trucks=n, loaders=l, unload_points=b,
               loading_hours=s['loading_hours'], trips=s['total_trips'],
               truck_hours=hours('truck', 'engaged_hours'), loader_hours=hours('loader', 'engaged_hours'),
               loader_active_hours=hours('loader', 'active_hours'), scale_hours=hours('scale', 'engaged_hours'),
               feeder_hours=hours('unload', 'engaged_hours'), feeder_active_hours=hours('unload', 'active_hours'),
               first_day_tons=contract['days'][0]['delivered_tons'],
               minimum_full_day_tons=min(full) if full else s['delivered_tons'],
               eligible=contract['eligible'], cost_uah=result['costs']['total'])
    if abs(cost_from_physical(row) - row['cost_uah']) > 1e-6:
        raise AssertionError('Калькуляція не відповідає фізичним показникам')
    return row


def baseline_namespace(path):
    """Завантажити лише визначення; масовий запуск і запис файлів виключено."""
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    functions = {'external_scale_traffic', 'is_break_time', 'time_until_break_end',
                 'wait_until_work_time', 'time_until_next_break',
                 'perform_operation_with_breaks', 'truck_process', 'run_scenario'}
    nodes = [n for n in tree.body if
             isinstance(n, (ast.Import, ast.ImportFrom)) or
             isinstance(n, ast.FunctionDef) and n.name in functions or
             isinstance(n, ast.Assign) and (n.lineno < 117 or
                 any(isinstance(t, ast.Name) and t.id == 'BREAK_INTERVALS' for t in n.targets))]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def compare_baseline(output, pd):
    namespace = baseline_namespace(ROOT / 'baseline/src/simulation/digital_twin.py')
    rows = []
    for n, l, b in [(4, 1, 1), (9, 2, 2), (11, 2, 2), (16, 2, 3), (20, 3, 3)]:
        for seed in range(100, 120):
            namespace['random'] = random.Random(seed)
            original, _ = namespace['run_scenario'](n, l, 0, b, 0)
            if seed == 100:
                namespace['random'] = random.Random(seed)
                repeated, _ = namespace['run_scenario'](n, l, 0, b, 0)
                if original != repeated:
                    raise AssertionError('Seed не відтворює ізольований запуск baseline')
            result = simulate(specification(n, l, b, seed), daily_min_tons=4000)
            tons = sum(t['cargo_tons'] for t in result['trips']
                       if t['finish_min'] < result['summary']['start_min'] + 1440)
            rows.append(dict(configuration=f'{n}_{l}_{b}', seed=seed,
                             baseline_first_day_tons=original['total_tons'], new_first_day_tons=tons))
    df = pd.DataFrame(rows)
    df.to_csv(output / 'baseline_runs.csv', index=False)
    aggregate = []
    for label, group in df.groupby('configuration'):
        old, new = group.baseline_first_day_tons.mean(), group.new_first_day_tons.mean()
        aggregate.append(dict(configuration=label, baseline_mean=old, new_mean=new,
                              difference_tons=new-old, difference_percent=100*(new-old)/old,
                              baseline_sd=group.baseline_first_day_tons.std(), new_sd=group.new_first_day_tons.std()))
    pd.DataFrame(aggregate).to_csv(output / 'baseline_comparison.csv', index=False)
    return aggregate


def generate(output, train_seeds, control_seeds, pd):
    records, daily = [], []
    for index, (n, l, b) in enumerate(itertools.product(range(4, 21), range(1, 4), range(1, 4)), 1):
        for phase, seeds in [('training', train_seeds), ('control', control_seeds)]:
            for seed in seeds:
                result = simulate(specification(n, l, b, seed), daily_min_tons=4000)
                row = observation(result, n, l, b)
                records.append(dict(phase=phase, **row))
                daily.extend(dict(phase=phase, configuration=row['configuration'], seed=seed, **day)
                             for day in result['contract']['days'])
        if index % 17 == 0:
            print(f'Складів техніки: {index}/153', flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output / 'physical_runs.csv', index=False)
    pd.DataFrame(daily).to_csv(output / 'daily.csv', index=False)
    return frame


def train_models(frame, output, pd):
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, r2_score
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler

    data = frame[frame.phase == 'training'].reset_index(drop=True)
    control = frame[frame.phase == 'control'].reset_index(drop=True)
    split = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42)
    train_i, test_i = next(split.split(data, groups=data.configuration))
    train, test = data.iloc[train_i], data.iloc[test_i]
    assert not set(train.configuration) & set(test.configuration)
    (output / 'split.json').write_text(json.dumps(dict(
        train_configurations=sorted(train.configuration.unique()),
        test_configurations=sorted(test.configuration.unique()),
        rule='Усі повтори одного складу лише з одного боку поділу'), indent=2), encoding='utf-8')

    def fit(training):
        scaler = StandardScaler().fit(training[TARGETS])
        reg = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=1)
        reg.fit(training[FEATURES], scaler.transform(training[TARGETS]))
        clf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=1)
        clf.fit(training[FEATURES], training.eligible)
        return dict(scaler=scaler, regressor=reg, classifier=clf, features=FEATURES, targets=TARGETS)

    def predict(bundle, rows):
        y = bundle['scaler'].inverse_transform(bundle['regressor'].predict(rows[FEATURES]))
        predictions = pd.DataFrame(np.maximum(y, 0), columns=TARGETS)
        predictions['cost_uah'] = [cost_from_physical(r) for r in predictions.to_dict('records')]
        predictions['eligible'] = bundle['classifier'].predict(rows[FEATURES])
        return predictions

    def evaluate(bundle, test_rows):
        p = predict(bundle, test_rows)
        metrics = {t: dict(mae=float(mean_absolute_error(test_rows[t], p[t])),
                           r2=float(r2_score(test_rows[t], p[t])))
                   for t in ['loading_hours', 'first_day_tons', 'minimum_full_day_tons', 'cost_uah']}
        metrics['contract'] = dict(accuracy=float(accuracy_score(test_rows.eligible, p.eligible)),
                                   labels=[False, True],
                                   confusion_matrix=confusion_matrix(test_rows.eligible, p.eligible,
                                                                    labels=[False, True]).tolist())
        return metrics, p

    heldout = fit(train)
    held_metrics, _ = evaluate(heldout, test)
    linear = LinearRegression().fit(train[FEATURES], train.loading_hours)
    linear_p = linear.predict(test[FEATURES])
    held_metrics['linear_loading_hours'] = dict(mae=float(mean_absolute_error(test.loading_hours, linear_p)),
                                                r2=float(r2_score(test.loading_hours, linear_p)))
    joblib.dump(heldout, output / 'heldout_model.joblib')

    # Спочатку зберігаємо рішення за ML; контрольні результати не визначають
    # параметри навчання, поріг відбору або налаштування алгоритмів.
    final = fit(data)
    configurations = data.drop_duplicates('configuration').reset_index(drop=True)
    predictions = predict(final, configurations)
    predictions.insert(0, 'configuration', configurations.configuration)
    predictions['predicted_admissible'] = predictions.eligible & (predictions.minimum_full_day_tons >= 4000)
    predictions.to_csv(output / 'predictions_153.csv', index=False)
    eligible = predictions[predictions.predicted_admissible]
    selected = eligible.sort_values(['cost_uah', 'configuration']).iloc[0].configuration if len(eligible) else None
    control_metrics, control_p = evaluate(final, control)
    pd.concat([control[['configuration', 'seed']].reset_index(drop=True), control_p.add_prefix('predicted_')], axis=1).to_csv(
        output / 'control_predictions.csv', index=False)
    stats = control.groupby('configuration').agg(runs=('eligible', 'size'), passed=('eligible', 'sum'),
        mean_cost=('cost_uah', 'mean'), mean_loading_hours=('loading_hours', 'mean')).reset_index()
    stats.to_csv(output / 'control_summary.csv', index=False)
    admissible = stats[stats.passed == stats.runs]
    reference = admissible.sort_values(['mean_cost', 'configuration']).iloc[0] if len(admissible) else None
    chosen = stats[stats.configuration == selected]
    decision = dict(ml_selected=selected,
                    ml_selected_controls=chosen.to_dict('records'),
                    cheapest_all_passed_in_control=reference.to_dict() if reference is not None else None,
                    qualification='Порівняння на заданій сітці та контрольних seed; не доведений оптимум очікуваних витрат')
    joblib.dump(final, output / 'final_model.joblib')
    report = dict(heldout_configurations=held_metrics, independent_seeds_known_configurations=control_metrics,
                  training_rows=len(data), control_rows=len(control), heldout_rows=len(test), decision=decision)
    (output / 'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--train-repeats', type=int, default=10)
    parser.add_argument('--control-repeats', type=int, default=5)
    parser.add_argument('--ml-library-dir', type=Path, help='Явний каталог локальних ML-залежностей')
    args = parser.parse_args()
    if min(args.train_repeats, args.control_repeats) < 2 or max(args.train_repeats, args.control_repeats) > 1000:
        parser.error('Від 2 до 1000 повторень на етап; діапазони seed не мають перетинатися')
    plan = dict(batch_tons=10000, daily_min_tons=4000, configurations=153,
                external_scale_load=0, road_delay_percent=0,
                training_seeds=list(range(1000, 1000+args.train_repeats)),
                control_seeds=list(range(10000, 10000+args.control_repeats)),
                model_parameters=dict(trees=300, random_state=42, class_weight='balanced'),
                scope='Постійний склад; ставка не є входом ML; зміна техніки не досліджується')
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if args.output is None:
        parser.error('Потрібен новий каталог --output')
    if args.ml_library_dir:
        sys.path.insert(0, str(args.ml_library_dir.resolve()))
    import pandas as pd
    import sklearn
    import numpy
    import simpy
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    paths = [Path(__file__), ROOT/'src/dissertation_model.py', ROOT/'src/daily_contract.py',
             ROOT/'src/resource_schedule.py', ROOT/'src/vessel_cycle.py', ROOT/'baseline/src/simulation/digital_twin.py']
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = dict(status='running', python=platform.python_version(), sklearn=sklearn.__version__,
                    pandas=pd.__version__, numpy=numpy.__version__, simpy=simpy.__version__,
                    source_sha256={str(p.relative_to(ROOT)): digest(p) for p in paths}, rates=Rates().__dict__,
                    conditions=Conditions().__dict__, started_unix=time.time())
    path = args.output / 'manifest.json'
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    try:
        compare_baseline(args.output, pd)
        print('Порівняння з добовою моделлю статті завершено', flush=True)
        frame = generate(args.output, plan['training_seeds'], plan['control_seeds'], pd)
        report = train_models(frame, args.output, pd)
        manifest.update(status='complete', output_sha256={p.name: digest(p) for p in args.output.iterdir() if p.name != 'manifest.json'},
                        finished_unix=time.time())
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        raise
    finally:
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
