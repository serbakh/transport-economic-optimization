"""Контрактний ML-експеримент з усіма зовнішніми умовами статті.

1836 сценаріїв: 153 склади × 3 рівні навантаження вагової × 4 дорожні
надбавки. Без --execute лише план. Склад техніки впродовж погрузки постійний.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

from src.dissertation_model import Conditions, Rates, simulate
from src.static_contract_study import (
    ROOT, TARGETS, baseline_namespace, cost_from_physical, observation, specification)

FEATURES = ['trucks', 'loaders', 'unload_points', 'external_scale_load', 'road_delay_percent']
CONTEXTS = list(itertools.product((0, 5, 50), (0, 5, 15, 30)))


def context_id(external, road):
    return f'scale_{external}_road_{road}'


def write_csv(path, rows):
    """Записати однорідну таблицю без залежності від ML-бібліотек."""
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def context_worker(task):
    """Окремий процес генерує одну пару зовнішніх умов та добові підсумки."""
    output, external, road, training_seeds, control_seeds = task
    output = Path(output)
    label = context_id(external, road)
    condition = Conditions(external_scale_load=external, road_delay_percent=road)
    records, daily = [], []
    for n, l, b in itertools.product(range(4, 21), range(1, 4), range(1, 4)):
        for phase, seeds in [('training', training_seeds), ('control', control_seeds)]:
            for seed in seeds:
                result = simulate(specification(n, l, b, seed), condition, daily_min_tons=4000)
                row = observation(result, n, l, b)
                row['resource_configuration'] = row['configuration']
                row.update(configuration=f'{n}_{l}_{b}_{external}_{road}', context=label,
                           external_scale_load=external, road_delay_percent=road)
                records.append(dict(phase=phase, **row))
                daily.extend(dict(phase=phase, configuration=row['configuration'], seed=seed, **day)
                             for day in result['contract']['days'])
    write_csv(output / f'{label}_runs.csv', records)
    write_csv(output / f'{label}_daily.csv', daily)
    return label, len(records)


def compare_baseline(output, pd):
    """Описове порівняння добового тоннажу в усіх 12 зовнішніх умовах."""
    namespace = baseline_namespace(ROOT / 'baseline/src/simulation/digital_twin.py')
    rows = []
    for external, road in CONTEXTS:
        for n, l, b in [(9, 2, 2), (11, 2, 2), (16, 2, 3)]:
            for seed in range(100, 105):
                namespace['random'] = random.Random(seed)
                original, _ = namespace['run_scenario'](n, l, external, b, road)
                result = simulate(specification(n, l, b, seed),
                                  Conditions(external_scale_load=external, road_delay_percent=road))
                boundary = result['summary']['start_min'] + 1440
                tons = sum(t['cargo_tons'] for t in result['trips'] if t['finish_min'] < boundary)
                rows.append(dict(context=context_id(external, road), resource_configuration=f'{n}_{l}_{b}',
                                 seed=seed, baseline_tons=original['total_tons'], new_tons=tons))
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'baseline_runs.csv', index=False)
    means = frame.groupby(['context', 'resource_configuration']).agg(
        baseline_mean=('baseline_tons', 'mean'), new_mean=('new_tons', 'mean'),
        baseline_sd=('baseline_tons', 'std'), new_sd=('new_tons', 'std')).reset_index()
    means['difference_percent'] = 100 * (means.new_mean / means.baseline_mean - 1)
    means.to_csv(output / 'baseline_comparison.csv', index=False)


def generate(output, plan, workers, pd):
    """Паралельне виконання не змінює seed або порядок рядків підсумку."""
    shards = output / 'contexts'
    shards.mkdir()
    tasks = [(str(shards), external, road, plan['training_seeds'], plan['control_seeds'])
             for external, road in CONTEXTS]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(context_worker, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            label, count = future.result()
            print(f'Умови {index}/12: {label}, прогонів {count}', flush=True)
    runs = pd.concat([pd.read_csv(p) for p in sorted(shards.glob('*_runs.csv'))], ignore_index=True)
    daily = pd.concat([pd.read_csv(p) for p in sorted(shards.glob('*_daily.csv'))], ignore_index=True)
    runs = runs.sort_values(['context', 'configuration', 'phase', 'seed']).reset_index(drop=True)
    daily = daily.sort_values(['configuration', 'phase', 'seed', 'day_from_start']).reset_index(drop=True)
    runs.to_csv(output / 'physical_runs.csv', index=False)
    daily.to_csv(output / 'daily.csv', index=False)
    return runs


def train_models(frame, output, pd):
    """Відкладені склади та нові seed — дві різні перевірки."""
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
    train_i, test_i = next(split.split(data, groups=data.resource_configuration))
    train, test = data.iloc[train_i], data.iloc[test_i]
    assert not set(train.resource_configuration) & set(test.resource_configuration)
    split_plan = dict(train_resource_configurations=sorted(train.resource_configuration.unique()),
                      test_resource_configurations=sorted(test.resource_configuration.unique()),
                      rule='Склад техніки повністю відкладено за всіх зовнішніх умов і seed')
    (output / 'split.json').write_text(json.dumps(split_plan, ensure_ascii=False, indent=2), encoding='utf-8')

    def fit(training):
        scaler = StandardScaler().fit(training[TARGETS])
        reg = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=6)
        reg.fit(training[FEATURES], scaler.transform(training[TARGETS]))
        clf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=6)
        clf.fit(training[FEATURES], training.eligible)
        return dict(scaler=scaler, regressor=reg, classifier=clf, features=FEATURES, targets=TARGETS)

    def predict(bundle, rows):
        y = bundle['scaler'].inverse_transform(bundle['regressor'].predict(rows[FEATURES]))
        p = pd.DataFrame(np.maximum(y, 0), columns=TARGETS)
        p['cost_uah'] = [cost_from_physical(r) for r in p.to_dict('records')]
        p['eligible'] = bundle['classifier'].predict(rows[FEATURES])
        return p

    def metrics(truth, p):
        result = {t: dict(mae=float(mean_absolute_error(truth[t], p[t])),
                          r2=float(r2_score(truth[t], p[t])))
                  for t in ['loading_hours', 'first_day_tons', 'minimum_full_day_tons', 'cost_uah']}
        matrix = confusion_matrix(truth.eligible, p.eligible, labels=[False, True])
        result['contract'] = dict(accuracy=float(accuracy_score(truth.eligible, p.eligible)),
                                  confusion_matrix=matrix.tolist(), labels=[False, True],
                                  false_admissions=int(matrix[0, 1]), false_rejections=int(matrix[1, 0]))
        return result

    held = fit(train)
    held_p = predict(held, test)
    held_metrics = metrics(test.reset_index(drop=True), held_p)
    linear = LinearRegression().fit(train[FEATURES], train.loading_hours)
    held_metrics['linear_loading_hours'] = dict(
        mae=float(mean_absolute_error(test.loading_hours, linear.predict(test[FEATURES]))))
    joblib.dump(held, output / 'heldout_model.joblib', compress=3)
    print('Перевірку на відкладених складах завершено', flush=True)
    final = fit(data)
    scenarios = data.drop_duplicates('configuration').reset_index(drop=True)
    p = predict(final, scenarios)
    p.insert(0, 'configuration', scenarios.configuration)
    p.insert(1, 'context', scenarios.context)
    p['predicted_admissible'] = p.eligible & (p.minimum_full_day_tons >= 4000)
    p.to_csv(output / 'predictions_1836.csv', index=False)
    joblib.dump(final, output / 'final_model.joblib', compress=3)
    predicted_controls = predict(final, control)
    all_metrics = metrics(control, predicted_controls)
    pd.concat([control[['configuration', 'context', 'seed']], predicted_controls.add_prefix('predicted_')], axis=1).to_csv(
        output / 'control_predictions.csv', index=False)
    summary = control.groupby(['context', 'configuration']).agg(
        runs=('eligible', 'size'), passed=('eligible', 'sum'), mean_cost=('cost_uah', 'mean'),
        mean_loading_hours=('loading_hours', 'mean')).reset_index()
    summary.to_csv(output / 'control_summary.csv', index=False)
    decisions, context_metrics = [], []
    for label, part in p.groupby('context'):
        allowed = part[part.predicted_admissible].sort_values(['cost_uah', 'configuration'])
        selected = allowed.iloc[0] if len(allowed) else None
        stats = summary[summary.context == label]
        valid = stats[stats.passed == stats.runs].sort_values(['mean_cost', 'configuration'])
        reference = valid.iloc[0] if len(valid) else None
        chosen = stats[stats.configuration == selected.configuration].iloc[0] if selected is not None else None
        decisions.append(dict(context=label,
            ml_configuration=selected.configuration if selected is not None else None,
            predicted_cost=float(selected.cost_uah) if selected is not None else None,
            control_passed=int(chosen.passed) if chosen is not None else None,
            control_runs=int(chosen.runs) if chosen is not None else None,
            control_cost=float(chosen.mean_cost) if chosen is not None else None,
            control_hours=float(chosen.mean_loading_hours) if chosen is not None else None,
            reference_configuration=reference.configuration if reference is not None else None,
            reference_cost=float(reference.mean_cost) if reference is not None else None,
            all_passed_configurations=len(valid)))
        mask = control.context == label
        context_metrics.append(dict(context=label, **metrics(control.loc[mask].reset_index(drop=True),
                                                             predicted_controls.loc[mask].reset_index(drop=True))))
    pd.DataFrame(decisions).to_csv(output / 'decisions_by_conditions.csv', index=False)
    report = dict(training_rows=len(data), control_rows=len(control), heldout_rows=len(test),
                  heldout_resource_configurations=len(set(test.resource_configuration)),
                  heldout_configurations=held_metrics, independent_seeds=all_metrics,
                  metrics_by_conditions=context_metrics, decisions=decisions)
    (output / 'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument('--ml-library-dir', type=Path)
    parser.add_argument('--train-existing', action='store_true', help='Навчити на вже згенерованих даних без їх заміни')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('Кількість процесів має бути додатною')
    plan = dict(batch_tons=10000, daily_min_tons=4000, contexts=CONTEXTS, configurations=1836,
                training_seeds=list(range(1000, 1010)), control_seeds=list(range(10000, 10005)),
                baseline_seeds=list(range(100, 105)), features=FEATURES, targets=TARGETS,
                model_parameters=dict(trees=300, random_state=42, class_weight='balanced'),
                scope='Постійний склад; 12 зовнішніх умов; ціни не є входом ML',
                admission_rule='Класифікатор: виконано; прогноз мінімального добового тоннажу >=4000')
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if args.output is None:
        parser.error('Потрібен каталог --output')
    if args.ml_library_dir:
        sys.path.insert(0, str(args.ml_library_dir.resolve()))
    import pandas as pd
    import numpy
    import sklearn
    import simpy
    paths = [Path(__file__), ROOT/'src/static_contract_study.py', ROOT/'src/dissertation_model.py',
             ROOT/'src/daily_contract.py', ROOT/'src/resource_schedule.py', ROOT/'src/vessel_cycle.py',
             ROOT/'baseline/src/simulation/digital_twin.py']
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if args.train_existing:
        if (args.output / 'evaluation.json').exists():
            raise FileExistsError('Готовий експеримент не перезаписується')
        saved = json.loads((args.output / 'plan.json').read_text(encoding='utf-8'))
        if saved['features'] != FEATURES or saved['configurations'] != 1836:
            raise ValueError('Невідповідний план даних')
        manifest = json.loads((args.output / 'manifest.json').read_text(encoding='utf-8'))
        frame = pd.read_csv(args.output / 'physical_runs.csv')
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
        manifest = dict(status='running', python=platform.python_version(), sklearn=sklearn.__version__,
            pandas=pd.__version__, numpy=numpy.__version__, simpy=simpy.__version__,
            source_sha256={str(p.relative_to(ROOT)): digest(p) for p in paths}, rates=asdict(Rates()),
            base_conditions=asdict(Conditions()), started_unix=time.time())
        (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        frame = None
    try:
        if frame is None:
            compare_baseline(args.output, pd)
            print('Порівняння з кодом статті за всіх зовнішніх умов завершено', flush=True)
            frame = generate(args.output, plan, args.workers, pd)
            manifest['generation_complete'] = True
            (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        report = train_models(frame, args.output, pd)
        manifest.update(status='complete', finished_unix=time.time(),
                        output_sha256={str(p.relative_to(args.output)): digest(p)
                                       for p in args.output.rglob('*') if p.is_file() and p.name != 'manifest.json'})
        print(json.dumps(dict(training=report['training_rows'], control=report['control_rows'],
                              heldout=report['heldout_configurations'], independent=report['independent_seeds'],
                              decisions=report['decisions']), ensure_ascii=False, indent=2))
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        raise
    finally:
        (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
