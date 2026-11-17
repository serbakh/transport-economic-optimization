"""Повторні прогони нової моделі з явними seed та окремими журналами.

Без --execute команда показує план. Незавершений прогін перериває серію
та позначається у маніфесті; його не змішують з успішними результатами.
"""
import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import statistics
import time

from src.dissertation_model import Conditions, Rates, calculate_costs, simulate, write_result
from src.resource_schedule import ScenarioSpec


def run_series(spec, seeds, output, *, conditions=Conditions(), rates=Rates(), start_min=495,
               daily_min_tons=None):
    """Записати траєкторії окремо: один seed — одна група для майбутнього ML."""
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError('Потрібно щонайменше два різні seed')
    if any(isinstance(s, bool) or not isinstance(s, int) for s in seeds):
        raise ValueError('Кожний seed має бути цілим числом')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Для серії потрібен новий або порожній каталог')
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'scenario': spec.to_dict(), 'seeds': seeds, 'start_min': start_min,
                'daily_min_tons': daily_min_tons,
                'status': 'running', 'completed_runs': 0,
                'note': 'Описові оцінки симуляції, не виробнича гарантія'}
    rows = []
    started = time.perf_counter()

    def save():
        (output/'series_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

    save()
    try:
        for seed in seeds:
            result = simulate(replace(spec, seed=seed), conditions, start_min=start_min,
                              daily_min_tons=daily_min_tons)
            result['costs'] = calculate_costs(result, rates)
            write_result(result, output/f'seed_{seed}')
            if result['summary']['status'] != 'complete':
                raise RuntimeError(f'Партію не завершено для seed={seed}; див. окремий журнал')
            rows.append({'scenario_id': spec.scenario_id, 'seed': seed,
                         'loading_hours': result['summary']['loading_hours'],
                         'all_returns_hours': result['summary']['all_returns_hours'],
                         'total_trips': result['summary']['total_trips'],
                         'cost_uah': result['costs']['total']})
            if daily_min_tons is not None:
                rows[-1].update(contract_eligible=result['contract']['eligible'],
                                failed_days=';'.join(map(str, result['contract']['failed_days'])))
            manifest['completed_runs'] = len(rows)
            save()
        with (output/'runs.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        stats = {name: {'mean': statistics.mean(r[name] for r in rows),
                        'sample_std': statistics.stdev(r[name] for r in rows),
                        'min': min(r[name] for r in rows), 'max': max(r[name] for r in rows)}
                 for name in ('loading_hours', 'all_returns_hours', 'cost_uah')}
        (output/'statistics.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        manifest.update(status='complete', elapsed_seconds=time.perf_counter()-started)
        if daily_min_tons is not None:
            manifest['contract_passed_runs'] = sum(r['contract_eligible'] for r in rows)
            manifest['contract_checked_runs'] = len(rows)
            manifest['contract_note'] = 'Частота в цій серії, не гарантія виконання контракту'
        save()
        return stats
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        save()
        raise


def main(argv=None):
    """Показати план або виконати серію за явною командою."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', type=Path, required=True, help='Конфігурація ресурсів JSON')
    parser.add_argument('--repeats', type=int, default=20, help='Кількість повторень, не менше двох')
    parser.add_argument('--first-seed', type=int, default=42, help='Перший seed послідовності')
    parser.add_argument('--output', type=Path, help='Новий каталог серії')
    parser.add_argument('--execute', action='store_true', help='Виконати заплановані прогони')
    parser.add_argument('--daily-min-tons', type=float, help='Норма тонн за кожну повну добу')
    args = parser.parse_args(argv)
    if args.repeats < 2:
        parser.error('Потрібно щонайменше два повторення')
    spec = ScenarioSpec.from_dict(json.loads(args.scenario.read_text(encoding='utf-8-sig')))
    seeds = list(range(args.first_seed, args.first_seed+args.repeats))
    if not args.execute:
        print(json.dumps({'mode': 'лише план', 'scenario': spec.to_dict(), 'seeds': seeds,
                          'daily_min_tons': args.daily_min_tons}, ensure_ascii=False, indent=2))
        return
    if args.output is None:
        parser.error('Для виконання потрібен --output')
    print(json.dumps(run_series(spec, seeds, args.output, daily_min_tons=args.daily_min_tons), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
