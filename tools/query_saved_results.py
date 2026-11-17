"""Переглянути збережені прогнози й перерахувати витрати без навчання ML."""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dissertation_model import Rates
from src.static_contract_study import cost_from_physical, TARGETS


def query(folder, trucks, loaders, unload_points, external, road, rates):
    key = f'{trucks}_{loaders}_{unload_points}_{external}_{road}'
    with (folder / 'predictions_1836.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    selected = next((row for row in rows if row['configuration'] == key), None)
    if selected is None:
        raise ValueError('У збереженій сітці немає такого поєднання; інтерполяція не виконується')
    def result(row):
        physical = {field: float(row[field]) for field in TARGETS}
        total = cost_from_physical(physical, rates)
        return dict(configuration=row['configuration'], **physical,
                    predicted_admissible=row['predicted_admissible'] == 'True',
                    cost_uah=total, unit_cost_uah=total / 10000)
    candidates = [result(row) for row in rows
                  if row['context'] == selected['context'] and row['predicted_admissible'] == 'True']
    candidates.sort(key=lambda row: (row['cost_uah'], row['configuration']))
    return dict(source='Збережені прогнози; без нового запуску ML або симуляції',
                batch_tons=10000, daily_min_tons=4000, rates=asdict(rates),
                selected=result(selected), eligible_count=len(candidates),
                recommendation=candidates[0] if candidates else None,
                qualification='Допуск за моделлю та заданою сіткою, не гарантія виробничого результату')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trucks', type=int, choices=range(4, 21), default=11)
    parser.add_argument('--loaders', type=int, choices=range(1, 4), default=2)
    parser.add_argument('--unload-points', type=int, choices=range(1, 4), default=2)
    parser.add_argument('--external', type=int, choices=(0, 5, 50), default=0)
    parser.add_argument('--road', type=int, choices=(0, 5, 15, 30), default=0)
    parser.add_argument('--rates', type=Path, help='JSON зі зміненими сценарними ставками')
    args = parser.parse_args()
    try:
        rates = Rates(**json.loads(args.rates.read_text(encoding='utf-8-sig'))) if args.rates else Rates()
        report = query(ROOT / 'results/conditions_contract_ml_2026_11_07', args.trucks,
                       args.loaders, args.unload_points, args.external, args.road, rates)
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
