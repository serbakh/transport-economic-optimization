"""Контрактна норма за послідовні повні доби від початку погрузки.

Останній неповний період показує залишок; на нього добова норма не
поширюється. Це перевірка траєкторії, а не гарантія майбутніх результатів.
"""
import math


def evaluate_contract(result, minimum_tons=4000):
    """Порахувати завершені розвантаження в інтервалах (початок, кінець]."""
    if (isinstance(minimum_tons, bool) or not isinstance(minimum_tons, (int, float))
            or not math.isfinite(minimum_tons) or minimum_tons <= 0):
        raise ValueError('Добова норма має бути додатним скінченним числом')
    summary = result['summary']
    start = summary['start_min']
    finished = summary['loading_finish_min']
    end = finished if finished is not None else start + summary['simulation_hours'] * 60
    if not all(math.isfinite(x) for x in (start, end)) or end < start:
        raise ValueError('Некоректний період спостереження')
    full_days = int((end - start) // 1440)
    days = []
    for i in range(full_days + int(end > start + full_days * 1440)):
        a, b = start + i * 1440, min(start + (i + 1) * 1440, end)
        # Маса задана з точністю 0,01 т; сума в цілих одиницях уникає
        # випадкового порушення рівності 4000 через арифметику float.
        tons = sum(round(t['cargo_tons'] * 100) for t in result['trips']
                   if a < t['finish_min'] <= b) / 100
        full = i < full_days
        days.append(dict(day_from_start=i + 1, from_min=a, until_min=b,
                         observed_hours=(b - a) / 60, full_day=full,
                         delivered_tons=tons,
                         meets_minimum=tons >= minimum_tons if full else None))
    failed = [d['day_from_start'] for d in days if d['meets_minimum'] is False]
    complete = summary['status'] == 'complete' and finished is not None
    return dict(minimum_tons_per_full_day=minimum_tons,
                period_basis='послідовні 24 години від початку погрузки',
                last_partial_day_exempt=True, full_days_checked=full_days,
                failed_days=failed, loading_complete=complete,
                eligible=complete and not failed, days=days)
