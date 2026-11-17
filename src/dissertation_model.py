"""Дисертаційна модель: змінні ресурси, повні рейси та індивідуальні витрати.

Час подій — абсолютні хвилини умовного календаря; інтервали ресурсів —
хвилини від команди на початок робіт. Історичний vessel_cycle не змінюємо.
Приклад: python -m src.dissertation_model --scenario examples/resource_schedule.json
Без --execute друкується лише план, без розрахунку та запису файлів.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import random

import simpy

from src.resource_schedule import ScenarioSpec
from src.daily_contract import evaluate_contract
from src.vessel_cycle import operate, wait_work


@dataclass(frozen=True)
class Conditions:
    """Експертні інтервали; це не оцінені виробничі розподіли.

    Кожна величина має власну одиницю: тонна, хвилина або відсоток.
    Рівні зовнішніх впливів задають сценарні умови, а не виміряну частку.
    """
    capacity_tons: tuple = (27, 28)
    load_min: tuple = (8, 10)
    road_min: tuple = (18, 25)
    unload_min: tuple = (12, 15)
    return_min: float = 20
    weigh_min: float = 3
    road_delay_percent: float = 0
    external_scale_load: float = 0

    def __post_init__(self):
        for name in ('capacity_tons', 'load_min', 'road_min', 'unload_min'):
            pair = getattr(self, name)
            if (not isinstance(pair, (tuple, list)) or len(pair) != 2 or
                any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in pair) or
                pair[0] <= 0 or pair[1] < pair[0]):
                raise ValueError(f'{name}: потрібні дві додатні скінченні межі в порядку зростання')
        if self.capacity_tons[0] < .01:
            raise ValueError('Місткість не може бути меншою за 0,01 т')
        for name in ('return_min', 'weigh_min', 'road_delay_percent', 'external_scale_load'):
            x = getattr(self, name)
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0:
                raise ValueError(f'{name}: потрібне невід’ємне скінченне число')
        if self.weigh_min <= 0 or self.external_scale_load >= 100:
            raise ValueError('Зважування має тривати понад нуль; зовнішнє навантаження — менше 100 %')


@dataclass(frozen=True)
class Rates:
    """Сценарні гривневі ставки. Персонал оплачується пропорційно часу.

    Зміна ставок не потребує повторної фізичної симуляції. Вартість оренди
    мехлопати вже включає оператора та пальне; вдруге їх не додаємо.
    """
    diesel_litre: float = 100
    litres_per_trip: float = 8
    driver_12h: float = 1250
    truck_maintenance_12h: float = 500
    loader_active_hour: float = 1900
    loader_maintenance_12h: float = 500
    scale_worker_12h: float = 1200
    feeder_operator_12h: float = 2000
    feeder_equipment_12h: float = 4200
    mechanic_12h: float = 2000
    electricity_kwh: float = 10
    feeder_kw: float = 40

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name}: ставка має бути невід’ємним скінченним числом')


def calculate_costs(result, rates=Rates()):
    """Застосувати ставки до журналів; незавершений прогін не є кошторисом партії."""
    if result['summary']['status'] != 'complete':
        return None
    rows = result['resource_usage']
    hours = lambda kind: sum(r['engaged_hours'] for r in rows if r['resource'] == kind)
    active = lambda kind: sum(r['active_hours'] for r in rows if r['resource'] == kind)
    parts = {
        'diesel': len(result['trips']) * rates.litres_per_trip * rates.diesel_litre,
        'drivers': hours('truck') * rates.driver_12h / 12,
        'truck_maintenance': hours('truck') * rates.truck_maintenance_12h / 12,
        'loader_rental': active('loader') * rates.loader_active_hour,
        'loader_maintenance': hours('loader') * rates.loader_maintenance_12h / 12,
        'scale_worker': hours('scale') * rates.scale_worker_12h / 12,
        'feeder_operators': hours('unload') * rates.feeder_operator_12h / 12,
        'feeder_equipment': hours('unload') * rates.feeder_equipment_12h / 12,
        'electricity': active('unload') * rates.feeder_kw * rates.electricity_kwh,
        # Один спільний механік залучений від команди до останньої тонни на судні.
        'mechanic': result['summary']['loading_hours'] * rates.mechanic_12h / 12,
    }
    return {'currency': 'UAH', 'basis': 'сценарні ставки, не підтверджена собівартість підприємства',
            'rates': asdict(rates), 'components': parts, 'total': sum(parts.values())}


def simulate(spec: ScenarioSpec, conditions=Conditions(), *, start_min=495,
             max_hours=24*60, observe_every_min=60, daily_min_tons=None):
    """Змоделювати партію та дочекатися повернення всіх залучених автомобілів.

    max_hours — захист від неможливого розкладу; це не строк замовлення.
    Розділені генератори стабілізують випадкові величини одного й того ж
    рейсу при зміні розкладу; виробничої причинності це саме по собі не доводить.
    """
    for name, x in [('start_min', start_min), ('max_hours', max_hours), ('observe_every_min', observe_every_min)]:
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0:
            raise ValueError(f'{name}: потрібне невід’ємне скінченне число')
    if start_min >= 1440 or max_hours <= 0 or observe_every_min <= 0:
        raise ValueError('Початок має належати добі; горизонт захисту та крок спостереження мають бути додатними')
    env = simpy.Environment(initial_time=start_min)
    cargo_done = env.event()
    total = round(spec.batch_tons * 100)
    remaining, delivered, next_id = total, 0, 0
    pools = {name: simpy.Store(env) for name in ('loader', 'scale', 'unload')}
    trips, operations, events, snapshots, units = [], [], [], [], {}
    truck_state = {}
    generators = {}

    def draw(key, bounds, digits=1):
        """Власний відтворюваний потік для одиниці техніки та виду операції."""
        if key not in generators:
            seed = int.from_bytes(hashlib.sha256(f'{spec.seed}:{key}'.encode()).digest()[:16], 'big')
            generators[key] = random.Random(seed)
        return round(generators[key].uniform(*bounds), digits)

    def record(event, **details):
        events.append({'time_min': env.now, 'elapsed_min': env.now-start_min, 'event': event, **details})

    def ready(row):
        return row['until_min'] is None or env.now < row['until_min']

    def wait_activation(row):
        """Після завершення партії майбутню техніку вже не залучаємо."""
        delay = max(0, row['from_min']-env.now)
        if delay:
            yield env.timeout(delay) | cargo_done
        if cargo_done.triggered:
            return False
        row['activated'] = True
        record('resource_added', resource=row['resource'], unit_id=row['unit_id'])
        return True

    def equipment(row):
        if not (yield env.process(wait_activation(row))):
            return
        yield pools[row['resource']].put(row['unit_id'])
        if row['until_min'] is not None:
            yield env.timeout(max(0, row['until_min']-env.now))
            # Вільну одиницю прибираємо негайно; зайнята зникне після release.
            if row['unit_id'] in pools[row['resource']].items:
                pools[row['resource']].items.remove(row['unit_id'])
            record('resource_retirement_requested', resource=row['resource'], unit_id=row['unit_id'])

    def reserve(truck_id):
        nonlocal remaining, next_id
        if remaining == 0:
            return None
        capacity = max(1, round(draw(f'{truck_id}:capacity', conditions.capacity_tons, 2)*100))
        amount = min(capacity, remaining)
        remaining -= amount
        next_id += 1
        return {'reservation_id': next_id, 'truck_id': truck_id, 'cargo_units': amount,
                'cargo_tons': amount/100, 'sampled_capacity_tons': capacity/100,
                'reserved_at_min': env.now}

    def service(name, duration, truck_id, rid, stage):
        """Окремо обліковуємо чергу, активну роботу та утримання під час перерв."""
        arrival = env.now
        if truck_id in truck_state:
            truck_state[truck_id] = f'waiting_{stage}'
        yield env.process(wait_work(env))
        queue_start = env.now
        while True:
            uid = yield pools[name].get()
            row = units[uid]
            if ready(row):
                break
            # Запит на ресурс і його вилучення могли мати однаковий момент.
        acquired = env.now
        row['busy'] = True
        if truck_id in truck_state:
            truck_state[truck_id] = stage
        segments = []
        yield env.process(operate(env, duration, segments))
        row['busy'] = False
        if truck_id != 'external':
            # Зовнішні зважування створюють чергу, але не оплачуються цим замовленням.
            row['last_work_min'] = env.now
            row['active_min'] += duration
            row['held_min'] += env.now-acquired
        operations.append({'resource': name, 'unit_id': uid, 'truck_id': truck_id,
                           'reservation_id': rid, 'stage': stage, 'arrival_min': arrival,
                           'acquired_min': acquired, 'released_min': env.now,
                           'queue_min': acquired-queue_start, 'pre_break_min': queue_start-arrival,
                           'active_min': duration, 'held_min': env.now-acquired, 'active_segments': segments})
        if ready(row):
            yield pools[name].put(uid)

    def truck(row):
        nonlocal delivered
        uid = row['unit_id']
        if not (yield env.process(wait_activation(row))):
            return
        truck_state[uid] = 'ready'
        reservation = reserve(uid)
        trip_no = 0
        while reservation is not None:
            trip_no += 1
            # Резервування наступного вантажу виконується до повернення,
            # як у статичній моделі: водій знає, чи має наступне завдання.
            trip_start = env.now
            load = draw(f'{uid}:load', conditions.load_min)
            road = draw(f'{uid}:road', conditions.road_min)
            road *= 1+draw(f'{uid}:delay', (0, conditions.road_delay_percent), 4)/100
            unload = draw(f'{uid}:unload', conditions.unload_min)
            rid = reservation['reservation_id']
            for name, duration, stage in [('scale', conditions.weigh_min, 'tare'),
                                          ('loader', load, 'load'), ('scale', conditions.weigh_min, 'gross')]:
                yield env.process(service(name, duration, uid, rid, stage))
            truck_state[uid] = 'outbound'
            yield env.timeout(road)
            yield env.process(service('unload', unload, uid, rid, 'unload'))
            delivered += reservation['cargo_units']
            entry = {**reservation, 'trip_no': trip_no, 'start_min': trip_start,
                     'finish_min': env.now, 'return_finish_min': None,
                     'load_min': load, 'road_min': road, 'unload_min': unload}
            trips.append(entry)
            record('cargo_delivered', truck_id=uid, reservation_id=rid, cargo_tons=entry['cargo_tons'])
            if delivered == total and not cargo_done.triggered:
                cargo_done.succeed(env.now)
                record('loading_finished')
            # Після запиту на виведення вже прийнятий рейс завершується,
            # але новий вантаж цьому автомобілю не призначається.
            following = reserve(uid) if ready(row) else None
            truck_state[uid] = 'returning'
            yield env.timeout(conditions.return_min)
            entry['return_finish_min'] = env.now
            row['last_work_min'] = env.now
            row['active_min'] += conditions.return_min + road
            reservation = following
        truck_state[uid] = 'released'
        record('truck_returned', truck_id=uid)

    def external_scale():
        """Фонове зважування: параметр не дорівнює фактичній частці зайнятості."""
        gap = conditions.weigh_min*(100-conditions.external_scale_load)/conditions.external_scale_load
        while not cargo_done.triggered:
            yield env.timeout(gap) | cargo_done
            if cargo_done.triggered:
                return
            yield env.process(service('scale', conditions.weigh_min, 'external', None, 'external'))

    def observe():
        """Знімок містить лише поточні дані; майбутні результати додаються окремо."""
        while True:
            yield env.timeout(0)
            snapshots.append({'time_min': env.now, 'elapsed_min': env.now-start_min,
                              'delivered_tons': delivered/100, 'unreserved_tons': remaining/100,
                              'in_transit_or_reserved_tons': (total-remaining-delivered)/100,
                              'queues': {k: len(p.get_queue) for k, p in pools.items()},
                              'free_resources': {k: len(p.items) for k, p in pools.items()},
                              'truck_states': dict(truck_state)})
            yield env.timeout(observe_every_min)

    processes = []
    for window in spec.resources:
        for index in range(window.count):
            uid = f'{window.group_id}:{index+1}'
            row = {'unit_id': uid, 'resource': window.resource, 'group_id': window.group_id,
                   'from_min': start_min+window.available_from_min,
                   'until_min': None if window.available_until_min is None else start_min+window.available_until_min,
                   'activated': False, 'busy': False, 'last_work_min': None, 'active_min': 0, 'held_min': 0}
            units[uid] = row
            process = env.process(truck(row) if window.resource == 'truck' else equipment(row))
            if window.resource == 'truck':
                processes.append(process)
    env.process(observe())
    if conditions.external_scale_load:
        env.process(external_scale())
    all_trucks = simpy.AllOf(env, processes)
    env.run(until=all_trucks | env.timeout(max_hours*60))
    complete = delivered == total and all_trucks.triggered
    loading_finish = cargo_done.value if cargo_done.triggered else None
    summary = {'scenario_id': spec.scenario_id, 'seed': spec.seed,
               'status': 'complete' if complete else 'incomplete', 'start_min': start_min,
               'loading_finish_min': loading_finish, 'all_returns_finish_min': env.now if complete else None,
               'loading_hours': (loading_finish-start_min)/60 if loading_finish is not None else None,
               'all_returns_hours': (env.now-start_min)/60 if complete else None,
               'simulation_hours': (env.now-start_min)/60, 'batch_tons': spec.batch_tons,
               'delivered_tons': delivered/100, 'unreserved_tons': remaining/100,
               'reserved_not_delivered_tons': (total-remaining-delivered)/100,
               'total_trips': len(trips)}
    usage = []
    for row in units.values():
        if not row['activated']:
            end = row['from_min']
        elif row['last_work_min'] is not None:
            end = row['last_work_min']
        else:
            # Залучена, але невикористана одиниця не стає безкоштовною.
            # Її очікування закінчується плановим виведенням або кінцем погрузки.
            end = min(x for x in (row['until_min'], loading_finish, env.now) if x is not None)
        usage.append({'resource': row['resource'], 'unit_id': row['unit_id'],
                      'activated': row['activated'], 'engaged_from_min': row['from_min'],
                      'last_work_min': row['last_work_min'], 'accounting_finish_min': end,
                      'engaged_hours': max(0, end-row['from_min'])/60,
                      'active_hours': row['active_min']/60, 'held_hours': row['held_min']/60})
    days = []
    if loading_finish is not None:
        for i in range(max(1, math.ceil((loading_finish-start_min)/1440))):
            a, b = start_min+i*1440, min(start_min+(i+1)*1440, loading_finish)
            days.append({'day_from_start': i+1, 'from_min': a, 'until_min': b,
                         'observed_hours': (b-a)/60, 'full_day': b-a == 1440,
                         'delivered_tons': sum(t['cargo_tons'] for t in trips if a < t['finish_min'] <= b)})
    result = {'summary': summary, 'scenario': spec.to_dict(), 'conditions': asdict(conditions),
              'trips': trips, 'operations': operations, 'resource_usage': usage,
              'events': events, 'snapshots': snapshots, 'daily': days}
    result['costs'] = calculate_costs(result)
    if daily_min_tons is not None:
        result['contract'] = evaluate_contract(result, daily_min_tons)
    return result


def write_result(result, output):
    """Записати один самодостатній результат і його походження без перезапису."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Каталог результатів має бути новим або порожнім')
    output.mkdir(parents=True, exist_ok=True)
    path = output/'result.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    sources = [Path(__file__), Path(__file__).with_name('resource_schedule.py'),
               Path(__file__).with_name('vessel_cycle.py'), Path(__file__).with_name('daily_contract.py')]
    manifest = {'python': platform.python_version(), 'simpy': simpy.__version__,
                'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                'output_sha256': {'result.json': hashlib.sha256(path.read_bytes()).hexdigest()}}
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main(argv=None):
    """План за замовчуванням; --execute явно дозволяє один прогін."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', type=Path, required=True, help='JSON із ресурсами та інтервалами доступності')
    parser.add_argument('--execute', action='store_true', help='Виконати симуляцію')
    parser.add_argument('--output', type=Path, help='Новий каталог результатів')
    parser.add_argument('--start-time', default='08:15', help='Час команди на погрузку, ГГ:ХХ')
    parser.add_argument('--max-hours', type=float, default=1440, help='Гранична тривалість захисного горизонту')
    parser.add_argument('--conditions', type=Path, help='JSON з експертними параметрами операцій')
    parser.add_argument('--rates', type=Path, help='JSON зі сценарними ставками в гривнях')
    parser.add_argument('--daily-min-tons', type=float, help='Мінімум тонн за кожні повні 24 години; наприклад 4000')
    args = parser.parse_args(argv)
    spec = ScenarioSpec.from_dict(json.loads(args.scenario.read_text(encoding='utf-8-sig')))
    try:
        hours, minutes = map(int, args.start_time.split(':'))
        if not (0 <= hours < 24 and 0 <= minutes < 60):
            raise ValueError
    except ValueError:
        parser.error('Час початку має бути у форматі ГГ:ХХ, від 00:00 до 23:59')
    if args.execute:
        if args.output is None:
            parser.error('Для виконання потрібен --output')
        conditions = Conditions(**json.loads(args.conditions.read_text(encoding='utf-8-sig'))) if args.conditions else Conditions()
        rates = Rates(**json.loads(args.rates.read_text(encoding='utf-8-sig'))) if args.rates else Rates()
        result = simulate(spec, conditions, start_min=hours*60+minutes, max_hours=args.max_hours,
                          daily_min_tons=args.daily_min_tons)
        result['costs'] = calculate_costs(result, rates)
        write_result(result, args.output)
        print(json.dumps(result['summary'], ensure_ascii=False, indent=2))
        if 'contract' in result:
            print(json.dumps(result['contract'], ensure_ascii=False, indent=2))
        if result['summary']['status'] != 'complete':
            raise SystemExit('Партію не завершено: перевірте розклад ресурсів і захисний горизонт')
    else:
        print(json.dumps({'mode': 'лише план, симуляцію не запущено', 'scenario': spec.to_dict(),
                          'start_time': args.start_time, 'daily_min_tons': args.daily_min_tons}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
