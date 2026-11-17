"""Дискретно-подієва симуляція погрузки завершеної суднової партії.

Це генератор прикладів для майбутнього машинного навчання, а не ML-модель.
Програма розігрує роботу техніки та вимірює час доставки всієї партії.
Вона поки не навчає прогнозну модель і не розраховує грошові витрати.

Шлях автомобіля: тара → навантаження → брутто → дорога → розвантаження.
Повернення на склад виконується лише перед наступним зарезервованим рейсом.
Модель зупиняється після останньої тонни на судні, а не через 24 години.

Одиниці: час середовища SimPy — хвилини; маса на вході/виході — тонни;
внутрішній облік маси — цілі соті тонни; підсумкові тривалості — години.
Наприклад, час 495 означає 08:15 першої умовної доби. Наступні доби
продовжують цей відлік, тому 1935 — це 08:15 наступної доби.

Логіку операцій і календар узято за основу з
baseline/src/simulation/digital_twin.py. Цей файл не імпортуємо:
імпорт baseline запустив би всю стару серію експериментів.
Поточна версія не додає сторонній потік вагової чи дорожні затримки.

Як читати код: Config → simulate → reserve/service/truck → write_results.
SimPy перемикає віртуальні процеси в місцях yield. env.timeout(20)
означає 20 модельних хвилин, а не очікування 20 хвилин на комп'ютері.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import simpy

# Календар повторюється щодоби. Кінець кожного інтервалу не включається:
# о 01:00, 08:15, 13:00 та 20:15 нові операції вже можна починати.
START = 495.0  # Початок роботи: 8 × 60 + 15 = 08:15.
# Нічний обід; ранкова пересмінка; денний обід; вечірня пересмінка.
BREAKS = ((0, 60), (480, 495), (720, 780), (1200, 1215))


def break_remaining(now):
    """Залишок поточної перерви у хвилинах; нуль, якщо зараз робочий час."""
    # Остача від ділення переводить будь-який день у хвилини його доби.
    minute = now % 1440
    return next((end - minute for start, end in BREAKS if start <= minute < end), 0)


def next_break(now):
    """Кількість хвилин до початку найближчої перерви."""
    if break_remaining(now):
        return 0
    day = now - now % 1440  # Початок поточної доби в загальній шкалі часу.
    # Переглядаємо і наступну добу, щоб коректно перейти через північ.
    return min(day + offset + start - now for offset in (0, 1440)
               for start, _ in BREAKS if day + offset + start > now)


def wait_work(env):
    """Відкласти дію до робочого часу; інші процеси SimPy продовжуються."""
    while break_remaining(env.now):
        yield env.timeout(break_remaining(env.now))


def operate(env, duration, active_segments):
    """Виконати duration активних хвилин операції з календарними перервами.

Ресурс уже отриманий у service і утримується до завершення операції,
зокрема під час її зупинки на обід. active_segments накопичує тільки
відрізки фактичного виконання, без часу призупинення.
За правилом baseline дозволено допрацювати до 1 хв після початку перерви.
"""
    yield env.process(wait_work(env))
    remaining = duration
    # Малий допуск запобігає зайвій ітерації через похибку дробового часу.
    while remaining > 1e-9:
        available = next_break(env.now)
        # Якщо після початку перерви залишилося б понад 1 хв — виконуємо
        # лише частину до перерви, чекаємо, потім продовжуємо залишок.
        chunk = remaining if remaining <= available + 1 else available
        start = env.now
        yield env.timeout(chunk)
        active_segments.append((start, env.now))
        remaining -= chunk
        if remaining > 1e-9:
            yield env.process(wait_work(env))


@dataclass(frozen=True)
class Config:
    """Параметри одного експерименту; frozen забороняє їх зміну після створення.

    seed фіксує послідовність псевдовипадкових чисел для повторення запуску.
    Він не є виробничою характеристикою чи майбутньою вхідною ознакою ML.
    """
    batch_tons: float = 10000
    trucks: int = 9
    loaders: int = 2
    unload_points: int = 2
    seed: int = 42

    def validate(self):
        """Перевірити обсяг і кількість ресурсів до запуску подій."""
        if not math.isfinite(self.batch_tons) or self.batch_tons <= 0:
            raise ValueError('batch_tons must be finite and positive')
        # Обсяг має бути кратним 0,01 т: саме таку точність має облік вантажу.
        if abs(self.batch_tons * 100 - round(self.batch_tons * 100)) > 1e-7:
            raise ValueError('batch_tons must be a multiple of 0.01 tonnes')
        for name in ('trucks', 'loaders', 'unload_points'):
            value = getattr(self, name)
            # У Python True є підтипом int, але не осмисленою кількістю машин.
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer')


def simulate(config=Config()):
    """Виконати один цикл і повернути підсумок та журнали без запису файлів.

    Усі змінні запуску локальні: повторний виклик не підхоплює старі рейси.
    Кількість вагових постів у цій версії фіксована — одна вагова.
    """
    config.validate()
    rng = random.Random(config.seed)  # Власний генератор цього запуску.
    env = simpy.Environment(initial_time=START)
    # Store містить номери вільних одиниць: наприклад, мехлопати [1, 2].
    # get() забирає номер; коли всі зайняті, автомобіль чекає у черзі.
    # put() повертає номер після роботи. Так можна вести облік кожної одиниці.
    pools = {}
    for name, count in (('scale', 1), ('loader', config.loaders),
                        ('unload', config.unload_points)):
        pool = simpy.Store(env, capacity=count)
        pool.items.extend(range(1, count + 1))
        pools[name] = pool
    total_units = round(config.batch_tons * 100)  # 10 000 т = 1 000 000 одиниць.
    remaining = total_units  # Ще НЕ зарезервований вантаж, не весь залишок на складі.
    delivered = 0  # Уже розвантажено на судно.
    # total_units - remaining - delivered: зарезервовано, але ще не доставлено.
    next_id = 0
    completed = env.event()  # Сигнал завершення, якого чекатиме env.run().
    trips, operations = [], []  # Рейси та окремі операції на ресурсах.

    def reserve():
        """Закріпити доступну частину вантажу за одним майбутнім рейсом.

        Резервування — диспетчерське рішення без витрат модельного часу.
        Це ще не фізична погрузка: вона відбудеться після тарування.
        """
        # Тут немає yield: SimPy не переключиться на іншу машину посеред
        # віднімання залишку. Два рейси не отримають одну й ту саму останню тонну.
        nonlocal remaining, next_id
        if remaining == 0:
            return None
        capacity_units = round(round(rng.uniform(27, 28), 2) * 100)
        cargo = min(remaining, capacity_units)  # Останній рейс може бути неповним.
        remaining -= cargo
        next_id += 1
        return {'reservation_id': next_id, 'reserved_at_min': env.now,
                'cargo_units': cargo, 'sampled_capacity_tons': capacity_units / 100}

    def service(resource, duration, truck_id, reservation_id, stage):
        """Одна операція: очікування робочого часу → черга → робота → звільнення.

        stage розрізняє тарування й брутто, хоча ресурс у них спільний — scale.
        pre_break — очікування до входу в чергу; queue — час самої черги
        (може охоплювати перерви); break_while_held — перерви після отримання
        ресурсу. Ці величини не додаються повторно одна до одної.
        """
        arrival = env.now
        yield env.process(wait_work(env))
        queue_start = env.now
        unit = yield pools[resource].get()  # За відсутності вільної одиниці чекаємо.
        acquired = env.now
        segments = []
        yield env.process(operate(env, duration, segments))
        released = env.now
        operations.append({
            'truck_id': truck_id, 'reservation_id': reservation_id,
            'stage': stage, 'resource': resource, 'unit_id': unit,
            'arrival_min': arrival, 'acquired_min': acquired,
            'released_min': released, 'pre_break_min': queue_start - arrival,
            'queue_min': acquired - queue_start, 'active_min': duration,
            'held_min': released - acquired,  # Активна робота + перерви на ресурсі.
            'break_while_held_min': released - acquired - duration,
            'active_segments': segments,
        })
        yield pools[resource].put(unit)  # Ресурс знову доступний наступній машині.

    def truck(truck_id):
        """Життєвий цикл одного автомобіля, що працює поряд з іншими.

        Рейси різних машин перекриваються в часі, але черги обмежують
        одночасне користування ваговою, мехлопатами та точками подачі.
        """
        nonlocal delivered
        reservation = reserve()
        trip_no = 0
        previous_finish = None
        while reservation is not None:
            trip_no += 1
            # Наступний вантаж резервуємо ще на причалі після розвантаження.
            # Якщо залишку немає, автомобіль завершує роботу. Це явне
            # припущення диспетчеризації: фінальні повернення не моделюємо.
            return_start = env.now
            if previous_finish is not None:
                yield env.timeout(20)
            start = env.now  # Початок поточного рейсу на складі, ПІСЛЯ повернення.
            # Рівномірні розподіли успадковані від baseline. Навіть останній
            # неповний кузов тут має звичайні тривалості погрузки/розвантаження.
            load_time = round(rng.uniform(8, 10), 1)
            travel_time = round(rng.uniform(18, 25), 1)
            unload_time = round(rng.uniform(12, 15), 1)
            rid = reservation['reservation_id']
            for resource, duration, stage in (
                ('scale', 3, 'tare'), ('loader', load_time, 'load'),
                ('scale', 3, 'gross')):
                yield env.process(service(resource, duration, truck_id, rid, stage))
            # Дорога не має власної черги; обіди не зупиняють уже початий рух.
            yield env.timeout(travel_time)
            yield env.process(service('unload', unload_time, truck_id, rid, 'unload'))
            delivered += reservation['cargo_units']  # Зараховуємо лише після розвантаження.
            trips.append({**reservation, 'truck_id': truck_id, 'trip_no': trip_no,
                          'cargo_tons': reservation['cargo_units'] / 100,
                          'return_before_min': 0 if previous_finish is None else 20,
                          'return_started_min': return_start,
                          'start_min': start, 'finish_min': env.now,
                          'load_min': load_time, 'travel_min': travel_time,
                          'unload_min': unload_time})
            previous_finish = env.now
            # Рівність точна завдяки цілочисельному обліку сотих тонни.
            if delivered == total_units:
                completed.succeed()
                return
            reservation = reserve()  # None завершує цикл саме цього автомобіля.

    # Реєструємо всі процеси перед запуском годинника. Це не виконання
    # повного циклу першої машини перед початком роботи другої.
    for truck_id in range(1, config.trucks + 1):
        env.process(truck(truck_id))
    env.run(until=completed)  # Межа визначається вантажем, а не заданим часом.
    # Сортування робить журнал зручним для читання; порядок резервування
    # не обов'язково збігається з порядком прибуття на судно.
    trips.sort(key=lambda row: row['reservation_id'])
    operations.sort(key=lambda row: (row['acquired_min'], row['resource'], row['unit_id']))
    hours = (env.now - START) / 60
    # active_hours — фізичні операції; held_hours — зайнятість разом з
    # перервами; reserved_batch_hours — умовне залучення на весь цикл.
    # Жоден з цих показників автоматично не встановлює оплачувані мотогодини.
    resources = []
    for name, count in (('scale', 1), ('loader', config.loaders),
                        ('unload', config.unload_points)):
        for unit in range(1, count + 1):
            rows = [r for r in operations if r['resource'] == name and r['unit_id'] == unit]
            resources.append({'resource': name, 'unit_id': unit,
                              'active_hours': sum(r['active_min'] for r in rows) / 60,
                              'held_hours': sum(r['held_min'] for r in rows) / 60,
                              'reserved_batch_hours': hours})
    # Дві бази для майбутньої економіки: тримати машину весь цикл або
    # звільняти після її останньої доставки. Для машини без рейсів друга — 0.
    trucks = []
    for unit in range(1, config.trucks + 1):
        rows = [r for r in trips if r['truck_id'] == unit]
        trucks.append({'truck_id': unit, 'trips': len(rows),
                       'last_unload_min': max((r['finish_min'] for r in rows), default=START),
                       'reserved_batch_hours': hours,
                       'until_last_unload_hours': (max((r['finish_min'] for r in rows), default=START) - START) / 60})
    # Активні години в підсумку СУМАРНІ за всіма одиницями ресурсу:
    # дві мехлопати за 1 календарну годину можуть дати 2 активні години.
    summary = {
        'config': asdict(config), 'external_scale_load': 0, 'road_delay_percent': 0,
        'start_min': START, 'finish_min': env.now, 'duration_hours': hours,
        'delivered_tons': delivered / 100, 'unreserved_tons': remaining / 100,
        'total_trips': len(trips), 'partial_trips': sum(r['cargo_tons'] < r['sampled_capacity_tons'] for r in trips),
        'last_reserved_cargo_tons': trips[-1]['cargo_tons'],
        'last_delivered_cargo_tons': max(trips, key=lambda r: r['finish_min'])['cargo_tons'],
        'inter_trip_return_hours': sum(r['return_before_min'] for r in trips) / 60,
        'loader_active_hours': sum(r['active_min'] for r in operations if r['resource'] == 'loader') / 60,
        'unload_active_hours': sum(r['active_min'] for r in operations if r['resource'] == 'unload') / 60,
    }
    return {'summary': summary, 'trips': trips, 'operations': operations,
            'resources': resources, 'trucks': trucks}


def write_results(result, directory):
    """Зберегти результати та дані для перевірки їх походження.

    JSON зберігає підсумок і вкладені відрізки операцій; CSV — плоскі таблиці.
    Файли можна аналізувати без повторного запуску симуляції.
    """
    directory = Path(directory)
    # Не перезаписуємо попередній експеримент випадково.
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty directory: {directory}')
    directory.mkdir(parents=True, exist_ok=True)
    for name in ('summary', 'operations'):
        (directory / f'{name}.json').write_text(
            json.dumps(result[name], ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    for name in ('trips', 'resources', 'trucks'):
        with (directory / f'{name}.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result[name][0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(result[name])
    # SHA-256 — контрольний відбиток вмісту, а не перевірка наукової правильності.
    # Зміна навіть коментаря змінить відбиток коду. Старий manifest описує
    # саме стару версію, якою було створено збережені результати.
    # Переноси рядків коду нормалізуються при читанні; результати пишемо з LF.
    metadata = {'python': platform.python_version(), 'simpy': simpy.__version__,
                'source_sha256_lf': hashlib.sha256(Path(__file__).read_text(encoding='utf-8').encode('utf-8')).hexdigest(),
                'output_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(directory.iterdir()) if p.is_file()}}
    (directory / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8', newline='\n')


def main():
    """Прочитати параметри запуску, виконати симуляцію та записати результат.

    Приклад з кореня проєкту:
    python src/vessel_cycle.py --output outputs/local/example --seed 42
    Без інших аргументів діють 10 000 т і конфігурація 9/2/2.
    Новий запуск потребує порожнього або нового каталогу результатів.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-tons', type=float, default=10000)
    parser.add_argument('--trucks', type=int, default=9)
    parser.add_argument('--loaders', type=int, default=2)
    parser.add_argument('--unload-points', type=int, default=2)
    args = parser.parse_args()
    result = simulate(Config(args.batch_tons, args.trucks, args.loaders, args.unload_points, args.seed))
    write_results(result, args.output)
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    # Прямий запуск виконує main; імпорт у тестах лише надає функції.
    main()
