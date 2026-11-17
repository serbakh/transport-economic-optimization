"""Формат майбутніх динамічних сценаріїв; цей модуль не запускає симуляцію.

Час — хвилини від початку замовлення, не від півночі.
Вікно задає план доступності. Фактичне звільнення зайнятої одиниці
майбутній рушій має записувати окремо, завершивши поточне завдання.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

RESOURCE_TYPES = ('truck', 'loader', 'scale', 'unload')


def nonnegative_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} має бути скінченним невід’ємним числом')
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} має бути скінченним невід’ємним числом')


@dataclass(frozen=True)
class ResourceWindow:
    """Група однакових одиниць: [available_from_min, available_until_min).

    None означає доступність до завершення партії. Групи додаються:
    два записи truck з count=9 і count=2 — це 11 машин під час перетину.
    group_id однозначно ідентифікує групу в межах сценарію.
    """
    group_id: str
    resource: str
    count: int
    available_from_min: float = 0
    available_until_min: float | None = None

    def __post_init__(self):
        if not isinstance(self.group_id, str) or not self.group_id.strip():
            raise ValueError('group_id має бути непорожнім рядком')
        if self.resource not in RESOURCE_TYPES:
            raise ValueError(f'resource має належати до {RESOURCE_TYPES}')
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 1:
            raise ValueError('count має бути додатним цілим числом')
        nonnegative_number(self.available_from_min, 'available_from_min')
        if self.available_until_min is not None:
            nonnegative_number(self.available_until_min, 'available_until_min')
            if self.available_until_min <= self.available_from_min:
                raise ValueError('Кінець доступності має бути пізнішим за початок')

    def scheduled_at(self, elapsed_min):
        nonnegative_number(elapsed_min, 'elapsed_min')
        return (self.available_from_min <= elapsed_min and
                (self.available_until_min is None or elapsed_min < self.available_until_min))


@dataclass(frozen=True)
class ScenarioSpec:
    """Контракт сценарію; описує умови, але не гарантує завершення роботи."""
    scenario_id: str
    batch_tons: float
    seed: int
    resources: tuple[ResourceWindow, ...]
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError('Підтримується лише schema_version=1')
        if not isinstance(self.scenario_id, str) or not self.scenario_id.strip():
            raise ValueError('scenario_id має бути непорожнім рядком')
        nonnegative_number(self.batch_tons, 'batch_tons')
        if self.batch_tons <= 0 or abs(self.batch_tons*100-round(self.batch_tons*100)) > 1e-7:
            raise ValueError('Маса партії має бути додатною та кратною 0,01 т')
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError('seed має бути цілим числом')
        if not isinstance(self.resources, tuple) or not all(isinstance(w, ResourceWindow) for w in self.resources):
            raise ValueError('resources має бути кортежем об’єктів ResourceWindow')
        ids = [w.group_id for w in self.resources]
        if len(ids) != len(set(ids)):
            raise ValueError('group_id має бути унікальним у межах сценарію')
        missing = set(RESOURCE_TYPES) - {w.resource for w in self.resources}
        if missing:
            raise ValueError(f'Відсутні типи ресурсів: {sorted(missing)}')

    def scheduled_counts(self, elapsed_min):
        """Плановий склад; не фактичний стан обладнання та не оплачуваний час."""
        nonnegative_number(elapsed_min, 'elapsed_min')
        return {name: sum(w.count for w in self.resources
                          if w.resource == name and w.scheduled_at(elapsed_min))
                for name in RESOURCE_TYPES}

    def to_dict(self):
        value = asdict(self)
        value['resources'] = [asdict(w) for w in self.resources]
        return value

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError('Сценарій має бути об’єктом JSON')
        fields = {'scenario_id', 'batch_tons', 'seed', 'resources', 'schema_version'}
        extra = set(value) - fields
        if extra:
            raise ValueError(f'Невідомі поля сценарію: {sorted(extra)}')
        required = fields - {'schema_version'}
        if required - set(value):
            raise ValueError(f'Відсутні поля сценарію: {sorted(required-set(value))}')
        if not isinstance(value['resources'], list):
            raise ValueError('resources має бути масивом JSON')
        windows = []
        for item in value['resources']:
            if not isinstance(item, dict):
                raise ValueError('Кожне вікно доступності має бути об’єктом JSON')
            try:
                windows.append(ResourceWindow(**item))
            except TypeError as exc:
                raise ValueError(f'Некоректні поля вікна доступності: {exc}') from exc
        return cls(**{**value, 'resources': tuple(windows)})


def main(argv=None):
    """Перевірити сценарій та показати плановий склад у моменти його зміни."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scenario', type=Path)
    args = parser.parse_args(argv)
    spec = ScenarioSpec.from_dict(json.loads(args.scenario.read_text(encoding='utf-8-sig')))
    boundaries = sorted({w.available_from_min for w in spec.resources} |
                        {w.available_until_min for w in spec.resources if w.available_until_min is not None})
    print(json.dumps({'status': 'configuration_only_not_simulated', 'scenario': spec.to_dict(),
                      'planned_changes': [{'elapsed_min': t, 'scheduled_counts': spec.scheduled_counts(t)}
                                          for t in boundaries]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
