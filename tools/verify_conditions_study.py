"""Перевірити походження, повноту та добові обмеження нового експерименту."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify(output):
    import numpy as np
    import pandas as pd
    from src.static_contract_study import cost_from_physical

    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    plan = json.loads((output / 'plan.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'complete', 'Експеримент не завершено'
    for name, digest in manifest['source_sha256'].items():
        assert hashlib.sha256((ROOT / name.replace('\\', '/')).read_bytes()).hexdigest() == digest, name
    for name, digest in manifest['output_sha256'].items():
        assert hashlib.sha256((output / name.replace('\\', '/')).read_bytes()).hexdigest() == digest, name
    for line in (ROOT / 'baseline/SHA256SUMS.txt').read_text(encoding='utf-8-sig').splitlines():
        digest, name = line.split(maxsplit=1)
        assert hashlib.sha256((ROOT / 'baseline' / name).read_bytes()).hexdigest() == digest, name
    runs = pd.read_csv(output / 'physical_runs.csv')
    days = pd.read_csv(output / 'daily.csv')
    keys = ['phase', 'configuration', 'seed']
    assert len(runs) == 27540 and not runs.duplicated(keys).any()
    training, control = runs[runs.phase == 'training'], runs[runs.phase == 'control']
    assert len(training) == 18360 and len(control) == 9180
    assert set(training.seed) == set(plan['training_seeds'])
    assert set(control.seed) == set(plan['control_seeds'])
    assert not set(training.seed) & set(control.seed)
    for frame, repeats in [(training, 10), (control, 5)]:
        assert frame.configuration.nunique() == 1836
        assert frame.context.nunique() == 12
        assert (frame.groupby('configuration').size() == repeats).all()
        assert (frame.groupby('context').configuration.nunique() == 153).all()
    assert np.allclose(days.groupby(keys).delivered_tons.sum(), 10000, rtol=0, atol=1e-6)
    full = days[days.full_day]
    calculated = full.assign(passed=full.delivered_tons >= 4000).groupby(keys).passed.all()
    assert calculated.sort_index().equals(runs.set_index(keys).eligible.sort_index())
    assert np.allclose(full.groupby(keys).delivered_tons.min().sort_index(),
                       runs.set_index(keys).minimum_full_day_tons.sort_index(), rtol=0, atol=1e-6)
    for row in runs.to_dict('records'):
        assert abs(cost_from_physical(row) - row['cost_uah']) < 1e-6
    split = json.loads((output / 'split.json').read_text(encoding='utf-8'))
    assert not set(split['train_resource_configurations']) & set(split['test_resource_configurations'])
    assert len(split['test_resource_configurations']) == 31
    test = training[training.resource_configuration.isin(split['test_resource_configurations'])]
    assert len(test) == 3720 and test.context.nunique() == 12
    decisions = pd.read_csv(output / 'decisions_by_conditions.csv')
    assert len(decisions) == 12 and decisions.context.nunique() == 12
    predictions = pd.read_csv(output / 'predictions_1836.csv')
    summary = pd.read_csv(output / 'control_summary.csv')
    for row in decisions.to_dict('records'):
        candidates = predictions[(predictions.context == row['context']) & predictions.predicted_admissible]
        if len(candidates):
            winner = candidates.sort_values(['cost_uah', 'configuration']).iloc[0]
            assert winner.configuration == row['ml_configuration']
            actual = summary[summary.configuration == winner.configuration].iloc[0]
            assert actual.passed == row['control_passed']
            assert actual.runs == row['control_runs']
    checks = dict(source_and_output_hashes=True, baseline_unchanged=True, rows=len(runs),
                  scenarios=1836, contexts=12, separate_seeds=True, full_day_contract_verified=True,
                  cost_reconstruction_verified=True, heldout_resource_configurations=31,
                  economic_selection_verified=True)
    (output / 'verification.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ml-library-dir', type=Path)
    args = parser.parse_args()
    if args.ml_library_dir:
        sys.path.insert(0, str(args.ml_library_dir.resolve()))
    print(json.dumps(verify(args.output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
