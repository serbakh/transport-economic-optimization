"""Build a static, traceable demonstration from saved results, without ML/simulation."""
from pathlib import Path
import csv, hashlib, json, statistics, subprocess

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/conditions_contract_ml_2026_11_07'
SITE=ROOT/'demo'
FIELDS=['loading_hours','trips','truck_hours','loader_hours','loader_active_hours','scale_hours','feeder_hours','feeder_active_hours','first_day_tons','minimum_full_day_tons','cost_uah']
def read(name):
    with (SOURCE/name).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
def numeric(row):
    return {k:float(row[k]) for k in FIELDS}
def main():
    SITE.mkdir(exist_ok=True)
    predictions=[dict(configuration=r['configuration'],context=r['context'],admissible=r['predicted_admissible']=='True',**numeric(r)) for r in read('predictions_1836.csv')]
    groups={}
    samples={}
    for r in read('physical_runs.csv'):
        if r['phase']!='control': continue
        groups.setdefault(r['configuration'],[]).append(r)
        if r['seed']=='10000': samples[r['configuration']]=dict(seed=10000,eligible=r['eligible']=='True',**numeric(r))
    controls={k:dict(runs=len(rows),passed=sum(r['eligible']=='True' for r in rows),**{f:statistics.mean(float(r[f]) for r in rows) for f in FIELDS}) for k,rows in groups.items()}
    days={}
    for r in read('daily.csv'):
        if r['phase']=='control' and r['seed']=='10000':
            days.setdefault(r['configuration'],[]).append(dict(day=int(r['day_from_start']),hours=float(r['observed_hours']),tons=float(r['delivered_tons']),full=r['full_day']=='True',passed=r['meets_minimum']=='True'))
    baseline=[dict(context=r['context'],configuration=r['resource_configuration'],old=float(r['baseline_mean']),new=float(r['new_mean']),difference=float(r['difference_percent'])) for r in read('baseline_comparison.csv')]
    names=['predictions_1836.csv','physical_runs.csv','daily.csv','baseline_comparison.csv','decisions_by_conditions.csv','evaluation.json','manifest.json']
    provenance={name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in names}
    data=dict(experiment='conditions_contract_ml_2026_11_07',research_commit='4dfa5d63f737bf0a3d610465a71bd9422c7634b3',baseline_commit='b8e2dfdadc81ca2e122bc8881a8db9d3a8e773e1',predictions=predictions,controls=controls,samples=samples,days=days,baseline=baseline,evaluation=json.loads((SOURCE/'evaluation.json').read_text(encoding='utf-8-sig')),provenance=provenance)
    notice='// Copyright (c) 2026 Serhii Bakhariev. SPDX-License-Identifier: MIT\n'
    (SITE/'data.js').write_text(notice+'window.STUDY='+json.dumps(data,ensure_ascii=False,separators=(',',':'))+';\n',encoding='utf-8')
    (SITE/'provenance.json').write_text(json.dumps({k:data[k] for k in ['experiment','research_commit','baseline_commit','provenance']},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Built demo: {len(predictions)} predictions, {len(controls)} control summaries, {len(samples)} recorded examples, {len(baseline)} baseline comparisons')
if __name__=='__main__': main()
