"""Prepare full-cycle replays and extend the retained terminal interface."""
from pathlib import Path
import csv,hashlib,json,re,shutil,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.dissertation_model import simulate,Conditions
from src.static_contract_study import specification

SITE=ROOT/'demo'
CASES=[(9,2,2,0,0),(11,2,2,0,0),(12,2,2,0,0),(16,2,3,0,0),
       (10,2,2,50,5),(12,2,2,50,5),(12,2,2,5,30),(11,2,2,5,15)]
def replays():
    source=ROOT/'results/conditions_contract_ml_2026_11_07/physical_runs.csv'
    with source.open(encoding='utf-8-sig') as f:
        controls={r['configuration']:r for r in csv.DictReader(f) if r['phase']=='control' and r['seed']=='10000'}
    scenes=[]
    for n,l,b,external,road in CASES:
        key=f'{n}_{l}_{b}_{external}_{road}'
        result=simulate(specification(n,l,b,10000),Conditions(external_scale_load=external,road_delay_percent=road),daily_min_tons=4000)
        expected=controls[key]
        for field,value in [('loading_hours',result['summary']['loading_hours']),('cost_uah',result['costs']['total'])]:
            assert abs(float(expected[field])-value)<1e-6,(key,field)
        by_res={}
        for op in result['operations']:
            if op['truck_id']!='external':by_res.setdefault(op['reservation_id'],[]).append(op)
        segments=[];trips=[]
        for trip in result['trips']:
            truck=int(trip['truck_id'].split(':')[-1]);number=trip['trip_no'];cargo=trip['cargo_tons']
            ops={op['stage']:op for op in by_res[trip['reservation_id']]}
            def add(state,a,z,phase=None,station=None):
                if z>a+1e-8:segments.append(dict(truck=truck,trip=number,cargo=cargo,state=state,start=a,end=z,phase=phase or state,station=station))
            for phase in ['tare','load','gross','unload']:
                op=ops[phase];station={'loader':'loader','unload':'unloader','scale':'scale'}[op['resource']]
                a=op['arrival_min'];q=a+op['pre_break_min']
                add('break_wait',a,q,phase,station);add('queue',q,op['acquired_min'],phase,station)
                add(phase,op['acquired_min'],op['released_min'],phase,station)
            add('travel',ops['gross']['released_min'],ops['unload']['arrival_min'])
            add('return',trip['finish_min'],trip['return_finish_min'])
            trips.append(dict(truck_id=truck,trip_id=number,cargo_weight=cargo,start_time=trip['start_min'],finish_time=trip['finish_min'],return_finish=trip['return_finish_min'],load_time=trip['load_min'],unload_time=trip['unload_min']))
        segments.sort(key=lambda s:(s['start'],s['truck'],s['end']))
        for truck in range(1,n+1):
            seq=sorted((s for s in segments if s['truck']==truck),key=lambda s:s['start'])
            assert all(abs(a['end']-z['start'])<1e-6 for a,z in zip(seq,seq[1:])),(key,truck)
        assert abs(sum(t['cargo_weight'] for t in trips)-10000)<1e-6
        scenes.append(dict(configuration=key,contract=True,trucks=n,loaders=l,unloaders=b,external=external,road=road,
          trips=trips,segments=segments,tonnes=10000,deliveries=len(trips),summary=result['summary'],days=result['contract']['days'],
          usage=result['resource_usage'],active=[dict(resource=o['resource'],segments=o['active_segments']) for o in result['operations'] if o['truck_id']!='external'],
          external_intervals=[dict(arrival=o['arrival_min']+o['pre_break_min'],start=o['acquired_min'],end=o['released_min']) for o in result['operations'] if o['truck_id']=='external'],
          expected_cost=result['costs']['total']))
    notice='// Copyright (c) 2026 Serhii Bakhariev. SPDX-License-Identifier: MIT\n'
    (SITE/'replays.js').write_text(notice+'window.CONTRACT_REPLAYS='+json.dumps(scenes,ensure_ascii=False,separators=(',',':'))+';\n',encoding='utf-8',newline='\n')
    print(f'Prepared {len(scenes)} complete replays; each matches saved seed 10000 duration and cost.')

def page():
    source=SITE/'source/terminal-resources-v3.html'
    html=source.read_text(encoding='utf-8-sig')
    html=html.replace('<title>Термінал · ресурси та прогнози</title>','<title>Термінал · ресурси, прогнози та повний цикл</title>')
    html=html.replace('</style>',(SITE/'contract.css').read_text(encoding='utf-8')+'\n</style>',1)
    switch='''<div class="model-switch" aria-label="Модель дослідження"><button id="mode-baseline">Модель статті · 24 години</button><button id="mode-contract" class="active">Партія 10 000 т</button><span id="model-caption"></span></div>'''
    html=html.replace('<main>','<main>'+switch,1)
    html=html.replace('<section id="resource-view">','<section id="resource-view"><div id="contract-milestones" class="contract-milestones" hidden></div>',1)
    html=html.replace('<div class="chart-foot">','<div id="daily-results" hidden></div><div class="chart-foot">',1)
    html=html.replace('<h3>Правила розрахунку</h3>','<h3>Правила розрахунку</h3>',1)
    html=html.replace('function stage(s){','function stage(s){if(s.phase)return s.phase;',1)
    html=html.replace('function load(n){','function load(n){if(window.isContractMode?.()){window.loadContract(n);return;}',1)
    html=html.replace("function renderForecast(){const v=values()","function renderForecast(){if(window.isContractMode?.())return;const v=values()",1)
    html=html.replace("const by=id=>document.getElementById(id);\n  const inputIds", "const by=id=>document.getElementById(id);\n  const inputIds",1)
    html=html.replace("time>=1935","time>=replayEnd()").replace('Math.min(1935,','Math.min(replayEnd(),').replace('??1935','??replayEnd()').replace('time===1935','time===replayEnd()')
    # Preserve the baseline chart, with a duration-aware horizontal axis in the new mode.
    html=html.replace('(t-495)/1440*(R-L)','(t-495)/(replayEnd()-495)*(R-L)')
    html=html.replace('[495,855,1215,1575,1935]','[0,1,2,3,4].map(i=>495+(replayEnd()-495)*i/4)')
    html=html.replace("t===1935?'end'","t===replayEnd()?'end'")
    html=html.replace("(t===1935?' (+1 доба)':'')","(t>=1935?' (+'+Math.floor((t-495)/1440)+' д.)':'')")
    html=html.replace('495+(xx-L)/(R-L)*1440','495+(xx-L)/(R-L)*(replayEnd()-495)')
    html=html.replace("const $=id=>", "function replayEnd(){return sc?.contract?sc.summary.all_returns_finish_min:1935;}\nconst $=id=>",1)
    html=html.replace('· 2 навантажувальні / 2 розвантажувальні ресурси','· ${sc.loaders||2} навантажувальні / ${sc.unloaders||2} розвантажувальні ресурси')
    html=html.replace("+(time>=1440?2:1)","+(Math.floor((time-495)/1440)+1)")
    html=html.replace("${time>=1440?2:1}","${Math.floor((time-495)/1440)+1}")
    # Static resource counts in the retained renderer become scenario-dependent.
    html=html.replace("k==='scale'?1:2","k==='scale'?1:k==='load'?(sc.loaders||2):(sc.unloaders||2)")
    html=html.replace("col.append(head,el('div','subtitle',subtitle))","col.append(head,el('div','subtitle',k==='unload'?'Комплектів подачі: '+(sc.unloaders||2):k==='load'?'Мехлопат: '+(sc.loaders||2):subtitle))")
    footer='<footer class="simple-footer">© 2026 Serhii Bakhariev · <a href="https://github.com/serbakh/transport-terminal-demo">Вихідний код</a> · <a href="LICENSE">MIT</a></footer>'
    html=html.replace('</main>','</main>'+footer,1)
    html=html.replace('</html>','<script src="data.js"></script><script src="replays.js"></script><script src="contract-math.js"></script><script src="contract.js"></script></html>')
    html=html.replace('</style>','</style><link rel="stylesheet" href="presentation.css">',1)
    html += '\n<script src="presentation.js"></script>\n'
    html=html.replace('Собівартість за врахованими статтями','Витрати на тонну за врахованими статтями')
    (SITE/'index.html').write_text(html,encoding='utf-8',newline='\n')
    # README is maintained separately from generated HTML.
    print('Built retained terminal interface with two model modes.')
if __name__=='__main__':
    if '--page-only' not in sys.argv:replays()
    page()
