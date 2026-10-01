#!/bin/bash
python3 - <<'PY'
import json,glob,os
R={}
for tag in ('base','clean','cleanmin'):
    fs=sorted(glob.glob(os.path.expanduser(f'~/kisoku-eval/results/_mn-{tag}/*/results_*.json')))
    if fs: R[tag]={t:m['4096,none'] for t,m in json.load(open(fs[0]))['results'].items() if t!='ruler'}
print('DONE' if len(R)==3 else 'PENDING', list(R))
if len(R)==3:
    print('%-18s %9s %9s %9s'%('task','penalty','no-pen','no-pen+min2'))
    for t in R['base']: print('%-18s'%t+''.join('%10.1f'%(100*R[g][t]) for g in R))
    print('%-18s'%'AVERAGE'+''.join('%10.1f'%(100*sum(R[g].values())/13) for g in R))
PY
