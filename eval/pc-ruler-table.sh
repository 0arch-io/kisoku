#!/bin/bash
python3 - <<'PY'
import json,glob,os,collections
# The 'ruler' group row only aggregates 4096, so average the 13 subtasks ourselves.
T=collections.defaultdict(dict)
for f in sorted(glob.glob(os.path.expanduser('~/kisoku-eval/results/*-ruler-*/*/results_*.json'))):
    name=f.split('/')[-3]; model,L=name.rsplit('-ruler-',1); d=json.load(open(f))
    v=[m[f'{L},none'] for t,m in d['results'].items() if t!='ruler' and isinstance(m.get(f'{L},none'),(int,float)) and m[f'{L},none']>=0]
    T[model][int(L)]=(sum(v)/len(v),len(v)) if v else None
Ls=[4096,8192,16384,32768,65536,131072]
print('%-16s'%'RULER avg'+''.join('%8d'%l for l in Ls))
for m,r in T.items(): print('%-16s'%m+''.join('%8s'%('-' if not r.get(l) else '%.1f'%(100*r[l][0])+('' if r[l][1]==13 else '*')) for l in Ls))
PY
tr '\r' '\n' < ~/logs/kisoku-eval.log | grep -E "^(===|---)" | tail -2
