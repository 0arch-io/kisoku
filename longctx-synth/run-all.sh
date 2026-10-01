#!/bin/bash
cd ~/synth; PY=~/.venv/bin/python; D=gs://kisoku-v2-training/datasets/longctx-synth-tasks
rm -rf out src; $PY gen_synth.py > gen.log 2>&1
$PY - > manifest.txt 2>&1 <<'PY'
import json,glob,os,collections
T=collections.Counter(); toks=[]; nqa=[]; err=0; kinds=collections.Counter(); files=sorted(glob.glob('out/part-*.arrayrecord'))
for f in glob.glob('out/*.json'):
    for s in json.load(open(f))['stats']:
        if 'error' in s: err+=1; continue
        T.update(s['tasks']); toks.append(s['tokens']); nqa.append(s['nqa']); kinds[s['kind']]+=1
n=len(files)
for i,f in enumerate(files): os.rename(f, f'out/longctx-synth-tasks-{i:05d}-of-{n:05d}.arrayrecord')
b=[0,8000,16000,32000,48000,10**9]; hist={f'{b[i]}-{b[i+1]}':sum(1 for t in toks if b[i]<=t<b[i+1]) for i in range(5)}
m={'records':len(toks),'shards':n,'total_tokens':sum(toks),'avg_tokens':sum(toks)/len(toks),'max_tokens':max(toks),'avg_qa_per_doc':sum(nqa)/len(nqa),'by_source':dict(kinds),'tasks':dict(T),'length_hist':hist,'errors':err}
json.dump(m,open('out/manifest.json','w'),indent=1); print(json.dumps(m,indent=1))
PY
mkdir -p meta; mv out/*.arrayrecord.json meta/; tar czf out/samples-and-stats.tgz meta; cp gen_synth.py out/
gcloud storage cp --quiet out/* $D/ > upload.log 2>&1; echo "UPLOAD rc=$? $(date -u)" >> gen.log
sudo shutdown -h +2
