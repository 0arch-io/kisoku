#!/bin/bash
export PATH=/usr/lib/wsl/lib:$HOME/ai/bin:$PATH; export TOKENIZERS_PARALLELISM=false
M=~/kisoku-eval/models/kisoku-1.6b-base-longA-2500
mk() { T=~/kisoku-eval/models/_kisoku-longA-$1; rm -rf $T; mkdir -p $T; for f in $M/*; do ln -s $f $T/; done; rm $T/generation_config.json; echo "$2" > $T/generation_config.json; }
mk clean   '{"bos_token_id": 128000, "eos_token_id": 128001, "pad_token_id": 128001}'
mk cleanmin '{"bos_token_id": 128000, "eos_token_id": 128001, "pad_token_id": 128001, "min_new_tokens": 2}'
for tag in clean cleanmin; do
  rm -rf ~/kisoku-eval/results/_mn-$tag
  for try in 1 2 3; do
    lm_eval --model hf --model_args "pretrained=$HOME/kisoku-eval/models/_kisoku-longA-$tag,dtype=bfloat16,max_length=32768" --tasks ruler --metadata '{"max_seq_lengths":[4096]}' --limit 30 --batch_size 1 --output_path ~/kisoku-eval/results/_mn-$tag > ~/logs/ruler-mn-$tag.log 2>&1
    ls ~/kisoku-eval/results/_mn-$tag/*/results_*.json >/dev/null 2>&1 && break; sleep 30
  done
done
python - <<'PY'
import json,glob,os
R={}; tags=('base','clean','cleanmin')
for tag in tags:
    fs=sorted(glob.glob(os.path.expanduser(f'~/kisoku-eval/results/_mn-{tag}/*/results_*.json')))
    if not fs: print('missing',tag); continue
    d=json.load(open(fs[0])); R[tag]={t:m['4096,none'] for t,m in d['results'].items() if t!='ruler'}
print('%-18s'%'task'+''.join('%10s'%t for t in R))
for t in R['base']: print('%-18s'%t+''.join('%10.1f'%(100*R[g][t]) for g in R))
print('%-18s'%'AVERAGE'+''.join('%10.1f'%(100*sum(R[g].values())/13) for g in R))
PY
