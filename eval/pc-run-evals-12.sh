#!/bin/bash
# Batch 12 (2026-10-05): RULER for the Kisoku Phase C final (64K-trained, step 1299), same settings as batches 8 and 11.
# Plain config for 4K-64K. For 128K a second model dir (same weights, symlinked) carries the YaRN x2 config
# (rope_type yarn, factor 2, original_max_position_embeddings 65536, max_position_embeddings 131072); it is also
# run at 64K and 4K to measure what static YaRN costs inside the trained range. Headline lengths run first.
export PATH=/usr/lib/wsl/lib:$HOME/ai/bin:$PATH
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=~/kisoku-eval/results; M=~/kisoku-eval/models; LOG=~/logs/kisoku-eval.log
SRC=/mnt/x/WSL/kisoku-hf/kisoku-1.6b-base-longC-1299; P=$M/kisoku-1.6b-base-longC-1299; Y=$M/kisoku-1.6b-base-longC-1299-yarn2
[ -f $P/model.safetensors ] || cp -r $SRC $M/
mkdir -p $Y && for f in $P/*; do ln -sf $f $Y/; done && rm $Y/config.json && python3 - $P/config.json $Y/config.json <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c['rope_parameters'] = {'rope_theta': c['rope_parameters']['rope_theta'], 'rope_type': 'yarn', 'factor': 2.0, 'original_max_position_embeddings': 65536}
c['rope_scaling'] = {'rope_type': 'yarn', 'factor': 2.0, 'original_max_position_embeddings': 65536}
c['max_position_embeddings'] = 131072
json.dump(c, open(sys.argv[2], 'w'), indent=2)
PY
echo "=== BATCH 12 (Kisoku Phase C RULER) START $(date -u)" >> $LOG
run() { local name=$1 pre=$2 len=$3 lim=$4 out="$R/$1-ruler-$3"
  if ls $out/*/results_*.json >/dev/null 2>&1; then echo "--- skip $name $len (done)" >> $LOG; return; fi
  for try in 1 2; do
    echo "--- $(date -u) $name ruler len=$len limit=$lim try=$try" >> $LOG
    lm_eval --model hf --model_args "pretrained=$pre,dtype=bfloat16,max_length=$len" --tasks ruler \
      --metadata "{\"max_seq_lengths\":[$len]}" --limit $lim --batch_size 1 --gen_kwargs max_gen_toks=128,max_new_tokens=128 --output_path "$out" >> $LOG 2>&1
    ls $out/*/results_*.json >/dev/null 2>&1 && { echo "--- $(date -u) $name $len done" >> $LOG; return; }
    sleep 60
  done
  echo "--- $(date -u) $name $len FAILED (no results file)" >> $LOG
}
run kisoku-longC $P 65536 50
run kisoku-longC-yarn2 $Y 131072 50
run kisoku-longC $P 32768 50
run kisoku-longC $P 16384 50
run kisoku-longC $P 8192 100
run kisoku-longC $P 4096 100
run kisoku-longC-yarn2 $Y 65536 50
run kisoku-longC-yarn2 $Y 4096 100
echo "=== BATCH 12 DONE $(date -u)" >> $LOG
