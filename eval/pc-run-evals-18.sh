#!/bin/bash
# Batch 18 (2026-10-07, RTX 4090 under WSL): the final long-context checkpoint (Phase C 1299, plain config) on the short-eval suite, so the
# chat-vs-base comparison in section 7B is against the right base; then plain-config RULER at 128K. Waits for batch 17. usage: pc-run-evals-18.sh
export PATH=/usr/lib/wsl/lib:$HOME/ai/bin:$PATH TOKENIZERS_PARALLELISM=false
R=~/kisoku-eval/results; M=~/kisoku-eval/models; LOG=~/logs/kisoku-eval-18.log; P=$M/kisoku-1.6b-base-longC-1299
while tmux has-session -t kisoku-eval17 2>/dev/null; do sleep 60; done
echo "=== BATCH 18 START $(date -u)" >> $LOG
run() { local name=$1 pre=$2 tasks=$3 fs=$4 bs=$5; shift 5
  ls $R/$name/*/results_*.json >/dev/null 2>&1 && { echo "--- skip $name" >> $LOG; return; }
  echo "--- $(date -u) $name tasks=$tasks fewshot=$fs bs=$bs" >> $LOG
  local fsarg=(); [ "$fs" != "default" ] && fsarg=(--num_fewshot "$fs")
  lm_eval --model hf --model_args "pretrained=$pre,dtype=bfloat16" --include_path ~/kisoku-eval/tasks --tasks "$tasks" "${fsarg[@]}" \
    --batch_size "$bs" --output_path "$R/$name" --log_samples "$@" >> $LOG 2>&1
  ls $R/$name/*/results_*.json >/dev/null 2>&1 && echo "--- $(date -u) $name done" >> $LOG || echo "--- $(date -u) $name FAILED" >> $LOG
}
ruler() { local name=$1 pre=$2 len=$3 lim=$4 out="$R/$1-ruler-$3"
  ls $out/*/results_*.json >/dev/null 2>&1 && { echo "--- skip $name $len" >> $LOG; return; }
  echo "--- $(date -u) $name ruler len=$len limit=$lim" >> $LOG
  lm_eval --model hf --model_args "pretrained=$pre,dtype=bfloat16,max_length=$len" --tasks ruler --metadata "{\"max_seq_lengths\":[$len]}" \
    --limit $lim --batch_size 1 --gen_kwargs max_gen_toks=128,max_new_tokens=128 --output_path "$out" >> $LOG 2>&1
  ls $out/*/results_*.json >/dev/null 2>&1 && echo "--- $(date -u) $name $len done" >> $LOG || echo "--- $(date -u) $name $len FAILED" >> $LOG
}
mkyarn() { [ -f $2/config.json ] && return; mkdir -p $2 && for f in $1/*; do ln -sf $(readlink -f $f) $2/; done && rm $2/config.json && python - $1/config.json $2/config.json $3 $4 <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); orig = int(sys.argv[3]); f = float(sys.argv[4])
theta = (c.get('rope_parameters') or {}).get('rope_theta', c.get('rope_theta', 1000000))
c['rope_parameters'] = {'rope_theta': theta, 'rope_type': 'yarn', 'factor': f, 'original_max_position_embeddings': orig}
c['rope_scaling'] = {'rope_type': 'yarn', 'factor': f, 'original_max_position_embeddings': orig}
c['max_position_embeddings'] = max(int(orig * f), orig)
json.dump(c, open(sys.argv[2], 'w'), indent=2)
PY
}
hub() { [ -f $M/$2/config.json ] || python -c "from huggingface_hub import snapshot_download; snapshot_download('$1', local_dir='$M/$2')" >> $LOG 2>&1; }
run kisoku-longC-core $P "hellaswag,arc_easy,arc_challenge,piqa,winogrande" 0 16; run kisoku-longC-gsm8k $P gsm8k 5 16; run kisoku-longC-humaneval $P humaneval 0 16 --confirm_run_unsafe_code
run kisoku-longC-bbhws $P bbh_ws default 16; run kisoku-longC-mmlu $P mmlu 5 8; run kisoku-longC-triviaqa $P triviaqa 5 16
ruler kisoku-longC $P 131072 50
echo "=== BATCH 18 DONE $(date -u)" >> $LOG
