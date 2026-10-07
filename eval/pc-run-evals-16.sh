#!/bin/bash
# Batch 16 (2026-10-06, RTX 4090 under WSL): the released chat model (pass 9) on HumanEval (executes code, so it runs here, not on the
# shared CTI box) and RULER with YaRN x2 at 4K, 32K and 64K, so the report's chat rows match the model that ships. Waits for batch 15a.
# usage: pc-run-evals-16.sh ACCESS_TOKEN
export PATH=/usr/lib/wsl/lib:$HOME/ai/bin:$PATH HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false
R=~/kisoku-eval/results; M=~/kisoku-eval/models; LOG=~/logs/kisoku-eval-16.log; C9=$M/kisoku-1.6b-chat-sft009; C9Y=$M/kisoku-1.6b-chat-sft009-yarn2
if [ ! -f $C9/model.safetensors ]; then mkdir -p $C9; for f in config.json generation_config.json tokenizer.json tokenizer_config.json special_tokens_map.json chat_template.jinja model.safetensors; do
  curl -sf -H "Authorization: Bearer $1" -o $C9/$f "https://storage.googleapis.com/kisoku-v2-training/hf/kisoku-1.6b-chat-sft009/$f" || { echo "download failed: $f" >> $LOG; exit 1; }; done; fi
while tmux has-session -t kisoku-eval15a 2>/dev/null; do sleep 60; done
echo "=== BATCH 16 (chat pass 9: HumanEval, RULER YaRN x2) START $(date -u)" >> $LOG
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
if [ ! -f $C9Y/config.json ]; then mkdir -p $C9Y && for f in $C9/*; do ln -sf $(readlink -f $f) $C9Y/; done && rm $C9Y/config.json && python - $C9/config.json $C9Y/config.json <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); theta = (c.get('rope_parameters') or {}).get('rope_theta', c.get('rope_theta'))
c['rope_parameters'] = {'rope_theta': theta, 'rope_type': 'yarn', 'factor': 2.0, 'original_max_position_embeddings': 65536}
c['rope_scaling'] = {'rope_type': 'yarn', 'factor': 2.0, 'original_max_position_embeddings': 65536}; c['max_position_embeddings'] = 131072
json.dump(c, open(sys.argv[2], 'w'), indent=2)
PY
fi
run kisoku-chat9-humaneval $C9 humaneval 0 16 --confirm_run_unsafe_code
ruler kisoku-chat9-yarn2 $C9Y 4096 100; ruler kisoku-chat9-yarn2 $C9Y 32768 50; ruler kisoku-chat9-yarn2 $C9Y 65536 50
echo "=== BATCH 16 DONE $(date -u)" >> $LOG
