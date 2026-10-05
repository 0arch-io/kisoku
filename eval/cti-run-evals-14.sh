#!/bin/bash
# Batch 14 (2026-10-05): the rest of the queue on the CTI GPU box (our container 951, 3x RTX PRO 6000 Blackwell shared with other
# services, so each stream is pinned to one card and kept small). Same lm_eval 0.4.13 + transformers 5.17 + settings as the 4090 batches.
# usage: cti-run-evals-14.sh STREAM   (A, B on card 2; C, D on card 1). A sanity rerun of llama-3.2-1b RULER 4K (4090: 73.5) checks
# that numbers from this box match the 4090. HumanEval is NOT run here (it executes generated code; that stays on the 4090 under WSL).
cd /models/kisoku-eval; export PATH=/models/kisoku-eval/venv/bin:$PATH HF_HOME=/models/kisoku-eval/hf TOKENIZERS_PARALLELISM=false
R=results; M=/models/kisoku-eval/models; LOG=logs/stream-$1.log
P=$M/kisoku-1.6b-base-longC-1299; Y=$M/kisoku-1.6b-base-longC-1299-yarn2; V1=$M/kisoku-v1-3b-base
if [ ! -f $Y/config.json ]; then mkdir -p $Y && for f in $P/*; do ln -sf $f $Y/; done && rm $Y/config.json && python - $P/config.json $Y/config.json <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c['rope_parameters'] = {'rope_theta': c['rope_parameters']['rope_theta'], 'rope_type': 'yarn', 'factor': 2.0, 'original_max_position_embeddings': 65536}
c['rope_scaling'] = {'rope_type': 'yarn', 'factor': 2.0, 'original_max_position_embeddings': 65536}
c['max_position_embeddings'] = 131072
json.dump(c, open(sys.argv[2], 'w'), indent=2)
PY
fi
python - $V1 <<'PY'
import json, os, sys
p = os.path.join(sys.argv[1], "generation_config.json"); g = json.load(open(p))
if "repetition_penalty" in g or g.get("do_sample"):
    json.dump(g, open(p + ".orig", "w"), indent=1)
    json.dump({"bos_token_id": 128000, "eos_token_id": 128001, "pad_token_id": 128001, "suppress_tokens": g["suppress_tokens"]}, open(p, "w"), indent=1)
PY
ruler() { local name=$1 pre=$2 len=$3 lim=$4 out="$R/$1-ruler-$3"
  ls $out/*/results_*.json >/dev/null 2>&1 && { echo "--- skip $name $len" >> $LOG; return; }
  echo "--- $(date -u) $name ruler len=$len limit=$lim" >> $LOG
  lm_eval --model hf --model_args "pretrained=$pre,dtype=bfloat16,max_length=$len" --tasks ruler --metadata "{\"max_seq_lengths\":[$len]}" \
    --limit $lim --batch_size 1 --gen_kwargs max_gen_toks=128,max_new_tokens=128 --output_path "$out" >> $LOG 2>&1
  ls $out/*/results_*.json >/dev/null 2>&1 && echo "--- $(date -u) $name $len done" >> $LOG || echo "--- $(date -u) $name $len FAILED" >> $LOG
}
run() { local name=$1 pre=$2 tasks=$3 fs=$4 bs=$5; shift 5
  ls $R/$name/*/results_*.json >/dev/null 2>&1 && { echo "--- skip $name" >> $LOG; return; }
  echo "--- $(date -u) $name tasks=$tasks fewshot=$fs bs=$bs" >> $LOG
  local fsarg=(); [ "$fs" != "default" ] && fsarg=(--num_fewshot "$fs")
  lm_eval --model hf --model_args "pretrained=$pre,dtype=${DT:-bfloat16}" --include_path /models/kisoku-eval/tasks --tasks "$tasks" "${fsarg[@]}" \
    --batch_size "$bs" --output_path "$R/$name" --log_samples "$@" >> $LOG 2>&1
  ls $R/$name/*/results_*.json >/dev/null 2>&1 && echo "--- $(date -u) $name done" >> $LOG || echo "--- $(date -u) $name FAILED" >> $LOG
}
echo "=== STREAM $1 START $(date -u)" >> $LOG
case $1 in
  A) export CUDA_VISIBLE_DEVICES=2; ruler kisoku-longC $P 32768 50; ruler kisoku-longC $P 16384 50 ;;
  B) export CUDA_VISIBLE_DEVICES=2; ruler llama-3.2-1b-ctisanity unsloth/Llama-3.2-1B 4096 100; ruler kisoku-longC $P 4096 100; ruler kisoku-longC $P 8192 100; ruler kisoku-longC-yarn2 $Y 4096 100 ;;
  # v1 runs in float32: its logits are tiny (untrained special rows sit at logit 0 above every real token) and bfloat16
  # rounding turns generation into "!!!!" (HumanEval 0.0, found 2026-10-05). The bf16 results are kept as results/_bf16-kisoku-v1-*.
  C) export CUDA_VISIBLE_DEVICES=1 DT=float32; run kisoku-v1-core $V1 "hellaswag,arc_easy,arc_challenge,piqa,winogrande" 0 16; run kisoku-v1-gsm8k $V1 gsm8k 5 16
     run kisoku-v1-bbhws $V1 bbh_ws default 8; run kisoku-v1-triviaqa $V1 triviaqa 5 8; run kisoku-v1-mmlu $V1 mmlu 5 8 ;;
  D) export CUDA_VISIBLE_DEVICES=1; ruler kisoku-longC-yarn2 $Y 65536 50 ;;
  # Added after YaRN x2 turned out to HELP at 64K (56.7 vs 44.7 plain) and cost nothing at 4K (76.1 vs 76.2): fill in the middle
  # lengths so one shipped config (YaRN) has a full row.
  E) export CUDA_VISIBLE_DEVICES=2; ruler kisoku-longC-yarn2 $Y 8192 100; ruler kisoku-longC-yarn2 $Y 16384 50 ;;
  F) export CUDA_VISIBLE_DEVICES=2; ruler kisoku-longC-yarn2 $Y 32768 50 ;;
esac
echo "=== STREAM $1 DONE $(date -u)" >> $LOG
