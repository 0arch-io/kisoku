#!/bin/bash
# Batch 15 (2026-10-06, RTX 4090 under WSL): the two YaRN replication runs that did not finish on the CTI box (batch 14, streams L and N):
# Kisoku final with YaRN x4 at 64K, and Qwen3 1.7B base with YaRN x2 (over its native 32K) at 64K. Same lm_eval settings as batch 14.
# Waits until the llama-server reference runs on this GPU are over (a second llama-server.exe next to the chat one means still busy).
export PATH=/usr/lib/wsl/lib:$HOME/ai/bin:$PATH TOKENIZERS_PARALLELISM=false
R=~/kisoku-eval/results; M=~/kisoku-eval/models; P=$M/kisoku-1.6b-base-longC-1299
# usage: pc-run-evals-15.sh 1|2   (1 = Kisoku YaRN x4, 2 = Qwen3 1.7B YaRN x2; run both at once). Starts once the 8B reference model is
# off the GPU: its GGUF is replaced by the 1B one when it finishes, and at most two llama-servers (chat + 1B reference) are left.
quiet=0; until [ $quiet -ge 3 ]; do n=$(/mnt/c/Windows/System32/tasklist.exe 2>/dev/null | grep -c llama-server); [ "$n" -le 2 ] && [ -f /mnt/x/WSL/llama/ref-llama1b-sys.gguf ] && quiet=$((quiet+1)) || quiet=0; sleep 60; done
LOG=~/logs/kisoku-eval-15-$1.log
echo "=== BATCH 15 (YaRN x4, Qwen3 1.7B YaRN 64K) START $(date -u)" >> $LOG
ruler() { local name=$1 pre=$2 len=$3 lim=$4 out="$R/$1-ruler-$3"
  ls $out/*/results_*.json >/dev/null 2>&1 && { echo "--- skip $name $len" >> $LOG; return; }
  echo "--- $(date -u) $name ruler len=$len limit=$lim" >> $LOG
  lm_eval --model hf --model_args "pretrained=$pre,dtype=bfloat16,max_length=$len" --tasks ruler --metadata "{\"max_seq_lengths\":[$len]}" \
    --limit $lim --batch_size 1 --gen_kwargs max_gen_toks=128,max_new_tokens=128 --output_path "$out" >> $LOG 2>&1
  ls $out/*/results_*.json >/dev/null 2>&1 && echo "--- $(date -u) $name $len done" >> $LOG || echo "--- $(date -u) $name $len FAILED" >> $LOG
}
# mkyarn SRC DST ORIG FACTOR: a copy of a model dir (weights symlinked) whose config applies YaRN at inference
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
if [ "$1" = 1 ]; then mkyarn $P $M/kisoku-longC-yarn4 65536 4; ruler kisoku-longC-yarn4 $M/kisoku-longC-yarn4 65536 50
else hub Qwen/Qwen3-1.7B-Base qwen3-1.7b; mkyarn $M/qwen3-1.7b $M/qwen3-1.7b-yarn2 32768 2; ruler qwen3-1.7b-yarn2 $M/qwen3-1.7b-yarn2 65536 50; fi
echo "=== BATCH 15 DONE $(date -u)" >> $LOG
