#!/bin/bash
# Build the SFT set with one process per source, then merge and upload.
# Runs the QUICK variant first; stops if it fails so the full run can't waste time on a broken script.
set -u
cd "$HOME/sft-build" || exit 1
PY="$HOME/maxtext/.venv/bin/python"
export HF_HOME="$HOME/sft-build/hf-home" HF_HUB_CACHE="$HOME/.cache/huggingface/hub"

build() {  # $1 = label
  local label=$1 pids=() fail=0
  for s in $($PY build_sft.py --list 2>/dev/null | tail -1); do
    $PY -u build_sft.py --source "$s" > "$HOME/logs/sft-$label-$s.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ "$fail" = 0 ] || { echo "$(date -u +%T) $label: a source FAILED"; grep -l Traceback "$HOME"/logs/sft-$label-*.log; return 1; }
  grep -h "kept" "$HOME"/logs/sft-$label-*.log | cut -c1-260
  $PY -u build_sft.py --merge 2>&1 | grep -v -i -e warn -e pytorch || return 1
}

echo "$(date -u +%T) quick build start"
QUICK=1 build quick || { echo "QUICK FAILED, stopping"; exit 1; }
echo "$(date -u +%T) full build start"
build full || { echo "FULL FAILED"; exit 1; }
echo "$(date -u +%T) uploading"
$PY -u "$HOME/bin/sft_data_sync.py" up || exit 1
echo "$(date -u +%T) ALL DONE"
