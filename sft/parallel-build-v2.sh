#!/bin/bash
# Build the final SFT set (build_sft_v2.py), one process per source, quick test build first.
# usage: SFT_DIR=/path PY=/path/to/python parallel-build-v2.sh    (SFT_DIR holds identity.jsonl and deepseek/*.jsonl)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; PY="${PY:-python3}"; export SFT_DIR="${SFT_DIR:-$HOME/sft-build}"
mkdir -p "$SFT_DIR/logs"; cd "$HERE" || exit 1
build() { local label=$1 pids=() fail=0
  for s in $($PY build_sft_v2.py --list 2>/dev/null | tail -1); do
    $PY -u build_sft_v2.py --source "$s" > "$SFT_DIR/logs/$label-$s.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ "$fail" = 0 ] || { echo "$(date -u +%T) $label: a source FAILED"; grep -l Traceback "$SFT_DIR"/logs/$label-*.log; return 1; }
  grep -h "kept" "$SFT_DIR"/logs/$label-*.log | cut -c1-300
  $PY -u build_sft_v2.py --merge 2>&1 | grep -v -i -e warn -e pytorch || return 1
}
echo "$(date -u +%T) quick build start"
QUICK=1 build quick || { echo "QUICK FAILED, stopping"; exit 1; }
[ -n "${QUICK_ONLY:-}" ] && exit 0
echo "$(date -u +%T) full build start"
build full || { echo "FULL FAILED"; exit 1; }
echo "$(date -u +%T) BUILD DONE"
