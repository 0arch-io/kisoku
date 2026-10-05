#!/bin/bash
# Run the Kisoku data generator: stage 1 (small targeted set), then stage 2 (the full set). Resumable; safe to rerun.
# usage: PY=/path/to/python sft/run-gen.sh
cd "$(dirname "$0")" || exit 1
PY="${PY:-python3}"
"$PY" -u gen_kisoku.py run --stage 1 >> ../data/gen2/_run.log 2>&1
"$PY" -u gen_kisoku.py run --stage 2 >> ../data/gen2/_run.log 2>&1
