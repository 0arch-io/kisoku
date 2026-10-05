#!/bin/bash
tmux new -d -s kisoku-eval12 bash /mnt/x/WSL/pc-run-evals-12.sh; sleep 420; tmux ls; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
tr '\r' '\n' < ~/logs/kisoku-eval.log | grep -a -E "^---|Error|error" | tail -3; tr '\r' '\n' < ~/logs/kisoku-eval.log | tail -1 | cut -c1-160
