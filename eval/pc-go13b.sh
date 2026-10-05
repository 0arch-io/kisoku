#!/bin/bash
tmux new -d -s kisoku-eval13b bash /mnt/x/WSL/pc-run-evals-13b.sh; sleep 3; tmux ls; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
