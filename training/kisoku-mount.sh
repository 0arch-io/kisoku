#!/bin/bash
# Remount the dataset bucket if it is not mounted (runs before every training start).
mkdir -p "$HOME/gcsfuse"
if ! mountpoint -q "$HOME/gcsfuse"; then
  gcsfuse --key-file="$HOME/gcs-key.json" --implicit-dirs -o ro kisoku-v2-training "$HOME/gcsfuse"
fi
mountpoint -q "$HOME/gcsfuse"
