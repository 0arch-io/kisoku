#!/bin/bash
# Remount the dataset bucket if it is not mounted (runs before every training start).
: "${KISOKU_BUCKET:?set KISOKU_BUCKET to your GCS bucket name (no gs:// prefix)}"
mkdir -p "$HOME/gcsfuse"
if ! mountpoint -q "$HOME/gcsfuse"; then
  gcsfuse --key-file="$HOME/gcs-key.json" --implicit-dirs -o ro ${KISOKU_BUCKET} "$HOME/gcsfuse"
fi
mountpoint -q "$HOME/gcsfuse"
