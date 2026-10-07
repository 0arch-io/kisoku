#!/usr/bin/env bash
# Kisoku 1.6B release flip. DRY RUN BY DEFAULT: prints every step, changes nothing.
# RUN=1 ./go-public.sh does it for real, in this order:
#   1. Hugging Face: make 0arch-io/kisoku-1.6b, -chat, -gguf and the dataset kisoku-1.6b-eval public
#   2. GitHub: make 0arch-io/kisoku public
#   3. 0arch.io: merge the kisoku-release branch into main and deploy
# Each step is idempotent, so a failed run can be repeated. Nothing here runs without RUN=1.
set -euo pipefail
RUN="${RUN:-0}"; ORG="${ORG:-0arch-io}"
HF="${HF:-$HOME/.local/bin/hf}"
HFPY="${HFPY:-$(head -1 "$HF" | sed 's/^#!//')}"
SITE="${SITE:-$HOME/Projects/Web/0arch-site}"
CF_ACCOUNT="${CLOUDFLARE_ACCOUNT_ID:-a5cf778848345609e846a5241cadf285}"
run() { if [ "$RUN" = 1 ]; then echo "+ $*"; "$@"; else echo "[dry-run] $*"; fi; }
[ "$RUN" = 1 ] && echo "# LIVE RUN" || echo "# DRY RUN (RUN=1 to execute)"

echo "# 1. Hugging Face repos -> public"
for r in "model:$ORG/kisoku-1.6b" "model:$ORG/kisoku-1.6b-chat" "model:$ORG/kisoku-1.6b-gguf" "dataset:$ORG/kisoku-1.6b-eval"; do
  t="${r%%:*}"; id="${r#*:}"
  run "$HFPY" -c "from huggingface_hub import HfApi; HfApi().update_repo_settings('$id', repo_type='$t', private=False); print('$id public')"
done

echo "# 2. GitHub repo -> public"
run gh repo edit "$ORG/kisoku" --visibility public --accept-visibility-change-consequences \
  --description "Kisoku 1.6B: a from-scratch language model trained solo on a TPU grant. Weights, code, evals and report." \
  --homepage "https://0arch.io/kisoku"

echo "# 3. 0arch.io -> release pages"
run git -C "$SITE" checkout main
run git -C "$SITE" pull --rebase
run git -C "$SITE" merge --no-edit kisoku-release
run git -C "$SITE" push
run env CLOUDFLARE_ACCOUNT_ID="$CF_ACCOUNT" npm --prefix "$SITE" run deploy

echo "# Done. Check: https://huggingface.co/$ORG/kisoku-1.6b  https://github.com/$ORG/kisoku  https://0arch.io/kisoku"
echo "# Then post: release/LAUNCH-POSTS.md"
