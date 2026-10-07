#!/usr/bin/env bash
# Kisoku 1.6B Hugging Face upload. DRY RUN BY DEFAULT: prints every command, runs nothing.
# Set RUN=1 to execute. All repos are created PRIVATE; flip to public by hand on release day.
#
#   ./upload.sh              # print the plan
#   RUN=1 ./upload.sh        # actually do it (needs `hf auth login` as 0arch-io, and gsutil read access)
#
# Steps: (1) download from the bucket into STAGE (read-only gsutil cp), (2) patch configs and add cards,
# (3) create private repos, (4) upload. Rows that are not ready are skipped with a note; see MANIFEST.md.
: "${KISOKU_BUCKET:?set KISOKU_BUCKET to your GCS bucket name (no gs:// prefix)}"
set -euo pipefail

ORG="${ORG:-0arch-io}"
BUCKET="${BUCKET:-gs://${KISOKU_BUCKET}}"
STAGE="${STAGE:-$HOME/kisoku-release-staging}"
REL="$(cd "$(dirname "$0")" && pwd)"          # this release/ directory (cards live here)
HF="${HF:-hf}"                                # use HF=huggingface-cli on older installs
RUN="${RUN:-0}"

run() { if [ "$RUN" = "1" ]; then echo "+ $*"; "$@"; else echo "[dry-run] $*"; fi; }
note() { echo "# $*"; }

[ "$RUN" = "1" ] && note "LIVE RUN" || note "DRY RUN (set RUN=1 to execute)"

# ---------- 1. download (read-only on the bucket) ----------
# dir name in bucket -> local name under STAGE
fetch() { # fetch <bucket subpath> <local dir>
  [ "${SKIP_FETCH:-0}" = 1 ] && return 0   # SKIP_FETCH=1 re-runs the upload steps against an existing STAGE
  run mkdir -p "$STAGE/$2"
  run gsutil -m cp -r "$BUCKET/$1/*" "$STAGE/$2/"
}
fetch hf/kisoku-1.6b-base-longC-1299 base-main
fetch hf/kisoku-1.6b-base-s1-198999  base-stage1
fetch hf/kisoku-1.6b-base-s3-242999  base-stage3
fetch hf/kisoku-1.6b-base-longA-2899 base-longA
fetch hf/kisoku-1.6b-base-longB-2859 base-longB
fetch hf/kisoku-1.6b-chat-sft009     chat-main
fetch hf/gguf-kisoku-1.6b-chat-sft009 gguf
fetch hf/gguf-kisoku-1.6b-base-longC-1299 gguf
fetch hf/kisoku-1.6b-chat-sft005 chat-sft005
fetch hf/kisoku-1.6b-chat-sft006 chat-sft006
fetch hf/kisoku-1.6b-chat-dpo38 chat-dpo38

# ---------- 2. patch configs, add cards ----------
# The exported config.json has rope_scaling null. Ship YaRN x2 over the 64K training length (DECISION in MANIFEST.md).
# The plain config is kept next to it as config.plain.json for people who stay under 32K.
patch_yarn() { # patch_yarn <dir>
  run python3 - "$STAGE/$1/config.json" <<'PY'
import json, shutil, sys
p = sys.argv[1]
shutil.copy(p, p.replace("config.json", "config.plain.json"))
c = json.load(open(p))
theta = c["rope_parameters"]["rope_theta"]
c["max_position_embeddings"] = 131072
c["rope_parameters"] = {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 65536, "rope_theta": theta}
c["rope_scaling"] = {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 65536}
json.dump(c, open(p, "w"), indent=2)
print("patched", p)
PY
}
patch_yarn base-main
patch_yarn chat-main
run cp "$REL/kisoku-1.6b/README.md"      "$STAGE/base-main/README.md"
run cp "$REL/kisoku-1.6b-chat/README.md" "$STAGE/chat-main/README.md"
run cp "$REL/kisoku-1.6b-gguf/README.md" "$STAGE/gguf/README.md"
note "Verify the patched config loads: python -c 'from transformers import AutoConfig; print(AutoConfig.from_pretrained(\"$STAGE/base-main\").rope_scaling)'"

# ---------- 3. create private repos ----------
for r in kisoku-1.6b kisoku-1.6b-chat kisoku-1.6b-gguf; do
  run "$HF" repo create "$ORG/$r" --repo-type model --private --exist-ok
done

# ---------- 4. upload ----------
run "$HF" upload "$ORG/kisoku-1.6b"      "$STAGE/base-main" . --repo-type model --commit-message "Kisoku 1.6B base (final, long-context Phase C step 1299)"
run "$HF" upload "$ORG/kisoku-1.6b-chat" "$STAGE/chat-main" . --repo-type model --commit-message "Kisoku 1.6B Chat preview (pass 9)"
run "$HF" upload "$ORG/kisoku-1.6b-gguf" "$STAGE/gguf" . --repo-type model --include "*.gguf" --include "README.md" --commit-message "GGUF builds"

# Per-stage checkpoints as branches of the base repo (DECISION: confirm which to publish; stage 2 is missing).
HFPY="${HFPY:-$(head -1 "$(command -v "$HF")" | sed 's/^#!//')}"   # the python that has huggingface_hub (the hf CLI's own)
branch() { # branch <repo> <branch name> <local dir>
  run "$HFPY" -c "from huggingface_hub import HfApi; HfApi().create_branch('$ORG/$1', branch='$2', exist_ok=True)"
  run "$HF" upload "$ORG/$1" "$STAGE/$3" . --repo-type model --revision "$2" --commit-message "checkpoint $2"
}
branch kisoku-1.6b stage1-step198999    base-stage1
branch kisoku-1.6b stage3-step242999    base-stage3
branch kisoku-1.6b long-phaseA-step2899 base-longA
branch kisoku-1.6b long-phaseB-step2859 base-longB
# stage2-step223999: MISSING, no stage 2 checkpoint in the bucket (report section 12 says so)
# Chat passes the report compares (decided 2026-10-07: 5, 6 and the preference-tuned pass)
branch kisoku-1.6b-chat sft005 chat-sft005
branch kisoku-1.6b-chat sft006 chat-sft006
branch kisoku-1.6b-chat dpo38  chat-dpo38

note "Done. Repos are private. Make public by hand after review: $HF repo settings, or the web UI."
