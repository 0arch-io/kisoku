# Kisoku 1.6B release manifest

Release date: Friday 2026-10-10. Nothing is public yet. Built 2026-10-07 from a read-only listing of `gs://kisoku-v2-training/{hf,runs,release}/` and the repo. Source of truth for numbers is `report/kisoku-report-draft.md`.

Bucket = `gs://kisoku-v2-training`. Status: **exists** (uploadable as is), **needs patch** (exists, config or card change needed), **needs conversion** (only a MaxText checkpoint exists), **missing** (nothing in the bucket).

## 1. Hugging Face artifacts

Existing 0arch-io repos for reference (read-only search): `kisoku-3b-base`, `kisoku-3b-sft`, `kisoku-3b`, `kisoku-3.2b-base`, `kisoku-1.6b-preview` (MIT) and `kisoku-1.6b-preview-GGUF`. No name clashes with the new repos. The old GGUF repo is named `-GGUF`, so [DECISION] rename `kisoku-1.6b-gguf` to `kisoku-1.6b-GGUF` for consistency (the cards and script use lowercase as requested).

| # | HF repo / revision | Source | Approx size | Public | Status |
|---|---|---|---|---|---|
| 1 | `0arch-io/kisoku-1.6b` main (final base, Phase C step 1299) | `hf/kisoku-1.6b-base-longC-1299/` (7 files) | 3.2 GB | yes | **needs patch**: config.json has `rope_scaling: null` and no YaRN, so the default-YaRN claim in the card is false until patched (script does it); add README |
| 2 | same repo, branch `stage1-step198999` | `hf/kisoku-1.6b-base-s1-198999/` | 3.2 GB | DECISION (default yes, branch) | exists. Not annealed (LR 3e-4 at stop) |
| 3 | same repo, branch `stage2-step223999` | none. `runs/kisoku-v2-1b-stage2-iter-backup` is 25 KB of iterator state only; `runs/kisoku-v2-1b-pretrain-001/checkpoints/` keeps only 239000 to 242999 | n/a | n/a | **missing**. Report section 12 promises stage 2. DECISION: drop the promise, or check the TPU VM and worker backups (`tools/worker0-backup-20261001/`) |
| 4 | same repo, branch `stage3-step242999` | `hf/kisoku-1.6b-base-s3-242999/` | 3.2 GB | DECISION (default yes, branch) | exists. This is the checkpoint behind the 10-benchmark table |
| 5 | same repo, branch `long-phaseA-step2899` | `hf/kisoku-1.6b-base-longA-2899/` (exported 2026-10-07, max positions 32768) | 3.2 GB | DECISION (default yes, branch) | exists |
| 6 | same repo, branch `long-phaseB-step2859` | `hf/kisoku-1.6b-base-longB-2859/` (exported 2026-10-07, max positions 65536) | 3.2 GB | DECISION (default yes, branch) | exists |
| 7 | `0arch-io/kisoku-1.6b-chat` main (pass 9, preview) | `hf/kisoku-1.6b-chat-sft009/` | 3.2 GB | yes | **needs patch**: config.json has `rope_scaling: null`; add README |
| 8 | chat branches `sft005`, `sft006`, `sft007`, `sft008`, `dpo38` | `hf/kisoku-1.6b-chat-sft005..008/`, `hf/kisoku-1.6b-chat-dpo38/` | 3.2 GB each | DECISION (default no; report compares them, so publishing 005, 006 and dpo38 is the useful subset) | exists. sft001 to 004 also exist in `hf/`, not discussed in the report |
| 9 | `0arch-io/kisoku-1.6b-gguf`: `kisoku-1.6b-chat-sft009-F16.gguf` | `hf/gguf-kisoku-1.6b-chat-sft009/` | 3.2 GB | yes | exists (only F16) |
| 10 | same: `kisoku-1.6b-chat-sft009-Q8_0.gguf`, `-Q4_K_M.gguf` | `hf/gguf-kisoku-1.6b-chat-sft009/` (quantized on the Mac 2026-10-07, 1.7 GB and 1.0 GB) | 1.7 GB, 1 GB | yes | exists |
| 11 | same: `kisoku-1.6b-base-longC-1299-F16 / Q8_0 / Q4_K_M.gguf` | `hf/gguf-kisoku-1.6b-base-longC-1299/` (F16 from worker 0, Q8 and Q4 from the Mac, 2026-10-07; plain rope config, YaRN is a llama.cpp runtime flag) | 3.2 / 1.7 / 1 GB | yes | exists (Q8/Q4 uploading) |
| 12 | `0arch-io/kisoku-1.6b-maxtext` (raw MaxText checkpoints for continued training: stage 1, stage 3, A, B, C final) | `runs/kisoku-v2-1b-{stage1,stage3,longctx-a,longctx-b,longctx-c}-final/` | about 12 GB each, 60 GB | DECISION (default no; the report promises "intermediate checkpoints", HF exports cover that) | exists |
| 13 | `0arch-io/kisoku-1.6b-eval` (dataset): raw per-sample results | `eval/results-20261005/`, `eval/cti-batch14/`, `eval/contamination-20261001/`; local `data/eval-results/*.tgz` (83 MB) | about 100 MB | yes | exists. Strip personal paths from logs first |
| 14 | `0arch-io/kisoku-1.6b-sft-data` (dataset): teacher-written conversations only | `sft/kisoku-gen2-20261005/` (about 142 MB local copy `data/gen2/`) and `sft/deepseek-gen-20261002/` | about 150 MB | DECISION | exists. Do not publish merged `sft/kisoku-sft-v10/` parquet (mixes public datasets with their own licences). Check DeepSeek API terms on redistributing outputs. Remove `_spend.json`, `_run*.log`, `_failed.txt` |
| 15 | same dataset or separate: held-out conversation test (297 scripts, judge outputs) | local `data/stress/` (240 MB; `scripts.jsonl`, `judge-*.jsonl`, `rollout-*.jsonl`) | up to 240 MB | DECISION (publish scripts, held-out ids and judge verdicts; rollouts optional) | exists |
| 16 | long-context synthetic task samples | `datasets/longctx-synth-tasks/` (22,386 docs, 575.5M tokens) | large | DECISION (default: generator plus a few hundred samples; text derives from PG19 and Dolma 3 PDFs) | exists |
| 17 | `0arch-io/kisoku-1.6b-preview` and `-preview-GGUF` | already public | n/a | already public | no action. Optionally add a pointer to the new repos |

Pretraining text (`datasets/*`) is not published. Nemotron sets forbid redistribution. Manifests only.

## 2. Code release (GitHub `0arch-io/kisoku`, currently private)

Publish: `training/` (MaxText configs, stage scripts, restart guard, export and fine-tune scripts, systemd units), `eval/` (batch scripts, `tasks/bbh_ws`, `yarn_rehearsal.py`, chat test scripts), `contamination/` (builder, scanner, summary, split-accuracy), `longctx-synth/` (generator, manifest), `sft/` (generators, builders, held-out eval, stress, typo noise, quiz scripts, `*-summary.json`), `report/`, README. Data manifests: `longctx-synth/manifest.json`, `sft/*-summary.json`, `contamination/summary.txt`, `contamination/split-accuracy.txt`, plus a manifest of pretraining sources, counts and mix weights (still to write, no text).

Do not publish: `data/` and `models/` (gitignored, 9.7 GB local), `eval/chat-tests/` outputs unless reviewed, `__pycache__`.

## 3. Must NOT be published: secret and personal-path scan

Searched for `api_key`, `token`, `.deepseek`, `sk-`, `hf_`, private-key blocks, cloud and Google key prefixes, IPs and home paths, across the working tree and the full git history (64 commits).

**No literal secrets found** in files or history. No key, token or key file is committed. `.gitignore` covers `*key*.json`, `hf_token`, `.env`, `data/`, `models/`.

Key file **references** (read from the machine at run time, safe but tell readers to supply their own):

| File | Hit |
|---|---|
| `sft/gen_kisoku.py` (lines 18 to 21) | reads `~/.deepseek-key` or `~/.ollama-key`; URLs `api.deepseek.com`, `ollama.com` |
| `sft/gen_sft.py` (line 18, 128) | reads `~/.ollama-key`, sends as Bearer |
| `longctx-synth/gen_synth.py` (line 155) | reads `~/hf_token` |
| `training/sft_data_sync.py`, `kisoku-mount.sh`, `kisoku-run.sh`, `kisoku-sft-run.sh`, `kisoku-dpo-run.sh`, `kisoku-smoke.sh`, `kisoku-next-stage.sh`, `kisoku-next-stage-c.sh`, `install-stage-c.sh`, `convert-bases.sh`, `convert-longA.sh`, `convert-longC.sh`, `convert-chat.sh`, `export-chat-shm.sh`, `dpo-export-chain.sh`, `build-gguf-chat.sh`, `systemd/kisoku-gcsfuse.service` | `$HOME/gcs-key.json` (GCP service account key path) |
| `.gitignore` | `hf_token`, `.env` (patterns, fine) |

Personal paths, hosts and identifiers to generalise before the repo goes public:

| File | Item |
|---|---|
| `eval/serve-4090.sh` | **Tailscale IP `100.94.140.6` and Windows login `jmmvx@`** in an ssh command. Remove or parametrise. Highest priority |
| `training/kisoku-v2-1b-sft.yml`, `kisoku-v2-1b-dpo.yml` | `/home/josephrodriguez/sft-data/...` (4 lines) |
| `training/*.sh`, `sft_data_sync.py`, `longctx-synth/*`, `contamination/*`, `sft/gen_kisoku.py`, `eval/*` | bucket name `kisoku-v2-training` (about 30 files). Replace with a variable |
| `eval/pc-*.sh`, `eval/cti-*.sh` | host names, WSL paths, `~/kisoku-eval`, `tmux` session names, CTI box references |
| `sft/quiz-heldout.jsonl` (tracked) | model outputs on 561 held-out questions; harmless but review |
| `data/gen2/_spend.json` | DeepSeek spend, `{"usd": 73.39, ...}`: report quotes under $20 for set 2 and $15.23 for set 1; do not publish the file, reconcile figures |
| `README.md` | already lists the pre-public checklist (generalise bucket paths, hosts, IPs) |

Also flag: `sft/*.py` mention DeepSeek by name in system prompts and filters (fine, but consistent with the report's disclosure).

## 4. Report claims the artifacts do not support yet

- Section 12 promises stage 1, 2 and 3 checkpoints. Stage 2 does not exist (row 3).
- Phase A final (2899) and Phase B final (2859) exported 2026-10-07 (rows 5 and 6). The old step-2500 Phase A export stays in the bucket, unused.
- Section 7 default config (YaRN x2) is not in any exported config.json (rows 1 and 7).
- GGUF: base and chat, F16 / Q8_0 / Q4_K_M, all in the bucket as of 2026-10-07 (rows 9 to 11).
- Chat card says short-suite on the final long-context checkpoint is unmeasured. `eval/pc-run-evals-18.sh` (batch 18, 2026-10-07) was queued to measure it; check its results and update the card and report.
- Report header says "Draft 3, 2026-10-05"; the chat tables (7B) are present, but section 10 still says "It has not been benchmarked yet" for the chat model. Fix the report before linking it.

## 5. Open decisions

1. Licence (`PLACEHOLDER_LICENCE`): earlier Kisoku repos use Apache-2.0 or MIT. Llama 3 tokenizer wording still undecided.
2. Default rope config: YaRN x2 (the cards assume yes) or plain.
3. Which per-stage checkpoints go public and how (branches vs repos); what to do about missing stage 2.
4. Raw MaxText checkpoints (60 GB): publish or not.
5. SFT and teacher-data release scope; DeepSeek output terms.
6. Held-out conversation test data scope.
7. Long-context synthetic data: samples only or full.
8. GGUF repo name casing, and whether to ship Q4_K_M without a quality check.
9. Chat default system prompt, and whether the Ollama Modelfile keeps the "created by 0ARCH" system line.
10. Dataset ids in the card front matter need verification on the Hub.
