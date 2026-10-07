# Kisoku

Working repository for Kisoku 1.6B, a language model pretrained from scratch by one person on a TPU v4-32 from Google's TPU Research Cloud.

**This repository is private until launch.** Before making it public, go through the checklist at the bottom.

## Layout

| Folder | What is in it |
|---|---|
| `report/` | Technical report draft and the benchmark scorecard page |
| `training/` | MaxText config, the stage run script (stages 1 to 3, long-context phases A, B, C), hand-off watchers, restart guard, export and fine-tuning scripts, systemd units |
| `eval/` | lm-evaluation-harness batch scripts used on the RTX 4090, RULER table script |
| `contamination/` | Verbatim-overlap audit: probe builder, shard scanner, summary, overlapping-versus-clean accuracy split |
| `longctx-synth/` | Generator for the synthetic long-context tasks used in Phase C, and its manifest |

## Setup

The scripts read the GCS bucket name from `KISOKU_BUCKET` (no `gs://` prefix): `export KISOKU_BUCKET=your-bucket`. The MaxText YAML files carry the literal `KISOKU_BUCKET` and `/home/USER`; the run scripts override those paths on the command line.

## Status (2026-10-07)

- Pretraining stages 1 to 3 done (about 0.5T tokens). Long-context Phases A (32K), B (64K) and C (64K plus synthetic tasks) done. Nine chat fine-tuning passes done; pass 9 is the preview.
- Same-harness comparison against Llama 3.2 1B, Gemma 3 1B, SmolLM2 1.7B, Qwen2.5 1.5B on ten benchmarks: see `report/`.
- Contamination audit over all pretraining shards done: see `contamination/summary.txt` and `split-accuracy.txt`.

## Before making this public

- Bucket paths, private IP addresses and host names are generalised (2026-10-07: `KISOKU_BUCKET`, `KISOKU_PC_HOST`). Re-run the scan in `release/MANIFEST.md` section 3 before flipping the repo.
- Confirm no credentials are present (none are committed; key files stay on the machines).
- Do not publish any text from datasets whose licence forbids redistribution (the Nemotron sets). Only manifests and counts.
- Replace placeholders in the report, add licence and citation sections.
