# Kisoku 1.6B: A Solo, From-Scratch Pretrain on a Free TPU Grant

**Joseph Rodriguez, 0ARCH.** Draft 2, 2026-10-01. Long-context training is still running, so sections 7 and 14 contain placeholders.

## Abstract

Kisoku 1.6B is an open base language model pretrained from scratch by one person on a TPU v4-32 from Google's TPU Research Cloud (TRC), with a single RTX 4090 used for evaluation. The model is a Qwen3-style decoder (22 layers, 2048 embedding width) using the Llama 3.2 tokenizer. It saw roughly 0.5 trillion tokens in three pretraining stages. On a 10-benchmark suite, run through the same harness on the same machine, it matches Meta's Llama 3.2 1B, with five wins each, while using about 18 times less training data (Llama 3.2 1B is reported at about 9 trillion tokens). The comparison has caveats that I state up front: Kisoku has 1.6B parameters against Llama's 1.24B, Llama 3.2 1B was distilled from larger models while Kisoku had no teacher, and Qwen2.5 1.5B and SmolLM2 1.7B are ahead of both on most tests. I also report a contamination audit of the pretraining corpus against the test sets (verbatim overlap exists, but accuracy on the clean subset is the same or higher), the recipe, and a candid log of what went wrong, including an evaluation bug that understated Kisoku's generation scores. A long-context extension to 64K tokens (with a planned 128K via YaRN) is in progress. Only preliminary 4K and 8K results exist, and I present them as such. Weights, intermediate checkpoints, training and evaluation code, and data manifests will be released.

## 1. Summary of results

The headline, stated as plainly as I can: one person, from scratch, matches Llama 3.2 1B on this suite (5 wins each) with about 18 times less training data, with everything open. This is a statement about token efficiency on a specific suite. It is not a claim that Kisoku is the best small model, and it is not a claim that it beats Meta.

All numbers below are for base (non-chat) checkpoints, measured with lm-evaluation-harness 0.4.13 in bfloat16 on one RTX 4090. Scores are percent correct. Generation tasks for Kisoku use the clean runs (no repetition penalty); see section 8 for why that matters.

| Benchmark | Kisoku 1.6B (stage 3) | Kisoku stage 1 | Llama 3.2 1B | Gemma 3 1B | SmolLM2 1.7B | Qwen2.5 1.5B |
|---|---|---|---|---|---|---|
| GSM8K (5-shot) | **15.3** | 6.4 | 5.8 | 2.0 | 30.0 | 60.7 |
| MMLU (5-shot) | **33.0** | 33.2 | 31.3 | 26.5 | 50.0 | 60.9 |
| ARC-Easy (0-shot) | **65.4** | 64.1 | 61.8 | 72.1 | 73.5 | 72.1 |
| ARC-Challenge (0-shot) | **39.8** | 39.4 | 36.9 | 38.0 | 46.9 | 45.2 |
| BBH (3-shot) | **29.2** | 28.2 | 28.3 | 28.4 | 31.4 | 21.4 |
| PIQA (0-shot) | 73.9 | 73.1 | **74.9** | 74.7 | 77.9 | 75.6 |
| WinoGrande (0-shot) | 57.3 | 57.3 | **60.5** | 59.2 | 66.3 | 62.9 |
| HellaSwag (0-shot) | 57.7 | 56.9 | **64.2** | 62.1 | 71.3 | 67.8 |
| HumanEval (pass@1) | 13.4 | 9.1 | **18.9** | 6.7 | not reported | 37.2 |
| TriviaQA (5-shot) | 22.8 | 22.4 | **40.7** | 35.7 | 49.7 | 39.7 |

Bold marks the winner of the Kisoku stage 3 versus Llama 3.2 1B pair only, not the row maximum. Kisoku wins GSM8K, MMLU, ARC-Easy, ARC-Challenge and BBH. Llama wins PIQA, WinoGrande, HellaSwag, HumanEval and TriviaQA. Several of Kisoku's wins are small: BBH by 0.9 points, MMLU by 1.7. I have not yet computed confidence intervals (see Open items), so I would not read those two as established. The large gaps are GSM8K in Kisoku's favor (+9.5) and TriviaQA (-17.9) and HellaSwag (-6.5) in Llama's favor.

Where Kisoku stands against the rest: SmolLM2 1.7B and Qwen2.5 1.5B are ahead of Kisoku on nearly every test, often by a wide margin (Qwen2.5 scores 60.7 on GSM8K and 37.2 on HumanEval). Gemma 3 1B is mixed against Kisoku. Stages 2 and 3 improved math and code a lot (GSM8K 6.4 to 15.3, HumanEval 9.1 to 13.4) and left general knowledge roughly flat.

Training tokens, as I understand the published figures (to be verified against primary sources before release):

| Model | Parameters | Training tokens | Notes |
|---|---|---|---|
| Kisoku 1.6B | 1.6B | about 0.5T | from scratch, no teacher |
| Gemma 3 1B | 1.0B | 2T | |
| Llama 3.2 1B | 1.24B | 9T | distilled from larger models |
| SmolLM2 1.7B | 1.7B | 11T | |
| Qwen2.5 1.5B | 1.5B | 18T | |

Kisoku's token figure covers pretraining stages 1 to 3 only (the long-context phases add about 12B tokens before Phase C; see section 4). [TBD: exact total token count and a score-versus-tokens figure.]

## 2. Model

Kisoku uses a Qwen3-style decoder block, implemented in MaxText and trained in JAX. The values I can state from my run configuration:

- Layers: 22
- Embedding width: 2048
- MLP width: 8192
- Attention heads: 16 query heads, 4 key/value heads (grouped-query attention), head dimension 128
- Input and output embeddings are tied
- Vocabulary: 128,256 (the Llama 3.2 tokenizer, taken from the unsloth/Llama-3.2-1B copy)
- RoPE base (theta): 5,000,000
- Pretraining context: 4096 tokens; extended to 32K and then 64K in the long-context phases
- Parameters: 1.6B (the "1B" in some of my run names is a leftover; the exported model is 1.6B)

[TBD: exact parameter count by component.]

I kept the RoPE base at 5M for the long-context phases. A minimum-theta table I consulted during planning puts the requirement for 64K at about 2.1M, so 5M leaves margin. The planned route to 128K is YaRN with factor 2 applied at inference over a 64K-trained model. I rehearsed that step on the 32K-trained Phase A checkpoint (section 7).

Export to Hugging Face format goes through a MaxText conversion that I wrapped with model-shape overrides, because MaxText's model name field only accepts a fixed set of names. I checked the export against the training framework in float32: argmax agreement was 100% and the maximum logit difference was 0.04. For base use, the exported end-of-sequence token is reset to the base end-of-text token (128001).

## 3. Data

All data was prepared as tokenized text records and streamed from cloud storage. I list the sources, what I know about their licences, and what I cannot redistribute.

**Stage 1 mix (steps 0 to 199,000).** Document weights from the run script: Nemotron-CC 0.35, Ultra-FineWeb 0.35, StarCoder 0.16, FineMath 0.06, OpenWebMath 0.04, MegaMath-Web-Pro 0.04.

**Stage 2 (to step 224,000)** shifted the mix toward more code and math: Nemotron-CC 0.30, Ultra-FineWeb 0.30, StarCoder 0.22, FineMath 0.07, OpenWebMath 0.05, MegaMath-Web-Pro 0.06.

**Stage 3 (to step 243,000)** added fresh math and reasoning data: Nemotron-CC-Math-v1 (the "4plus" subset, about 8.9M documents) and OpenThoughts3 text (1.2M records). The stage 3 mixture in the run script is Nemotron-CC 0.27, Ultra-FineWeb 0.28, StarCoder 0.22, FineMath 0.05, OpenWebMath 0.03, MegaMath 0.05, OpenThoughts3 0.012, Nemotron-CC-Math 0.14. These weights select documents, not tokens, which matters because OpenThoughts3 documents are about 12K tokens on average against 1K to 2K for web text. A first draft of the stage 3 line gave OpenThoughts3 a weight of 0.10, which would have made it roughly 60% of the stage 3 tokens; I corrected it to about 0.012 (roughly 15% of tokens).

**Long-context data (about 13.1B tokens available):**

| Source | Records | Tokens | Licence |
|---|---|---|---|
| The Stack v1 repository-level concatenation (via ProLong) | 69,120 | about 4.5B | [TBD: confirm] |
| Dolma 3 Longmino science PDFs | 145,526 | about 5.7B | ODC-By |
| PG19 books | 28,602 | about 2.9B | Apache-2.0 |

I deliberately excluded ProLong's "book" subset because it is built from SlimPajama books, which derive from a pirated corpus.

**Licences and redistribution.**
- Nemotron-CC-Math-v1 is under the NVIDIA Data Agreement for Model Training. Models trained on it may be published, but the data itself must not be redistributed, so my data manifests for it will list sources and counts without the text. The corpus was built using Phi-4 (MIT).
- [TBD: licences for Nemotron-CC, Ultra-FineWeb, StarCoder, FineMath, MegaMath, OpenWebMath and OpenThoughts3, and a statement of which can be redistributed.]
- The Llama 3.2 tokenizer carries Meta's licence terms, which apply to the released model.
- [TBD: provenance of OpenThoughts3 reasoning traces. If they were generated by other models, I will say so here, because it bears on the "no teacher" framing in section 10.]

The full corpus used in the contamination scan (section 6) was 784 shards and about 1.66 TB of text across 11 datasets.

## 4. Training

**Hardware and framework.** A TPU v4-32 (four hosts) from Google's TRC, running MaxText with data parallelism over fully sharded parameters. Evaluation and later data work ran on one RTX 4090 under WSL. Throughput in the first long-context phase was 62.9 TFLOP/s per device, and the original pretrain log showed about 105 TFLOP/s per device.

**Optimizer.** Muon, as I recorded for the supervised fine-tuning run ("same as pretrain"). [TBD: Muon hyperparameters, weight decay, gradient clipping, and the embedding/output optimizer split.]

**Schedule.** The learning rate schedule was written for 243,000 steps, with a warmup-stable-decay shape: a peak of 3e-4, with decay starting at 0.9 times 243,000 = 218,700 and ending at 9.9e-5 at step 243,000. [TBD: stage 1 warmup length.] Each step was about 2.1M tokens at 4096-token sequences.

**Stage 1 (steps 0 to 198,999, about 414B tokens).** Loss ended around 1.81 to 1.85. Step time was about 13.28 s. Because the schedule was written for 243,000 steps, the model ended stage 1 at the full 3e-4 learning rate, not annealed. The run was interrupted many times (section 8); it logged 41 starts.

**Stage 2 (steps 199,000 to 223,999).** Resumed from the stage 1 checkpoint including data iterator state, at 14.9 s per step. LR stayed at 3e-4. Loss was about 1.70 to 1.76. It ran from 2026-09-22 to 2026-09-26.

**Stage 3 (steps 224,000 to 242,999).** Started 2026-09-27 at about 13.7 to 14.8 s per step, later about 13.3. Loss on the first steps jumped to 2.0 to 2.15 on the new data and settled to about 1.47 by the end (last step 242,999: 1.468). The learning rate decayed from 3e-4 along the schedule above. Two changes were needed here: the weight correction described in section 3, and turning off MaxText's document truncation (section 8). Stage 3 finished on 2026-09-30 with a clean exit.

**Long-context Phase A (32K).** Started from the stage 3 final parameters with a fresh optimizer. 2,900 steps (about 6B tokens) at 38.5 s per step, sequence length 32,768, per-device batch 1 with gradient accumulation 4, full rematerialization, vocabulary tiling 8, flash (splash) attention. Learning rate cosine from 1e-4 to 1e-5 with 2% warmup. Token mix 60% long, 40% short, where the short share is the stage 3 mixture and the long share is code repositories, science PDFs and PG19. Loss started at 3.75 (positions beyond 4K were untrained, so a high start is expected), and fell to about 2.0 to 2.3 by step 130. Phase A ended on 2026-10-01 at about 10:51 UTC.

**Long-context Phase B (64K).** Started 2026-10-01 11:04 UTC from the Phase A final checkpoint (fresh optimizer). 2,860 steps (about 6B tokens) at 65.8 s per step, sequence length 65,536, gradient accumulation 2, vocabulary tiling 16, 1% warmup, same cosine LR range as Phase A. It uses the same mixture as Phase A. At the time of writing it is at step 658 and expected to finish around 2026-10-03 15:25 UTC. [TBD: Phase B final loss and checkpoint.]

**Long-context Phase C (64K with synthetic tasks).** Queued to start automatically when Phase B ends, 1,300 steps (about one day), learning rate cosine from 5e-5 to 5e-6 with 2% warmup, initialized from the Phase B final. Token shares: short 40%, code 15%, science PDFs 15%, PG19 12%, synthetic tasks 18%. Details in section 7. [TBD: Phase C actual steps, time and loss.]

A few training-system facts that affected results. Checkpoints were written every 1,000 steps in stages 1 to 3 (every 250 in the long phases), and only the latest five were retained, so I copied out the final checkpoint of each stage by hand. A crash restart resumes from the run's own latest checkpoint, not from the initial load path.

## 5. Evaluation method

**Harness.** lm-evaluation-harness 0.4.13, bfloat16, one RTX 4090 under WSL, base checkpoints. Every model, including the baselines, was run through this setup. Baselines: Llama 3.2 1B (unsloth base copy), Gemma 3 1B (pretrained), SmolLM2 1.7B, Qwen2.5 1.5B.

**Settings.** HellaSwag, ARC-Easy, ARC-Challenge, PIQA and WinoGrande: 0-shot, length-normalized accuracy. MMLU, GSM8K and TriviaQA: 5-shot. BBH: 3-shot with exact match after whitespace removal, using a custom copy of the task. HumanEval: pass@1 with greedy decoding. Generation tasks used a fixed batch size of 32 (16 for the TriviaQA and BBH batch).

**Test set sizes** (from the contamination audit): GSM8K 1,319; MMLU 14,042; ARC-Easy 2,376; ARC-Challenge 1,172; HellaSwag 10,042; PIQA 1,838; WinoGrande 1,267; HumanEval 164; TriviaQA 17,944. HumanEval has only 164 problems, so one problem is 0.6 points, and I would treat HumanEval gaps of a few points with caution. [TBD: bootstrap confidence intervals for every cell.]

**Exclusions and why.**
- DROP is excluded. In this harness and datasets version its targets load as the CSV header string, so any score would be meaningless.
- SmolLM2's HumanEval run returned 0.6%. Its generations never stop, which is a harness interaction, so I left the number out rather than publish a misleading figure.
- BBH as shipped scored 0.0 for every model because exact match did not strip the leading space (" False" against "False"). I used a custom task copy with a whitespace-removing filter, and a probe of 8 items scored 7 correct before I ran it at scale.
- Qwen2.5's BBH score (21.4) is lower than the others and I do not understand it. I reran it with the corrected generation settings and it did not change, so I report it as the harness gave it.
- Kisoku generation scores use the clean runs (section 8).

**Known comparability limits.** The harness, prompts and shot counts differ from the numbers each lab publishes, so my baseline numbers will not match their model cards. They are meant to be compared with each other, not with those cards.

## 6. Contamination audit

**Question.** Do the test sets appear in the pretraining data, and does that explain the scores?

**Method.** For each test item I built two probes: the first 80 and last 80 characters after normalization. Items shorter than 50 characters were skipped. This gave 61,366 probes across GSM8K, MMLU, ARC, HellaSwag, PIQA, WinoGrande, HumanEval and TriviaQA. I scanned all 784 shards of the pretraining corpus (about 1.66 TB of text, 11 datasets) with an Aho-Corasick matcher, with no errors. An item counts as hit when at least one probe matches verbatim. The scripts and per-item results will be released.

**Overlap.**

| Benchmark | Items | Probed | Any-probe hit | Both-probe hit |
|---|---|---|---|---|
| GSM8K | 1,319 | 1,319 | 27.6% | 26.8% |
| MMLU | 14,042 | 12,331 | 28.7% | 19.2% |
| ARC-Easy | 2,376 | 2,061 | 20.9% | 20.1% |
| ARC-Challenge | 1,172 | 1,053 | 26.3% | 25.7% |
| HumanEval (prompts) | 164 | 164 | 25.6% | 11.0% |
| HellaSwag | 10,042 | 10,042 | 6.9% | 0.2% |
| TriviaQA | 17,944 | 14,227 | 3.3% | 0.6% |
| PIQA | 1,838 | 1,589 | 2.6% | 0.4% |
| WinoGrande | 1,267 | 1,267 | 0.5% | 0.5% |

The hits come from homework-help and quiz web pages in Nemotron-CC, Ultra-FineWeb, FineMath, MegaMath, Nemotron-CC-Math and OpenWebMath. These are the Common Crawl-derived families that most labs train on, so I expect the baselines to share this overlap. A HumanEval "solution" probe matched 97%, but that is not meaningful: generic code such as a Fibonacci function matches itself everywhere. I do not use it.

**Does overlap inflate the scores?** I split each test set into overlapping and clean items and computed accuracy on each, using per-sample logs.

| Benchmark | Kisoku stage 3 (overlap / clean) | Llama 3.2 1B | SmolLM2 1.7B | Qwen2.5 1.5B | Items (overlap / clean) |
|---|---|---|---|---|---|
| GSM8K | 10.7 / 16.4 | 7.3 / 5.5 | 31.0 / 29.7 | 64.8 / 61.3 | 364 / 955 |
| MMLU | 30.8 / 33.2 | 30.5 / 30.9 | 45.5 / 50.5 | 53.8 / 62.4 | 3,406 / 8,127 |
| ARC-Easy | 72.1 / 70.8 | 68.4 / 65.8 | 79.1 / 77.5 | 77.4 / 74.5 | 430 / 1,946 |
| ARC-Challenge | 40.1 / 37.3 | 34.7 / 30.5 | 47.7 / 43.4 | 43.0 / 40.6 | 277 / 895 |
| HellaSwag | 43.2 / 44.3 | 45.0 / 48.2 | 52.7 / 53.3 | 49.4 / 50.1 | 694 / 9,348 |
| PIQA | 90.2 / 74.0 | 92.7 / 74.9 | 97.6 / 76.6 | 97.6 / 75.1 | 41 / 1,797 |

Kisoku does not score higher on overlapping items on GSM8K, MMLU or HellaSwag. On the clean GSM8K subset it scores 16.4 against Llama's 5.5. On ARC the overlapping items score 1 to 3 points higher for every model, including the baselines, which suggests those items are simply easier. The 41 overlapping PIQA items are easy for every model (90 to 98%). Stage 1 on GSM8K: 5.2 overlapping against 6.4 clean.

Two details on this table. First, it uses per-sample accuracy (plain accuracy, not the length-normalized metric in section 1), so the HellaSwag and ARC figures are lower than in the main table. Second, the Kisoku GSM8K numbers here come from the earlier runs that still had the repetition penalty on (section 8), so they sit slightly below the clean numbers; the direction of the finding does not depend on that. [TBD: redo the split with the clean generation logs.] Gemma 3 1B has no per-sample logs here, and the stage 1 MMLU split is missing.

**Caveat.** This audit detects only exact verbatim matches of 80-character probes. It does not detect paraphrases, translations, reformatted copies, or answer-only leakage. A clean result here means "no evidence of verbatim leakage driving the scores", not "no contamination". I also have not yet run the same check on the supervised fine-tuning datasets. [TBD: SFT data check against the same probes.]

## 7. Long context

**Status.** Everything in this section is preliminary. Phase B (64K) is running and Phase C has not started. The numbers below come from a mid-Phase-A checkpoint (step 2500 of 2,900, trained to 32K), not from the final model.

**Recipe.** Extend in stages: 4K to 32K (Phase A), then 64K (Phase B), then 64K with synthetic aggregation tasks (Phase C), each stage loading the previous final parameters with a fresh optimizer. Keep the RoPE base at 5M. Mix about 60% long and 40% short documents, with long data from code repositories, science PDFs and books. Disable document truncation so long documents are split, not cut. Then apply YaRN with factor 2 at inference to reach about 128K. The design draws on the open recipes I studied: SmolLM3 (YaRN for the last doubling), ProLong (60/40 long/short mix; short-only SFT is safe), and Olmo 3 (science PDFs plus synthetic aggregation tasks in the long mix; arXiv 2512.13961). [TBD: formal citations for SmolLM3 and ProLong.]

**Synthetic task design.** Phase C includes 18% synthetic documents. I wrote my own generator; its templates are my own and are not RULER's. It takes a science PDF or a PG19 book, trims it to a target length (8K, 16K, 32K or 56K tokens, with weights 15/20/30/35%, never above 60K so a 64K chunk never separates the questions from the document), plants lines in it, and appends 8 to 24 question-answer pairs whose answers are computed exactly from the final text:

- Note lookup: planted notes such as "the registration number of the amber anchor is 482913", with 3 to 30 keys, some repeated with 2 to 4 values, asked as single lookups, "list every value", and multi-query.
- Definition chains: 2 to 6 hops ("Let X be 41822", "Let Y be equal to X"), asking which names share a value and what a given name's value is.
- Word counts: exact occurrence counts, most frequent of four words, and the top five content words.
- Text position: the sentence that follows a quoted one, and which of three rare words appears first.

Answers come from the final document text, not from the planted data alone, and the generator verifies that chain hops appear in order. Question styles vary ("Question/Answer", "Q/A", and lead-in completion), and some documents carry a header sentence. The finished set is 22,386 documents (15,200 science, 7,186 PG19), 575.5M tokens, average 25,709 tokens, maximum 55,189 tokens, about 13.6 question-answer pairs per document, zero generation errors. The generator will be released.

I want to be clear about what this means for evaluation. These tasks resemble parts of RULER (needle lookup, counting, variable tracking). A model trained on them should do better on RULER partly because of format familiarity, and I will say so wherever I report Phase C numbers. To keep the claim honest I plan to report a held-out suite alongside RULER (BABILong and LongBench v2, both available in the harness) and to report every task and length, not just the average. [TBD: held-out long-context results.]

**Preliminary RULER results.** The harness's built-in RULER (13 tasks), 100 samples per task at 4K and 8K, greedy decoding with no repetition penalty, 128-token generation cap, average over 13 tasks. All models ran in the same setup.

| Model | 4K | 8K |
|---|---|---|
| Qwen3.5 2B base | 91.6 | [TBD] |
| Qwen3 1.7B base | 89.4 | 84.2 |
| Qwen3.5 0.8B base | 86.7 | 83.0 |
| Qwen3 0.6B base | 84.2 | 73.5 |
| Llama 3.2 1B | 73.5 | 67.4 |
| **Kisoku (Phase A, step 2500)** | **71.9** | **59.3** |
| LFM2.5 1.2B | 63.1 | 54.6 |
| Gemma 3 1B | 59.6 | 43.8 |

At short lengths Kisoku trails the Qwen3 family by about 12 to 25 points and sits below Llama 3.2 1B (71.9 against 73.5 at 4K, 59.3 against 67.4 at 8K), above LFM2.5 and Gemma 3 1B. Its drop from 4K to 8K (12.6 points) is larger than Llama's (6.1) and similar to Qwen3 0.6B's (10.7); I do not yet know why. I do not expect a long-context headline from the 4K and 8K numbers. The case for the long-context work rests on 64K and beyond, where most competing small models are out of their native range; that is a hypothesis until I measure it.

Part of the gap is format, not ability. In diagnostic runs the model often stops immediately on counting tasks or rambles ("Answer: Answer:") on question tasks, which looks like base-model format habit that instruction data should fix. A forced minimum of two new tokens only added 0.8 points at 4K (71.5 against 70.7 on 30 samples), so I do not use it.

**YaRN rehearsal.** Before the 64K model exists, I rehearsed the last step on the Phase A checkpoint, which was trained to 32K: a passkey test (a five-digit key hidden in filler text, five depths per length), with the plain config and with YaRN factor 2 over a 32,768 base.

| Prompt length | Plain | YaRN x2 |
|---|---|---|
| 16K | 4 of 5 | 2 of 5 |
| 30K | 4 of 5 | 5 of 5 |
| 47K | 2 of 5 | 5 of 5 |
| 60K | 0 of 5 | 5 of 5 |

YaRN carried retrieval to almost twice the trained length, which is the same mechanism planned for 64K to 128K. It also appears to hurt short prompts (2 of 5 at 16K), a known weakness of static YaRN, so the scaled config will probably ship as a separate long-context option. Five trials per cell is a rehearsal, not a result. A 116K-token prompt peaked at 15 GB in bfloat16, so 128K fits on one 24 GB card.

**Planned and missing.**
- [TBD: RULER at 16K and 32K (50 samples per task), and 64K and 128K, for the final Phase B and Phase C checkpoints and the same baselines]
- [TBD: passkey retrieval and perplexity curves at 32K, 64K and about 100K, with YaRN]
- [TBD: Granite 4.0 1B and Falcon-H1 1.5B baselines, which publish no RULER numbers]
- [TBD: per-task breakdown and confidence intervals]

Published long-context numbers from other sources (for example, third-party RULER results for Qwen3.5 2B and for a Llama 3.2 1B research fine-tune, arXiv 2412.18860) use different harnesses and are not comparable to mine, so I will rerun every baseline instead of quoting them. For models whose native range is 32K, I will label scores beyond 32K as extrapolated.

## 8. What went wrong

This section is long on purpose. Each item cost time, and most are the kind of thing a first-time solo run hits.

**1. gcsfuse served zero pages (four incidents, roughly 50 hours lost).** Training data was streamed from a cloud bucket through a gcsfuse mount. Occasionally a failed read (an HTTP 500 or 503 retry) left a page of zeros in the mount's cache. The training loader read a corrupted record, logged "chunk header hash mismatch (stored 0x0000000000000000)" and exited cleanly on that one worker. The other three workers hung at the shutdown barrier, systemd restarted everything from the last 1,000-step checkpoint, and the run died at the same step again, forever. The tail of the log still looked healthy, so the only reliable health check was whether the newest checkpoint folder was advancing, and I learned to count "START" lines. Incident 1 (2026-09-02 to 09-04) cost about 30 hours before I found the cause by comparing bytes at the failing offset against the cloud copy. I then added a restart guard: a state file records the latest checkpoint and the number of consecutive starts without a new one; on the third such start the script remounts gcsfuse and restarts training. The guard fired for incidents 2 (09-04, about 9 hours), 3 (09-10 to 09-11, about 8 hours) and 4 (09-21, about 2.5 hours). Lesson: the guard works but waiting for the third start is too slow. Firing on the second would have saved about 2.5 hours per incident. The total over the run was 41 starts and four guard fires.

**2. The billing outage (2026-09-09, about 3 hours).** The billing account went delinquent and cloud storage began returning 403 errors. The run died at step 122,733. This looks exactly like the gcsfuse failure in systemd's behavior (restart loops, one worker dies first), and my first restart attempt failed that way. I paid, the account reopened about 24 minutes after the run died, and the next automatic restart recovered from the 122,000 checkpoint. Lesson: check the billing state before debugging the mount.

**3. The stage switch that was lost.** My plan had three stages (to 199,000, 224,000 and 243,000 steps), and the stage-switch launcher was lost during a laptop migration. When stage 1 stopped at step 198,999, my notes recorded "PRETRAIN FINISHED". I found on 09-22 that only stage 1 had run and the data mixture for stages 2 and 3 had never been launched. One consequence is that the stage 1 checkpoint was never annealed. The learning rate was still at 3e-4 when it stopped, because the decay was scheduled for the end of stage 3. The stage 1 checkpoint was already at risk: the run retains only five checkpoints, which would have deleted it about 11 hours into stage 2, so I copied it out first. Resuming stages 2 and 3 from stage 1 with a different data mixture also meant checking that the data iterator state would restore.

**4. The truncation default.** MaxText's `use_truncation` defaults to true, so in stages 1 and 2 every document was cut at 4096 tokens and the rest discarded. That is mostly harmless for web text, but OpenThoughts3 documents average about 12K tokens, and cutting them removes the final answers. I set it false for stage 3 and the long phases. Doing that swaps a map step for a chunking step in the data pipeline, which adds a state layer, and restoring the old iterator state then raised `KeyError: 'parent'`. The fix was to wrap each saved iterator state in the new layer's expected structure, which I tested on a toy pipeline first and then applied. I backed up the original iterator state beforehand.

**5. The SFT evaluation crash.** My first preview fine-tuning run crashed at step 1. Its step-0 evaluation requested 20 evaluation steps, but each host's evaluation shard held only about 10 packed batches. One host ran out and stopped early while the others kept launching, which triggers a TPU "unexpected peer, different launch id" halt. The rule I now follow: with this input pipeline, the number of eval steps must fit the smallest host's shard, or evaluation must be turned off. Related smaller bugs from that work: a chat template that put a start-of-sequence token before every conversation round (fixed with a per-message flag and checked on 300 of 300 sample rows), and a Hugging Face streaming job that hung forever after finishing its work and cost about 45 minutes (fixed by ending the script with an explicit hard exit).

**6. Evaluation gotchas.**
- `--batch_size auto` on generation tasks (GSM8K, HumanEval) overflowed the 4090 under WSL with "CUDA error: device not ready" and produced no results file, wasting 16 to 50 minutes per attempt. Use a fixed batch (32) for generation tasks, and auto only for log-likelihood tasks. The stage 1 MMLU run crashed twice for the same reason until I fixed the batch at 16.
- BBH scoring 0.0 for everyone, DROP loading its CSV header as the answer, and SmolLM2's HumanEval never stopping (section 5).
- RULER's `--limit` flag with several sequence lengths only scored 4096, because the harness orders documents by length. I now run one length per run.
- Qwen models ship a generation config with a 2,048-token output limit that overrides the harness's own limit. At 17 to 30 seconds per sample this made RULER runs slow. Passing the limit on the command line did not get through, so I removed the setting from the cached configs. Qwen2.5's earlier GSM8K, HumanEval and BBH runs had used the 2,048 default and I reran them.
- The harness's RULER needs two extra Python packages and tokenizer data; my first RULER batch failed in seconds on all five models for lack of them.
- Long 4090 jobs must run under tmux, not in a foreground ssh session, which dropped.

**7. A generation config silently overrode the harness, and the repetition penalty.** My Hugging Face exports of Kisoku shipped chat sampling defaults in `generation_config.json` (sampling on, temperature 0.7, top-p 0.9, repetition penalty 1.1), written by my own conversion script. The harness forces greedy decoding, but it did not remove the repetition penalty, and none of the baselines had one. The penalty discourages copying tokens from the prompt, so it hurt every Kisoku generation score (GSM8K, HumanEval, BBH, TriviaQA and RULER) and none of the log-likelihood scores (MMLU, ARC, HellaSwag, PIQA, WinoGrande). I found it on 2026-10-01 while diagnosing why the model answered with an empty string on a word-counting task. A controlled A/B on the Phase A checkpoint at 4K (30 samples per task) moved RULER from 63.5 to 70.7. Counting-task accuracy went from 4.0 to 41.3, and the "empty answer" had mostly been the penalty pushing the end-of-sequence token to the top. On the short generation benchmarks the effect was small: GSM8K 14.6 to 15.3, HumanEval unchanged at 13.4, BBH unchanged at 29.2. I reran every Kisoku generation evaluation with a base-only generation config and report those. The numbers I had briefly written down before the fix (RULER 67.0 at 4K and 56.8 at 8K) were wrong and are discarded. The base export scripts now overwrite the generation config with token ids only. Even without the penalty, the top token after a counting prompt is the end-of-text token (probability 0.20), which I attribute to base-model format habit.

## 9. Cost and compute

**TPU.** A TPU v4-32 from Google's TRC, free of charge. The grant period formally ended (my notes give 2026-08-11 for the original grant, and a renewal was declined for capacity), and the TPU kept running because it had never been reclaimed. From 2026-10-01 I treated it as revocable at any moment: all state is backed up in cloud storage, and the long-context phases were planned so that Phase B is the last essential use of the TPU. [TBD: total TPU hours or chip-hours used.]

**Cloud storage.** The grant covers TPU time only. The ongoing cost of the project was cloud storage and read operations: about $85 to $100 per month in total, of which my notes itemize about $44 per month of read operations from streaming data through gcsfuse and about $16 per month for about 787 GB stored. This was flat per day, not growing.

**Other spending.** A short-lived CPU VM for data preparation and the contamination scan. The Nemotron-CC-Math prep used about 15 minutes of VM time, roughly $1; the contamination scan ran about six hours (all 784 shards) and I have not itemized its cost. A supervised-fine-tuning data job through a hosted API reached $15.23 for 20,900 examples at last count. These are for the chat model and not part of the base pretrain. All evaluations ran on a personal RTX 4090. [TBD: total out-of-pocket dollar figure.]

## 10. Limitations

- **Parameter and teacher differences.** Kisoku is 1.6B against Llama 3.2 1B's 1.24B. Llama 3.2 1B was distilled from larger models and Kisoku was not, but some of my data sources were themselves built with other models (Nemotron-CC-Math used Phi-4, and OpenThoughts3 is a reasoning dataset whose provenance I still have to confirm). "No teacher" means no distillation loss and no teacher logits in my training, not "no other model touched the data".
- **Qwen and SmolLM2 are ahead.** On most tests Qwen2.5 1.5B and SmolLM2 1.7B beat Kisoku by large margins. The result is about tokens, not rank.
- **Weak on knowledge recall.** TriviaQA 22.8 against 40.7 for Llama, and HellaSwag 57.7 against 64.2. Kisoku has fewer tokens, and fact recall appears to scale with them.
- **Single run, no ablations.** One training run per configuration, no seeds, no ablations. I chose Muon, the data mix and the schedule from reading, not from experiments of my own. The stage 1 checkpoint was not annealed.
- **Statistical uncertainty.** No confidence intervals yet. Several differences, including BBH and MMLU against Llama and everything on HumanEval, are within what I would expect from noise.
- **Contamination audit limits.** Exact matches only (section 6), and the SFT data has not been checked.
- **Long context is unproven.** The 4K numbers show Kisoku below the Qwen3 family and slightly below Llama 3.2 1B. Synthetic training tasks resemble the benchmark.
- **Short-context regression after long-context training** has not been measured. [TBD: rerun the 10-benchmark suite on the final long-context checkpoint.]
- **Not evaluated:** safety and bias behavior, multilingual ability, instruction following, tool use. The chat preview is a rough demonstration, not a finished assistant.
- **One evaluation setup.** One harness version, one machine, bfloat16, and my own prompt settings.

## 11. What a v3 would do with more compute

Everything here is a plan, not a result.

- Train on far more tokens. Kisoku saw about 0.5T tokens against 2T to 18T for its peers, and knowledge tasks are where it trails. A run on the order of 2 to 3T tokens is the obvious ask.
- Run the ablations I could not afford: Muon against AdamW on a roughly 10B-token proxy, and the data-mix choices.
- Anneal properly: put the decay inside the planned run instead of ending stage 1 at full learning rate.
- Treat contamination checks and held-out long-context evaluation as part of the pipeline from the start, including a check of fine-tuning data.
- Move the checkpoint exports, generation config and evaluation harness into tested scripts so that the failures in section 8 do not recur.
- Apply fine-tuning data and reinforcement learning only after the base model and its evaluation are fixed, with data that teaches the model to say when it does not know.

## 12. Releases

All links are placeholders until release. A preview chat model (a supervised fine-tune of the stage 1 base) is already public on Hugging Face under 0arch-io, with GGUF quantizations.

- Final base weights, bf16 safetensors, Hugging Face format: [TBD: link]
- Stage 1, stage 2 and stage 3 base checkpoints: [TBD: link]
- Long-context checkpoints (Phase A, B, C): [TBD: link]
- GGUF builds for llama.cpp and Ollama: [TBD: link]
- Training code, configs and run scripts (MaxText configuration, stage scripts, restart guard): [TBD: link]
- Evaluation scripts and raw per-sample results: [TBD: link]
- Contamination audit scripts and results: [TBD: link]
- Synthetic long-context task generator and samples: [TBD: link]
- Data manifests (sources, counts and mix weights; no text for sources whose licences forbid redistribution): [TBD: link]
- Model licence for the final weights: [TBD: licence; note the Llama 3.2 tokenizer terms]

## 13. Acknowledgements

This work was made possible by Google's TPU Research Cloud, which provided the TPU v4-32 that all pretraining ran on. I also thank the authors of MaxText, lm-evaluation-harness, and the open datasets and recipes this work builds on (Nemotron-CC, Ultra-FineWeb, StarCoder, FineMath, OpenWebMath, MegaMath, OpenThoughts3, Dolma 3, ProLong, PG19, SmolLM3 and Olmo 3). [TBD: formal citations.]

## 14. Open items

Every placeholder in this draft:

1. Exact total pretraining token count, and the score-versus-tokens figure.
2. Exact parameter count by component.
3. Muon hyperparameters, weight decay, clipping, stage 1 warmup length.
4. Licences for Nemotron-CC, Ultra-FineWeb, StarCoder, FineMath, MegaMath, OpenWebMath, OpenThoughts3 and The Stack v1 long-context subset, plus redistribution status.
5. Provenance of OpenThoughts3 reasoning traces (affects the "no teacher" wording).
6. Bootstrap confidence intervals for all benchmark cells.
7. Contamination split recomputed with clean (no repetition penalty) generation logs; stage 1 MMLU split; Gemma split.
8. Contamination check of the supervised fine-tuning data.
9. Phase B and Phase C final loss, steps, time and checkpoints.
10. RULER: 16K, 32K, 64K, 128K tables for final checkpoints and all baselines (and Qwen3.5 2B at 8K).
11. Held-out long-context results (BABILong, LongBench v2), passkey and perplexity curves with YaRN to about 100K and 128K.
12. Granite 4.0 1B and Falcon-H1 1.5B long-context baselines.
13. RULER per-task breakdown and confidence intervals.
14. Re-run the 10-benchmark suite on the final long-context checkpoint to measure short-context regression.
15. Verify the training-token figures for baselines against primary sources, and add formal citations (SmolLM3, ProLong, Olmo 3, arXiv 2412.18860, and others).
16. Total TPU hours or chip-hours, and total out-of-pocket dollar cost.
17. All release links and the final weights licence.
18. Figures: loss curves across stages, score versus tokens.
