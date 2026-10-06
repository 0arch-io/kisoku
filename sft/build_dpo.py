"""Preference pairs for DPO from the stress-test judgments (stress.py rollout + judge).

For every turn the judge marked bad in a TRAIN script: prompt = the conversation up to that user message, rendered in the
training chat format (literal <|begin_of_text|> and headers, so the trainer must run with add_bos=false add_eos=false);
chosen = the judge's rewrite; rejected = what the model actually said. Test scripts (stress.is_test) are never used.
usage: build_dpo.py OUT_DIR TAG [TAG ...]      e.g. build_dpo.py ~/sft-build/kisoku-dpo-v1 sft005 sft006"""
import json, random, sys
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq
import stress

HDR = lambda role: f"<|start_header_id|>{role}<|end_header_id|>\n\n"
EOT = "<|eot_id|>"

out, tags = Path(sys.argv[1]).expanduser(), sys.argv[2:]
rows, skipped = [], 0
for tag in tags:
    for r, j in stress.load(tag):
        if stress.is_test(r["id"]):
            continue
        m = r["messages"]
        for x in j["turns"]:
            if x["ok"] or not x.get("better"):
                continue
            k = 2 * (x["n"] - 1)                       # index of the user message this bad turn answered
            if k + 1 >= len(m) or m[k + 1]["role"] != "assistant":
                skipped += 1; continue
            hist, bad, good = m[:k + 1], m[k + 1]["content"].strip(), x["better"].strip()
            if any("<|" in y["content"] for y in hist) or "<|" in bad or "<|" in good or not bad or bad == good:
                skipped += 1; continue
            prompt = "<|begin_of_text|>" + "".join(HDR(y["role"]) + y["content"].strip() + EOT for y in hist) + HDR("assistant")
            rows.append({"prompt": prompt, "chosen": good + EOT, "rejected": bad + EOT, "source": f"{tag}:{r['id']}:{x['n']}"})

# hold out whole scripts (not single turns) for the eval split, so no conversation is in both
random.Random(0).shuffle(rows)
ids = sorted({x["source"].split(":")[1] for x in rows})
ev = set(random.Random(1).sample(ids, max(1, len(ids) // 40)))
train = [x for x in rows if x["source"].split(":")[1] not in ev]
evals = [x for x in rows if x["source"].split(":")[1] in ev]
out.mkdir(parents=True, exist_ok=True)
for name, part in (("train", train), ("eval", evals)):
    pq.write_table(pa.Table.from_pylist(part), out / f"{name}-00000-of-00001.parquet")
summary = {"tags": tags, "train_pairs": len(train), "eval_pairs": len(evals), "skipped": skipped,
           "mean_chars": {k: round(sum(len(x[k]) for x in rows) / len(rows)) for k in ("prompt", "chosen", "rejected")}}
(out / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary))
