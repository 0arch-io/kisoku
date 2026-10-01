#!/bin/bash
# Table of every finished lm-eval result: rows = tasks, columns = models. Picks the main metric per task generically.
~/ai/bin/python - <<'PY'
import json, glob, os
PREF = ["acc_norm,none", "exact_match,ws", "exact_match,strict-match", "exact_match,flexible-extract", "exact_match,remove_whitespace", "exact_match,get-answer", "exact_match,none", "pass@1,create_test", "f1,none", "acc,none"]
rows = {}
for f in sorted(glob.glob(os.path.expanduser("~/kisoku-eval/results/*/*/results_*.json"))):
    name = f.split("/results/")[1].split("/")[0]; model = name.rsplit("-", 1)[0]
    j = json.load(open(f)); r = j["results"]; groups = set(j.get("groups", {}) or {})
    for task, v in r.items():
        if task.startswith(("mmlu_", "bbh_fewshot", "bbh_ws_", "drop")): continue
        m = next((v[k] for k in PREF if k in v and isinstance(v[k], (int, float))), None)
        if m is None: continue
        rows.setdefault(task, {})[model] = round(m * 100, 1)
models = sorted({m for t in rows.values() for m in t})
order = ["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande", "mmlu", "gsm8k", "humaneval", "triviaqa", "bbh_ws"]
print("task".ljust(14) + "".join(m.ljust(15) for m in models))
for t in order + [t for t in rows if t not in order]:
    if t in rows: print(t.ljust(14) + "".join(str(rows[t].get(m, "-")).ljust(15) for m in models))
PY
