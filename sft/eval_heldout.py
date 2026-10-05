"""Before/after on the held-out quiz items (never trained on). 'Before' = the grades stored in quiz-heldout.jsonl
(chat-sft-001). 'After' = the model now behind llama-server on :8911. usage: eval_heldout.py quiz-heldout.jsonl"""
import json, sys, os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import quiz_kisoku as q

rows = [json.loads(l) for l in open(sys.argv[1])]
before = [(r["q"], r["grade"]) for r in rows]
with ThreadPoolExecutor(8) as ex:
    after = list(ex.map(q.ask, [dict(r) for r in rows]))
c = Counter((b[1], a["grade"]) for b, a in zip(before, after))
for g in ("wrong", "right"):
    n = sum(v for (b, a), v in c.items() if b == g)
    print(f"was {g} ({n}): now " + ", ".join(f"{a} {100*c[(g,a)]/n:.0f}%" for a in ("right", "wrong", "abstain")))
tot = len(rows); A = Counter(a["grade"] for a in after); B = Counter(b[1] for b in before)
print("overall before:", {k: f"{100*v/tot:.0f}%" for k, v in B.items()}, "| after:", {k: f"{100*v/tot:.0f}%" for k, v in A.items()})
for a in [a for b, a in zip(before, after) if b[1] == "right" and a["grade"] == "abstain"][:3]: print("LOST:", a["q"], "->", a["answer"][:120])
for a in [a for b, a in zip(before, after) if b[1] == "wrong" and a["grade"] == "wrong"][:3]: print("STILL WRONG:", a["q"], "->", a["answer"][:120])
sys.stdout.flush(); os._exit(0)
