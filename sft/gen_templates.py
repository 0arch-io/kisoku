#!/usr/bin/env python3
"""Reply phrasings for the quiz-based additions (build_additions.py), written by the teacher model instead of by hand.
Each kind gets ~40 short variants with placeholders. Output: data/gen2/_templates.json"""
import json, re, sys
from pathlib import Path
import gen_kisoku as g

KINDS = {
    "IDK": ("an assistant that does not know the answer to a specific factual question and says so honestly instead of guessing; one or two sentences; some suggest checking a reliable source or sharing context", []),
    "IDK_SUBJ": ("an assistant that does not have reliable information about a named subject and says so honestly instead of guessing; one or two sentences; must mention the subject", ["{s}"]),
    "UNFAMILIAR": ("an assistant that does not recognize a name or term at all (a product, method, person or place it has never seen) and says so instead of inventing a description; one to three sentences; most ask the user for context; must mention the term", ["{s}"]),
    "CORRECT_USER": ("a USER telling an assistant its last answer was wrong and giving the right answer; short, casual, like a real chat message; must contain the right answer", ["{g}"]),
    "CORRECT_ASST": ("an assistant that gave a wrong answer, was corrected by the user, and now owns the mistake plainly and restates the correct answer; one or two sentences; no grovelling; must contain the correct answer", ["{g}"]),
    "PUSH_USER": ("a USER insisting that an assistant's answer is wrong and proposing a different answer; short and casual; must contain the proposed answer", ["{a}"]),
    "HOLD_ASST": ("an assistant that gave a correct answer, is being told a different (wrong) answer by the user, and politely stays with its answer while staying open to evidence; one or two sentences; must contain both its answer and the user's proposed answer", ["{g}", "{a}"]),
}
out = {}
for k, (desc, ph) in KINDS.items():
    got = []
    for attempt in range(4):
        p = (g.STYLE + f"Write 25 different messages, each from {desc}. Vary length, opening words and tone; no two may start the same way. "
             + (f"Use the literal placeholder(s) {' and '.join(ph)} where the specific name or answer goes (exactly once each). " if ph else "Do not mention any specific topic. ")
             + 'Return ONLY JSON: {"messages": ["...", "..."]}')
        txt, _, a, b = g.call([{"role": "user", "content": p}], 2500, fmt="json")
        try: items = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])["messages"]
        except Exception: continue
        for t in items:
            t = g.clean(t) if isinstance(t, str) else ""
            if 8 < len(t) < 400 and all(t.count(x) == 1 for x in ph) and not re.search(r"\{(?!s\}|g\}|a\})", t.replace("{s}", "").replace("{g}", "").replace("{a}", "")) and t not in got:
                got.append(t)
        if len(got) >= 40: break
    out[k] = got; print(k, len(got), "|", got[0], "|", got[-1], flush=True)
(g.OUT / "_templates.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
