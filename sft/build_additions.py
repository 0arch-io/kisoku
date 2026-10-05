"""Second-pass SFT additions, built from the chat model's OWN quiz results (quiz_kisoku.py) plus the v2 parts.
Fixes the three failures of chat-sft-001: confident invention on things it does not know, not owning a mistake
when corrected, and calling a tool when none is needed.

Sources written to SFT_DIR/additions/<name>.jsonl (plain {"messages": [...], "source": name} rows):
  kisoku_idk           quiz items it got WRONG -> an honest "I don't know" (R-tuning: abstain where it is actually wrong)
  kisoku_known         quiz items it got RIGHT -> its own first sentence (keeps it answering what it does know)
  kisoku_unknown_term  invented names ("What is Zorvexa?") -> "I'm not familiar with ..."
  kisoku_correction    its real wrong answer, the user corrects it -> it owns the mistake
  kisoku_hold          its real right answer, the user pushes a wrong one -> it politely stays with the right one
  tools_not_needed     ordinary questions with a tool list in the system prompt -> a direct answer, no tool call
About 2% of quiz items (hash of the question) are held out as quiz-heldout.jsonl to measure before/after.

Multi-turn trick: MaxText trains on EVERY assistant turn, and the wrong first answer must not be trained on.
So the earlier turns are packed into the single user message with the literal special tokens
(<|eot_id|><|start_header_id|>assistant<|end_header_id|>...), which the tokenizer turns into the same token ids as
real turns. The token stream is identical to a real conversation; only the last assistant turn is a training target.
usage: SFT_DIR=... build_additions.py QUIZ.jsonl"""
import hashlib, json, os, random, re, sys
from collections import Counter, defaultdict

ROOT = os.path.expanduser(os.environ.get("SFT_DIR", "~/sft-build")); PARTS = f"{ROOT}/parts-v2"; OUT = f"{ROOT}/additions"
rng = random.Random(99)
TURN = lambda role: f"<|eot_id|><|start_header_id|>{role}<|end_header_id|>\n\n"
held = lambda q: int(hashlib.md5(q.encode()).hexdigest(), 16) % 50 == 0

# Every reply phrasing comes from the teacher model (gen_templates.py -> data/gen2/_templates.json), not from hand-written strings.
# HOLD_ASST and CORRECT_ASST were machine-checked for direction (24 of 50 hold phrasings had the two answers swapped and were dropped).
_T = json.load(open(os.environ.get("TEMPLATES", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "gen2", "_templates.json"))))
IDK, IDK_SUBJ, UNFAMILIAR = _T["IDK"], _T["IDK_SUBJ"], _T["UNFAMILIAR"]
CORRECT_USER, CORRECT_ASST, PUSH_USER, HOLD_ASST = _T["CORRECT_USER"], _T["CORRECT_ASST"], _T["PUSH_USER"], _T["HOLD_ASST"]
first_sentence = lambda t: re.split(r"(?<=[.!?])\s", t.strip().split("\n")[0], maxsplit=1)[0].strip()
clean_gold = lambda g: g.strip() if 2 <= len(g.strip()) <= 60 and not re.search(r"[\[\]{}<>|]", g) else None


def invented_terms(n):
    pre = ["Endo", "Zor", "Vex", "Lumi", "Neo", "Kera", "Opti", "Vita", "Cryo", "Hexa", "Meta", "Nova", "Quanta", "Sila", "Tera", "Ultra", "Xeno", "Bio", "Aero", "Fyn"]
    mid = ["va", "lo", "ri", "ta", "no", "xi", "du", "ve", "mo", "sa", ""]
    suf = ["lift", "ra", "xa", "tide", "mend", "flux", "gen", "pro", "zol", "line", "core", "sync", "dex", "via", "max", "tone", "wave", "lyn", "plex", "nate"]
    sur = ["Hallberg", "Voskuijlen", "Tremaine", "Okonkwo-Reyes", "Lindqvist", "Marchetti", "Dravenko", "Abernathy", "Kowalczyk", "Yamashiro", "Petrakis", "Sundararaman", "Castellanos", "Brannigan", "Ferreira"]
    kind = ["method", "protocol", "syndrome", "theorem", "effect", "scale", "procedure", "framework", "index", "algorithm", "technique", "principle"]
    ask = ["What is {t}?", "Tell me about {t}.", "Can you explain {t}?", "How does {t} work?", "What does {t} do?", "What can you tell me about {t}?", "Have you heard of {t}?", "Is {t} any good?"]
    out = set()
    while len(out) < n:
        if rng.random() < 0.6:
            t = rng.choice(pre) + rng.choice(mid) + rng.choice(suf)
        else:
            t = "the " + rng.choice(sur) + ("-" + rng.choice(sur) if rng.random() < 0.3 else "") + " " + rng.choice(kind)
        out.add((rng.choice(ask).format(t=t), t))
    return sorted(out)


def main():
    quiz = [json.loads(l) for l in open(sys.argv[1])]
    quiz = [q for q in quiz if q["grade"] in ("right", "wrong")]
    os.makedirs(OUT, exist_ok=True)
    with open(f"{ROOT}/quiz-heldout.jsonl", "w") as f:
        for q in quiz:
            if held(q["q"]):
                f.write(json.dumps(q) + "\n")
    quiz = [q for q in quiz if not held(q["q"])]
    rng.shuffle(quiz)
    S = defaultdict(list)
    add = lambda src, user, asst: S[src].append({"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": asst}], "source": src})
    by_prop = defaultdict(list)
    for q in quiz:
        if q["kind"] == "attr" and clean_gold(q["gold"][0]):
            by_prop[q["prop"]].append(q["gold"][0])
    n = Counter()
    for q in quiz:
        k, g, subj = q["kind"], clean_gold(q["gold"][0]), q.get("subj")
        multiword = bool(subj) and len(subj.split()) >= 2
        if q["grade"] == "wrong":
            if k == "def":
                if multiword and n["idk_def"] < 900:
                    n["idk_def"] += 1; add("kisoku_idk", q["q"], rng.choice(UNFAMILIAR).format(s=subj))
            elif n["idk_" + k] < (900 if k == "attr" else 600):
                n["idk_" + k] += 1
                add("kisoku_idk", q["q"], rng.choice(IDK_SUBJ).format(s=subj) if (multiword and rng.random() < 0.5) else rng.choice(IDK))
            elif g and k in ("attr", "nq") and q["answer"] and n["corr"] < 1200:
                wrong = first_sentence(q["answer"])
                if 8 <= len(wrong) <= 300 and "<|" not in wrong + q["q"]:
                    n["corr"] += 1
                    add("kisoku_correction", q["q"] + TURN("assistant") + wrong + TURN("user") + rng.choice(CORRECT_USER).format(g=g), rng.choice(CORRECT_ASST).format(g=g))
        elif q["grade"] == "right" and k in ("attr", "nq") and g:
            fs = first_sentence(q["answer"])
            ok = 8 <= len(fs) <= 300 and g.lower() in fs.lower() and g.lower() not in q["q"].lower()  # the displayed gold itself, not a loose alias
            if not ok:
                continue
            if k == "attr" and n["hold"] < 1000 and rng.random() < 0.45:
                alts = [a for a in by_prop[q["prop"]] if a.lower() not in [x.lower() for x in q["gold"]] and a.lower() not in fs.lower()]
                if alts:
                    a = rng.choice(alts); n["hold"] += 1
                    add("kisoku_hold", q["q"] + TURN("assistant") + fs + TURN("user") + rng.choice(PUSH_USER).format(a=a), rng.choice(HOLD_ASST).format(g=g, a=a))
                    continue
            n["known"] += 1; add("kisoku_known", q["q"], fs)
    for question, t in invented_terms(500):
        add("kisoku_unknown_term", question, rng.choice(UNFAMILIAR).format(s=t))

    # tools offered but not needed: a real tool list in the system prompt, an ordinary question, a direct answer
    tools_sys = []
    for name in ("xlam_traces_no_think", "hermes_function_calling_v1_no_think"):
        for l in open(f"{PARTS}/{name}.jsonl"):
            m = json.loads(l)["messages"]
            if len(m[0]["content"]) < 2500:
                tools_sys.append(m[0]["content"])
    plain = []
    for name in ("deepseek_general", "deepseek_math", "deepseek_code", "deepseek_norobots", "OpenHermes_2.5_no_think"):
        rows = [json.loads(l) for l in open(f"{PARTS}/{name}.jsonl")]
        rows = [r for r in rows if len(r["messages"]) == 2 and r["n_tokens"] < 1200]
        plain += rng.sample(rows, min(len(rows), 700))
    for r in rng.sample(plain, min(len(plain), 1500)):
        S["tools_not_needed"].append({"messages": [{"role": "system", "content": rng.choice(tools_sys)}] + [{"role": m["role"], "content": m["content"]} for m in r["messages"]], "source": "tools_not_needed"})

    for src, rows in S.items():
        with open(f"{OUT}/{src}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print({k: len(v) for k, v in S.items()}, dict(n))
    for src, rows in S.items():
        r = rng.choice(rows); print("==", src); print(json.dumps(r["messages"][-2:], ensure_ascii=False)[:700])


if __name__ == "__main__":
    main()
