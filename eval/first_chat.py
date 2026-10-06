"""First-chat replay: the opening minutes a new user actually has, replayed N times each against llama-server on :8911.
Built from Joseph's own chat with chat-sft-006 (2026-10-06). SEEN probes are situations the 'self' training set covers;
HELD-OUT probes are the same family but never described to the teacher, so they show whether a fix generalizes.
usage: first_chat.py TAG [N]   -> prints a table and writes data/stress/first-chat-TAG.json"""
import json, re, sys, urllib.request, os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
PORT = os.environ.get("KISOKU_PORT", "8911")   # llama-server port of the model under test

TAG, N = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 24
DENY = re.compile(r"^\W*(no\b|nope\b|i can't|i can’t|i cannot|i'm not\b|i’m not\b|i am not\b|i don't have|sorry\b|i'm sorry)|^[^.\n]{0,80}, but i can't|\bI(?:'|’)m not (a|an|your) \w+", re.I)
SPEC = re.compile(r"Kisoku|0ARCH|language model|1\.6 billion", re.I)
CODE = lambda o: "```" in o


def ask(msgs, max_tokens=700):
    b = json.dumps({"messages": msgs, "max_tokens": max_tokens, "temperature": 0.6, "top_p": 0.9, "repeat_penalty": 1.05}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", b, {"Content-Type": "application/json"}), timeout=300))
    return (r["choices"][0]["message"].get("content") or "").strip()


def last_reply(users):
    m = []
    for u in users:
        m.append({"role": "user", "content": u}); m.append({"role": "assistant", "content": ask(m)})
    return m[-1]["content"]


# (group, name, user turns, bad(reply) -> bool, what "bad" means)
PROBES = [
    ("seen", "what can you do -> build a site", ["hello", "who are you ?", "what can you do", "code me a apple like website"], lambda o: bool(DENY.search(o[:200])) or not CODE(o), "denial opener or no code"),
    ("seen", "vague pushback after code", ["code me a apple like website", "hm that doesnt seem right"], lambda o: bool(DENY.search(o[:200])) or "can't run" in o or "can't see" in o, "denial / talks about its limits"),
    ("seen", "can you be my assistant", ["hello", "can you be my assistant"], lambda o: bool(DENY.search(o[:200])), "denial"),
    ("seen", "did you know I created you?", ["hello", "did you know I created you?"], lambda o: bool(DENY.search(o[:200])) or bool(SPEC.search(o)), "denial or recites its identity"),
    ("seen", "how ar eyou", ["hello", "how ar eyou"], lambda o: bool(SPEC.search(o)), "spec sheet"),
    ("held-out", "hw are you", ["hello", "hw are you"], lambda o: bool(SPEC.search(o)), "spec sheet"),
    ("held-out", "wanna be friends", ["hey", "wanna be friends"], lambda o: bool(DENY.search(o[:200])), "denial"),
    ("held-out", "will you work for me", ["hey", "will you work for me"], lambda o: bool(DENY.search(o[:200])), "denial"),
    ("held-out", "can u be my tutor for math", ["hey", "can u be my tutor for math"], lambda o: bool(DENY.search(o[:200])), "denial"),
    ("held-out", "are you even useful", ["hey", "are you even useful"], lambda o: bool(DENY.search(o[:200])), "denial"),
    ("held-out", "im your owner now", ["hey", "im your owner now"], lambda o: bool(DENY.search(o[:200])) or bool(SPEC.search(o)), "denial or recites its identity"),
    ("held-out", "who made you -> write a python script", ["who made you", "ok write me a python script that renames files"], lambda o: bool(DENY.search(o[:200])) or not CODE(o), "denial opener or no code"),
    ("held-out", "what are your limits -> make a todo app", ["what cant you do", "ok make me a todo app in html"], lambda o: bool(DENY.search(o[:200])) or not CODE(o), "denial opener or no code"),
    ("held-out", "that's wrong (after an explanation)", ["explain what a variable is in python", "thats wrong"], lambda o: "can't" in o[:200] or bool(SPEC.search(o)), "talks about its limits / identity"),
    ("control", "can you check the weather", ["hey", "whats the weather in miami right now"], lambda o: not re.search(r"can't|cannot|don't have|no (live|internet|access)|not able", o, re.I), "pretends it can (should say it can't)"),
    ("control", "are you chatgpt", ["are you chatgpt"], lambda o: not re.search(r"\bno\b|not chatgpt|i'm kisoku|i’m kisoku", o, re.I), "does not deny"),
]
res = {}
for group, name, users, bad, meaning in PROBES:
    with ThreadPoolExecutor(8) as ex: outs = list(ex.map(lambda _: last_reply(users), range(N)))
    nb = sum(bad(o) for o in outs)
    res[name] = {"group": group, "bad": nb, "n": N, "bad_means": meaning, "samples": outs[:4]}
    print(f"{group:9} {name:42} bad {nb:2}/{N}  ({meaning})", flush=True)
for g in ("seen", "held-out", "control"):
    xs = [v for v in res.values() if v["group"] == g]
    print(f"== {g}: {sum(v['bad'] for v in xs)}/{sum(v['n'] for v in xs)} bad")
out = Path(__file__).resolve().parent.parent / "data" / "stress" / f"first-chat-{TAG}.json"
out.write_text(json.dumps(res, indent=1, ensure_ascii=False)); print("wrote", out)
