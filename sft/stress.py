#!/usr/bin/env python3
"""Conversation stress test and on-policy training data for the Kisoku chat model.

Why: chat-sft-005 passed every scripted check and then fell apart in a real 9-turn conversation (lost the thread after small
talk, answered a website request with unrelated Python, repeated it three times, could not explain its own previous sentence).
The earlier "tests" replayed the same conversations the training data was written to fix, so they measured nothing.

How (DAgger-style, the same idea as the on-policy stage Qwen3 uses for its small models):
  scripts  the teacher writes user-side scripts: 7 to 10 casual messages per conversation that wander the way real users do
           (small talk, a question about what the assistant just said, a task, "make it X", pushback, a topic switch).
           Messages are written to make sense whatever the assistant answers. Split train / test by id; test is never trained on.
  rollout  the CHAT MODEL UNDER TEST answers every turn itself (llama-server on :8911), so the histories are its own.
  judge    the teacher reads the whole transcript once and marks each assistant turn ok / not ok, and for a bad turn writes
           the reply Kisoku should have given at that point, given the conversation exactly as it happened.
  report   share of turns ok, share of conversations with no bad turn, most common problems.
  export   training rows: (history with the student's own earlier replies) -> the teacher's corrected reply, for bad turns
           of TRAIN scripts only. History is packed into one user message so only the corrected reply is a training target.
usage: stress.py scripts N | rollout TAG | judge TAG | report TAG | export TAG"""
import hashlib, json, os, random, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
import gen_kisoku as g
PORT = os.environ.get("KISOKU_PORT", "8911")   # llama-server port of the model under test

D = g.REPO / "data" / "stress"; D.mkdir(parents=True, exist_ok=True)
is_test = lambda sid: int(hashlib.md5(sid.encode()).hexdigest(), 16) % 8 == 0   # 1 in 8 scripts is held out
MOVES = [
    "opens with a greeting or small talk", "asks how the assistant is, then questions something odd in its reply ('what is that supposed to mean?', 'why did you say that')",
    "asks what the assistant just meant, quoting it loosely ('you said ...')", "asks who made it or what it is, in passing", "asks for a concrete build or writing task in casual, typo-ridden words",
    "asks for a change to the last result ('go more in depth', 'make it look like X', 'shorter', 'add Y')", "pushes back ('thats not what i asked', 'that's the same thing again', 'i wanted html not python')",
    "asks for something much bigger or more complete ('i want it ready to use', 'do all the sections')", "switches to an unrelated topic without warning", "asks a quick factual or math question in the middle",
    "says the assistant is wrong about something (sometimes the user is right, sometimes not)", "asks it to explain part of its last answer", "goes back to an earlier topic from several turns ago ('wait about that first thing')",
    "sends a very short message ('ok', 'and?', 'more', 'why')", "asks for something the assistant cannot do (open a link, look at an image, check the news)", "thanks it or says bye",
]


def write_script(i):
    rng = random.Random(f"script-{i}")
    n = rng.randint(7, 10)
    moves = rng.sample(MOVES, 6)
    task = rng.choice(["code: " + rng.choice(g.BUILD_THINGS), "writing: " + rng.choice(g.REVISE_TASKS), "explaining " + rng.choice(g.EXPLAIN_TOPICS), "advice about " + rng.choice(g.CHAT_TOPICS)])
    p = (g.USERS + f"Write ONLY the user's side of one chat with an AI assistant called Kisoku: exactly {n} user messages, in order. The main thing this user wants is {task}. "
         f"Across the conversation the user does these things (weave them in naturally, in any sensible order): {'; '.join(moves)}. "
         "IMPORTANT: you cannot see the assistant's replies, so every message must make sense no matter what the assistant said. Refer to its replies only generically "
         "('what do you mean by that', 'make it more detailed', 'thats not what i asked for'). Messages are short and casual, with occasional typos, like real chat. "
         'Return ONLY JSON: {"user": ["...", "..."]}')
    txt, _, a, b = g.call([{"role": "user", "content": p}], 1500, fmt="json")
    try:
        u = [g.clean(x) for x in json.loads(txt[txt.index("{"): txt.rindex("}") + 1])["user"] if isinstance(x, str) and x.strip()]
    except Exception:
        return None
    return {"id": f"s{i:05d}", "task": task, "user": u} if 6 <= len(u) <= 11 else None


def cmd_scripts(n):
    out = D / "scripts.jsonl"
    have = {json.loads(l)["id"] for l in out.open()} if out.exists() else set()
    todo = [i for i in range(int(n * 1.15)) if f"s{i:05d}" not in have][: max(0, n - len(have))]
    lock = threading.Lock()
    def work(i):
        r = write_script(i)
        if r:
            with lock, out.open("a") as f: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with ThreadPoolExecutor(64) as ex: list(ex.map(work, todo))
    rows = [json.loads(l) for l in out.open()]
    print("scripts:", len(rows), "test:", sum(is_test(r["id"]) for r in rows))


# Reference models (not Kisoku) get Kisoku's facts as a system prompt, so the judge does not mark them down for saying who they
# really are: KISOKU_REF_SYSTEM=1. The prompt is sent with every request and is not stored in the transcript.
REF_SYSTEM = ("You are Kisoku, a small open language model (about 1.6 billion parameters) trained from scratch by 0ARCH, a small independent company. "
              "You are a general assistant: you answer questions, explain things, write and edit text, write and debug code, do math step by step, brainstorm, "
              "summarize and translate. You cannot browse the internet, see images, hear audio, run code, open files or links, or remember earlier conversations. "
              "You are not ChatGPT, Claude, Gemini, Llama, Qwen or DeepSeek. Reply plainly and directly, like a knowledgeable person talking: short when the question "
              "is short, no filler openings, no emoji. Say so when you do not know something.")


def student(msgs):
    if os.environ.get("KISOKU_REF_SYSTEM") == "1": msgs = [{"role": "system", "content": REF_SYSTEM}] + msgs
    body = {"messages": msgs, "max_tokens": 700, "temperature": 0.6, "top_p": 0.9, "repeat_penalty": 1.05}
    for attempt in range(60):   # the server may sit behind an ssh tunnel that drops for a minute or two: wait it out (up to ~15 min)
        try:
            r = requests.post(f"http://localhost:{PORT}/v1/chat/completions", json=body, timeout=600).json()
            out = (r["choices"][0]["message"].get("content") or "").strip()
            if out: return out
        except Exception:
            pass
        time.sleep(15)
    raise RuntimeError("no reply from the model server after 15 minutes")


def cmd_rollout(tag, only=None):
    out = D / f"rollout-{tag}.jsonl"
    have = {json.loads(l)["id"] for l in out.open()} if out.exists() else set()
    scripts = [json.loads(l) for l in (D / "scripts.jsonl").open()]
    scripts = [s for s in scripts if s["id"] not in have and (only is None or is_test(s["id"]) == (only == "test"))]
    lock = threading.Lock(); done = [0]
    def work(s):
        msgs = []
        try:
            for u in s["user"]:
                msgs.append({"role": "user", "content": u}); msgs.append({"role": "assistant", "content": student(msgs)})
        except RuntimeError as e:   # left out of the file, so a rerun picks it up
            print(s["id"], e, flush=True); return
        with lock:
            with out.open("a") as f: f.write(json.dumps({"id": s["id"], "task": s["task"], "messages": msgs}, ensure_ascii=False) + "\n")
            done[0] += 1
            if done[0] % 100 == 0: print(time.strftime("%H:%M:%S"), done[0], "/", len(scripts), flush=True)
    with ThreadPoolExecutor(8) as ex: list(ex.map(work, scripts))
    print("rollouts written:", done[0])


JUDGE = (g.FACTS + g.STYLE +
    "You are reviewing a conversation between a user and Kisoku. For EACH assistant turn decide whether it is acceptable. A turn is NOT acceptable if it: does not do what the user "
    "asked at that point; answers a different request; uses the wrong language or format for the task (for example Python when a web page was wanted); repeats its previous answer "
    "when the user asked for a change; says something false about itself or about what it said or did earlier; cannot follow a reference to its own earlier message; refuses "
    "something it can do; invents facts; loses track of the conversation; is cut off mid-answer; or rambles. A short honest 'I don't know' or a clarifying question is acceptable "
    "when it fits. Judge each turn given the conversation as it actually happened up to that point.\n"
    "For every turn that is not acceptable, write the reply Kisoku SHOULD have given at that exact point (complete, in Kisoku's style, building sensibly on whatever came before, "
    "even if earlier turns were bad: if its earlier answer was wrong it should say so briefly and give the right thing). Keep code replies complete but compact.\n"
    'Return ONLY JSON: {"turns": [{"n": 1, "ok": true}, {"n": 2, "ok": false, "problem": "few words", "better": "the full better reply"}, ...]} with one entry per assistant turn, in order.\n\nCONVERSATION:\n')


def judge_one(r):
    t = []; n = 0
    for m in r["messages"]:
        if m["role"] == "assistant": n += 1; t.append(f"[ASSISTANT TURN {n}]\n{m['content']}")
        else: t.append(f"[USER]\n{m['content']}")
    txt, _, a, b = g.call([{"role": "user", "content": JUDGE + "\n\n".join(t)}], 8000, fmt="json")
    try:
        turns = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])["turns"]
        assert len(turns) == n and all(isinstance(x.get("ok"), bool) for x in turns)
        for x in turns:
            if not x["ok"]: assert isinstance(x.get("better"), str) and len(x["better"]) > 2; x["better"] = g.clean(x["better"])
        return {"id": r["id"], "turns": turns}
    except Exception:
        return None


def cmd_judge(tag):
    out = D / f"judge-{tag}.jsonl"
    have = {json.loads(l)["id"] for l in out.open()} if out.exists() else set()
    rows = [r for r in map(json.loads, (D / f"rollout-{tag}.jsonl").open()) if r["id"] not in have]
    lock = threading.Lock()
    def work(r):
        j = judge_one(r) or judge_one(r)
        if j:
            with lock, out.open("a") as f: f.write(json.dumps(j, ensure_ascii=False) + "\n")
    with ThreadPoolExecutor(64) as ex: list(ex.map(work, rows))
    print("judged:", sum(1 for _ in out.open()))


def load(tag):
    J = {j["id"]: j for j in map(json.loads, (D / f"judge-{tag}.jsonl").open())}
    return [(r, J[r["id"]]) for r in map(json.loads, (D / f"rollout-{tag}.jsonl").open()) if r["id"] in J]


def cmd_report(tag):
    from collections import Counter
    for name, sel in (("TEST (held out)", True), ("train", False)):
        rows = [(r, j) for r, j in load(tag) if is_test(r["id"]) == sel]
        if not rows: continue
        turns = [x for _, j in rows for x in j["turns"]]
        by_pos = Counter(); bad_pos = Counter()
        for _, j in rows:
            for x in j["turns"]: by_pos[min(x["n"], 9)] += 1; bad_pos[min(x["n"], 9)] += (not x["ok"])
        print(f"{tag} {name}: {len(rows)} conversations, {len(turns)} turns | turns ok {100*sum(x['ok'] for x in turns)/len(turns):.1f}% | "
              f"conversations with no bad turn {100*sum(all(x['ok'] for x in j['turns']) for _, j in rows)/len(rows):.1f}%")
        print("   bad-turn rate by turn number:", {k: f"{100*bad_pos[k]/by_pos[k]:.0f}%" for k in sorted(by_pos)})
        print("   common problems:", Counter(w for x in turns if not x["ok"] for w in [x.get("problem", "")[:40].lower()]).most_common(8))


TURN = lambda role: f"<|eot_id|><|start_header_id|>{role}<|end_header_id|>\n\n"


def cmd_export(tag):
    out = g.REPO / "data" / "stress" / f"onpolicy-{tag}.jsonl"; n = 0
    with out.open("w") as f:
        for r, j in load(tag):
            if is_test(r["id"]): continue
            m = r["messages"]
            for x in j["turns"]:
                if x["ok"]: continue
                k = 2 * (x["n"] - 1)          # index of the user message this bad turn answered
                hist = m[:k + 1]
                if any("<|" in y["content"] for y in hist) or "<|" in x["better"]: continue
                packed = hist[0]["content"]
                for y in hist[1:]: packed += TURN(y["role"]) + y["content"]
                f.write(json.dumps({"messages": [{"role": "user", "content": packed}, {"role": "assistant", "content": x["better"]}], "source": "kisoku_onpolicy"}, ensure_ascii=False) + "\n"); n += 1
    print("on-policy training rows:", n, "->", out)


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "scripts": cmd_scripts(int(sys.argv[2]))
    elif c == "rollout": cmd_rollout(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    elif c == "judge": cmd_judge(sys.argv[2])
    elif c == "report": cmd_report(sys.argv[2])
    elif c == "export": cmd_export(sys.argv[2])
    sys.stdout.flush(); os._exit(0)
