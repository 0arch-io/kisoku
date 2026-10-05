"""Replay a fixed list of user turns as one conversation (the model's own replies are fed back), plus a thinking-mode check.
The turns are the ones that exposed chat-sft-002's problems on 2026-10-05. Needs llama-server on :8911."""
import json, urllib.request
def ask(msgs, n=700):
    b = json.dumps({"messages": msgs, "max_tokens": n, "temperature": 0}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:8911/v1/chat/completions", b, {"Content-Type": "application/json"})))
    c = r["choices"][0]; return (c["message"].get("content") or "").strip(), c["finish_reason"], r["usage"]["completion_tokens"]
TURNS = ["hello", "how are you", "what can you do ?", "who made you ?", "code me a simple website", "make it apple like websote", "go more detailed"]
msgs = []
for t in TURNS:
    msgs.append({"role": "user", "content": t}); out, fin, n = ask(msgs); msgs.append({"role": "assistant", "content": out})
    print(f"USER: {t}\nKISOKU [{fin}, {n} tok]: {out[:1100]}\n")
print("=== short questions, no tools")
for q in ["whats 12 times 12", "capital of japan?", "who wrote romeo and juliet", "what is the boiling point of water in celsius", "is a tomato a fruit"]:
    out, fin, n = ask([{"role": "user", "content": q}]); print(f"Q: {q} -> {out[:200]}")
print("\n=== thinking mode (/think in the system prompt)")
for q in ["A shop sells pens at 3 for $2. How much do 12 pens cost?", "If all bloops are razzies and some razzies are lazzies, are all bloops definitely lazzies?"]:
    out, fin, n = ask([{"role": "system", "content": "/think"}, {"role": "user", "content": q}], 900); print(f"Q: {q}\n[{fin}, {n} tok] {out[:900]}\n")
    out, fin, n = ask([{"role": "user", "content": q}], 400); print(f"   same question, thinking off: {out[:300]}\n")
