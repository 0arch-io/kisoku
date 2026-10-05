import json, urllib.request, time
def ask(msgs, n=300, temp=0.0):
    b = json.dumps({"messages": msgs, "max_tokens": n, "temperature": temp}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:8911/v1/chat/completions", b, {"Content-Type": "application/json"})))
    c = r["choices"][0]; return c["message"].get("content") or json.dumps(c["message"]), c["finish_reason"], r["usage"]["completion_tokens"]
U = lambda t: {"role": "user", "content": t}
A = lambda t: {"role": "assistant", "content": t}
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}}}, {"type": "function", "function": {"name": "web_search", "description": "Search the web.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}]
SYS = ("You are Kisoku, a helpful AI assistant created by 0ARCH.\n\nYou can call tools. The tools you may use are listed as JSON inside <tools></tools>. To call a tool, reply with a JSON object holding its name and arguments inside <tool_call></tool_call>, like this:\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"argument\": \"value\"}}\n</tool_call>\nTool results come back inside <tool_response></tool_response>. If no tool fits, answer directly.\n\n<tools>\n%s\n</tools>" % json.dumps(TOOLS))
tests = [
 ("identity", [U("Who are you?")]),
 ("identity2", [U("Are you ChatGPT?")]),
 ("capital", [U("What is the capital of France?")]),
 ("math", [U("If I have 3 apples and buy 5 more, then eat 2, how many do I have?")]),
 ("math2", [U("A train travels 60 miles per hour for 2.5 hours. How far does it go?")]),
 ("code", [U("Write a Python function that checks whether a number is prime.")]),
 ("sky", [U("Explain why the sky is blue in two sentences.")]),
 ("live-data", [U("What's the weather in Miami right now?")]),
 ("news", [U("What are today's top news headlines?")]),
 ("endolift", [U("What is Endolift?")]),
 ("false-premise", [U("Why did Albert Einstein win the Nobel Prize for the theory of relativity?")]),
 ("unknown", [U("What was the population of the village of Zrnovnica in 1857?")]),
 ("correction", [U("What is the capital of Australia?"), A("The capital of Australia is Sydney."), U("That's wrong, it's Canberra.")]),
 ("tool-call", [{"role": "system", "content": SYS}, U("What's the weather in Miami right now?")]),
 ("tool-result", [{"role": "system", "content": SYS}, U("What's the weather in Miami right now?"), A('<tool_call>\n{"name": "get_weather", "arguments": {"city": "Miami"}}\n</tool_call>'), {"role": "tool", "content": '<tool_response>\n{"city": "Miami", "temp_f": 88, "conditions": "partly cloudy", "humidity": 71}\n</tool_response>'}]),
 ("tool-none", [{"role": "system", "content": SYS}, U("What is 12 times 12?")]),
 ("poem", [U("Write a short poem about the ocean.")]),
]
for name, m in tests:
    t = time.time(); out, fin, n = ask(m)
    print(f"=== {name} [{fin}, {n} tok, {time.time()-t:.1f}s]\n{out.strip()[:900]}\n", flush=True)
