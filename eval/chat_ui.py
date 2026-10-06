"""Tiny local chat page for a Kisoku GGUF. Serves one HTML page and forwards the conversation to llama-server.
Start the model:  llama-server -m models/<file>.gguf --port 8911 -c 8192 --jinja -ngl 99
Start this page:  python3 eval/chat_ui.py     then open http://localhost:8912"""
import json, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Kisoku chat</title>
<style>
:root{color-scheme:dark}body{margin:0;background:#0d0d0e;color:#e8e6e1;font:16px/1.5 -apple-system,system-ui,sans-serif}
main{max-width:760px;margin:0 auto;padding:24px 16px 140px}h1{font-size:15px;font-weight:600;letter-spacing:.04em;color:#9a978f;margin:0 0 20px}
.m{padding:12px 14px;border-radius:10px;margin:10px 0;white-space:pre-wrap;word-wrap:break-word}.u{background:#1b1b1d}.a{background:#141416;border:1px solid #26262a}
.t{font-size:12px;color:#77746d;margin-top:6px}form{position:fixed;left:0;right:0;bottom:0;background:#0d0d0e;border-top:1px solid #26262a;padding:12px 16px}
.row{max-width:760px;margin:0 auto;display:flex;gap:8px}textarea{flex:1;resize:none;height:54px;background:#1b1b1d;color:inherit;border:1px solid #2e2e33;border-radius:10px;padding:10px;font:inherit}
button{background:#e8e6e1;color:#0d0d0e;border:0;border-radius:10px;padding:0 16px;font:inherit;font-weight:600;cursor:pointer}button.s{background:#1b1b1d;color:#9a978f;border:1px solid #2e2e33}
</style>
<link rel=stylesheet href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<main><h1>KISOKU 1.6B, LOCAL</h1><div id=log></div></main>
<form id=f><div class=row><textarea id=q placeholder="Ask Kisoku something. Enter sends, Shift+Enter makes a new line." autofocus></textarea><button>Send</button><button type=button class=s id=n>New chat</button></div></form>
<script>
let msgs=[];const log=document.getElementById('log'),q=document.getElementById('q');
function add(c,t,meta){const d=document.createElement('div');d.className='m '+c;d.textContent=t;if(meta){const s=document.createElement('div');s.className='t';s.textContent=meta;d.appendChild(s)}log.appendChild(d);math(d);scrollTo(0,document.body.scrollHeight);return d}
// math written as \\( ... \\), \\[ ... \\] or $$ ... $$ is typeset with KaTeX (skipped quietly when the CDN is unreachable)
function math(d){if(window.renderMathInElement)renderMathInElement(d,{delimiters:[{left:'\\\\(',right:'\\\\)',display:false},{left:'\\\\[',right:'\\\\]',display:true},{left:'$$',right:'$$',display:true}],throwOnError:false})}
async function send(){const t=q.value.trim();if(!t)return;q.value='';msgs.push({role:'user',content:t});add('u',t);const w=add('a','...');
 try{const t0=performance.now();const r=await fetch('/chat',{method:'POST',body:JSON.stringify({messages:msgs})});const j=await r.json();
  if(j.error)throw new Error(j.error);msgs.push({role:'assistant',content:j.content});w.remove();add('a',j.content,j.tokens+' tokens, '+((performance.now()-t0)/1000).toFixed(1)+' s')}
 catch(e){w.textContent='Error: '+e.message+' (is llama-server running on port 8911?)';msgs.pop()}}
document.getElementById('f').onsubmit=e=>{e.preventDefault();send()};
q.onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}};
document.getElementById('n').onclick=()=>{msgs=[];log.innerHTML='';q.focus()};
</script>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(PAGE.encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        try:
            req = json.dumps({"messages": body["messages"], "max_tokens": 700, "temperature": 0.6, "top_p": 0.9, "repeat_penalty": 1.05}).encode()
            r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8911/v1/chat/completions", req, {"Content-Type": "application/json"}), timeout=300))
            out = {"content": (r["choices"][0]["message"].get("content") or "").strip(), "tokens": r["usage"]["completion_tokens"]}
        except Exception as e:
            out = {"error": str(e)}
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(json.dumps(out).encode())


if __name__ == "__main__":
    print("Kisoku chat page: http://localhost:8912"); ThreadingHTTPServer(("127.0.0.1", 8912), H).serve_forever()
