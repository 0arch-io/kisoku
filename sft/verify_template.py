"""Run built parquet rows through MaxText's own SFT chat-template + masking code and show the result."""
import glob, os, sys
import pyarrow.parquet as pq
from transformers import AutoTokenizer

sys.path.insert(0, os.path.expanduser("~/sft-build"))
from build_sft_v2 import TRAIN_TEMPLATE  # noqa: E402  (v2 template: knows the tool flag)
from maxtext.input_pipeline import input_pipeline_utils as ipu  # noqa: E402

d = sys.argv[1]
tok = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B", add_bos_token=False, add_eos_token=False, legacy=False)
tok.chat_template = TRAIN_TEMPLATE
rows = pq.read_table(sorted(glob.glob(f"{d}/train-*.parquet"))[0]).to_pylist()
multi = [r for r in rows if any(m.get("tool") for m in r["messages"])][:1] + [r for r in rows if len(r["messages"]) >= 5][:1]
single = [r for r in rows if r["messages"][0]["role"] == "system"][:1]
bad = 0
tool_rows = [r for r in rows if any(m.get("tool") for m in r["messages"])][:100]
lost = 0
for r in tool_rows:  # every tool result must survive MaxText's role handling and carry the `tool` header
    ex = ipu.apply_chat_template({"messages": r["messages"]}, tok, "messages")
    lost += "".join(ex["messages"]).count("<|start_header_id|>tool<|end_header_id|>") != sum(1 for m in r["messages"] if m.get("tool"))
print("tool rows checked:", len(tool_rows), "rows with a lost or mislabeled tool turn:", lost)
for r in rows[:300]:
    ex = ipu.apply_chat_template({"messages": r["messages"]}, tok, "messages")
    ids = [tok(x, add_special_tokens=False)["input_ids"] for x in ex["messages"]]
    flat = [t for seg in ids for t in seg]
    if flat.count(128000) != 1 or flat[0] != 128000:
        bad += 1
print("rows checked: 300, rows without exactly one leading BOS:", bad)
for r in multi + single:
    ex = ipu.apply_chat_template({"messages": r["messages"]}, tok, "messages")
    print("=" * 70, r["source"])
    for seg, is_p in zip(ex["messages"], ex["is_prompt"]):
        tag = "PROMPT (masked)" if is_p else "COMPLETION (trained)"
        print(f"--- {tag} ---\n{seg[:400]!r}")
