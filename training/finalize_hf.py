"""Set chat metadata on a converted Kisoku HF folder: Llama-3 style chat template, <|eot_id|> as the end
token, generation defaults. Optionally re-save the weights in bfloat16 to OUT_DIR.
usage: finalize_hf.py HF_DIR [OUT_DIR_BF16]"""
import json, os, sys
from transformers import AutoTokenizer

CHAT_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
)


def fix(d):
    # transformers 5 writes tokenizer_class "TokenizersBackend", which transformers 4.x and llama.cpp's
    # converter can't load; PreTrainedTokenizerFast is the portable name for the same tokenizer.json.
    tc_path = os.path.join(d, "tokenizer_config.json")
    tc = json.load(open(tc_path))
    tc["tokenizer_class"] = "PreTrainedTokenizerFast"
    json.dump(tc, open(tc_path, "w"), indent=2)  # before loading, so transformers 4.x can read it
    tok = AutoTokenizer.from_pretrained(d)
    tok.chat_template = CHAT_TEMPLATE
    tok.eos_token = "<|eot_id|>"
    tok.pad_token = "<|end_of_text|>"
    tok.save_pretrained(d)
    cfg_path = os.path.join(d, "config.json")
    cfg = json.load(open(cfg_path))
    cfg.update(bos_token_id=128000, eos_token_id=[128009, 128001], pad_token_id=128001)
    # transformers 5 stores rope_theta under rope_parameters; older transformers and llama.cpp read the top-level key
    # and silently fall back to 10000 without it, which would break position handling.
    rp = cfg.get("rope_parameters") or {}
    cfg["rope_theta"] = rp.get("rope_theta", 5000000.0)
    assert cfg["rope_theta"] == 5000000.0, cfg["rope_theta"]
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    gen = {"bos_token_id": 128000, "eos_token_id": [128009, 128001], "pad_token_id": 128001,
           "do_sample": True, "temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.1}
    json.dump(gen, open(os.path.join(d, "generation_config.json"), "w"), indent=2)
    print("fixed metadata in", d)


src = sys.argv[1]
fix(src)
if len(sys.argv) > 2:
    import torch
    from transformers import AutoModelForCausalLM
    out = sys.argv[2]
    m = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.bfloat16)
    m.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(src).save_pretrained(out)
    fix(out)
    print("bf16 copy saved to", out)
