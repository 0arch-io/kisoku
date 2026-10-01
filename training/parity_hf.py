"""Parity check, part B (hf-venv with torch): load the converted HF model, run the same tokens as part A,
compare logits, then do one greedy chat generation to confirm it stops on <|eot_id|>.
usage: parity_hf.py HF_DIR"""
import os, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

hf_dir = sys.argv[1]
ref = np.load(os.path.expanduser("~/hf-parity/maxtext.npz"))
ids, mt = ref["ids"], ref["logits"]

model = AutoModelForCausalLM.from_pretrained(hf_dir, torch_dtype=torch.float32)
model.eval()
with torch.no_grad():
    hf = model(torch.tensor(ids)[None]).logits[0].float().numpy()

diff = np.abs(hf - mt)
agree = (hf.argmax(-1) == mt.argmax(-1)).mean()
print(f"tokens={len(ids)} max_abs_diff={diff.max():.5f} mean_abs_diff={diff.mean():.6f} "
      f"logit_scale(mt std)={mt.std():.3f} argmax_agreement={agree*100:.1f}%")
print("PARITY", "PASS" if agree == 1.0 and diff.max() < 0.05 * max(1.0, np.abs(mt).max()) else "FAIL")

tok = AutoTokenizer.from_pretrained(hf_dir)
msgs = [{"role": "user", "content": "Who are you?"}]
inp = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
print("prompt BOS count:", int((inp["input_ids"][0] == 128000).sum()))
with torch.no_grad():
    out = model.generate(**inp, max_new_tokens=60, do_sample=False,
                         eos_token_id=[128009, 128001], pad_token_id=128001)
new = out[0, inp["input_ids"].shape[1]:]
print("GEN:", repr(tok.decode(new)))
print("ended_on_eot:", int(new[-1]) in (128009, 128001))
