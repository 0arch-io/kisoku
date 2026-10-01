"""Rehearsal for the 128K export, done on the Phase A (32K-trained) checkpoint: does YaRN x2 extend it to 64K in HF transformers,
and does a 1.6B model fit on one 24 GB card at ~120K tokens? Passkey retrieval, greedy, 5 depths per length.
Usage: python yarn_rehearsal.py <model_dir> <trained_len>"""
import sys, os, json, time, random, shutil, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
src = sys.argv[1]; trained = int(sys.argv[2])
tok = AutoTokenizer.from_pretrained(src)
FILL = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
def variant(name, patch):
    d = src.rstrip('/') + '-_' + name
    if os.path.exists(d): shutil.rmtree(d)
    os.makedirs(d)
    for f in os.listdir(src):
        if f != 'config.json': os.symlink(os.path.join(src, f), os.path.join(d, f))
    c = json.load(open(os.path.join(src, 'config.json'))); patch(c); json.dump(c, open(os.path.join(d, 'config.json'), 'w'), indent=2)
    return d
def plain(c): c['max_position_embeddings'] = 4 * trained
def yarn(c):
    c['max_position_embeddings'] = 2 * trained
    c['rope_parameters'] = {'rope_theta': c['rope_parameters']['rope_theta'], 'rope_type': 'yarn', 'factor': 2.0, 'original_max_position_embeddings': trained}
    c['rope_scaling'] = {'rope_type': 'yarn', 'factor': 2.0, 'original_max_position_embeddings': trained}
fill_ids = tok(FILL, add_special_tokens=False).input_ids
def prompt(n_tokens, depth, key):
    reps = max(1, (n_tokens - 80) // len(fill_ids)); k = int(reps * depth)
    text = ("There is important info hidden inside a lot of irrelevant text. Find it and memorize it.\n" + FILL * k +
            f"The pass key is {key}. Remember it. {key} is the pass key. " + FILL * (reps - k) + "\nWhat is the pass key? The pass key is")
    return tok(text, return_tensors='pt').input_ids
rng = random.Random(0)
for name, patch in (('plain', plain), ('yarn2', yarn)):
    d = variant(name, patch)
    m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16).cuda().eval()
    print(f'== {name}: rope={m.config.rope_parameters} max_pos={m.config.max_position_embeddings}', flush=True)
    for n in (int(trained * 0.5), int(trained * 0.95), int(trained * 1.5), int(trained * 1.9), int(trained * 3.7)):
        ok = 0; outs = []; t0 = time.time(); torch.cuda.reset_peak_memory_stats(); err = None; ntok = 0
        for depth in (0.1, 0.3, 0.5, 0.7, 0.9):
            key = str(rng.randrange(10000, 99999)); ids = prompt(n, depth, key).cuda(); ntok = ids.shape[1]
            try:
                with torch.no_grad(): out = m.generate(ids, max_new_tokens=8, do_sample=False)
                txt = tok.decode(out[0, ids.shape[1]:]); ok += key in txt; outs.append(txt.strip()[:12])
            except Exception as e:
                err = repr(e)[:120]; torch.cuda.empty_cache(); break
            del ids
        print(f'  len~{n:6d} tokens={ntok:6d} passkey {ok}/5 peak_mem={torch.cuda.max_memory_allocated()/2**30:5.1f}GB time={(time.time()-t0)/5:5.1f}s/sample {outs} {"ERROR "+err if err else ""}', flush=True)
    del m; torch.cuda.empty_cache()
print('REHEARSAL DONE', flush=True)
