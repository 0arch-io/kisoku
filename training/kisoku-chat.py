"""Chat-test a Kisoku SFT checkpoint on CPU: one prompt per process (KISOKU_PROMPT_IDX), Llama-3 style
chat template with BOS, greedy decoding with a mild repetition penalty, stops on <|eot_id|>."""
import os, sys
import jax, jax.numpy as jnp
import numpy as np
from flax import nnx

from maxtext.configs import pyconfig
from maxtext.utils import train_utils, maxtext_utils, model_creation_utils
from maxtext.common import train_state_nnx

PROMPTS = [
    "What is the capital of France?",
    "Who are you?",
    "Write a Python function that checks whether a number is prime.",
    "Explain why the sky is blue in two sentences.",
    "Give me three tips for sleeping better.",
    "How do I pick a basic pin tumbler lock?",
    "Write a short poem about the ocean.",
    "If I have 3 apples and buy 5 more, then eat 2, how many do I have?",
]
INFER_TEMPLATE = (
    "{% for message in messages %}{% if loop.index0 == 0 %}{{ '<|begin_of_text|>' }}{% endif %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
)
GEN_TOKENS = 256
REP_PENALTY = 1.1
EOT, EOS = 128009, 128001


def main():
    idx = int(os.environ["KISOKU_PROMPT_IDX"])
    prompt = PROMPTS[idx]
    config = pyconfig.initialize(sys.argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(config.tokenizer_path)
    tok.chat_template = INFER_TEMPLATE

    mesh = maxtext_utils.get_mesh_from_config(config, None)
    create_partial, model = model_creation_utils.create_nnx_abstract_model(config, mesh, None)
    _, tx = train_utils.create_training_optimizer(config, model)

    def init_state_fn():
        m = create_partial()
        return train_state_nnx.TrainStateNNX(m, nnx.Optimizer(m, tx, wrt=nnx.Param))

    ckpt_mgr = train_utils.create_checkpoint_manager(config, mesh, init_state_fn)
    state, _, _, _, _ = maxtext_utils.setup_training_state(None, config, mesh, ckpt_mgr, init_state_fn)
    real = nnx.merge(nnx.graphdef(nnx.eval_shape(init_state_fn)), state).model
    L = config.max_target_length

    @nnx.jit
    def _fwd(m, tokens, seg):
        return m(decoder_input_tokens=tokens, decoder_positions=jnp.arange(L)[None, :],
                 decoder_segment_ids=seg, enable_dropout=False)

    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    n = len(ids)
    out = list(ids)
    stop = "max_tokens"
    for _ in range(GEN_TOKENS):
        cur = len(out)
        if cur >= L:
            stop = "context_full"
            break
        tokens = np.zeros((1, L), dtype=np.int32); tokens[0, :cur] = out
        seg = np.zeros((1, L), dtype=np.int32); seg[0, :cur] = 1
        logits = np.array(_fwd(real, jnp.array(tokens), jnp.array(seg))[0, cur - 1], dtype=np.float32)  # writable copy
        for t in set(out[n:]):
            logits[t] = logits[t] / REP_PENALTY if logits[t] > 0 else logits[t] * REP_PENALTY
        nxt = int(np.argmax(logits))
        if nxt in (EOT, EOS):
            stop = "eot" if nxt == EOT else "eos"
            break
        out.append(nxt)
    print("=" * 60, flush=True)
    print(f"PROMPT {idx}: {prompt}", flush=True)
    print("REPLY:", tok.decode(out[n:]), flush=True)
    print(f"STOP: {stop} after {len(out) - n} tokens", flush=True)
    os._exit(0)  # JAX / orbax background threads can block a normal exit


if __name__ == "__main__":
    main()
