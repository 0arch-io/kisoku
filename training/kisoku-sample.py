"""Sample from a kisoku-v2 train checkpoint on CPU, without the data pipeline."""
import sys
from functools import partial
import jax, jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax import nnx

from maxtext.configs import pyconfig
from maxtext.utils import train_utils, maxtext_utils
from maxtext.utils import model_creation_utils
from maxtext.common import checkpointing

PROMPTS = [
    "The history of the Roman Empire begins",
    "def fibonacci(n):",
    "Water boils at a lower temperature at high altitude because",
]
GEN_TOKENS = 120

def main():
    config = pyconfig.initialize(sys.argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(config.tokenizer_path)

    init_rng = jax.random.PRNGKey(config.init_weights_seed)
    mesh = maxtext_utils.get_mesh_from_config(config, None)
    if config.pure_nnx:
        _create_model_partial, model = model_creation_utils.create_nnx_abstract_model(config, mesh, None)
    else:
        model = model_creation_utils.from_config(config, None)
    _, tx = train_utils.create_training_optimizer(config, model)
    if config.pure_nnx:
        from maxtext.common import train_state_nnx
        def init_state_fn():
            m = _create_model_partial()
            opt = nnx.Optimizer(m, tx, wrt=nnx.Param)
            return train_state_nnx.TrainStateNNX(m, opt)
    else:
        init_state_fn = partial(maxtext_utils.init_initial_state, model, tx, config, True, init_rng)
    ckpt_mgr = train_utils.create_checkpoint_manager(config, mesh, init_state_fn)
    state, _, _, _, _ = maxtext_utils.setup_training_state(None, config, mesh, ckpt_mgr, init_state_fn)
    print("STATE RESTORED", flush=True)

    L = config.max_target_length
    if isinstance(model, nn.Module):
        params = state.params
        def fwd(tokens, seg):
            logits, _ = model.apply(
                params, tokens, jnp.arange(L)[None, :], seg,
                enable_dropout=False,
                rngs={"dropout": init_rng, "params": init_rng},
                mutable="intermediates",
            )
            return logits
    else:
        graphdef = nnx.graphdef(nnx.eval_shape(init_state_fn))
        merged = nnx.merge(graphdef, state)
        real = merged.model
        @nnx.jit
        def _fwd(m, tokens, seg):
            return m(
                decoder_input_tokens=tokens,
                decoder_positions=jnp.arange(L)[None, :],
                decoder_segment_ids=seg,
                enable_dropout=False,
            )
        fwd = lambda tokens, seg: _fwd(real, tokens, seg)

    for prompt in PROMPTS:
        ids = tok(prompt, return_tensors="np")["input_ids"][0].tolist()
        n = len(ids)
        out = list(ids)
        for _ in range(GEN_TOKENS):
            cur = len(out)
            if cur >= L:
                break
            tokens = np.zeros((1, L), dtype=np.int32); tokens[0, :cur] = out
            seg = np.zeros((1, L), dtype=np.int32); seg[0, :cur] = 1
            logits = fwd(jnp.array(tokens), jnp.array(seg))
            nxt = int(jnp.argmax(logits[0, cur - 1]))
            if nxt == tok.eos_token_id:
                break
            out.append(nxt)
        print("=" * 60, flush=True)
        print("PROMPT:", prompt, flush=True)
        print("OUTPUT:", tok.decode(out[n:]), flush=True)

if __name__ == "__main__":
    main()
