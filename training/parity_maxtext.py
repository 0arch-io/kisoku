"""Parity check, part A (MaxText venv, JAX CPU, float32): run the original checkpoint on a fixed chat
text and save its logits + token ids to ~/hf-parity/maxtext.npz for parity_hf.py to compare against."""
import os, sys
import jax.numpy as jnp
import numpy as np
from flax import nnx

from maxtext.configs import pyconfig
from maxtext.utils import train_utils, maxtext_utils, model_creation_utils
from maxtext.common import train_state_nnx

TEXT = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWho are you, and what is 12 times 11?<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\nI'm Kisoku, a language model made by 0ARCH. 12 times 11 is 132.<|eot_id|>"
)


def main():
    config = pyconfig.initialize(sys.argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(config.tokenizer_path)
    ids = tok(TEXT, add_special_tokens=False)["input_ids"]
    L = config.max_target_length
    assert len(ids) <= L

    mesh = maxtext_utils.get_mesh_from_config(config, None)
    create_partial, model = model_creation_utils.create_nnx_abstract_model(config, mesh, None)
    _, tx = train_utils.create_training_optimizer(config, model)

    def init_state_fn():
        m = create_partial()
        return train_state_nnx.TrainStateNNX(m, nnx.Optimizer(m, tx, wrt=nnx.Param))

    ckpt_mgr = train_utils.create_checkpoint_manager(config, mesh, init_state_fn)
    state, _, _, _, _ = maxtext_utils.setup_training_state(None, config, mesh, ckpt_mgr, init_state_fn)
    real = nnx.merge(nnx.graphdef(nnx.eval_shape(init_state_fn)), state).model

    tokens = np.zeros((1, L), dtype=np.int32); tokens[0, : len(ids)] = ids
    seg = np.zeros((1, L), dtype=np.int32); seg[0, : len(ids)] = 1
    logits = real(decoder_input_tokens=jnp.array(tokens), decoder_positions=jnp.arange(L)[None, :],
                  decoder_segment_ids=jnp.array(seg), enable_dropout=False)
    logits = np.array(logits[0, : len(ids)], dtype=np.float32)
    os.makedirs(os.path.expanduser("~/hf-parity"), exist_ok=True)
    np.savez(os.path.expanduser("~/hf-parity/maxtext.npz"), ids=np.array(ids), logits=logits)
    print("SAVED", logits.shape, "argmax head:", logits.argmax(-1)[:12].tolist(), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
