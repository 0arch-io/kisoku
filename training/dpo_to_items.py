"""Rewrite a Tunix DPO checkpoint (<run>/checkpoints/<step>/model_params, an nnx state of TunixMaxTextAdapter: keys
('base', ..., 'value')) as the params-only layout the SFT checkpoints use (<step>/items: ('params', 'params', ...)), so
convert-chat.sh / export-chat-shm.sh can export it unchanged. Runs on CPU (JAX_PLATFORMS=cpu), safe while the TPU trains.
usage: JAX_PLATFORMS=cpu python dpo_to_items.py gs://$KISOKU_BUCKET/runs/<run>/checkpoints/<step>"""
import sys
import numpy as np
import orbax.checkpoint as ocp

step_dir = sys.argv[1].rstrip("/")
ckptr = ocp.PyTreeCheckpointer()
import jax
meta = ckptr.metadata(f"{step_dir}/model_params")
meta = getattr(meta, "item_metadata", None) or getattr(meta, "tree", None) or meta
# restore every leaf as a host numpy array (no device sharding needed on CPU)
state = ckptr.restore(f"{step_dir}/model_params",
                      restore_args=jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), meta))


def strip(node):
    """('base', ..., 'value') -> nested params dict without the adapter's 'base' and nnx 'value' wrappers; drops rngs."""
    if isinstance(node, dict):
        if set(node) == {"value"}:
            return np.asarray(node["value"])
        return {k: strip(v) for k, v in node.items() if k != "rngs" and k != "dropout"}
    return np.asarray(node)


params = strip(state["base"])
leaves = []
def walk(n, p=()):
    if isinstance(n, dict):
        for k, v in n.items(): walk(v, p + (k,))
    else:
        leaves.append((p, n.shape, n.dtype))
walk(params)
assert len(leaves) == 13, leaves  # 11 decoder tensors (scanned layers) + final norm + embedding
for p, s, d in leaves:
    print("/".join(p), s, d)
ckptr.save(f"{step_dir}/items", {"params": {"params": params}})
print("WROTE", f"{step_dir}/items")
