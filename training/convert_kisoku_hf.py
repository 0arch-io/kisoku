"""Convert a Kisoku v2 MaxText checkpoint to Hugging Face (Qwen3ForCausalLM layout + Llama-3.2 tokenizer),
using maxtext.checkpoint_conversion.to_huggingface with Kisoku's config swapped in. Usage (worker 0, JAX on CPU):
  python convert_kisoku_hf.py ~/kisoku-v2-1b.yml model_name=qwen3-1.7b override_model_config=True \
    base_num_kv_heads=4 base_mlp_dim=8192 base_num_decoder_layers=22 vocab_size=128256 rope_max_timescale=5000000 \
    load_parameters_path=gs://.../items base_output_directory=/path/out scan_layers=True weight_dtype=float32 \
    ici_fsdp_parallelism=1 per_device_batch_size=1 skip_jax_distributed_system=True --hf_model_path=unsloth/Llama-3.2-1B
"""
import transformers
from absl import app

from maxtext.checkpoint_conversion import to_huggingface as th
from maxtext.checkpoint_conversion.utils import hf_model_configs, hf_shape, param_mapping
from maxtext.utils import globals as mt_globals

# MaxText only accepts model_name values from a fixed list, so Kisoku runs under the "qwen3-1.7b" key
# (same decoder block: QK-norm, GQA, SwiGLU, tied embeddings). Its tables are replaced in THIS process only,
# and the differing sizes are passed on the command line with override_model_config=True.
NAME = SRC = "qwen3-1.7b"

hf_shape.HF_SHAPE[NAME] = hf_shape.QWEN_HF_WEIGHTS_TO_SHAPE  # table has no qwen3-1.7b key; all Qwen3 dense use this
hf_model_configs.HF_MODEL_CONFIGS[NAME] = transformers.Qwen3Config(
    vocab_size=128256,
    hidden_size=2048,
    intermediate_size=8192,
    num_hidden_layers=22,
    num_attention_heads=16,
    num_key_value_heads=4,
    head_dim=128,
    hidden_act="silu",
    max_position_embeddings=4096,
    rms_norm_eps=1.0e-6,
    rope_theta=5000000.0,
    tie_word_embeddings=True,
    attention_bias=False,
    bos_token_id=128000,
    eos_token_id=128009,
    torch_dtype="bfloat16",
)
mt_globals.HF_IDS[NAME] = "unsloth/Llama-3.2-1B"

if __name__ == "__main__":
    app.run(th.main)
