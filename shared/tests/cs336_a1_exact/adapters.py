from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SHARED_ADAPTERS_PATH = Path(__file__).resolve().parents[1] / "adapters.py"
_spec = importlib.util.spec_from_file_location("_minillm_shared_test_adapters", _SHARED_ADAPTERS_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"could not load MiniLLM shared adapters from {_SHARED_ADAPTERS_PATH}")
_shared = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _shared
_spec.loader.exec_module(_shared)

run_train_bpe = _shared.run_train_bpe
get_tokenizer = _shared.get_tokenizer

run_get_batch = _shared.run_get_batch

run_linear = _shared.run_linear
run_embedding = _shared.run_embedding
run_rmsnorm = _shared.run_rmsnorm
run_swiglu = _shared.run_swiglu
run_silu = _shared.run_silu
run_softmax = _shared.run_softmax
run_cross_entropy = _shared.run_cross_entropy

run_rope = _shared.run_rope
run_scaled_dot_product_attention = _shared.run_scaled_dot_product_attention
run_multihead_self_attention = _shared.run_multihead_self_attention
run_multihead_self_attention_with_rope = _shared.run_multihead_self_attention_with_rope
run_transformer_block = _shared.run_transformer_block
run_transformer_lm = _shared.run_transformer_lm

get_adamw_cls = _shared.get_adamw_cls
run_get_lr_cosine_schedule = _shared.run_get_lr_cosine_schedule
run_gradient_clipping = _shared.run_gradient_clipping

run_save_checkpoint = _shared.run_save_checkpoint
run_load_checkpoint = _shared.run_load_checkpoint
