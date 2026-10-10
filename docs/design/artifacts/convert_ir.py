"""R3: native OV IR from google/embeddinggemma-2 torch weights via ov.convert_model (FX).

optimum-cli is unusable here (optimum-intel 2.2.0 pins transformers<5.6, which
predates embedding_gemma2; under 5.19 its qwen2_vl import breaks). The full
config is multimodal whose __post_init__ needs torchvision (broken binary), so
load the text tower with a bare EmbeddingGemma2TextConfig and a remapped
state_dict (composite checkpoint prefix stripped).
"""
import json
import sys
import time

sys.path.insert(0, "/root/eg2-bench")

import torch
import openvino as ov
from safetensors.torch import load_file
from transformers.models.embedding_gemma2.configuration_embedding_gemma2 import (
    EmbeddingGemma2TextConfig,
)
from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import (
    EmbeddingGemma2TextModel,
)

SNAP = "/root/eg2-bench/hf/hub/models--google--embeddinggemma-2/snapshots/914f7f89142e33e77833254d9c9b90c3cef7303b"
OUT = "/root/eg2-bench/ov_ir/openvino_model.xml"

t0 = time.perf_counter()
sd = load_file(SNAP + "/model.safetensors")
print(f"checkpoint {len(sd)} tensors, {time.perf_counter() - t0:.1f}s")
sample = [k for k in sd if "embed_tokens" in k]
print("embed key sample:", sample[:3])
prefix = sample[0][: -len("embed_tokens.weight")]
print("detected prefix:", repr(prefix))

text_sd = {}
for k, v in sd.items():
    if not k.startswith(prefix):
        continue
    short = k[len(prefix):]
    if "vision" in short or "audio" in short or "multi_modal_projector" in short:
        continue
    text_sd[short] = v.to(torch.float32)
del sd
print(f"text tensors: {len(text_sd)}")

cfg = EmbeddingGemma2TextConfig(**json.load(open(SNAP + "/config.json"))["text_config"])
t0 = time.perf_counter()
model = EmbeddingGemma2TextModel(cfg)
model.load_state_dict(text_sd, strict=True)
model.eval()
print("loaded:", type(model).__name__, f"{time.perf_counter() - t0:.1f}s")

t0 = time.perf_counter()
S = 400  # fixed padded seq (corpus max = 385); batch dim made dynamic post-convert
ov_model = ov.convert_model(
    model,
    example_input=(
        torch.ones(1, S, dtype=torch.long),
        torch.ones(1, S, dtype=torch.long),
    ),
)
print(f"convert_model {time.perf_counter() - t0:.1f}s")
ov_model.reshape({"input_ids": [-1, S], "attention_mask": [-1, S]})
for inp in ov_model.inputs:
    print("input", sorted(inp.get_names()), str(inp.get_partial_shape()))
print("outputs:", [o.get_names() for o in ov_model.outputs])

t0 = time.perf_counter()
ov.save_model(ov_model, OUT, compress_to_fp16=True)
print(f"save_model fp16 {time.perf_counter() - t0:.1f}s -> {OUT}")
