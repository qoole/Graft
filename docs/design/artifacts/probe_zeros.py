"""Test hypothesis: OV leaves optional placeholder inputs (image/video/audio_features) uninitialized."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/eg2-bench")
import embed_gpu as eg

eg.MODEL_DIR = Path("/root/eg2-bench/model_fp16")
eg.MODEL = "model_fp16.onnx"

chunks = [json.loads(l) for l in open("/root/eg2-bench/head50.chunks.jsonl")]
texts = [eg.doc_prompt(c["text"], title=c["name"]) for c in chunks]
ref = np.load("/root/eg2-bench/ort_fp16_50.npy")

compiled, tok = eg.load("GPU")
enc = tok(texts, padding=True, truncation=True, max_length=eg.MAX_LEN, return_tensors="np")
B, T = enc["input_ids"].shape

for inp in compiled.inputs:
    print("input", sorted(inp.get_names()), inp.get_element_type(), inp.get_partial_shape())

feed = {n: enc[n] for inp in compiled.inputs for n in inp.get_names() if n in enc}
res = compiled(feed)
mask = enc["attention_mask"][..., None].astype(np.float32)
pooled = (res["last_hidden_state"] * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)


def cosreport(emb, tag):
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    cos = (emb * ref).sum(1) / (np.linalg.norm(emb, axis=1) * np.linalg.norm(ref, axis=1))
    print(f"{tag}: min={cos.min():.6f} mean={cos.mean():.6f} max={cos.max():.6f}")


cosreport(pooled, "baseline (text-only feed)")

for name in ["image_features", "video_features", "audio_features"]:
    for inp in compiled.inputs:
        if name in inp.get_names():
            dt = {"f32": np.float32, "f16": np.float16, "f64": np.float64}.get(
                inp.get_element_type().get_type_name(), np.float32
            )
            zeros = np.zeros((B, inp.get_partial_shape().get_dimension(1).get_length()), dtype=dt)
            feed[name] = zeros

res2 = compiled(feed)
pooled2 = (res2["last_hidden_state"] * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
cosreport(pooled2, "with zero placeholders")
