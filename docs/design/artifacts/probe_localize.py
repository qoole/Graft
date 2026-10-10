"""Diagnose where OpenVINO diverges: device (CPU vs GPU) x output (pooled vs graph sentence_embedding)."""
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


def cosreport(emb, tag):
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    cos = (emb * ref).sum(1) / (
        np.linalg.norm(emb, axis=1) * np.linalg.norm(ref, axis=1)
    )
    print(f"{tag}: min={cos.min():.6f} mean={cos.mean():.6f} max={cos.max():.6f}")


enc_all = None
for device in ["CPU", "GPU"]:
    compiled, tok = eg.load(device)
    if enc_all is None:
        enc_all = tok(
            texts, padding=True, truncation=True, max_length=eg.MAX_LEN, return_tensors="np"
        )
    feed = {
        n: enc_all[n]
        for inp in compiled.inputs
        for n in inp.get_names()
        if n in enc_all
    }
    res = compiled(feed)
    hidden = res["last_hidden_state"]
    mask = enc_all["attention_mask"][..., None].astype(np.float32)
    pooled = (hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
    cosreport(pooled, f"{device} last_hidden_state masked-mean")
    sent = np.asarray(res["sentence_embedding"])
    cosreport(sent, f"{device} sentence_embedding(graph)")
