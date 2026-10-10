#!/usr/bin/env python3
"""Cross-box agreement (CPU int8 ONNX vs GPU OpenVINO Battlemage) + MRL storage.

1. Join 50 shared ids: cpu50.jsonl (embed_cpu.py, 768d) vs small_rs.npy rows
   0..49 (same file order as small_rs.chunks.jsonl). Cosine per pair, min/mean.
2. Storage: truncate GPU vectors to 768/512/256/128, renormalize, save f32,
   measure bytes/node.
"""
import json
import os

import numpy as np

D = "/build/eg2-bench/bench-gpu"

# --- 1. agreement ---
cpu = {}
with open(f"{D}/cpu50.jsonl") as f:
    for line in f:
        r = json.loads(line)
        cpu[r["id"]] = np.asarray(r["vector"], dtype=np.float32)

chunks = [json.loads(l) for l in open(f"{D}/small_rs.chunks.jsonl")]
gpu_all = np.load(f"{D}/small_rs.npy")
assert gpu_all.shape == (len(chunks), 768), gpu_all.shape

sims = []
for i, c in enumerate(chunks[:50]):
    cid = c["id"]
    g = gpu_all[i]
    v = cpu[cid]
    s = float(np.dot(g, v) / (np.linalg.norm(g) * np.linalg.norm(v)))
    sims.append(s)
sims = np.asarray(sims)
print(f"AGREEMENT n={len(sims)} min={sims.min():.6f} mean={sims.mean():.6f} "
      f"max={sims.max():.6f} valid_threshold_0.99={'PASS' if sims.min() > 0.99 else 'FAIL'}")

# --- 2. storage ---
for repo in ["small_rs", "small_ts"]:
    emb = np.load(f"{D}/{repo}.npy")
    n = emb.shape[0]
    print(f"STORAGE {repo} n={n}")
    for dim in [768, 512, 256, 128]:
        e = emb[:, :dim].copy()
        e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-12
        out = f"{D}/{repo}.{dim}d.npy"
        np.save(out, e.astype(np.float32))
        size = os.path.getsize(out)
        norms = np.linalg.norm(e, axis=1)
        print(f"  dim={dim} bytes={size} bytes_per_node={size/n:.1f} "
              f"norms=[{norms.min():.4f},{norms.max():.4f}]")
