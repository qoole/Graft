#!/usr/bin/env python3
"""CPU embedding harness for embeddinggemma-2 int8 ONNX.

Reads JSONL {"id","name","text"} on stdin, applies the document prompt
"title: {name} | text: {text}", runs the ONNX model, does masked mean pooling +
L2 normalize (MRL: truncate to first --dims of 768 then re-normalize), and
writes JSONL {"id","vector","dim"} to stdout.
"""
import argparse
import json
import sys
import time

import numpy as np

MODEL_DIR = "/build/eg2-bench/data/onnx-model"
DOC_PROMPT = "title: {name} | text: {text}"
MAX_LEN = 2048  # embeddinggemma context; model_max_length is unbounded in config


def pool_normalize(last_hidden, attention_mask, dims):
    mask = attention_mask[:, :, None].astype(np.float32)
    emb = (last_hidden * mask).sum(1) / np.clip(mask.sum(1), 1e-9, None)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    if dims < emb.shape[1]:
        emb = emb[:, :dims]  # MRL: truncate then re-normalize
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    return emb


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dims", type=int, default=768, choices=[768, 512, 256, 128])
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    import onnxruntime as ort

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    sess = ort.InferenceSession(f"{MODEL_DIR}/model_quantized.onnx",
                                providers=["CPUExecutionProvider"])

    def encode(records):
        prompts = [DOC_PROMPT.format(name=r["name"], text=r["text"]) for r in records]
        enc = tok(prompts, padding=True, truncation=True, max_length=MAX_LEN,
                  return_tensors="np", return_token_type_ids=False)
        feed = {"input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
                "image_features": np.zeros((0, 512), np.float32),
                "video_features": np.zeros((0, 512), np.float32),
                "audio_features": np.zeros((0, 512), np.float32)}
        last_hidden = sess.run(["last_hidden_state"], feed)[0]
        return pool_normalize(last_hidden, enc["attention_mask"], args.dims)

    total = 0
    t0 = time.perf_counter()
    records = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
        if len(records) >= args.batch:
            total += len(records)
            for r, e in zip(records, encode(records)):
                print(json.dumps({"id": r["id"], "vector": e.tolist(), "dim": args.dims}))
            records = []
    if records:
        total += len(records)
        for r, e in zip(records, encode(records)):
            print(json.dumps({"id": r["id"], "vector": e.tolist(), "dim": args.dims}))
    sys.stdout.flush()

    elapsed = time.perf_counter() - t0
    rate = total / elapsed if elapsed > 0 else 0.0
    print(f"RATE {rate:.2f} {total}", file=sys.stderr)


if __name__ == "__main__":
    main()
