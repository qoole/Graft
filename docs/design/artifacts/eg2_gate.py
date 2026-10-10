"""Gate + throughput harness for EmbeddingGemma-2 routes.

Reuses embed_gpu.py helpers verbatim (prompts, tokenizer, masked-mean pooling,
L2 normalize); only the model file and device are parameterised.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/eg2-bench")
import embed_gpu as eg


def _run_batch_fixed(compiled, tok, texts, seq_len):
    # identical math to eg._run_batch, but every batch padded to a fixed length
    # (required by the static-seq IR); masked mean ignores pads, so semantics match
    enc = tok(
        texts, padding="max_length", truncation=True, max_length=seq_len,
        return_tensors="np",
    )
    feed = {
        n: enc[n]
        for inp in compiled.inputs
        for n in inp.get_names()
        if n in enc
    }
    res = compiled(feed)
    hidden = res[list(res)[0]]
    mask = enc["attention_mask"][..., None].astype(np.float32)
    summed = (hidden * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    emb = summed / counts
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def embed_fixed(compiled, tok, texts, batch_size, seq_len):
    embs = [
        _run_batch_fixed(compiled, tok, texts[i: i + batch_size], seq_len)
        for i in range(0, len(texts), batch_size)
    ]
    return np.concatenate(embs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--gate-chunks", default="/root/eg2-bench/head50.chunks.jsonl")
    ap.add_argument("--chunks", default="/root/eg2-bench/small_rs.chunks.jsonl")
    ap.add_argument("--ref", default="/root/eg2-bench/ort_fp16_50.npy")
    ap.add_argument("--bench", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default=None, help="save full-sample embeddings npy")
    ap.add_argument("--pad-to", type=int, default=None,
                    help="fixed padded seq len for static-seq IRs (e.g. 400)")
    args = ap.parse_args()

    eg.MODEL_DIR = Path(args.model_dir)
    eg.MODEL = args.model
    compiled, tok = eg.load(args.device)
    embed_fn = (
        (lambda texts, batch: embed_fixed(compiled, tok, texts, batch, args.pad_to))
        if args.pad_to
        else (lambda texts, batch: eg.embed(compiled, tok, texts, batch))
    )

    # accuracy gate: 50 chunks vs fp16 reference
    chunks = [json.loads(l) for l in open(args.gate_chunks)]
    texts = [eg.doc_prompt(c["text"], title=c["name"]) for c in chunks]
    emb = embed_fn(texts, args.batch)
    ref = np.load(args.ref)
    cos = (emb * ref).sum(1) / (
        np.linalg.norm(emb, axis=1) * np.linalg.norm(ref, axis=1)
    )
    gate_pass = bool(cos.min() >= 0.99)
    print(
        f"GATE n=50 min={cos.min():.6f} mean={cos.mean():.6f} max={cos.max():.6f} "
        f"-> {'PASS' if gate_pass else 'FAIL'}",
        flush=True,
    )

    # warm throughput on first N chunks
    all_chunks = [json.loads(l) for l in open(args.chunks)]
    n_bench = min(args.bench, len(all_chunks))
    btexts = [eg.doc_prompt(c["text"], title=c["name"]) for c in all_chunks[:n_bench]]
    embed_fn(btexts[: 2 * args.batch], args.batch)  # warmup
    t0 = time.perf_counter()
    embed_fn(btexts, args.batch)
    dt = time.perf_counter() - t0
    print(
        f"THROUGHPUT n={n_bench} batch={args.batch} warm {dt:.3f}s "
        f"-> {n_bench / dt:.1f} chunks/s",
        flush=True,
    )

    # optional full sample
    if args.out:
        ftexts = [eg.doc_prompt(c["text"], title=c["name"]) for c in all_chunks]
        t0 = time.perf_counter()
        full = embed_fn(ftexts, args.batch)
        full_dt = time.perf_counter() - t0
        np.save(args.out, full.astype(np.float32))
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(
            f"FULL n={len(all_chunks)} wall={full_dt:.3f}s shape={full.shape} "
            f"peak_rss_mb={rss / 1024:.0f}"
        )


if __name__ == "__main__":
    main()
