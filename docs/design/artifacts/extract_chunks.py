#!/usr/bin/env python3
"""Extract FULL node sets to the chunk convention (mirrors gen_nodes_magnetico.py).

Read-only on the repos. {"id": node id, "name": node id,
"text": (signature or summary + "\n\n" + crux.code)[:6000]}
"""
import json

JOBS = [
    ("/home/arch/projects/magnetico-rs-v2/graft/.graph/wiring.json",
     "/build/eg2-bench/bench-gpu/small_rs.chunks.jsonl"),
    ("/home/arch/projects/ai-job/graft/.graph/wiring.json",
     "/build/eg2-bench/bench-gpu/small_ts.chunks.jsonl"),
]
CAP = 6000

for wiring, out in JOBS:
    with open(wiring, encoding="utf-8") as f:
        nodes = json.load(f)["nodes"]
    count = 0
    with open(out, "w", encoding="utf-8") as f:
        for n in nodes:
            head = n.get("signature") or n.get("summary") or ""
            body = (n.get("crux") or {}).get("code") or ""
            text = (head + "\n\n" + body)[:CAP]
            f.write(json.dumps({"id": n["id"], "name": n["id"], "text": text},
                               ensure_ascii=False) + "\n")
            count += 1
    print(f"{out}: {count} chunks")
