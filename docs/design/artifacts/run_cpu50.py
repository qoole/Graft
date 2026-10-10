import sys, json
sys.path.insert(0, "/root/eg2-bench")
import numpy as np
import embed_gpu as eg
real = sys.stdout
sys.stdout = sys.stderr
chunks = [json.loads(l) for l in open("/root/eg2-bench/head50.chunks.jsonl")]
prompts = [eg.doc_prompt(c["text"], title=c["name"]) for c in chunks]
compiled, tok = eg.load("CPU")
embs = [eg._run_batch(compiled, tok, prompts[i:i+32]) for i in range(0, len(prompts), 32)]
embs = np.concatenate(embs, axis=0)
print("shape", embs.shape)
np.save(real.buffer, embs.astype(np.float32))
real.flush()
