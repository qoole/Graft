import sys, time, json
sys.path.insert(0, "/root/eg2-bench")
import numpy as np
import embed_gpu as eg

chunks_path = sys.argv[1]
chunks = [json.loads(l) for l in open(chunks_path) if l.strip()]
# exact documented doc prompt: "title: {name} | text: {text}"
prompts = [eg.doc_prompt(c["text"], title=c["name"]) for c in chunks]
n = len(prompts)
bs = 32

real_stdout = sys.stdout
sys.stdout = sys.stderr  # keep the .npy stream clean; eg.load prints via print()

t_all = time.perf_counter()
compiled, tok = eg.load("GPU")
t0 = time.perf_counter()
_ = eg.embed(compiled, tok, prompts, bs)  # warm-up pass
t_warmup = time.perf_counter() - t0
t0 = time.perf_counter()
embs = eg.embed(compiled, tok, prompts, bs)  # measured warm pass
t_warm = time.perf_counter() - t0
t_total = time.perf_counter() - t_all

print(f"REPO {chunks_path} n={n} warmup_pass_s={t_warmup:.3f} warm_pass_s={t_warm:.3f} "
      f"warm_chunks_per_s={n/t_warm:.2f} total_process_s={t_total:.2f}")
norms = np.linalg.norm(embs, axis=1)
print(f"shape={embs.shape} dtype={embs.dtype} norms min={norms.min():.6f} max={norms.max():.6f}")
print(f"DONE {chunks_path}")

np.save(real_stdout.buffer, embs.astype(np.float32))
real_stdout.flush()
