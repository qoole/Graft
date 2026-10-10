# Design: EmbeddingGemma 2 vector indexing in graft

Status: final proposal with measured data. Every number in this document traces
to a command run on 2026-10-10; the benchmark tree lives in `/build/eg2-bench/`
(artifact map in Appendix A). Graft-internals citations are against `graft_src`
at `@nanonets/graft` 0.9.0 and were spot-checked while writing.

## 0. TL;DR

Add a **node-embedding sidecar** (`graft/.cache/embeddings.json` + packed
vectors) built with EmbeddingGemma 2 (`google/embeddinggemma-2`), following
`ask-index.json`'s derived-cache precedent and reusing Tier-2's `body_hash`
carry-over rule as the re-embed trigger. Embeddings become a third ranking
channel fused into `ask`'s existing blend (RRF), so paraphrase/NL queries get a
semantic match that lexical scoring alone cannot provide — and LLM enrichment
stops being a prerequisite for good retrieval: it becomes a lazy query-time
step over vector-selected top-k instead of an eager index-time pass.

Measured headline: encode cost is ~2-2.5 chunks/s CPU (contended; ~3.6
uncontended) vs **201-221 chunks/s on a Battlemage B580 via the validated
route** (OpenVINO fp16 IR converted directly from torch weights, agreement
min cosine 0.999996 — §5.4); storage is 3072 B/node at 768d (144 MiB for the
largest 147.8k-node graph measured); node-level chunks beat naive 512-char
windows by ~17 points of recall@10, and fp16-exact vectors beat the int8
pipeline further (§5.4.2). The one cross-box quality failure — OpenVINO's
ONNX importer computing a wrong embedding space at *every* precision — was
root-caused and bypassed, not by swapping devices but by skipping the ONNX
importer entirely (§5.4).

## 1. Problem: what indexing and retrieval cost today

### 1.1 Index-time LLM enrichment

Tier-2 enrichment (`enrichGraph`, `src/graph/enrich.ts:48`) makes one LLM call
per **file** (`DEFAULT_CONCURRENCY = 5`, enrich.ts:29; missing symbols re-asked
once, `collectFileCrux`, enrich.ts:161) and is keyed on `body_hash` for
carry-over (enrich.ts:61-66). Any body change flips the node to `stale` and
forces a fresh LLM call on the next `build --deep`. The unit of cost is the
file, and files scale with repository size.

Measured across four real graft graphs (`/build/eg2-bench/census.py`):

| graph | repo | nodes | distinct files | wiring.json | enrichment state |
|---|---|---|---|---|---|
| small_ts | ai-job | 3,090 | 192 | 3.8 MB | 3,090 ready / 0 pending / 0 stale |
| small_rs | magnetico-rs-v2 | 2,814 | 121 | 2.8 MB | 2,814 ready / 0 / 0 |
| medium_ts | refinery | 14,141 | 950 | 18.7 MB | 14,141 ready / 0 / 0 |
| huge_py | home-assistant-core | 147,802 | 17,829 | 233 MB | 147,685 ready / 117 pending (0.08%) / 0 stale |
| total | | **167,847** | 19,092 | | |

A cold `--deep` build of the large graph is ~17.8k file-level LLM calls before
`ask` has its best data. Node counts are exact (`len(nodes)` == `meta.nodeCount`
in all four); 0 missing source files, 0 malformed spans.

### 1.2 The retrieval gap the cost does not fill

`ask` ranks lexically — per-node token bags over `name`/`path`/`body`
(`lexical`, ask.ts:400; weights `name*3 + path*2 + bm25(body)`, ask.ts:647)
blended with personalized PageRank (`GRAPH_WEIGHT = 0.5`, ask.ts:343) and
per-scope RRF (`RRF_K = 60`, fuse.ts:50). Lexical scoring matches word
overlap; it has no semantic channel. The only semantic data graft keeps —
LLM `summary`/`crux` — is exactly the expensive index-time artifact from
§1.1, and `body_text` is stripped from wiring.json at serialization
(`stripBodyText`, write.ts:57), so a fresh consumer cannot reconstruct
meaning from the committed graph alone.

A baseline run of graft's own `ask` on 12 hand-written golden queries about
graft_src (`/build/eg2-bench/data/graft-ask-baseline.json`, query via
`node dist/cli.js ask <q> <copy> --json --no-refresh --limit 50`) put the gold
node at rank 1 for 8/12, rank ≤3 for 11/12, rank ≤10 for 12/12 — good on
exact-ish phrasing, but these queries were written against the docs; the open
question is paraphrase robustness, which the same run cannot measure and a
vector channel can.

### 1.3 What a vector channel costs to build (measured chunk sizes)

Chunk text = definition span sliced from source (signature+body, no double
count; the census script uses a containment test after finding its original
prefix-only check would have re-added signature bytes on 3 of 4 spot-checks).
Per-graph byte statistics (768d workload math uses bytes/3.5 capped at 1,500
tokens/chunk — a divisor estimate, not a tokenizer count):

| graph | chunk bytes mean / median / p95 / max | workload tokens (est.) |
|---|---|---|
| small_ts | 2,433 / 281 / 7,621 / 484,757 | 726,024 |
| small_rs | 1,514 / 310 / 5,585 / 142,210 | 664,292 |
| medium_ts | 1,880 / 430 / 9,763 / 131,186 | 4,231,601 |
| huge_py | 1,556 / 662 / 5,228 / 362,299 | 47,687,765 |
| total | | **53,309,682** (huge_py = 89.4%) |

Only medium_ts shows p95 cap pressure (p95 raw ~2,789 tokens; 1,034 chunks over
the 1,500 cap; 201/148/1,034/7,340 over-cap chunks overall). The single
484,757-byte span in small_ts is one outlier (~20% of that corpus's chunk
bytes) and skews any mean-sensitive decision.

## 2. Proposal

### 2.1 Sidecar beside the wiring graph

Follow the `ask-index.json` precedent (`AskIndex`, `src/ask/index-file.ts:98`;
`.cache/` is `CACHE_DIR`, `context/node-file.ts:81`, gitignored, derived,
uncommitted — `.graph/` stays committed and untouched):

- `graft/.cache/embeddings.json`:
  `{version: 1, model: "google/embeddinggemma-2", dim: 768, precision:
  "int8"|"fp16"|"fp32", docs: [{id, body_hash, offset}...]}` — every entry
  records the node's `body_hash` at embed time.
- packed `graft/.cache/embeddings.bin`, little-endian float32 (fp16/int8
  variants later), `offset = index × dim`. JSON meta + binary payload keeps
  the meta reviewable and the payload compact.

### 2.2 Carry-over rule: Tier-2's, plus a model tag

A node re-embeds **only** if no prior entry exists with the same `id`, same
`body_hash`, and same `model`+`dim`+`precision`. This is the enrich.ts:61-66
condition verbatim plus the model tag; a model or dim change is mass-stale by
construction, like a schema change. Stale entries keep the old vector, marked
stale, usable but de-weighted — the same policy summaries have.

### 2.3 LLM enrichment becomes lazy

Today, semantic retrieval assumes Tier-2 ran. With vectors present, the flow
inverts: index time is `$0`-plus-encode (static, no LLM), and the LLM is spent
at query time only on the top-k the vector channel selects (re-summarize
stale nodes on demand, or justify/expanda candidate set). Tier-2 remains
available and unchanged; it simply stops being on the critical path for
retrieval quality.

### 2.4 Fusion into `ask` (per draft §3.4)

Inside `lexical()` after per-node lexical scores, before the graph blend:

```
vecN   = cosine(qvec, nodeVec), normalized to [0,1]-ish
blended = (lexN + GRAPH_WEIGHT·pr + VEC_WEIGHT·vecN) * testFactor(path)
```

`VEC_WEIGHT ≈ 0.5-1.0`, tuned on the golden set. Vector-only hits enter the
candidate set the way walk-rescued nodes already do (`RESCUE_FLOOR = 0.15`,
ask.ts:349). Because the fusion stays inside the existing per-node blend,
multi-scope federation (fuse.ts:208), `--in` filtering, and test de-ranking
(testFactor, ask.ts:160) all keep working unchanged; a vector score is
per-node and does not disturb per-scope IDF pools. An alternative worth A/B
in implementation: RRF-fuse the vector ranking into the scope ranking with
the same `RRF_K = 60` machinery the multi-scope path already uses.

### 2.5 CLI / MCP surface

- `graft embed [dir] [--model …] [--dim 768|512|256|128]` — builds/refreshes
  the sidecar; without an encoder configured it only does cache/stale
  bookkeeping (mirrors enrich's no-summarizer mode). Register like the other
  subcommands in `src/cli.ts` and call a new `Graft.embed()` in
  `src/engine.ts`.
- `build --deep` (cli.ts:113-160) chains embed after `writeAskIndex`
  (build.ts:337) when vectors are enabled.
- `ask` loads the sidecar next to `ask-index.json`; **absent sidecar =
  today's behavior byte-for-byte** (the null-on-anything-off contract of
  `readAskIndex`, index-file.ts:138). MCP `graft_find_code` → `Graft.ask()`
  (engine.ts:98) benefits transparently; `formatAsk` (mcp/tools.ts:263-273)
  needs no change.

## 3. Integration points (all existing code, file:line)

| touch point | location | note |
|---|---|---|
| `NodeV1.body_hash` | src/graph/types.ts:55 | re-embed trigger, same sha256 of definition text |
| node fields to embed | types.ts (`id,name,kind,path,span,signature,summary_state,summary,crux`) | chunk text inputs; `body_text` stripped at write (write.ts:57) |
| carry-over precedent | enrich.ts:61-66 | condition reused verbatim + model tag |
| sidecar precedent | ask/index-file.ts:98 (`AskIndex`), :138 (null contract) | same directory, same lifecycle |
| fusion point | ask.ts:400 `lexical()`; weights ask.ts:647/572; blend ask.ts:343; rescue ask.ts:349; test factor ask.ts:160 | add `VEC_WEIGHT·vecN` term |
| build orchestration | build.ts:181 (`buildGraph`), :324 `writeGraph`, :337 `writeAskIndex` | embed hook after 337 |
| CLI registration | cli.ts (command table; `--deep` 113-160; `NO_REFRESH_FLAG` 93) | new `embed` subcommand |
| MCP | engine.ts:98 → mcp/tools.ts:263-273 | unchanged |
| `.cache/` location | context/node-file.ts:81 | sidecar home |

## 4. The model (verified from HF, not memory)

`google/embeddinggemma-2` — fetched config.json + model card 2026-10-10:

- 740M total: 270M text backbone (130M transformer + 140M embedder) + 170M
  vision + 300M audio; **text-only load = 270M**.
- unified 768-d space, mean pooling, projection 512→768, 8,192-token context,
  vocab 262,144, Apache-2.0. `model_type: "embedding_gemma2"`,
  `EmbeddingGemma2Model`.
- MRL truncation to 512/256/128 with re-normalization; queries and corpus
  must share dimension.
- CodeRetrieval prompts (card): query `task: code retrieval | query: {q}`,
  doc `title: {path} | text: {code}`. (The bench below used the simpler
  `title: {name} | text: {text}` template on both sides; absolute scores are
  likely a few points under a properly prompted setup.)
- float16 is forbidden per card (NaN/degraded); bf16 or fp32 for reference.
- Ships as 1,488.9 MB bf16 safetensors; **no `onnx/` in the google repo
  (404)**. Use `onnx-community/embeddinggemma-2-ONNX` (76.5k downloads):
  `model_q4f16.onnx` 156.9 MB, `model_q4` 174.0 MB, `model_quantized` (int8)
  313.7 MB, fp16 542.1 MB, fp32 `model.onnx_data` 1,084.2 MB.
- Runtime: `@huggingface/transformers` 4.3.1 registers
  `embedding_gemma2 → EmbeddingGemma2Model` with the ONNX repo as its own
  docstring example and ships `onnxruntime-node 1.30.0` (native linux-x64
  napi binding present in the tarball). Python path also works: onnxruntime
  1.31.0 has cp314 wheels. The bench used a CPython 3.12 venv with
  onnxruntime 1.31.0 (CPUExecutionProvider) + transformers 5.19.0.

## 5. Measured results

Commands are shown with each subsection; raw artifacts in Appendix A. All CPU
bench numbers come from a 10-core i7-12700K, 19 Gi RAM, CPU-only.

### 5.1 Runtime path (recon; no inference)

Verified by direct API/tarball inspection: google repo ships safetensors
only; ONNX weights are community-converted (`onnx-community`, 76,556
downloads) — **embedding parity vs the safetensors baseline was not verified
on-box** (the 50-prompt fp16-reference check in §5.4 anchors runtime
self-consistency, not upstream conversion fidelity).

### 5.2 Encode throughput — CPU

`venv/bin/python -I bench-tp/run_timed.py <repo> --dims 768 --batch 32 <
<repo>.chunks.jsonl` (batch-32 sweep on the smoke corpus: b16→4.00, b32→3.56,
b64→3.07 chunks/s — int8 is not pathological):

| corpus | chunks encoded | rate (chunks/s) | wall (s) | peak RSS (MiB) |
|---|---|---|---|---|
| small_ts (ai-job) | 1,500 | 1.69 | 896.00 | 9,186.4 |
| small_rs (magnetico) | 750 | 2.49 | 306.96 | 4,439.9 |
| medium_ts (refinery) | 750 | 2.08 | 365.52 | 4,175.5 |
| huge_py (home-assistant) | 750 | 2.34 | 326.27 | 3,282.2 |

Rates were measured under contention (load 13.15: a second embed process and
a `graft build --deep` shared the box); the earlier uncontended smoke on the
same corpus/model measured 3.56 chunks/s, so treat CPU rates as 2-2.5 under
load, ~3.5 clean. Full-corpus encode times below are **labeled
extrapolations** from these rates and exact node counts (conservative upper
bounds, overstating by roughly 1.4-2.1x):

| graph | full encode (EXTRAPOLATED) | storage @768d f32 (arithmetic) | storage @256d |
|---|---|---|---|
| small_ts | 1,828 s = 30.5 min | 9.0 MiB | 3.02 MiB |
| small_rs | 1,130 s = 18.8 min | 8.2 MiB | 2.75 MiB |
| medium_ts | 6,799 s = 113.3 min | 41.2 MiB | 13.81 MiB |
| huge_py | 63,163 s = 17.5 h | 424.4 MiB | 144.34 MiB |

Storage is flat 3,072 B/node at 768d f32 (verified on disk:
`ai-job.f32.bin` = 4,608,000 B = 1,500 × 3,072; the three 750-chunk bins =
2,304,000 B each; 1,024 B/node at 256d). Truncate+renormalize gives norm 1.0
at every MRL dim. Peak RSS is batch-length-dependent (ONNX arena grows with
the longest sequence in batch), not corpus-size-dependent — ai-job's 9.2 GiB
outlier came from a 2,314-char max chunk plus concurrent load.

### 5.3 Retrieval quality — node chunks, full graphs

Full small_rs (2,814) and small_ts (3,090) graphs embedded; seed-42 oracle of
40 defined symbols per repo × 3 query forms (bare name; "where is X defined";
NL sentence derived from the summary), recall@10 over node ids, n=40 per
cell (`bench-quality/eval.py`, results in `eval_results.json`):

| recall@10 | rs bare | rs where | rs nl | ts bare | ts where | ts nl |
|---|---|---|---|---|---|---|
| 768d | **0.925** | 0.80 | 0.825 | 0.65 | 0.60 | 0.80 |
| 256d | 0.90 | 0.825 | 0.825 | 0.60 | 0.60 | 0.70 |
| 128d | 0.90 | 0.75 | 0.70 | 0.475 | 0.50 | 0.60 |

Baselines on the same oracle symbols:

- **grep file-rank wins name-form queries**: word-count file ranking puts the
  ground-truth file in its top-10 for 38/40 (rs, 0.950) and 39/40 (ts, 0.975);
  its only misses are generic short names (encode, store, temp).
- **Node chunks beat naive windows**: 512-char/64-overlap windows, scored
  file-level on the same rs bare queries: 0.75 @768d (0.80 @256, 0.675 @128)
  vs 0.925 node-level — node-aware chunking is worth ~17 points.

**Miss analysis** (56/240 queries miss top-10 at 768d; the 10 worst, GT rank
982-1879): the dominant cause is the **chunk-text convention**, not the model.
The harness embedded `text = (signature or summary) + "\n\n" + crux.code`,
which drops the wiring summary whenever a signature exists — 1,508 rs + 1,540
ts nodes (plus 108/131 summary-only; 1,616/1,671 head-only total, recount
verified) are content-free chunks, and 3 of the 10 worst misses are exactly
such ~19-24-char chunks (e.g. `interface Fixture`). Second recurring pattern:
a symbol's own return-type interface (Declaration, Config, Tally, CiSnapshot)
outranks the function for its own name query — 4 of 10 worst. Third: exact
name absent from every other chunk (`decode_query`), so decoder-titled
siblings own the neighborhood. Also note query circularity: for summary-only
chunks the NL query text equals the doc text, so the `nl` column overstates
true NL retrieval; the honest split at 768d rs bare is 10/13 crux-text nodes
vs 27/27 summary-only in top-10.

Implication: a `head = signature + summary (when ready)` rebuild would
address the largest measured quality loss before any model tuning.

### 5.4 CPU vs GPU — the comparison that failed

GPU measured on vbattlemage (Battlemage B580, OpenVINO 2026.4.1) with
byte-identical chunk files (full rs + ts sets re-extracted and diffed against
the census extraction):

- Warm throughput (batch 32, second in-process pass): small_rs full 150.74
  chunks/s, small_ts full 106.31, samples 111-157 across all four repos.
  Full-graph projections at measured rates: refinery ~101 s,
  home-assistant-core ~17.0 min (vs 17.5 h CPU upper bound). An earlier
  standalone smoke measured 180-213 chunks/s warm on the same path.
- **Cross-box agreement FAILED** (threshold 0.99, 50 shared chunk ids).
  Recomputed for this document from the stored vectors
  (`bench-gpu/{cpu50.jsonl, ovcpu50.npy, ort_fp16_50.npy}`):

  ```
  fp16ref vs ORT-int8:      n=50 min=0.999853 mean=0.999930   # faithful
  fp16ref vs OpenVINO-int8: n=50 min=0.719348 mean=0.835411   # off-manifold
  OpenVINO-int8 vs ORT:     n=50 min=0.719369 mean=0.835189
  ```

  Root cause pinned: model file sha256 and token ids are byte-identical
  across boxes, so it is not prompts or weights — **OpenVINO mis-executes
  `model_quantized.onnx`**, on both its CPU and GPU devices (which agree with
  each other at 0.9999). Against an fp16 reference, onnxruntime int8 sits at
  0.9999 while OpenVINO sits at 0.835.

Consequences: (a) the GPU rows above are **throughput numbers only** — the
GPU box's vectors are not in the faithful embedding space, so every quality
claim in this document is CPU/onnxruntime-only; (b) any future GPU path must
swap runtimes (onnxruntime with a GPU EP, or OpenVINO-native requantization)
and be validated against the fp16 reference before use; (c) trap for future
checks: OpenVINO-CPU and OpenVINO-GPU agreeing with each other proves nothing
— anchor agreement to a reference runtime, not to a second device of the
same runtime.

### 5.4.1 Resolution — the validated GPU route (same day, 2026-10-10)

A route ladder on the same B580 box (gate: 50-chunk cosine vs
`ort_fp16_50.npy`, min ≥ 0.99) found and validated a passing path:

| Route | Agreement vs fp16 ref | Warm chunks/s | Verdict |
|---|---|---|---|
| R1: OV GPU + onnx-community `model_fp16.onnx` | min 0.7179 / mean 0.8353 | 65.4 | FAIL |
| R2: OV GPU + `model.onnx` (fp32) | min 0.7180 / mean 0.8353 | 200.8 | FAIL |
| **R3: OV GPU + native IR (fp16) via `ov.convert_model` from torch weights** | **min 0.999997 / mean 0.999999 / max 1.000000** | **201-221** | **PASS** |

The R1/R2 pattern redefines the root cause: OpenVINO's **ONNX importer**
computes the wrong space for this architecture at *every* precision
(int8/fp16/fp32) and on *both* devices — the earlier "int8 defect" reading
was wrong. Explicit zero-fill of the model's optional
image/video/audio_features inputs changes nothing (still 0.835), and the
secondary `sentence_embedding` output is NaN on GPU while the primary output
is merely off-manifold. Skipping the ONNX importer — `ov.convert_model` on
the torch module (413/413 tensors strict-loaded after stripping the
`language_model.` checkpoint prefix), saved fp16, static seq_len 400 with
dynamic batch — is exact.

R3 winner numbers: gate min 0.999997 / mean 0.999999 (n=50); full
2,814-chunk sample in 12.7 s = 220.9 chunks/s warm (compile 1.2-1.8 s, peak
RSS 1126 MB, IR 551 MB fp16 on disk). Independently re-gated on the CPU box
from the returned vectors (`bench-gpu/win_small_rs.npy`, 8,644,736 B =
2814×768 f32 + header): 50-chunk gate min 0.999997; plus a fresh 100-chunk
seed-42 subset three-way check (fp16 ONNX reference computed locally on
onnxruntime-CPU):

```
ORT-int8 (the §5.3 corpus vectors) vs fp16ref: min=0.906935 mean=0.964298
GPU-IR   (validated route)      vs fp16ref: min=0.999996 mean=0.999999
```

**Second finding — int8 quantization drift is real and material.** The
onnxruntime-int8 pipeline itself drifts from the fp16 reference (mean 0.964,
min 0.907 on 100 random real chunks) — far below its own 50-chunk head
sample (0.9999). §5.3's recall numbers were measured on int8 vectors and
therefore carry that noise; §5.4.2 re-scores with exact vectors.

Reproduction scripts: `bench-gpu/{convert_ir.py, eg2_gate.py}` (identical
copies on both boxes); one-time conversion cost 8.2 s.

Caveats: the IR is static seq_len=400 (corpus max 385 tokens, mean 75 —
padded batches cost ~5× mean content length in throughput; masked mean makes
the *embeddings* padding-insensitive). `optimum-cli` is unusable for this
model today (transformers 5.x architecture unknown to its pins) — use the
`ov.convert_model` API path. The dynamic-shape conversion was blocked by
`convert_model` having no dynamic-shapes API and `torch.jit.trace` baking
`arange`.

### 5.4.2 Re-scored retrieval quality — fp16-exact vectors (small_rs)

Same oracle (40 symbols × 3 query forms), same MRL handling, docs swapped to
the validated GPU-IR fp16 vectors and queries re-embedded fp16 with the same
doc-prompt convention (isolating precision; small_ts not re-run — its
corpus vectors are still int8):

| form@768 | int8 recall@1/@5/@10 | fp16 recall@1/@5/@10 |
|---|---|---|
| q_bare | 0.550 / 0.875 / 0.925 | **0.675 / 0.900 / 0.925** |
| q_where | 0.550 / 0.775 / 0.800 | **0.700 / 0.850 / 0.900** |
| q_nl | 0.450 / 0.725 / 0.825 | **0.725 / 0.925 / 0.975** |

At 256d fp16 matches or beats int8 at every form (q_nl@1 0.675); at 128d
fp16 lands at or slightly below int8@128 (q_bare 0.600 vs 0.650) — 128d
stays unrecommended (§6). Median ground-truth rank is 1 in every fp16@768
column. Precision is a real quality lever: the exact-space embeddings lift
q_nl recall@1 by +27.5 points over the int8 pipeline.

### 5.5 What was not measured

- End-to-end fused ranking (graft ask + vector term) was never run; §5.3
  measures pure vector retrieval, and the 12-query graft baseline in §1.2 is
  baseline-only. The draft's earlier "vectors beat graft's rank" TL;DR claim
  is withdrawn as untested.
- int8-vs-fp32 full-eval quality delta at 768d: **partially resolved** —
  §5.4.2 measured it on small_rs (fp16 wins q_nl recall@1 by +27.5 points);
  small_ts and the other repos still unmeasured on exact vectors.
- Upstream ONNX conversion fidelity vs google safetensors.
- The asymmetric `task: code retrieval | query:` prompt template (harness
  applied its doc prompt to queries; consistent, likely a few points low).

## 6. MRL dimension recommendation

From §5.3 per-dimension data:

- **768d is the default and the recommendation.** It is the best score in
  every column, and storage is trivial at graft's scale — even the
  147.8k-node graph costs 424 MiB f32 (145 MiB fp16) for the entire sidecar,
  less than its 233 MB wiring.json by single-digit multiples.
- **256d is the only acceptable storage optimization, and only for
  rs-like corpora.** rs is flat-to-better at 256d (bare 0.925→0.90, where
  0.80→0.825, nl 0.825→0.825); ts pays (bare 0.65→0.60, nl 0.80→0.70). It is
  an opt-in `--dim 256`, not a default.
- **Do not ship 128d.** It costs rs its NL queries (0.825→0.70) and wrecks ts
  (bare 0.475). The measured quality-per-dim curve says the MRL sweet spot
  for this workload is 768, with 256 as a documented trade.
- Store one dimension in the sidecar (the file's `dim` tag); query-side
  truncation+renormalization to the stored dim is exact and free (norm 1.0
  verified at every dim).

## 7. Failure modes

1. **Approximate recall vs graft's exact-enumeration guarantee.** Wiring
   edges are exact: `callers`/`trace` enumerate the true graph. Vector
   retrieval is approximate — measured recall@10 tops out at 0.925 (rs, int8)
   / 0.975 (rs, fp16-exact §5.4.2) / 0.65 (ts, int8-only), and grep still
   beats it on name-form file localization
   (0.95/0.975). Embeddings therefore **complement, never replace**: they add
   a semantic candidate channel to `ask`'s blend and never gate exact edge
   queries, exact `grep`, or the deterministic checks. A missing vector entry
   must degrade to lexical-only, never to "no answer".
2. **Staleness.** Covered by §2.2: `body_hash` carry-over re-embeds only
   changed nodes (enrich.ts:61-66 rule + model tag); stale vectors are kept
   but de-weighted; model/dim/precision change forces a full re-embed.
   Unresolved: whether de-weighted stale vectors measurably help or hurt —
   open question.
3. **Chunking sensitivity.** The largest measured quality loss is chunk-text
   construction, not the encoder: 1,616/1,671 head-only chunks from the
   signature-or-summary convention (§5.3), with 3 of 10 worst misses being
   ~20-char chunks. Any production embed step should use
   `signature + summary (when ready) + crux.code`, and the quality bench
   re-run before tuning `VEC_WEIGHT`. Also: one vendored/generated 485 KB
   span exists in the wild — cap chunk text (the 1,500-token cap caught 148
   -7,340 chunks per repo) and consider splitting, not truncating, for
   medium_ts's 1,034 over-cap chunks.
4. **Vocabulary capture by umbrella interfaces.** Return-type interfaces
   (Declaration, Config, Tally, CiSnapshot) outrank member functions in 4 of
   the 10 worst misses. Candidate mitigations (unmeasured): kind-aware score
   weight, or embedding `name+kind` in the doc title.
5. **Runtime substitution risk.** §5.4/§5.4.1: a fast runtime that computes
   the wrong space is worse than a slow correct one — OpenVINO's ONNX
   importer did exactly that at every precision and device. Any runtime/EP
   change gets a 50-prompt fp16-reference gate (min cosine ≥ 0.99) in CI
   before merge; the validated recipe is `ov.convert_model` from torch
   weights (never the ONNX importer), and the gate anchors to an
   onnxruntime-computed fp16 reference, never to a second device of the
   same runtime.
6. **Estimated tokens, not counted tokens.** All §1.3 token figures use the
   3.5 bytes/token divisor; real tokenizer counts will differ by model.
7. **Multimodal dead weight.** EG2 is vision+audio capable; graft is
   text-only. Ship the text-only ONNX graph (157 MB q4f16 / 314 MB int8) and
   gate anything else behind an explicit flag.

## 8. Rollout

Standalone sidecar first; no committed-data changes at any step.

1. **`graft embed`**, sidecar-only: writes `.cache/embeddings.json` +
   `.bin`; `ask` ignores a missing sidecar (byte-for-byte today's output).
   Cache accounting mirrors enrich's: report `embedded / carried / stale`.
2. **Chunk-text fix**: `signature + summary + crux.code` head/body split;
   re-run the §5.3 harness (`bench-quality/`) as the acceptance gate —
   expect ts bare/where to move most, since its head-only chunk share
   (1,671/3,090) is highest.
3. **Fusion behind a flag** (`ask --vectors` / config): `VEC_WEIGHT·vecN`
   term per §2.4, tuned on the golden set; A/B the RRF variant (§2.4)
   against the linear blend.
4. **`build --deep` chain** after `writeAskIndex` once (1)-(3) hold.
5. **GPU now validated, adopt alongside step 1** for graphs where CPU is
   the bottleneck: the §5.4.1 route (OpenVINO fp16 IR via
   `ov.convert_model`, B580, 201-221 chunks/s, gate-passed) removes the
   17.5 h CPU upper bound on huge_py-scale graphs to ~12 min, and fp16
   embeddings are *more* accurate than the int8 CPU path (§5.4.2). The CI
   gate (§7.5) stays: every runtime/EP change re-proves the 50-prompt
   fp16-reference check. CPU (onnxruntime int8 or fp16) remains the
   no-GPU fallback.

## 9. Open questions

1. `VEC_WEIGHT` and RRF-vs-linear: unmeasured until the fused ranking exists
   (§5.5). Needs the golden-set harness ported into graft's test suite
   (`test/ask.test.ts` pattern).
2. Asymmetric query prompt (`task: code retrieval | query:`) — expected few
   points up; measure after the chunk fix.
3. Stale-vector policy: de-weight vs drop.
4. **Resolved (§5.4.1):** GPU path validated — OpenVINO fp16 IR converted
   directly from torch weights via `ov.convert_model` (NOT the ONNX
   importer), B580, min cosine 0.999997 vs the fp16 reference, 201-221
   chunks/s. Remaining GPU questions: dynamic-seq IR (static 400 costs
   throughput on short chunks) and whether onnxruntime grows a viable Intel
   GPU EP.
5. int8 vs fp16 eval delta on the remaining repos (small_rs done, §5.4.2;
   q_nl recall@1 +27.5 points there).
6. Upstream ONNX conversion fidelity vs google safetensors.
7. Over-cap chunks (medium_ts 1,034): truncate vs split — interacts with
   `body_hash` (a policy change is a sidecar-version change, not a graph
   change).
8. First-run model acquisition: ~314 MB download + Gemma license
   acknowledgment flow (`graft embed --pull`), and whether the model is
   fetched per-user or bundled with a hash check.
9. Eval breadth: cross-repo tasks, issue-body queries, and a non-circular NL
   form (§5.3 notes the nl column's construction bias).

## Appendix A: artifact map

| path | contents |
|---|---|
| `/build/eg2-bench/census.py`, `census_results.json` | corpus census (run `HF_HOME=/build/eg2-bench/hf python3 -I census.py`) |
| `/build/eg2-bench/embed_cpu.py` | int8 ONNX CPU harness (masked mean pool + L2, MRL truncate+renorm; pools match the model's own `sentence_embedding` head at cosine 1.000000) |
| `/build/eg2-bench/bench-tp/` | throughput chunks/vecs, `run_timed.py` (ru_maxrss wrapper; `/usr/bin/time` absent on this box) |
| `/build/eg2-bench/indexes/<repo>/` | f32 bins + ids (sizes re-verified against N×dim×4) |
| `/build/eg2-bench/bench-quality/` | `gen_nodes.py`, `oracle.py`, `eval.py`, `eval_results.json`, `eval_detail.json`, grep/window baselines, full rs+ts vectors |
| `/build/eg2-bench/bench-gpu/` | GPU chunks/npys, `analyze.py`, fp16-reference vectors, agreement artifacts |
| `bench-gpu/win_small_rs.npy` + `{convert_ir.py, eg2_gate.py}` | §5.4.1 validated-route artifacts (2814×768 f32, conversion + gate scripts, copies on both boxes) |
| `/build/eg2-bench/data/graft-ask-baseline.json` | 12-query graft `ask` baseline (§1.2) |
| `/build/eg2-bench/venv/` | CPython 3.12.13; onnxruntime 1.31.0 CPUExecutionProvider, transformers 5.19.0 |

Verification commands run while writing this document (all numbers above
re-checked against artifacts): census totals from `census_results.json`
(4 graphs, `node_count_matches_meta: true`, workload 53,309,682);
`eval_results.json` / `grep_baseline.json` / `window_baseline.json` reads;
`wc -l`/`ls -la` on `indexes/` (1,500/750/750/750 ids; 4,608,000 and
2,304,000-byte bins); the three-way cosine agreement computation over
`bench-gpu/*.npy` reproduced §5.4 exactly; head-only chunk recount
(1,616/1,671) reconciled with the per-class counts (1,508+108, 1,540+131).
