# Design: EmbeddingGemma 2 vector indexing in graft

Status: final proposal with measured data. Every number in this document traces
to a command run on 2026-10-10; the benchmark tree lives in `/build/eg2-bench/`
(artifact map in Appendix A). Graft-internals citations are against `graft_src`
at `@nanonets/graft` 0.9.0 and were spot-checked while writing.

**§1-§9 are as of 2026-10-10 and unchanged except two in-place corrections
flagged `[RESOLVED]` / `partially answered` (§5.5 first bullet, §9 item 1).
§10 was added 2026-10-11 and reports the fused-ranking A/B; §3's `src/ask/`
line numbers are still cited against 0.9.0 and have **not** been re-pinned
against the §10 implementation worktree — re-pin before merge.**

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

**Addendum 2026-10-11 (§10): the fusion has now been built and measured.** A
flag-gated linear blend (`VEC_WEIGHT`, sidecar read inside the opt-in guard,
byte-identical output with the flag off) passes 240 oracle queries on small_rs
+ small_ts at every weight swept. Measured best `VEC_WEIGHT = 1.5`: pooled
file hit@5 0.912 → 0.933, node hit@5 0.896 → 0.925, node MRR@10 0.815 → 0.844,
at a cost of +2-12 ms median latency (noisy; see §10.5). The measured effect
is small, concentrated in one of two corpora, and mostly *reorders* files the
lexical ranker had already retrieved rather than finding new ones. §10 does
**not** clear the channel for default-on shipping: it recommends an opt-in
`--vector` flag, off by default, on ts-like corpora only, and states the four
confounds that would have to be re-run before anyone turns it on by default.

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

- **[RESOLVED 2026-10-11 — see §10; corrected in place.]** This bullet said
  end-to-end fused ranking had never been run. It has now: a flag-gated linear
  blend, `VEC_WEIGHT` swept over 7 settings, 240 queries across small_rs +
  small_ts. §10 reports the result, including that the pre-registered gate
  could not be met on the baseline that was measured. §5.3 still measures pure
  vector retrieval only, and the 12-query baseline in §1.2 is still
  baseline-only; the draft's original "vectors beat graft's rank" TL;DR claim
  remains withdrawn as untested on its own terms.
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

1. `VEC_WEIGHT` and RRF-vs-linear: **partially answered (§10).** The linear
   blend is built and swept; `VEC_WEIGHT = 1.5` is the measured best on the
   benchmark, but the curve is flat from 0.30 to 1.5 and the measured effect
   is small enough that §10 recommends shipping it opt-in rather than tuned.
   **Still open: RRF-vs-linear.** The RRF variant named in §2.4 was not built
   or run — Stage 1 was declared saturated against the oracle-union ceiling
   (0.933), which made the later protocol stages moot on *this* baseline. If
   that baseline is ever re-run on a weaker arm A, RRF becomes live again.
   Still outstanding: the golden-set harness ported into graft's test suite
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

## 10. Addendum: the fused-ranking A/B, measured 2026-10-11

This section is new as of 2026-10-11 and does not rewrite §1-§9. It reports the
first end-to-end measurement of the §2.4 fusion inside `graft ask`, corrects one
claim in §5.5 in place (that bullet now carries a `[RESOLVED]` marker), and
gives a go/no-go.

Everything below was re-derived from the raw per-query dump while writing this
section, not taken from the run's own summary. Commands are given with each
claim; artifacts are in Appendix A.

### 10.0 What was built and what it guarantees

The prototype implements §2.4's linear blend exactly as written — no RRF, no
form-conditional weights, no schema change:

- `src/ask/vector-index.ts` (new) — sidecar reader. Returns `null` on missing,
  unparseable, unknown-version, dim-mismatch, model-mismatch, truncated-bin or
  duplicate-id, mirroring the `readAskIndex` null-contract of §2.5.
- `src/ask/ask.ts` — `AskOptions` gains `vectorWeight` / `queryVector` /
  `queryModel`. The sidecar is read **inside the opt-in guard**, so the
  no-flag path never touches the file. The blend is one additive term at
  `ask.ts:768`: `lexN + GRAPH_WEIGHT * pr + vecWeight * vecN`, with `vecN`
  max-normalised over the candidate set. `VECTOR_WEIGHT_CEILING = 1.5`.
- `src/ask/fuse.ts` — same term on the multi-scope path, via a defaulted
  `vectorWeight = 0` parameter so existing callers are byte-identical.
- `src/cli.ts` — `--vector <weight[@path]>`, threaded through **both** the
  workspace-federation branch (which returns early and would otherwise drop
  it) and `engine.ask`.
- `src/engine.ts` — `Graft.ask` retyped from a hand-written inline option
  literal to `Omit<AskOptions, 'contextDir'>` + spread.

Two of those are worth calling out because they are properties, not features.
The opt-in guard placement is what makes the byte-identity claim in §2.5 hold
mechanically rather than by convention. And the `engine.ts` retype fixed a
pre-existing bug: the old inline literal silently dropped every option it did
not name, and had **already** been swallowing `graphRank`. Any graft option
added since that wrapper was written has probably never reached `ask()` from the
CLI; the retype makes that class of drop impossible rather than merely absent.

Evidence:

| claim | command | result |
|---|---|---|
| flag-off byte-identity, sidecar present | `env -u GRAFT_NO_REFRESH node --import tsx /build/eg2-bench/ab/parity.mts` | `flag-off parity: 240/240 byte-identical, 0 differ` |
| same, at the scored limit of 50 | `... /build/eg2-bench/ab/audit/parity50.mts` | `FLAG-OFF PARITY @limit50: 240/240 byte-identical, 0 differ` / `avg hits returned: A=48.03 B_flagoff=48.03` |
| full suite | `env -u GRAFT_NO_REFRESH TMPDIR=/build/eg2-bench/ab/tmp node --import tsx --test test/*.test.ts` | `tests 517 / pass 517 / fail 0 / skipped 0 / duration_ms 48843.6` |
| the new tests alone | same, `test/ask-vector.test.ts` | `tests 24 / pass 24 / fail 0 / duration_ms 360.6` |
| missing sidecar degrades, never throws | `env -u GRAFT_NO_REFRESH node --import tsx /build/eg2-bench/ab/guard12.mts` | `arm A (shipped) hit@5 = 12/12 OK` / `arm B (flag on, sidecar MISSING) hit@5 = 12/12, threw 0 times -> degrades to flag-off, no throw` |

`env -u GRAFT_NO_REFRESH` is required on this box: `GRAFT_NO_REFRESH=1` is
exported ambiently, `refresh.ts:63` then short-circuits `ensureFreshGraph`, and
2 unrelated refresh tests fail. It is not a property of the vector channel.

The guard probe also produced an independent confirmation of the version gap
in §10.6: on the same 12 queries, arm A's gold ranks came back as
`[3,4,1,2,1,1,3,1,1,1,1,1]` while the §1.2 baseline file recorded
`[1,1,1,1,1,1,1,1,2,3,3,4]`. Same code path, different baseline.

### 10.1 Headline: 240 queries, two corpora, seven weights

Arm A is `ask` as shipped. Arm B is the same binary with the flag on. Both
measured in-process against the same warm graph, 0 errors in every arm.

```
w      f@5      f@10     nodeHit@5  MRR@10   median
0      0.9125   0.9458   0.8958     0.8149   35.4 ms
1.5    0.9333   0.9417   0.9250     0.8435   37.8 ms
```

Per-cell file hit@5 (counts out of 40; entry-level, see §10.4):

| cell | A | B | Δ |
|---|---|---|---|
| small_rs:q_bare | 38/40 | 38/40 | 0.0 |
| small_rs:q_where | 38/40 | 38/40 | 0.0 |
| small_rs:q_nl | 38/40 | 38/40 | 0.0 |
| small_ts:q_bare | 33/40 | 36/40 | **+7.5pt** |
| small_ts:q_where | 33/40 | 35/40 | **+5.0pt** |
| small_ts:q_nl | 39/40 | 39/40 | 0.0 |

Per-cell node MRR@10:

| cell | A | B | Δ |
|---|---|---|---|
| small_rs:q_bare | 0.786 | 0.828 | +4.2 |
| small_rs:q_where | 0.786 | 0.795 | +0.9 |
| small_rs:q_nl | 0.921 | 0.900 | **−2.1** |
| small_ts:q_bare | 0.734 | 0.800 | +6.6 |
| small_ts:q_where | 0.732 | 0.812 | +8.0 |
| small_ts:q_nl | 0.929 | 0.925 | −0.4 |

All of §10.1 re-derived by `python3 -I /build/eg2-bench/ab/audit/recompute.py`
over `per-query.json`; its output matches these tables exactly, so the run's
own summary is reproducible rather than merely reported.

### 10.2 The weight sweep, and the one number that should not be read as a signal

| w | f@5 | f@10 | nodeHit@5 | MRR@10 | median (run 1) | median (run 2) |
|---|---|---|---|---|---|---|
| 0 | 0.9125 | 0.9458 | 0.8958 | 0.8149 | 35.4 ms | 36.4 ms |
| 0.15 | 0.9208 | 0.9417 | 0.9042 | 0.8161 | 47.0 ms | 37.8 ms |
| 0.30 | 0.9292 | 0.9375 | 0.9167 | 0.8230 | 47.7 ms | 39.6 ms |
| 0.50 | 0.9292 | 0.9417 | 0.9167 | 0.8289 | 43.8 ms | 47.3 ms |
| 0.75 | 0.9292 | 0.9417 | 0.9167 | 0.8376 | 34.9 ms | 46.6 ms |
| 1.00 | 0.9292 | 0.9417 | 0.9167 | 0.8410 | 40.2 ms | 46.8 ms |
| 1.50 | 0.9333 | 0.9417 | 0.9250 | 0.8435 | 37.8 ms | 44.6 ms |

Quality is **bit-identical across both runs** at every weight; only latency
moves. Two consequences:

1. **f@5 is flat from 0.30 to 1.00.** All four of those arms score exactly
   0.9292 (223/240). `VEC_WEIGHT = 1.5` is the argmax at 0.9333 (224/240) —
   5 queries above arm A, but only **1 query** above w=0.5. A one-query margin
   is not a margin that survives a different oracle sample. If a weight is
   ever chosen, 1.5 is defensible as the measured argmax; 0.5 is equally
   defensible on this curve.
2. **The latency column is noise and must not be tuned on.** The same arm at
   w=0.75 is 34.9 ms in run 1 and 46.6 ms in run 2 — an 11.7 ms swing that
   exceeds the entire budget. Arm A itself moved 35.4 → 36.4 ms between runs.
   The honest statement is "somewhere between −1 and +12 ms, not resolvable
   with this harness", not "+2.3 ms at w=1.5".

### 10.3 The pre-registered gate cannot be met on this baseline

The run was gated against a rule registered before it: pooled file hit@5
+3.0pt over arm A. Every setting was rejected on that gate alone; gates 2-5
(pass: no `q_bare` regression, pooled MRR@10 above A, no cell down more than
5pt, latency inside 25 ms) passed at all seven weights. `stopping-rule-verify.py`
prints `PRE-REGISTERED RULE: NO SHIP`.

**Do not publish that NO SHIP as evidence against the channel.** The oracle
union over the same per-query dump bounds what is winnable at all:

```
ORACLE UNION (A|B per-query) = 224/240 = 0.9333; A=219/240; headroom=5
```

Only 5 of 240 queries are winnable by *any* implementation on this baseline.
+3.0pt is 7.2 queries. The gate is therefore unsatisfiable by arithmetic, and
B reaching exactly the union value means it captured 5/5 of the headroom that
existed. The rule was calibrated against a weaker arm A (0.883) that had ~8.4pt
of headroom; on the baseline actually measured it has 2.08pt.

A second, independent reason the verdict is unsafe to quote: the rule is
implemented in `stopping-rule-verify.py:35` as a *relative* test
(`if d_f5 < 0.030`, where `d_f5` is measured against **this run's** arm A),
while the registered form is an *absolute* threshold, `>= 0.913`, which is
0.883 + 3.0pt. Under the absolute form, B@1.5 at 0.9333 **passes**. The
substitution is invisible in `diff stopping-rule.py stopping-rule-verify.py`,
which shows only the input-path line changing. Whoever re-derives these gates
should re-check the semantics, not the diff.

### 10.4 What the +2.1pt actually is — a metric artefact, not new retrieval

The harness scores `files.slice(0, 5).includes(gtFile)` against a list with one
entry per **result node**, not per distinct file. A single file that ranks five
nodes occupies all five slots; the same file ranked once occupies one. That
makes entry-level file hit@k strictly harsher than distinct-file hit@k, and it
makes any reordering *within* an already-retrieved file look like a recall win.

Recomputing both arms with the file list de-duplicated, order preserved:

| metric | A | B | Δ |
|---|---|---|---|
| entry-level hit@5 (as scored) | 219/240 = 0.9125 | 224/240 = 0.9333 | **+5** |
| **distinct-file hit@5** | 229/240 = 0.9542 | 230/240 = 0.9583 | **+1** |
| entry-level hit@10 (as scored) | 227/240 = 0.9458 | 226/240 = 0.9417 | **−1** |
| **distinct-file hit@10** | 229/240 = 0.9542 | 231/240 = 0.9625 | **+2** |

Flipping the de-duplicated lists query by query, the entire distinct-file
movement is:

- **2 gains**, both the same query symbol — `guardFor` →
  `tests/container-interior.test.ts`, in `small_ts:q_bare` and
  `small_ts:q_where`. Arm A did not return that file at all; arm B returned it
  at distinct-file rank 5.
- **1 loss** — `small_rs:q_bare` `fmt` → `src/classifier/sensitive.rs`,
  A rank 4 → B rank 6, pushed out of the top 5.

The five entry-level "gains" are three query symbols (`listOf`,
`interruptedTurn`, `logged`) in one repo, and every one is a pure slot
promotion of a node arm A had already placed: gold-node ranks move **6→2,
6→3, 6→4, 7→2, 7→4**. Not one of them is a file the lexical ranker failed to
retrieve. So the honest reading of the headline is: **the vector channel
reordered files the lexical ranker had already retrieved, and retrieved exactly
one new file across 240 queries in one of two corpora.** Note also that the
reported hit@10 regression is an artefact of the same counting rule — at
distinct-file granularity hit@10 *improves* by 2.

(The one genuinely new file, `guardFor`, is a case the other way round: arm B
puts it at distinct-file rank 5 but its gold *node* still sits at rank 22-23.
The file is findable; the specific definition is not. If a caller cares about
the symbol rather than the file, this channel did not find it.)

That is a real ranking effect and worth having. It is not evidence that
embeddings find files lexical ranking misses, which is the claim §0 and §7.1
were reaching for.

### 10.5 Controls

| control | result | reading |
|---|---|---|
| grep, same query set | `grep_baseline.json` → `small_rs 0.95`, `small_ts 0.975`, pooled **0.9625** | Neither arm clears it: A f@10 0.9458, B 0.9417. The vector blend does not beat a name match, and was already behind before it was added. N/A for `q_where`/`q_nl` — grep is name-only. |
| flag-off parity | 240/240 byte-identical at limit 8 **and** at limit 50 | The §2.5 byte-identity claim holds mechanically with a valid sidecar on disk. |
| missing sidecar | 12/12, 0 throws | Degrades to flag-off; §7.1's "must degrade to lexical-only, never to no answer" holds. |
| mutation pins | 3 probes | Deleting `+ vecWeight * vecN` from ask.ts kills the single-scope pin and only that one; from fuse.ts, kills the multi-scope pin and only that one; reverting the `engine.ts` retype kills both forwarding tests. Each fails on its own line. |

The grep control is the one this section should end on. Every quality gate
passes, and the channel still does not beat `grep` on the benchmark's own
queries.

### 10.6 Confounds — four things invalidate this measurement as a ship decision

1. **Wrong baseline.** This ran on worktree `d4ec9d7` (v0.8.2), not the live
   checkout at `7b8acc5` (v0.9.0). `git rev-list --count d4ec9d7..7b8acc5`
   → **72**, and `git diff --stat d4ec9d7 7b8acc5 -- src/ask/` →
   `ask.ts` 59 lines changed, `fuse.ts` +8/−1, `index-file.ts` +2/−1. Those
   are exactly the fusion and ranking files under test. Arm A already scores
   0.9125 here against the 0.883 the protocol recorded for the live checkout,
   and arm A's own hit@10 is 0.9458 here — so most of what the vector channel
   was meant to win has already been banked by lexical/ranking work in those
   72 commits. **Re-run arm A on `7b8acc5` before quoting any of these numbers
   as a property of the channel.**
2. **The embedded text is not what the design specifies.** The stored vectors
   embed `crux.code` capped at 6000 chars; the draft's §3.3
   (`/build/eg2-bench/design-embeddinggemma2.md`, "Embed-time input text")
   specifies `title: {path} | text:` + signature + `summary` + definition body
   from `node.span`, capped at ~2,000 chars. `body_text` is absent from
   `wiring.json` on all 2814 nodes (stripped by `stripBodyText()`), so
   `crux.code` was used as an offline proxy. Every recall number above is
   measured against text the shipping encoder would never see. The title field
   is also unpinned: the benchmark used the bare symbol name where the draft
   says the path, and those two score cosine 0.953-0.963 against each other.
3. **Latency is not comparable to the protocol's budget.** Measured in-process
   and warm against a 35 ms baseline; the 25 ms gate was calibrated against a
   344-584 ms CLI-subprocess baseline. The delta measurement is internally
   sound (same process, same A) but "inside the 25 ms budget" is not a claim
   this run can support.
4. **Both corpora are single-scope.** The multi-scope blend is covered by the
   `fuse.ts` unit test and one 2-scope synthetic fixture, never by a real
   multi-scope benchmark. §2.4's claim that federation "keeps working
   unchanged" is untested at scale.

Two shipping surfaces are also deliberately absent: `graft ask` ships no
encoder, so `--vector` consumes a caller-produced query-vector JSON, and there
is no `graft embed` sidecar writer in the CLI — the sidecars used here were
written by an offline script. Both are prerequisites for a real shipping path,
not part of the fusion.

### 10.7 Go / no-go

**Recommendation: conditional GO — ship the flag, off by default. Do not enable
the channel anywhere until the four §10.6 confounds are cleared.**

Ship it, because the engineering half is real and paid for: the null-contract
reader, the opt-in guard placement, byte-identical flag-off behaviour verified
at two limits, the federation forwarding fix, and 24 tests including three
mutation pins that each die on their own line. Every quality gate that could be
measured passed at every weight. The one gate that failed failed for an
arithmetic reason (§10.3), and the effect that did occur is a genuine ranking
improvement in the corpus that §5.3 already identified as weakest.

Do not enable it by default, because at distinct-file granularity the channel
retrieved **one** file in 240 queries and lost one (§10.4), it does not beat
`grep` (§10.5), and the effect was measured against a baseline 72 commits
behind the one this document targets (§10.6). A default-on channel whose
demonstrated contribution is reordering is a bad trade against 8.6 MB of bin
read per query and an extra sidecar to keep fresh.

Conditions, explicitly:

| | |
|---|---|
| **Flag** | `--vector <weight[@path]>`, off by default. No config key in this pass; a config switch would outlive the evidence. |
| **Weight** | `1.5` is the measured argmax and is fine to ship as the default *once enabled*. `0.5` is 1 query behind on a curve that is flat from 0.30 up, and is the better choice if anyone objects to the ceiling constant. Do not re-tune on the §10.2 latency column — it is noise. |
| **Enable on** | ts-like corpora — those with a high head-only chunk share, where §5.3 showed pure-vector weakness and where both of the entry-level gains and the single distinct-file gain landed. |
| **Leave off** | rs-like corpora. All three small_rs cells are frozen at 38/40 with zero movement, and at distinct-file granularity the channel *loses* a file there (`fmt`). There is no evidence of benefit and one small measured cost. |
| **Query text** | Gains landed only on the `q_bare` and `q_where` forms in ts; `q_nl` moved zero in every cell of both corpora. Do not read that as evidence about NL queries in general — §10.4 shows the whole gain is rank-order within already-retrieved files, and §10.6(2) means the embedded text was not the design's text. It is one more reason the "paraphrase queries get a match lexical cannot provide" motivation of §0 remains unproven by this run. |

Work required before anyone flips it on, in order:

1. **Re-run arm A on `7b8acc5`.** Everything in §10.1-§10.5 is relative to a
   displaced baseline. On the current code the headroom may be zero, in which
   case this section's conclusion is settled by that and nothing else is
   worth running.
2. **Re-run with the draft's §3.3 embed-time text**, i.e. an encoder run over
   `title + signature + summary + span body` at ~2,000 chars. This requires a
   sidecar writer in the CLI, which does not exist yet.
3. **Score at distinct-file granularity** as well as entry-level. Report both.
   Entry-level counts 5 queries of gain here; distinct-file counts 1.
4. **Fix the gate before re-registering it.** Absolute `>= 0.913`, derived
   from the arm A actually used, applied to the arm A actually used — and
   recompute the union ceiling on that baseline rather than inheriting +8.4pt.
5. **One multi-scope corpus**, since §2.4's federation claim is still untested
   at scale and the multi-scope blend has never seen real multi-scope traffic.

If (1) shows the headroom on `7b8acc5` is under ~2pt, the correct outcome is to
ship the flag, document it, and leave it off indefinitely — the flag-off
byte-identity guarantee makes that a cheap option, which is most of why this
recommendation is safe.

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
| `/build/eg2-bench/design-embeddinggemma2.md` | the earlier draft; §3.3 is the canonical embed-time input text that §10.6(2) compares against |
| `/build/eg2-bench/ab/ask-ab.mts`, `ab-verify.json`, `ab-results.json` | §10 A/B driver and its two runs. `ab-results.json` = original run, `ab-verify.json` = independent re-run. Quality is identical in both; only latency differs (§10.2) |
| `/build/eg2-bench/ab/per-query.json` | 240 rows × full ranked file/node lists for both arms — the substrate for every §10 number |
| `/build/eg2-bench/ab/audit/recompute.py` | independent recomputation of §10.1-§10.4 from `per-query.json`, including the distinct-file pass |
| `/build/eg2-bench/ab/parity.mts`, `audit/parity50.mts` | flag-off byte-identity at limit 8 and at the scored limit 50 |
| `/build/eg2-bench/ab/guard12.mts` | missing-sidecar degradation probe (12 queries, `graft_src`, no sidecar) |
| `/build/eg2-bench/ab/stopping-rule.py`, `stopping-rule-verify.py`, `ceiling.py`, `ceiling-verify.py`, `analyze.py` | the registered 5-gate rule and the oracle-union ceiling — see §10.3 for why the gate's verdict is not quotable |
| `/build/eg2-bench/ab/make-sidecars.py` | offline sidecar writer from the benchmark's stored vectors; stands in for the `graft embed` writer that does not exist yet (§10.6) |
| `/build/eg2-bench/bench-quality/grep_baseline.json` | `{"small_rs": 0.95, "small_ts": 0.975}` — the grep control in §10.5 |
| `/build/worktrees/graft_src/eg2-fusion-ab` | the implementation worktree, on `d4ec9d7` (v0.8.2). **Not merged.** 517/517 tests pass there; the branch is `ab/vector-fusion` |

Verification commands run while writing this document (all numbers above
re-checked against artifacts): census totals from `census_results.json`
(4 graphs, `node_count_matches_meta: true`, workload 53,309,682);
`eval_results.json` / `grep_baseline.json` / `window_baseline.json` reads;
`wc -l`/`ls -la` on `indexes/` (1,500/750/750/750 ids; 4,608,000 and
2,304,000-byte bins); the three-way cosine agreement computation over
`bench-gpu/*.npy` reproduced §5.4 exactly; head-only chunk recount
(1,616/1,671) reconciled with the per-class counts (1,508+108, 1,540+131).
