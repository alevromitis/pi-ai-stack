# Plan: disk cache for YNNPACK packed weights

Status: **stopped at the Phase 0 gate (2026-10-08). Not worth building.** See "Phase 0
results" below. Nothing in this stack has changed.

## Phase 0 results (2026-10-08, Pi 5, gemma4-e2b, CPU, 4 threads, 2k context)

How it was measured:

- gdb breakpoints on YNNPACK's exported symbols (`ynn_create_subgraph`,
  `ynn_optimize_subgraph`, `ynn_subgraph::fold_constants`, `ynn_runtime::build`,
  `ynn_runtime::reshape`, `ynn_runtime::setup`, `eliminate_common_subgraphs`,
  `ynn_create_runtime`, `ynn_invoke_runtime`, `ynn_define_tensor`);
- gdb overhead corrected at about 0.49 ms per breakpoint hit.

The scripts are in `~/litert-bench/p0/` on rpi, with a throwaway image `litert-lm-rpi5:gdb`.

**One YNNPACK graph build** (1,539 delegate partitions; gdb time, then corrected time):

| Phase | Time | Cacheable? |
|---|---|---|
| Main runtimes: Slinky pipeline build | 5.1 s / about 4.4 s | no |
| `fold_constants` (weight packing) | 1.9 s / about 1.4 s | **yes, the only part** |
| Optimize (fusion etc.) | 1.3 s / about 0.5 s | no |
| Define, create, setup, CSE | about 1.6 s raw, almost all of it gdb overhead | no |

**Gate:** packing is about 20% of a build and well under 5 s, so the gate fails.

- At best a cache saves about 1.5 s per engine init: roughly 7 s becomes roughly 5.5 s.
- The benchmark's "Init time" (14.7 s) covers two builds, because its engine-plus-session
  setup builds the graphs twice.
- `serve` builds them only once per engine init: the logs show 9 builds for 9 engine inits
  across 60 requests, never once per conversation. So the saving would only be felt at
  container start or on a model/settings change.

**Step 2 passed, for the record:**

- 627 unique constant tensors (742 MB; all 496 tensors of 1 MB or more) are pointers into
  the mmapped `model.litertlm`;
- 11.5 MB of constants are on the heap; scalars are on the stack (`COPY_DATA`);
- file-offset cache keys would therefore have worked.

**Memory:** a cache would move about 0.5-0.75 GB of packed weights from anonymous to
file-backed memory. Decode reads every weight on every token, so those pages stay hot and
resident. The saving would be reclaimability under pressure, not a smaller working set.

**XNNPACK without its cache**, for comparison (`--cache no` vs `--cache disk`):

| | `--cache no` | `--cache disk` |
|---|---|---|
| Init | 33.7 s | 0.75 s |
| RssAnon peak | 4.8 GB | 0.97 GB |
| Prefill | 70 tok/s | 98 tok/s |
| Decode | 6.3 tok/s | 8.6 tok/s |

XNNPACK's packing is expensive. YNNPACK's is cheap (about 1.4 s), which is why its cache
gap matters much less.

**Plain YNNPACK rerun:** 147.6 / 9.0 tok/s, init 14.7 s, peak RssAnon 1,462 MB. This
matches the 2026-10-07 run.

**Better lever if init time ever matters:** the Slinky pipeline build (about 4.4 s per
engine init). It is not a weight-cache problem; it would need an upstream runtime or
pipeline cache in YNNPACK/Slinky.

## Goal and expected payoff

The Dockerfile patches `serve` to use YNNPACK (`enable_ynnpack=True`). Benchmarked on the
Pi on 2026-10-08 (gemma4-e2b, CPU, 4 threads, 1024 prefill / 256 decode, 2k context):

| | XNNPACK (with its disk cache) | YNNPACK (no cache) |
|---|---|---|
| Prefill / decode | 108 / 8.4 tok/s | 143 / 8.7 tok/s |
| Init time | 0.7 s | 15.7 s |
| Peak RssAnon / RssFile | 967 / 981 MB | 1431 / 956 MB |

A YNNPACK disk cache could at best:

- cut init time (15.7 s down to about 1 s), which matters at container start and whenever
  `serve` re-initialises the engine (model or settings change);
- move about 460 MB of packed weights from anonymous memory to file-backed mmap that the
  kernel can reclaim.

It will **not** change prefill or decode speed, because the packed layout is identical
either way.

## What upstream has (checked 2026-10-08, upstream main and litert-lm 0.17.1)

There is no YNNPACK weight cache anywhere, and no option to plumb one:

- **XNNPACK** `ynnpack/subgraph/subgraph.cc`: weights are packed by a `pack_b` subgraph
  node that `ynn_subgraph::fold_constants()` evaluates during `ynn_optimize_subgraph()`.
  The result is a heap `slinky::raw_buffer::make(...)`, which is why it shows as RssAnon.
  - An in-process cache already exists: `constant_cache`, a
    `std::map<cache_key, std::weak_ptr<slinky::raw_buffer>>` (about L530-651), plus
    `fold_constants` (about L695-921).
  - Its `cache_key` identifies weights by **raw pointer** (`input_pointer`). That
    deduplicates prefill and decode subgraphs inside one process, but a pointer is not
    stable across processes. This is the hook to extend.
- **TFLite/LiteRT YNNPACK delegate** `delegates/ynnpack/ynnpack_delegate.cc`:
  - constant tensors go to `ynn_define_tensor(..., tensor.data.raw, /*flags=*/0, ...)`
    (about L120-125), so they are pointers into the mmapped `.litertlm`, not copies;
  - it calls `ynn_optimize_subgraph` and then `ynn_create_runtime` (about L271-273);
  - `TfLiteYNNPackDelegateOptions` has no cache field.
- **LiteRT** `litert/runtime/litert_cpu_options.h`: `LiteRtYnnpackOptionsT` has no cache
  field. Only the XNNPACK options carry `weight_cache_file_path`.
- **LiteRT-LM** `runtime/executor/llm_executor_settings_utils.cc` (about L284-290): calls
  `SetEnableYNNPack(...)`, then sets only the `.xnnpack_cache` file. The installed
  `liblitert-lm.so` has XNNPACK (`MMapWeightCacheProvider`) and ML Drift caches only.
- **Issues and discussions:** none found requesting or implementing this.
  [snnn/ynnpack-model-lab](https://github.com/snnn/ynnpack-model-lab) lists "a
  packed-weight disk cache" only as a possible future experiment.
- **Template to copy:** XNNPACK's `MMapWeightCacheProvider`
  (`tensorflow/lite/delegates/xnnpack/weight_cache.{h,cc}`). It uses stable buffer
  identifiers instead of pointers, a file named `<model>_<mtime>_<size>.xnnpack_cache`,
  build-then-mmap, and a stale flag.

Line numbers above are from main as of 2026-10-08. Re-find them at the pinned revisions.

### Build pins

| | litert-lm v0.17.1 (installed) | v0.18.0 |
|---|---|---|
| `LITERT_REF` | `9fe5be45564c868408e6514c8aabb83e211a0911` | `26895c9fbcc25c43faa8c1a98cd1fd28951602c3` |
| `TENSORFLOW_REF` | `d9a8da74b4c3de28a39ab34ad007838d6bc30c67` | same |
| XNNPACK | unresolved, find in Phase 1 | `d89ef6669a14db203b3b7935b1b3862cb63fb6df` |
| slinky | unresolved | `2c9e129fc82215530af851cfd8d3094de0519241` |

- The upstream arm64 wheel builds natively on an 8-core arm64 runner
  (`.github/workflows/nightly-linux-arm64.yml`) with:
  `bazel build --config=linux_arm64 --define=litert_enable_ynnpack=true --define=litert_runtime_link_mode=dynamic -c opt //python/litert_lm:wheel`.
- Without `use_prebuilt_c_api`, `liblitert-lm.so` is built from `//c:litert-lm`. YNNPACK is
  statically linked into it, so only that one `.so` needs replacing.

## Design (Phase 2)

Keep the patch inside **XNNPACK/YNNPACK only**, so LiteRT and LiteRT-LM need no changes.
Apply it with `--override_repository=XNNPACK=<patched checkout>` and switch it on with an
env var.

1. **Enable:** `YNN_CONSTANT_CACHE_DIR=<dir>`. If unset, behave exactly as stock.
2. **Stable identity:** in `build_chain`, translate each `input_pointer` to
   `(dev, inode, size, mtime, offset)` of the file mapping that contains it. Parse
   `/proc/self/maps` once and cache the result.
   - A pointer in anonymous memory makes that constant non-persistable; it keeps the
     current in-memory path.
   - `input_value` (`YNN_VALUE_FLAG_COPY_DATA`) switches from `std::hash` to XXH3.
3. **Op identity:** serialize `op_key.op` as variant index plus op fields, for an allowlist
   of the op types that actually occur in constant chains (log them in Phase 2a; expected
   `pack_b`, maybe convert, transpose or dequantize). Ops outside the allowlist are not
   persisted.
4. **Storage:** one segment file per `fold_constants` call, named
   `<model-basename>_<size>_<mtime>/<xxh3 of sorted cache keys>.ynn_cache`.
   - Header: magic, format version, YNNPACK build id, CPU ISA flags.
   - Then entries `{key, rank, elem_size, dims[], offset, size}`, with data aligned to
     `YNN_ALLOCATION_ALIGNMENT`.
   - Written to a temp file, fsynced, then renamed.
   - The same subgraph produces the same key set, so a warm start finds its segment
     directly. No global index is needed.
5. **Load:** mmap the segment `PROT_READ, MAP_PRIVATE`.
   - Wrap each entry as a `slinky::raw_buffer_ptr` (a `shared_ptr`) whose deleter holds a
     `shared_ptr` to the mapping, then put it in `cached_values` exactly as an in-process
     hit would be.
   - Any header mismatch or short file means the segment is ignored and rebuilt.
6. **Writing on a miss:** after `runtime.invoke()`, copy each newly folded constant into
   the segment, then re-point `values[i].data` at the mmapped copy. That frees the heap
   buffer and makes the memory file-backed on the first run too.

## Phases

### Phase 0: is it worth it? (no build; needs `litert-lm-1` stopped for a few minutes, ask first)

1. Find where YNNPACK's 15.7 s init goes. **Gate:** if packing is under about 60% of init
   time, or under 5 s, stop and record the result here. Init that comes from Slinky
   pipeline building is not helped by a cache.
   - Cheap proxy: run the `~/litert-bench` harness with XNNPACK and `--cache no` against
     `--cache disk`. That shows what packing costs XNNPACK.
   - Direct check: poor-man's profiler with host `gdb` (installed on the Pi; `perf` is
     not). Attach about 30 times during YNNPACK init and bucket the stacks into `pack` or
     `fold_constants` versus `slinky` pipeline build versus other.
2. Confirm that most YNNPACK constants come from pointers inside the `model.litertlm`
   mappings. Check with gdb breakpoint sampling of `ynn_define_tensor` data pointers
   against `/proc/<pid>/maps`.

### Phase 1: reproduce the stock build (decision needed: where to build)

1. Resolve the XNNPACK and slinky pins for v0.17.1: grep the LiteRT and TF workspace files
   at the refs above, or run `bazel query` the repository rule.
2. Build `//c:litert-lm` (or the full wheel) at tag `v0.17.1` for linux arm64. Build-host
   options:
   - **GitHub Actions `ubuntu-24.04-arm` on a fork.** Fast and free, but it publishes a
     public fork, so ask before creating it.
   - **The Pi natively**, with `--jobs=2` and a RAM cap while `litert-lm-1` is stopped.
     Slow, likely many hours, and it takes the service down.
   - **A rented arm64 VM.**
   - **QEMU on the PC**: impractically slow for Bazel; avoid.
3. Swap the self-built `.so` into a test image and rerun the 2k benchmark. It must match
   the PyPI build within noise before any patching.

### Phase 2: implement (in the XNNPACK checkout)

1. Add logging to list op types in constant chains and the share of
   pointer-in-file-mapping inputs for gemma4-e2b.
2. Implement the design above. New code goes in `ynnpack/subgraph/constant_disk_cache.{h,cc}`;
   the hooks go in `build_chain`, `fold_constants` and the cache lookup and insert.
3. Add tests in `ynnpack/subgraph/test/`:
   - fold, persist, reload in a fresh subgraph, and get bit-identical buffers;
   - a corrupt, truncated or stale-header file falls back;
   - an unset env var gives stock behaviour.

### Phase 3: validate on the Pi (ask before stopping `litert-lm-1`)

With `~/litert-bench` at 2k context, compare:

- stock;
- patched with a cold cache (first run builds it);
- patched with a warm cache.

Pass criteria:

- warm init time at most 2 s;
- prefill and decode within noise of stock;
- warm RssAnon at least 300 MB lower than stock YNNPACK;
- greedy outputs token-identical to stock.

Then check the 32k-context serve peak stays under the 7 GB cap.

### Phase 4: roll out

1. Dockerfile: install the patched `.so` or wheel instead of `pip install litert-lm==0.17.1`.
   The Python patches (serve `enable_ynnpack`, prefix cache, reasoning) stay as they are.
2. Compose: set `YNN_CONSTANT_CACHE_DIR=/root/.litert-lm/ynn-cache`, which persists in the
   `litert-registry` volume.
3. README: document the cache, how to clear it, and the rebuild needed on every litert-lm
   upgrade. Rollback is to unset the env var or revert to the PyPI wheel.

### Phase 5 (optional): upstream

Open an issue, or a PR, on google/XNNPACK proposing a persistent YNNPACK constant cache,
with the Phase 3 numbers. If accepted, the local patch can be dropped.

## Costs and risks

- **Effort:** Phase 0 about 1 h; Phase 1 0.5-2 days (Bazel arm64 build is the biggest
  unknown); Phase 2 1-2 days of C++; Phases 3-4 about half a day.
- **Ongoing:** a self-built binary means rebasing the patch and rebuilding on every
  litert-lm upgrade. v0.18.0 already pins a different XNNPACK.
- **Correctness:** a stale cache would give wrong outputs. Mitigate with a strict header
  (build id, ISA, model identity), atomic writes, and the token-identical check in Phase 3.
- **Memory pressure:** file-backed weights can be evicted and re-read from disk mid-decode,
  which causes latency spikes. XNNPACK has the same exposure today.
- **Fallback:** if weights turn out not to be pointers into the model mapping (Phase 0
  step 2), keys need content hashing. That reads all weights on every start and erases
  most of the init-time gain.
