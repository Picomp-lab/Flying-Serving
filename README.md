# Flying Serving

**Dynamic DP↔TP switching for LLM serving — built on [vLLM](https://github.com/vllm-project/vllm).**

Flying Serving lets a single deployment **switch its parallelism strategy at runtime**, per wave of requests, *without reloading weights*. Engines run as **Data‑Parallel (DP)** replicas for high aggregate throughput, and transparently **merge into a larger Tensor‑Parallel (TP)** group when a request would benefit from it (e.g. long‑context, latency‑sensitive prefill), then split back. You get **TP‑class latency when you need it and DP‑class throughput when you don't**, on the same GPUs and the same resident weights.

> This repository is a fork of vLLM. The upstream README is preserved as [`README_vLLM.md`](README_vLLM.md).

---

## Why

A serving deployment normally commits to one parallelism layout:

| | First‑token latency (TTFT) | Per‑token latency (TPOT) | Throughput | KV cache / request |
|---|---|---|---|---|
| **Tensor Parallel (TP)** | **low** (prefill split across GPUs) | **low** (compute split) | lower | smaller (sharded) |
| **Data Parallel (DP)** | high (prefill on one GPU) | high under load | **high** (independent replicas) | larger (full replica) |

The gap widens with model size and context length. Measured on Llama‑3‑70B at 4096‑token prompts and 1 req/s, static TP has **2.1× lower TTFT** and **2.1× lower TPOT** than static DP — but push past saturation and static DP wins every metric, because queueing rather than prefill then decides latency. Picking one strategy statically leaves performance on the table whenever the load changes.

**Flying Serving removes the static choice.** It keeps the throughput of DP at steady state and borrows the latency of TP on demand, by merging and splitting engines live.

---

## How it works

1. **Resident weights, zero reload.** Each engine loads the full model once (DP layout). When engines merge, each one's TP shard is *activated in place* from the already‑resident replica — no second copy, no reload. This is the key to switching in milliseconds instead of re‑initializing the model.

2. **A dynamic TP communicator (`_DTP`).** At merge time the participating engines' GPUs form a single TP group and run the model's per‑layer all‑reduces across engine boundaries, exactly like native TP.

3. **Lock‑step scheduling.** The merged engines must execute identical batches. A lightweight cross‑engine barrier aligns admission so every engine builds the same TP batch each step.

4. **Three weight‑sharding strategies** (`VLLM_DTP_SHARDING`):

   | mode | GEMM | extra memory | decode speed | use when |
   |---|---|---|---|---|
   | `materialize` | one fused GEMM on a cached contiguous shard | + one shard copy (resident) | native | the shard copy fits |
   | `view` | one GEMM per weight slice (zero‑copy) | **none** | slower (multi‑GEMM) | memory is tight |
   | **`reorder`** *(recommended)* | one fused GEMM on an **in‑place reordered** shard | **none** | **native** | always, for unquantized models |

   `reorder` is the best of both: on the first forward after a merge it permutes the resident weight rows **in place** (column‑chunked, ~tens of MB transient) so each engine's shard becomes one contiguous block — a single native‑speed cuBLAS GEMM — with **zero extra resident memory**. Rows are restored on split, so pure‑DP execution is byte‑for‑byte unchanged. On Llama‑3‑70B this lets the merged TP run at near‑native decode speed *and* fit where `materialize` OOMs.

> Flying Serving implements the system described in the paper [*Flying Serving: On‑the‑Fly Parallelism Switching for Large Language Model Serving*](https://arxiv.org/abs/2602.22593) (Gao et al., 2026). See [Citation](#citation).

---

## Install

Flying Serving is a fork of vLLM; install it the same way (from source, since the dynamic kernels run in eager mode):

```bash
git clone https://github.com/Picomp-lab/Flying-Serving.git
cd Flying-Serving
pip install -e .            # or: VLLM_USE_PRECOMPILED=1 pip install -e .
```

Requirements match upstream vLLM (CUDA GPUs with NVLink recommended for the cross‑engine all‑reduce). The examples below use 4× H‑class GPUs.

---

## Tutorial: Llama‑3‑70B on 4 GPUs

We run `meta-llama/Meta-Llama-3-70B-Instruct` on 4 GPUs three ways and compare on one workload: **in=4096 / out=256**, the long‑context shape where the choice of parallelism actually matters. All three carry the same flags — `--enforce-eager` (the dynamic switch is not compatible with CUDA graph capture today) and `--no-enable-prefix-caching` (a baseline with caching on skips prefill work the merged path cannot, which invents a win that is not there).

### 1. Static Data Parallel (throughput baseline)

Two replicas, each Tensor‑Parallel‑2 (70B does not fit on one GPU):

```bash
vllm serve meta-llama/Meta-Llama-3-70B-Instruct \
    --data-parallel-size 2 --tensor-parallel-size 2 \
    --enforce-eager --gpu-memory-utilization 0.90 \
    --max-model-len 8192 --no-enable-prefix-caching --port 8000
```

### 2. Static Tensor Parallel (latency baseline)

One TP‑4 group across all 4 GPUs:

```bash
vllm serve meta-llama/Meta-Llama-3-70B-Instruct \
    --tensor-parallel-size 4 \
    --enforce-eager --gpu-memory-utilization 0.90 \
    --max-model-len 8192 --no-enable-prefix-caching --port 8000
```

### 3. Flying Serving (dynamic DP↔TP) ⭐

Launch as 2 DP engines (each TP‑2) **in dynamic mode**, with the `reorder` sharding strategy. Engines run as DP replicas and merge into TP‑4 on demand:

```bash
VLLM_DTP_WORK_MODE=width \
VLLM_DTP_SHARDING=reorder \
vllm serve meta-llama/Meta-Llama-3-70B-Instruct \
    --data-parallel-size 2 --tensor-parallel-size 2 \
    --enforce-eager --gpu-memory-utilization 0.90 \
    --max-model-len 8192 --no-enable-prefix-caching --port 8000
```

Merge the two engines into one TP‑4 group, and split them back:

```bash
curl -X POST localhost:8000/dtp_switch -H 'Content-Type: application/json' -d '{"width": 2}'   # both engines -> one TP-4 group
curl -X POST localhost:8000/dtp_switch -H 'Content-Type: application/json' -d '{"width": 1}'   # back to 2 DP replicas
curl -s localhost:8000/dtp_status
```

A switch is carried *by a request*: on an idle deployment the POST returns immediately and nothing moves until the next request arrives, which then carries it (measured: 0.19 s to merge, 0.02 s to split). Requests use the ordinary OpenAI‑compatible API and are unaffected by which mode is active:

```bash
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{
  "model": "meta-llama/Meta-Llama-3-70B-Instruct",
  "prompt": "Explain tensor parallelism in one paragraph.",
  "max_tokens": 256
}'
```

### What you get

Llama‑3‑70B, 4× H200, in=4096 / out=256, Poisson arrivals. **Mean TTFT / P99 TTFT / median TPOT, in ms.** Flying switches on the first request of each run, so the merge is inside every measured window:

| req/s | Static DP (DP2×TP2) | Static TP4 | **Flying (reorder)** |
|---|---|---|---|
| 0.5 | 635 / 1884 / 32.4 | 352 / 880 / 18.5 | **356 / 874 / 21.0** |
| 1 | 786 / 2169 / 44.2 | 367 / 991 / 20.9 | **370 / 990 / 24.8** |
| 1.5 | 1211 / 4015 / 63.6 | 512 / 1519 / 27.2 | **518 / 1519 / 31.1** |
| saturated | 21132 / 40128 / 113.7 | 22299 / 43869 / 115.5 | **22329 / 43918 / 117.1** |

Throughput at saturation: static DP **12 865** tok/s, static TP4 12 417, Flying 12 298 — −4% against DP, **−1% against the TP4 layout it emulates**.

Read the table top to bottom and the design falls out of it:

- **Below saturation, TP is worth having.** At 1 req/s it halves TTFT (786 → 367 ms), more than halves P99 TTFT (2169 → 991 ms) and halves TPOT (44 → 21 ms) against static DP.
- **Flying captures nearly all of it** — within 1% of static TP on TTFT at every rate, and its P99 is at or below static TP's (874 vs 880, 990 vs 991). TPOT costs 14–19% more than static TP, which is the per‑layer all‑reduce running across engine boundaries; against static DP it is still ~2× better.
- **At saturation the advantage inverts and TP has nothing left.** Static DP wins throughput (+4%), TTFT (−5%) and TPOT (−2%). Queueing, not prefill, sets TTFT once arrivals exceed capacity, and DP has twice the aggregate prefill capacity and twice the concurrency budget (`--max-num-seqs` is per engine).

So the deployment rule is: **merge before the queue builds, split once it has.** That is the decision Flying Serving makes switchable at runtime instead of at launch. The switch itself is cheap enough not to enter into it — 0.2–1.0 s to merge, and every engine of a group flips within 10 ms of the others (see [Switch methods](#switch-methods)).

Numbers for gpt‑oss‑120B (MXFP4 MoE, DP4×TP1 → TP4) are in `dtp_bench/out/`: the same shape holds, with a larger merged‑TP penalty (−4…−7% against static TP) because it merges four engines rather than two.

---

## Configuration

All knobs are environment variables, read at startup / per step.

| variable | values | default | effect |
|---|---|---|---|
| `VLLM_DTP_WORK_MODE` | `original` \| `tp_mode` \| `width` \| `switch_test` | `original` | `tp_mode` enables dynamic DP↔TP merging. `original` = stock vLLM behavior. `width` takes the layout from the `/dtp_switch` endpoint (see [Merge width](#merge-width-more-than-one-tp-group)). `switch_test` is a validation harness (see [Switch methods](#switch-methods)). |
| `VLLM_DTP_SHARDING` | `materialize` \| `view` \| `reorder` | `materialize` | Merged‑TP weight strategy (see table above). `reorder` recommended for unquantized models. |
| `VLLM_DTP_SYNC_EVERY` | integer ≥ 1 | `1` | Cadence of the cross‑engine admission barrier. **Lower → lower TTFT**; **higher → lower TPOT** (the barrier cost is amortized over more steps). |
| `VLLM_DTP_MERGE_SETS` | `"0,1;2,3;0,1,2,3"` | binary hierarchy | Which engine subsets may merge into one TP group. Default is every aligned power‑of‑two range plus the full merge. |
| `VLLM_DTP_SWITCH_METHOD` | `hard-preempt` \| `sequential` | `hard-preempt` | How a switch treats in‑flight work (see [Switch methods](#switch-methods)). |
| `VLLM_DTP_TO_TP_METHOD` / `VLLM_DTP_TO_DP_METHOD` | `sequential` \| `hard-preempt` | `VLLM_DTP_SWITCH_METHOD` | Per‑direction override. `switch_test` mode only. |
| `VLLM_DTP_SWITCH_PERIOD` | integer ≥ 1 | `40` | Requests between mode flips. `switch_test` mode only. |
| `VLLM_DTP_DEBUG` | `0` \| `1` | `0` | Log what each merged engine received and scheduled every step. The merged engines must build an identical batch; this is what shows you when they do not. |

**Tuning guidance.** The barrier is the one piece of *extra* cross‑engine communication Flying Serving adds (the per‑layer all‑reduce is intrinsic to TP). `VLLM_DTP_SYNC_EVERY` trades first‑token against per‑token latency. Measured on Llama‑3.1‑8B at low load:

| `VLLM_DTP_SYNC_EVERY` | TPOT | TTFT |
|---|---|---|
| **1** *(default)* | 9.15 ms | **28.8 ms** |
| 2 | 8.46 ms | 33.0 ms |
| 4 | **8.31 ms** | 41.7 ms |

`=1` is the default: it is the only setting with no admission stall, and the ~0.7 ms of TPOT it costs is the smaller half of the trade against the ~4 ms of TTFT that `=2` adds. Raise it for decode‑heavy workloads where TPOT dominates.

Quantized models (e.g. MoE / mxfp4) automatically fall back to the `materialize`/`view` path for layers `reorder` does not support.

`VLLM_DTP_WORK_MODE` also accepts several experimental routing policies used for the paper's experiments — `traffic_load`, `request_length`, `custom`, `manipulated` — which switch on hard‑coded request counts. They are not intended for serving; the three in the table above are.

---

## Merge width: more than one TP group

**Merge width** `w` is how many engines form one TP group; the engines split into `dp_size / w` groups, all of them serving. Width 1 is plain DP, width `dp_size` is a single merged group. Any width in between runs several merged groups side by side, which is what makes a 4 → 2 → 1 → 2 → 4 cycle possible on one deployment.

```bash
curl -X POST localhost:8000/dtp_switch -H 'Content-Type: application/json' -d '{"width": 2, "method": "hard-preempt"}'
curl -s localhost:8000/dtp_status
# {"merge_width":1,"groups":[],"dp_size":4,"pending_switches":[]}
```

A width change always passes through width 1: merged engines re-shard from their own resident DP replica, so a group splits back to DP before re-merging at another width. `2 → 4` is rejected with a 400 asking you to go through 1. Requests are routed round-robin across the groups at the current width, and each one goes to every engine of its group — a merged group executes an identical batch on all of its engines.

Which subsets may merge is set by `VLLM_DTP_MERGE_SETS`. The default is the aligned binary hierarchy (`0,1` / `2,3` / `0,1,2,3` at `dp_size=4`), which is linear in `dp_size` rather than the 246 subsets the full enumeration would give at `dp_size=8` — each one costs a communicator.

### Performance by merge width

Llama‑3.1‑8B, 4× H200, one deployment cycled `1 → 2 → 4 → 2 → 1`, each width measured against the **static layout it emulates**, identical client calls and server flags on both sides (`dtp_bench/perf_all.sh`):

| merge width | emulates | saturated tok/s (merged / static) | single‑stream TTFT (merged / static) | TPOT (merged / static) |
|---|---|---|---|---|
| 1 | DP4×TP1 | 73 744 / 71 943 — on par | 109 / 110 ms | 6.80 / 6.81 ms |
| 2 | DP2×TP2 | 61 900 / 67 853 — **−9%** | 73 / 72 ms | 9.00 / 7.53 ms |
| 4 | DP1×TP4 | 50 620 / 58 508 — **−13%** | 54 / 52 ms | 10.06 / 7.34 ms |

- **Unmerged costs nothing.** Width 1 matches static DP4×TP1 on every metric; the machinery is inert until a switch is asked for.
- **Merging captures the prefill win in full.** TTFT falls 109 → 73 → 54 ms as width goes 1 → 2 → 4, within 1–3% of the static layout at every width.
- **The cost lands on decode and grows with merge degree** — TPOT +19% at width 2, +37% at width 4 against static. It is the per‑layer all‑reduce running across engine boundaries, paced each step by the slowest engine. The penalty is smaller on larger models: Llama‑3‑70B at merge degree 2 pays ~1% (see the [tutorial](#what-you-get)).

**Known limits.**

- **Correctness across a width change is a fixed bug worth knowing about.** A width‑4 merge following a width‑2 merge in the same process used to degrade the model by +9.5% NLL, silently — no error, no hang, coherent output. One rank kept the previous episode's `block_size` while its cache view followed the new one, so it wrote a token at `b*32+t` and read it back at `b*64+t`, and attention returned zeros for blocks that width had never touched. The layout transition now runs from `worker_set_dtp_group_state` (`dtp_apply_layout` / `dtp_undo_layout`), which fires on every engine of the group at the step the flag flips whether or not it has work — the old site, `execute_model`'s `dtp_context`, was skipped on any step that scheduled no tokens, which is exactly what a hard‑preempt exit produces. Covered by `tests/v1/engine/test_dtp_subset_merge.py`.
- **The invariant to re-check after touching the KV layout**: `slot_mapping[i]` must equal `block_table[0,0] * block_size` for the block size the *cache view* uses.
- **Greedy text will not catch that class of bug.** Over a 4 → 2 → 1 → 2 → 4 cycle a fixed‑prompt probe agreed at every width while width 4 was degraded. Compare teacher‑forced NLL against the equivalent *static* layout instead (`dtp_bench/dtp_ppl.sh`, `static_ppl.sh`). Reference values on Llama‑3.1‑8B (267 tokens, 3 passages), each measured *after* an episode at a different width: width 1 → 1.991508 (static TP1 1.991508), width 2 → 1.993130 (static TP2 1.993130), width 4 → 1.990033 (static TP4 1.990033). Every merge reproduces its static equivalent to six decimals, and splitting back to DP is bit‑exact — a merge never damages the resident weights.
- **gpt‑oss is only verified at one merge width.** gpt‑oss‑120B (MXFP4 MoE, DP4×TP1 → TP4, `materialize`) completed 36 saturated and rate‑limited benchmarks with zero failed requests, so width 4 works. Width 2 has never been re-tested on it, and an earlier wedge — merged engines holding different batches, one running while the rest wait — was reproduced on gpt‑oss‑20B and is still unexplained. Treat anything other than a full merge on gpt‑oss as unverified.
- **The dynamic path requires `--enforce-eager`.** This tree cannot start with compilation enabled: a plain `DP=1 TP=4` server dies in engine init with a Dynamo error in stock `model_executor/parameter.py`. It reproduces with no DTP environment variables and the switching code inert, so it predates this work — but it does mean none of these numbers can be compared against a cudagraph‑enabled deployment.
- **Multi‑node subsets are not supported.** The per‑subgroup barrier rendezvous binds on `data_parallel_master_ip`, so a subset whose engine 0 is on another node cannot form. Multi‑node deployments fall back to the single full‑merge group.

---

## Switch methods

A mode switch has to decide what happens to the requests already in flight. Two methods are implemented:

| method | in‑flight requests | switch latency | cost |
|---|---|---|---|
| **`hard-preempt`** *(default)* | preempted immediately; their KV is freed and their generated tokens are folded into their prompt so they resume by re‑prefilling | sub‑second at any load | the preempted work is recomputed |
| `sequential` | the switch waits for the *cohort* of requests running when the switch request arrived — later arrivals are not waited for | as long as that cohort's slowest member | none |

**`hard-preempt` is the default** (`DEFAULT_SWITCH_METHOD` in `vllm/v1/engine/core_client.py`, override with `VLLM_DTP_SWITCH_METHOD`). `sequential` is not a safe default: it never fires at all on an idle deployment, and it has no working TP→DP path — its exit vote is not collective, so the admission barrier times out.

**Measured** on Llama‑3.1‑8B, DP4×TP1, 4× H200, 73 switches. A switch is timed in two parts because they have different causes: *drain* is `POST /dtp_switch` to the first engine transitioning, *cutover* is first engine to last.

| workload (concurrency × in/out) | drain, DP→TP | drain, TP→DP | cutover |
|---|---|---|---|
| idle | 0.19 s | 0.04 s | ≤10 ms |
| 4 × (256/64) | 0.34 s | 0.39 s | ≤10 ms |
| 64 × (256/64) | 0.29 s | 0.09 s | ≤10 ms |
| 4 × (4096/512) | 1.04 s | 3.13 s | ≤10 ms |
| 32 × (4096/512) | 0.74 s | 3.08 s | ≤10 ms |

- **Cutover is constant and workload‑independent.** Across all 69 successful switches the first engine and the last were ≤10 ms apart, median below the 5 ms sampling resolution. That is what the admission barrier and `dtp_switch_agreed` buy: every engine of a group flips on the same step whether the deployment is idle or saturated.
- **`hard-preempt` cost tracks context length, not concurrency.** 16× the concurrency (4 → 64 streams) changed nothing (0.34 → 0.29 s); 16× the context (256 → 4096) tripled it (0.34 → 1.04 s). It frees KV and folds generated tokens, so the cost is *how much KV there is*, not how many requests.
- **A switch is not a service outage.** A streaming canary held across each switch saw a worst‑case gap of 0.01–0.43 s (1.06 s in the heaviest case) — far less than the drain, because the deployment keeps serving in the old mode while draining. That is the number to put in an SLO, not the drain.

**An idle deployment does not switch until a request arrives.** In 12 idle switches nothing happened for a full 15 s window at zero traffic; each completed only once a request was injected to carry it (then 0.19 s / 0.02 s). A switch is carried *by* a request — inherent to the design, not a stall.

`sequential` was measured on the same rig, but its drain numbers are not comparable: the streaming canary was itself in the cohort, so those runs timed "wait for a 1500‑token request to finish" (8.9 s light, 16.1 s heavy) rather than the workload's own request lengths. What is clean is that it timed out 3/3 on an idle deployment and 1/1 in the TP→DP direction.

`hard-preempt` is not free:

- **Throughput.** Preempted work is re‑prefilled. Over 200 requests with a switch every ~20 (a deliberately pathological cadence), 31.9 s vs 10.2 s at 48 output tokens and 107.7 s vs 49.0 s at 512.
- **Greedy determinism.** Re‑prefilling changes batch composition, which changes all‑reduce order, which changes the floating‑point result. Distinct outputs for one fixed prompt across a run: 1 (`sequential`/48 tok), 2 (`hard-preempt`/48), 5 (`sequential`/512), 7 (`hard-preempt`/512). Divergence appears only in later paragraphs.

### Limitation: DP→TP `hard-preempt` defers, it does not cancel

A DP request lives on exactly one engine. A merged group can only execute requests that *every* engine holds, so a request preempted by a DP→TP switch cannot join the merged batch — it is set aside and resumes at the next TP→DP switch. If a deployment enters TP mode and never leaves, those requests wait indefinitely. The scheduler logs a `WARNING` naming them on entry to TP mode so this is visible rather than looking like a hang.

### Validating a change to the switch paths

`switch_test` flips the mode every `VLLM_DTP_SWITCH_PERIOD` requests, so a few hundred requests cover many switches in both directions — the shipped policies place their switch points thousands of requests apart, which is why these paths went unexercised for so long.

```bash
VLLM_DTP_WORK_MODE=switch_test \
VLLM_DTP_TO_TP_METHOD=hard-preempt \
VLLM_DTP_TO_DP_METHOD=hard-preempt \
VLLM_DTP_SWITCH_PERIOD=20 \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --data-parallel-size 2 --tensor-parallel-size 2 \
    --enforce-eager --gpu-memory-utilization 0.90 \
    --max-model-len 8192 --no-enable-prefix-caching --port 8000
```

Drive it with concurrent load and check that every request completes and none exceeds its `max_tokens`. Keep traffic flowing for the whole run: if the load stops while the engines are merged, deferred requests (above) have no switch to resume at and will look like a hang.

---

## Reproducing the numbers

Three measurement mistakes produced large, confident, and wrong results while this system was being benchmarked. They are easy to repeat:

- **Prefix caching favours the baselines.** Merged‑TP requests must skip the prefix cache (each engine's cache is populated independently and the KV is re‑blocked on merge, so a hit on one engine and a miss on another produces differently shaped batches and hangs the merged forward). A baseline with caching on therefore skips prefill work that Flying cannot. Always pass `--no-enable-prefix-caching` to **every** configuration being compared.
- **Under SLURM, `srun --overlap` steps get 1 CPU by default.** That alone halves throughput. Request cores explicitly for the server and client steps.
- **The first sweep after a server start is ~13% slow.** Discard it, or a cold baseline against a warm experiment will invent a win that is not there.

---

## Where the code lives

Flying Serving is implemented as focused changes inside the vLLM engine:

| area | file | what |
|---|---|---|
| weight sharding (materialize / view / reorder) | `vllm/model_executor/layers/linear.py` | per‑switch shard activation; in‑place column‑chunked reorder; dual‑mode column/row apply |
| dynamic TP communicator | `vllm/distributed/parallel_state.py` | one `_DTP` group per mergeable engine subset; `get_dtp_group()` |
| switch protocol + lock‑step barrier | `vllm/v1/engine/core.py` | DP↔TP transition, per‑group admission barrier, `dtp_switch_agreed()` |
| merged KV layout | `vllm/v1/worker/gpu_model_runner.py`, `vllm/v1/worker/gpu_worker.py` | `dtp_apply_layout()` / `dtp_undo_layout()`, driven from the state‑flip RPC |
| merge‑width control plane | `vllm/entrypoints/openai/api_server.py` | `/dtp_switch`, `/dtp_status` |
| DTP‑aware scheduling | `vllm/v1/core/sched/scheduler.py` | identical‑batch scheduling, `TP_execute_number`, `sequential` / `hard-preempt` switch methods |
| generation budget across a switch | `vllm/v1/request.py`, `vllm/v1/core/sched/utils.py` | tokens folded into the prompt by a preemption still count against `max_tokens` |
| cross‑engine sync primitives | `vllm/config/parallel.py` | count / state all‑reduces over the DP group |
| request routing | `vllm/v1/engine/core_client.py` | `work_mode` dispatch |

---

## Related work

Flying Serving builds directly on **vLLM** and is informed by a line of work on parallelism and latency/throughput trade‑offs in LLM serving:

- **vLLM** — Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023. The engine, scheduler, and PagedAttention KV cache this fork extends.
- **Megatron‑LM** — Shoeybi et al., *Megatron‑LM: Training Multi‑Billion Parameter Language Models Using Model Parallelism*, 2019. The tensor‑parallel layout the merged group reproduces.
- **Orca** — Yu et al., *Orca: A Distributed Serving System for Transformer‑Based Generative Models*, OSDI 2022. Iteration‑level (continuous) batching.
- **DistServe** — Zhong et al., *Disaggregating Prefill and Decoding for Goodput‑optimized LLM Serving*, OSDI 2024, and **Splitwise** — Patel et al., *Splitwise: Efficient Generative LLM Inference Using Phase Splitting*, ISCA 2024. Phase disaggregation as an alternative way to reconcile TTFT and TPOT; Flying Serving instead reconfigures parallelism in place.
- **LoongServe** — Wu et al., *LoongServe: Efficiently Serving Long‑Context Large Language Models with Elastic Sequence Parallelism*, SOSP 2024. The closest in spirit — elastic parallelism per request — applied to sequence parallelism rather than DP↔TP.
- **Flying Serving** — Gao et al., *Flying Serving: On‑the‑Fly Parallelism Switching for Large Language Model Serving*, 2026 ([arXiv:2602.22593](https://arxiv.org/abs/2602.22593)). The paper this repository implements.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{gao2026flyingservingontheflyparallelism,
      title={FLYING SERVING: On-the-Fly Parallelism Switching for Large Language Model Serving},
      author={Shouwei Gao and Junqi Yin and Feiyi Wang and Wenqian Dong},
      year={2026},
      eprint={2602.22593},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2602.22593},
}
```

---

## Acknowledgements & license

This project is a fork of [vLLM](https://github.com/vllm-project/vllm) and inherits its **Apache‑2.0** license (see [`LICENSE`](LICENSE)). All credit for the underlying serving engine goes to the vLLM team and contributors. The Flying Serving additions (dynamic DP↔TP switching, zero‑copy resharding, lock‑step scheduling) are released under the same license.
