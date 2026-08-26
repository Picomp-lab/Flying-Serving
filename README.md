# Flying Serving

**Dynamic DP↔TP switching for LLM serving — built on [vLLM](https://github.com/vllm-project/vllm).**

Flying Serving lets a single deployment **switch its parallelism strategy at
runtime**, per wave of requests, *without reloading weights*. Engines run as
**Data‑Parallel (DP)** replicas for high aggregate throughput, and transparently
**merge into a larger Tensor‑Parallel (TP)** group when a request would benefit
from it (e.g. long‑context, latency‑sensitive prefill), then split back. You get
**TP‑class latency when you need it and DP‑class throughput when you don't**, on
the same GPUs and the same resident weights.

> This repository is a fork of vLLM. The upstream README is preserved as
> [`README_vLLM.md`](README_vLLM.md).

---

## Why

A serving deployment normally commits to one parallelism layout:

| | First‑token latency (TTFT) | Per‑token latency (TPOT) | Throughput | KV cache / request |
|---|---|---|---|---|
| **Tensor Parallel (TP)** | **low** (prefill split across GPUs) | **low** (compute split) | lower | smaller (sharded) |
| **Data Parallel (DP)** | high (prefill on one GPU) | high under load | **high** (independent replicas) | larger (full replica) |

The gap widens with model size and context length. On Llama‑3‑70B at
4096‑token prompts, static TP has **~2× lower TTFT** and **~2.5× lower TPOT** than
static DP — but static DP sustains more concurrent requests. Picking one strategy
statically leaves performance on the table whenever the workload mix changes.

**Flying Serving removes the static choice.** It keeps the throughput of DP at
steady state and borrows the latency of TP on demand, by merging and splitting
engines live.

---

## How it works

1. **Resident weights, zero reload.** Each engine loads the full model once (DP
   layout). When engines merge, each one's TP shard is *activated in place* from
   the already‑resident replica — no second copy, no reload. This is the key to
   switching in milliseconds instead of re‑initializing the model.

2. **A dynamic TP communicator (`_DTP`).** At merge time the participating
   engines' GPUs form a single TP group and run the model's per‑layer all‑reduces
   across engine boundaries, exactly like native TP.

3. **Lock‑step scheduling.** The merged engines must execute identical batches.
   A lightweight cross‑engine barrier aligns admission so every engine builds the
   same TP batch each step.

4. **Three weight‑sharding strategies** (`VLLM_DTP_SHARDING`):

   | mode | GEMM | extra memory | decode speed | use when |
   |---|---|---|---|---|
   | `materialize` | one fused GEMM on a cached contiguous shard | + one shard copy (resident) | native | the shard copy fits |
   | `view` | one GEMM per weight slice (zero‑copy) | **none** | slower (multi‑GEMM) | memory is tight |
   | **`reorder`** *(recommended)* | one fused GEMM on an **in‑place reordered** shard | **none** | **native** | always, for unquantized models |

   `reorder` is the best of both: on the first forward after a merge it permutes
   the resident weight rows **in place** (column‑chunked, ~tens of MB transient)
   so each engine's shard becomes one contiguous block — a single native‑speed
   cuBLAS GEMM — with **zero extra resident memory**. Rows are restored on split,
   so pure‑DP execution is byte‑for‑byte unchanged. On Llama‑3‑70B this lets the
   merged TP run at near‑native decode speed *and* fit where `materialize` OOMs.

> Flying Serving implements the system described in the paper
> [*Flying Serving: On‑the‑Fly Parallelism Switching for Large Language Model
> Serving*](https://arxiv.org/abs/2602.22593) (Gao et al., 2026). See
> [Citation](#citation).

---

## Install

Flying Serving is a fork of vLLM; install it the same way (from source, since the
dynamic kernels run in eager mode):

```bash
git clone https://github.com/Picomp-lab/Flying-Serving.git
cd Flying-Serving
pip install -e .            # or: VLLM_USE_PRECOMPILED=1 pip install -e .
```

Requirements match upstream vLLM (CUDA GPUs with NVLink recommended for the
cross‑engine all‑reduce). The examples below use 4× H‑class GPUs.

---

## Tutorial: Llama‑3‑70B on 4 GPUs

We run `meta-llama/Meta-Llama-3-70B-Instruct` on 4 GPUs three ways and compare.
All commands use `--enforce-eager` (the dynamic switch is not compatible with CUDA
graph capture today).

### 1. Static Data Parallel (throughput baseline)

Two replicas, each Tensor‑Parallel‑2 (70B does not fit on one GPU):

```bash
vllm serve meta-llama/Meta-Llama-3-70B-Instruct \
    --data-parallel-size 2 --tensor-parallel-size 2 \
    --enforce-eager --gpu-memory-utilization 0.90 \
    --max-model-len 8192 --port 8000
```

### 2. Static Tensor Parallel (latency baseline)

One TP‑4 group across all 4 GPUs:

```bash
vllm serve meta-llama/Meta-Llama-3-70B-Instruct \
    --tensor-parallel-size 4 \
    --enforce-eager --gpu-memory-utilization 0.90 \
    --max-model-len 8192 --port 8000
```

### 3. Flying Serving (dynamic DP↔TP) ⭐

Launch as 2 DP engines (each TP‑2) **in dynamic mode**, with the `reorder`
sharding strategy. Engines run as DP replicas and merge into TP‑4 on demand:

```bash
VLLM_DTP_WORK_MODE=tp_mode \
VLLM_DTP_SHARDING=reorder \
vllm serve meta-llama/Meta-Llama-3-70B-Instruct \
    --data-parallel-size 2 --tensor-parallel-size 2 \
    --enforce-eager --gpu-memory-utilization 0.90 \
    --max-model-len 8192 --port 8000
```

Send a request (OpenAI‑compatible API, same for all three):

```bash
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{
  "model": "meta-llama/Meta-Llama-3-70B-Instruct",
  "prompt": "Explain tensor parallelism in one paragraph.",
  "max_tokens": 256
}'
```

### What you get (online serving, in=4096 / out=256, 4× H200)

Mean TTFT / P99 TTFT / median TPOT (ms), Poisson arrivals:

| req/s | Static DP | Static TP4 | **Flying (reorder)** |
|---|---|---|---|
| 1 | 670 / 2329 / 37.9 | 356 / 1107 / 19.7 | **373 / 1008 / 22.4** |
| 2 | 820 / 2390 / **78.4** | 395 / 1182 / 29.3 | **424 / 1215 / 31.5** |
| 4 | 2662 / 4313 / 65.0 | 1133 / 2204 / 54.2 | **1133 / 2262 / 55.4** |

**Flying Serving tracks static TP within a few percent at these loads, while
static DP collapses on long context** — up to **2.3× worse P99 TTFT** and **2.5×
worse TPOT**. And unlike static TP, the dynamic deployment can still fall back to
DP‑mode throughput when prompts are short and the queue is deep.

**This holds while neither configuration is saturated, which is the regime the
table covers.** Push past it and the picture changes: on Llama‑3.1‑8B (same
in=4096 / out=256, 4× H200) Flying tops out around **76k tok/s** (79k at
`VLLM_DTP_SYNC_EVERY=1`) while static TP‑4 is still unsaturated above **177k
tok/s**. The merged path couples two engine loops at every layer's all‑reduce, so
its ceiling is set by the slower of the two engines each step. At rate 12, where
neither is saturated, the two are within 2%. Size the deployment for the
unsaturated regime, or keep enough DP headroom that the merged episodes stay
short.

On the 8B class the residual gap to native TP at decode is the design's
irreducible cost (two engine loops coupled at the per‑layer all‑reduce), ~1–2 ms
of TPOT; `reorder` removes the GEMM/memory penalty entirely (validated
byte‑identical to a single fused GEMM, and runs Llama‑70B at 0.92 GPU utilization
with no OOM where `materialize` cannot).

---

## Configuration

All knobs are environment variables, read at startup / per step.

| variable | values | default | effect |
|---|---|---|---|
| `VLLM_DTP_WORK_MODE` | `original` \| `tp_mode` \| `switch_test` | `original` | `tp_mode` enables dynamic DP↔TP merging. `original` = stock vLLM behavior. `switch_test` is a validation harness (see [Switch methods](#switch-methods)). |
| `VLLM_DTP_SHARDING` | `materialize` \| `view` \| `reorder` | `materialize` | Merged‑TP weight strategy (see table above). `reorder` recommended for unquantized models. |
| `VLLM_DTP_SYNC_EVERY` | integer ≥ 1 | `1` | Cadence of the cross‑engine admission barrier. **Lower → lower TTFT**; **higher → lower TPOT** (the barrier cost is amortized over more steps). |
| `VLLM_DTP_TO_TP_METHOD` | `sequential` \| `hard-preempt` | `sequential` | How a DP→TP switch treats in‑flight DP work. `switch_test` mode only. |
| `VLLM_DTP_TO_DP_METHOD` | `sequential` \| `hard-preempt` | `hard-preempt` | Same, for TP→DP. `switch_test` mode only. |
| `VLLM_DTP_SWITCH_PERIOD` | integer ≥ 1 | `40` | Requests between mode flips. `switch_test` mode only. |

**Tuning guidance.** The barrier is the one piece of *extra* cross‑engine
communication Flying Serving adds (the per‑layer all‑reduce is intrinsic to TP).
`VLLM_DTP_SYNC_EVERY` trades first‑token against per‑token latency. Measured on
Llama‑3.1‑8B at low load:

| `VLLM_DTP_SYNC_EVERY` | TPOT | TTFT |
|---|---|---|
| **1** *(default)* | 9.15 ms | **28.8 ms** |
| 2 | 8.46 ms | 33.0 ms |
| 4 | **8.31 ms** | 41.7 ms |

`=1` is the default: it is the only setting with no admission stall, and the
~0.7 ms of TPOT it costs is the smaller half of the trade against the ~4 ms of
TTFT that `=2` adds. Raise it for decode‑heavy workloads where TPOT dominates.

Quantized models (e.g. MoE / mxfp4) automatically fall back to the
`materialize`/`view` path for layers `reorder` does not support.

`VLLM_DTP_WORK_MODE` also accepts several experimental routing policies used for
the paper's experiments — `traffic_load`, `request_length`, `custom`,
`manipulated` — which switch on hard‑coded request counts. They are not intended
for serving; the three in the table above are.

> **Removed:** `VLLM_DTP_ASYNC_BARRIER` (overlap the barrier with the forward,
> consuming a 1‑step‑stale result). It was an unfinished experiment that killed
> the engine under load, and it put a second code path through the barrier — the
> one piece of the system where a divergence between engines deadlocks. The
> synchronous barrier is now the only path.

---

## Switch methods

A mode switch has to decide what happens to the requests already in flight. Two
methods are implemented, chosen per direction:

| method | in‑flight requests | switch latency | cost |
|---|---|---|---|
| **`sequential`** | drained — the switch waits for them to finish | scales with how long they run | none |
| **`hard-preempt`** | preempted immediately; their KV is freed and their generated tokens are folded into their prompt so they resume by re‑prefilling | near‑immediate | the preempted work is recomputed |

Measured on Llama‑3.1‑8B, DP2×TP2 on 4× H200, time from the switch request being
dispatched to TP mode being live (1 s log resolution):

| DP→TP method | 512‑token requests in flight | 48‑token requests in flight |
|---|---|---|
| `hard-preempt` | 0, 0, 0, 0, 1, 0 s | ≤1 s |
| `sequential` | 3, 3, 2, 1 s | ≤1 s |

The gap *is* the drain: with short requests the two are indistinguishable, and
`sequential` only falls behind once the in‑flight work is long — which is exactly
the case a latency‑driven switch cares about.

`hard-preempt` is not free:

- **Throughput.** Preempted work is re‑prefilled. Over 200 requests with a switch
  every ~20 (a deliberately pathological cadence), 31.9 s vs 10.2 s at 48 tokens
  and 107.7 s vs 49.0 s at 512.
- **Greedy determinism.** Re‑prefilling changes batch composition, which changes
  all‑reduce order, which changes the floating‑point result. Distinct outputs for
  one fixed prompt across a run: 1 (`sequential`/48 tok), 2 (`hard-preempt`/48),
  5 (`sequential`/512), 7 (`hard-preempt`/512). Divergence appears only in later
  paragraphs.

### Limitation: DP→TP `hard-preempt` defers, it does not cancel

A DP request lives on exactly one engine. A merged group can only execute
requests that *every* engine holds, so a request preempted by a DP→TP switch
cannot join the merged batch — it is set aside and resumes at the next TP→DP
switch. If a deployment enters TP mode and never leaves, those requests wait
indefinitely. The scheduler logs a `WARNING` naming them on entry to TP mode so
this is visible rather than looking like a hang.

### Validating a change to the switch paths

`switch_test` flips the mode every `VLLM_DTP_SWITCH_PERIOD` requests, so a few
hundred requests cover many switches in both directions — the shipped policies
place their switch points thousands of requests apart, which is why these paths
went unexercised for so long.

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

Drive it with concurrent load and check that every request completes and none
exceeds its `max_tokens`. Keep traffic flowing for the whole run: if the load
stops while the engines are merged, deferred requests (above) have no switch to
resume at and will look like a hang.

---

## Reproducing the numbers

Three measurement mistakes produced large, confident, and wrong results while
this system was being benchmarked. They are easy to repeat:

- **Prefix caching favours the baselines.** Merged‑TP requests must skip the
  prefix cache (each engine's cache is populated independently and the KV is
  re‑blocked on merge, so a hit on one engine and a miss on another produces
  differently shaped batches and hangs the merged forward). A baseline with
  caching on therefore skips prefill work that Flying cannot. Always pass
  `--no-enable-prefix-caching` to **every** configuration being compared.
- **Under SLURM, `srun --overlap` steps get 1 CPU by default.** That alone
  halves throughput. Request cores explicitly for the server and client steps.
- **The first sweep after a server start is ~13% slow.** Discard it, or a cold
  baseline against a warm experiment will invent a win that is not there.

---

## Where the code lives

Flying Serving is implemented as focused changes inside the vLLM engine:

| area | file | what |
|---|---|---|
| weight sharding (materialize / view / reorder) | `vllm/model_executor/layers/linear.py` | per‑switch shard activation; in‑place column‑chunked reorder; dual‑mode column/row apply |
| dynamic TP communicator | `vllm/distributed/parallel_state.py` | `_DTP` group creation and `get_dtp_group()` |
| switch protocol + lock‑step barrier | `vllm/v1/engine/core.py` | DP↔TP transition, admission barrier |
| DTP‑aware scheduling | `vllm/v1/core/sched/scheduler.py` | identical‑batch scheduling, `TP_execute_number`, `sequential` / `hard-preempt` switch methods |
| generation budget across a switch | `vllm/v1/request.py`, `vllm/v1/core/sched/utils.py` | tokens folded into the prompt by a preemption still count against `max_tokens` |
| cross‑engine sync primitives | `vllm/config/parallel.py` | count / state all‑reduces over the DP group |
| request routing | `vllm/v1/engine/core_client.py` | `work_mode` dispatch |

---

## Related work

Flying Serving builds directly on **vLLM** and is informed by a line of work on
parallelism and latency/throughput trade‑offs in LLM serving:

- **vLLM** — Kwon et al., *Efficient Memory Management for Large Language Model
  Serving with PagedAttention*, SOSP 2023. The engine, scheduler, and
  PagedAttention KV cache this fork extends.
- **Megatron‑LM** — Shoeybi et al., *Megatron‑LM: Training Multi‑Billion Parameter
  Language Models Using Model Parallelism*, 2019. The tensor‑parallel layout the
  merged group reproduces.
- **Orca** — Yu et al., *Orca: A Distributed Serving System for Transformer‑Based
  Generative Models*, OSDI 2022. Iteration‑level (continuous) batching.
- **DistServe** — Zhong et al., *Disaggregating Prefill and Decoding for
  Goodput‑optimized LLM Serving*, OSDI 2024, and **Splitwise** — Patel et al.,
  *Splitwise: Efficient Generative LLM Inference Using Phase Splitting*, ISCA 2024.
  Phase disaggregation as an alternative way to reconcile TTFT and TPOT; Flying
  Serving instead reconfigures parallelism in place.
- **LoongServe** — Wu et al., *LoongServe: Efficiently Serving Long‑Context Large
  Language Models with Elastic Sequence Parallelism*, SOSP 2024. The closest in
  spirit — elastic parallelism per request — applied to sequence parallelism
  rather than DP↔TP.
- **Flying Serving** — Gao et al., *Flying Serving: On‑the‑Fly Parallelism
  Switching for Large Language Model Serving*, 2026
  ([arXiv:2602.22593](https://arxiv.org/abs/2602.22593)). The paper this
  repository implements.

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

This project is a fork of [vLLM](https://github.com/vllm-project/vllm) and inherits
its **Apache‑2.0** license (see [`LICENSE`](LICENSE)). All credit for the
underlying serving engine goes to the vLLM team and contributors. The Flying
Serving additions (dynamic DP↔TP switching, zero‑copy resharding, lock‑step
scheduling) are released under the same license.
