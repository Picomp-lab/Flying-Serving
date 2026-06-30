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
VLLM_DTP_SYNC_EVERY=2 \
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

**Flying Serving tracks static TP within a few percent across all loads, while
static DP collapses on long context** — up to **2.3× worse P99 TTFT** and **2.5×
worse TPOT**. And unlike static TP, the dynamic deployment can still fall back to
DP‑mode throughput when prompts are short and the queue is deep.

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
| `VLLM_DTP_WORK_MODE` | `original` \| `tp_mode` | `original` | `tp_mode` enables dynamic DP↔TP merging. `original` = stock vLLM behavior. |
| `VLLM_DTP_SHARDING` | `materialize` \| `view` \| `reorder` | `materialize` | Merged‑TP weight strategy (see table above). `reorder` recommended for unquantized models. |
| `VLLM_DTP_SYNC_EVERY` | integer ≥ 1 | `2` | Cadence of the cross‑engine admission barrier. **Lower → lower TTFT**; **higher → lower TPOT** (the barrier cost is amortized over more steps). |
| `VLLM_DTP_ASYNC_BARRIER` | `0` \| `1` | `0` | Overlap the admission barrier with the model forward (1‑step‑stale). Lowers TPOT and raises throughput at the cost of higher TTFT. Experimental. |

**Tuning guidance.** The barrier is the one piece of *extra* cross‑engine
communication Flying Serving adds (the per‑layer all‑reduce is intrinsic to TP).
`VLLM_DTP_SYNC_EVERY` trades first‑token vs. per‑token latency: `=1` for the
lowest TTFT, `=4` for the lowest TPOT on decode‑heavy workloads, `=2` is a good
default. Quantized models (e.g. MoE / mxfp4) automatically fall back to the
`materialize`/`view` path for layers `reorder` does not support.

---

## Where the code lives

Flying Serving is implemented as focused changes inside the vLLM engine:

| area | file | what |
|---|---|---|
| weight sharding (materialize / view / reorder) | `vllm/model_executor/layers/linear.py` | per‑switch shard activation; in‑place column‑chunked reorder; dual‑mode column/row apply |
| dynamic TP communicator | `vllm/distributed/parallel_state.py` | `_DTP` group creation and `get_dtp_group()` |
| switch protocol + lock‑step barrier | `vllm/v1/engine/core.py` | DP↔TP transition, admission barrier (sync / async) |
| DTP‑aware scheduling | `vllm/v1/core/sched/scheduler.py` | identical‑batch scheduling, `TP_execute_number` |
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
