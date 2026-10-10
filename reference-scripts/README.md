# Reference Scripts

These are worked, verified examples of PD (prefill/decode) disaggregated deployment
on the Firmus AI Cloud cluster (Slurm + enroot/pyxis), progressing from a minimal
smoke test to a full 235B, 16-GPU topology. Use them as a starting point, not as a
required template.

## These are examples, not requirements

The Task Rules do not mandate any of the choices demonstrated here:

- **Containers are not required.** These scripts use enroot/pyxis containers because
  that is the environment the organizing committee validated on. A self-built
  environment on shared storage is equally valid (see the Task Rules' Submission
  Guidelines).
- **SGLang is not required.** These scripts use SGLang because it is the stack the
  organizing committee used to validate the cluster and the benchmark harness.
  vLLM, TensorRT-LLM, and combinations of either with Dynamo are equally valid
  choices under the Task Rules' Optimization Guidelines.
- **These exact PD splits, TP/EP degrees, and GPU counts are not required.** They
  are working data points, not a recommendation. Part of the task is finding a
  better one.

## What each script demonstrates

| Script | Demonstrates |
|---|---|
| `01-smoke-image.sbatch` | Container starts, GPU visible, SGLang importable. The first thing to verify before anything else. |
| `02-smoke-serve.sbatch` | A single non-disaggregated server instance comes up and answers `/health`. |
| `03-smoke-serve-aiperf.sbatch` | A benchmarking client can reach a served instance and complete a request/response cycle. |
| `04-pd-minimal-2node.sbatch` | The smallest possible PD deployment: 1 prefill worker + 1 decode worker + router, across 2 nodes. |
| `05-pd-tp-variant-1node.sbatch` | Prefill and decode workers can use different tensor-parallel degrees (TP=4 prefill, TP=2 decode here), on a single node. |
| `06-pd-moe-example.sbatch` | The MoE-specific `--ep-size` flag, demonstrated on a small MoE model (Qwen1.5-MoE-A2.7B) before touching the full 235B model. |
| `07-pd-235b-8plus8.sbatch` | A full 235B, 16-GPU PD deployment: 4 prefill workers (TP=2) + 2 decode workers (TP=4, EP=4). This is the topology walked through in `3.1 Qwen3 Application Notes - Firmus AI Cloud.md`. |
| `08-pd-235b-12plus4-dynamo-match.sbatch` | The same 235B model with a different worker ratio (6 prefill + 1 decode), matching the split used by the public Dynamo Hopper disaggregated recipe, as a second reference point. |
| `09-pd-235b-fatp-peak-bench.sbatch` | A third split (2 wide prefill workers at TP=4 instead of 4 narrower ones), run with an AIPerf call shaped like the official benchmark (closed-loop, fixed ISL/OSL) but at a shortened duration and swept concurrency, for exploratory testing rather than the official scored re-run. |

## Prerequisites assumed by these scripts

- Container images already staged at `/scratch/enroot-images/sglang-harbor.sqsh`
  and `/scratch/enroot-images/aiperf.sqsh`.
- Model weights already staged under `/scratch/models/`, e.g.
  `/scratch/models/Qwen3-235B-A22B-Instruct-2507-FP8`.
- A Slurm cluster with enroot/pyxis and InfiniBand between nodes.

## Pass/fail convention

Every script's Slurm exit code is the verdict (0 = pass). None of them wrap the
verification logic in shell tricks that could hide a failure. If a script fails,
check the job's `.out`/`.err` log for the specific step that failed.

## Suggested order

Run `01` through `03` first to confirm the basic container/GPU/serving/client path
works on your own environment, before attempting any PD topology. From there, `04`
and `05` are the simplest PD deployments to adapt. `06` through `09` are all at
model scales and GPU counts closer to the actual task, and are more useful as
reference points once your own deployment is close to working.
