# Firmus reference runs: 09 layout, N=48

Two organiser runs of the 09 layout on Firmus AI Cloud (2×8 H200, nodes g1 + g4): 2 prefill workers at TP=4 on one node, 2 decode workers at TP=4, EP=4 on the other, SGLang PD with NIXL, router on port 8000. The deployment is `3.1-firmus-scripts/09-pd-235b-fatp-peak-bench.sbatch`.

| Run | Date (UTC+8) | Benchmark | `good_request_fraction` | Good output tok/s | Requests (good / total) |
|---|---|---|---:|---:|---:|
| `09-12579/` | 2026-08-22 17:42 | official settings; tokenizer given as the local model path, no `--workers-max` | 0.974 | 1,834.4 | 1,153 / 1,184 |
| `09-25013/` | 2026-10-20 18:23 | official command, unmodified (`official-aiperf.sh`) | 0.973 | 1,832.4 | 1,134 / 1,165 |

Good output tok/s = `goodput` × 200, from `profile_export_aiperf.json`. Both runs used AIPerf 0.12.0; the full client configuration of each run is recorded in `input_config` inside that file.

25013 is the Firmus baseline in `5. Common Baselines.md`. 12579 is the same layout two months earlier, kept for comparison.

## What is in each folder

- `profile_export_aiperf.json` / `.csv`, `profile_export_console.txt`, `phases/` (warm-up and profiling separately), `logs/aiperf.log`: AIPerf output.
- `prefill-*.log`, `decode-*.log`, `router.log`, `worker.sh`: the serving side.
- `<job>.out`, `<job>.err`: Slurm output.
- `09-25013/firmus-09-official.sbatch`, `09-25013/official-aiperf.sh`: the job script of 25013 (comments translated to English). The job script of 12579 was not kept.

Not included: AIPerf's `inputs.json` (228 MB; regenerated from `--random-seed 2026`) and the per-request `profile_export.jsonl`.
