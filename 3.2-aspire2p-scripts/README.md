# ASPIRE 2A+ reference scripts

PBS job scripts for the steps in [3.2 Qwen3 Application Notes - ASPIRE 2A+.md](../3.2%20Qwen3%20Application%20Notes%20-%20ASPIRE%202A+.md). Like Firmus's `reference-scripts/`, they are worked examples, not requirements.

| Script | Demonstrates | Firmus counterpart |
|---|---|---|
| `01-smoke-image.pbs` | Container starts, GPU visible, SGLang importable. | `01-smoke-image.sbatch` |
| `02-smoke-serve.pbs` | A single non-disaggregated server answers `/health`. | `02-smoke-serve.sbatch` |
| `03-smoke-serve-aiperf.pbs` | AIPerf completes two requests against a served instance. | `03-smoke-serve-aiperf.sbatch` |
| `04-pd-minimal-2node.pbs` | 1 prefill + 1 decode + router across two nodes, one GPU each. | `04-pd-minimal-2node.sbatch` |
| `09-pd-235b-fatp.pbs` | Qwen3-235B FP8 on 16 GPUs: 2 prefill TP=4 + 2 decode TP=4/EP=4, then the official AIPerf command at concurrency `N`. | `09-pd-235b-fatp-peak-bench.sbatch` |
| `official-aiperf.sh` | The official AIPerf command from the Task Rules, called by `09`. | |

Before submitting:

- Replace `<project-id>` with your project.
- Submit from the directory where you want the `.o` file and the run directory; `09` also expects `official-aiperf.sh` there: `qsub -v N=32 09-pd-235b-fatp.pbs`.
- For `09`, generate the HTTPS proxy file in `~/.config/enroot/environ.d/` first (one line in any job script, see the notes), so the AIPerf container can fetch the tokenizer.

Each script's PBS exit status is the verdict (0 = pass).
