#!/bin/bash
#SBATCH -J smoke-serve
#SBATCH -p hpcai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:10:00
#SBATCH --gres=gpu:1
#SBATCH --propagate=NONE
#SBATCH -o runs/%j.out
#SBATCH -e runs/%j.err
#
# See README.md in this folder for how this script fits into the task.
#
# Serve Qwen3-0.6B just long enough to answer /health, then exit.
# Pass = /health 200 before READY_SECS. Exit code is the verdict.
set -euo pipefail

IMAGE=/scratch/enroot-images/sglang-harbor.sqsh
MODEL=/scratch/models/Qwen3-0.6B
PORT=30000
READY_SECS=180
ROOT=${SLURM_SUBMIT_DIR}
HOME_BIND=${ROOT}/home

export MODEL PORT READY_SECS

echo "[$(date -Iseconds)] job=${SLURM_JOB_ID} user=$(whoami) host=$(hostname)"
echo "image=${IMAGE} model=${MODEL} port=${PORT}"
test -r "${IMAGE}"
test -f "${MODEL}/config.json"
test -f "${MODEL}/model.safetensors"

srun --container-image="${IMAGE}" \
  --container-mounts="/scratch:/scratch,${HOME_BIND}:${HOME}" \
  bash -c 'set -euo pipefail
    echo "in-container host=$(hostname) user=$(whoami)"
    python3 -m sglang.launch_server \
      --model-path "${MODEL}" \
      --host 127.0.0.1 --port "${PORT}" \
      --tp-size 1 --mem-fraction-static 0.4 \
      --context-length 4096 --trust-remote-code >&2 &
    timeout "${READY_SECS}" curl -sf \
      --retry-connrefused --retry-all-errors \
      --retry-delay 2 --retry 100 --max-time 2 \
      -o /dev/null "http://127.0.0.1:${PORT}/health"
    echo "READY after ${SECONDS}s"
  '
