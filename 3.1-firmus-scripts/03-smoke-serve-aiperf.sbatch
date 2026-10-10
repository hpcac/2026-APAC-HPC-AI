#!/bin/bash
#SBATCH -J smoke-serve-aiperf
#SBATCH -p hpcai
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH --gres=gpu:1
#SBATCH --propagate=NONE
#SBATCH -o runs/%j.out
#SBATCH -e runs/%j.err
#
# See README.md in this folder for how this script fits into the task.
#
# One node: sglang serve, then 2 aiperf requests. Pass = aiperf exits 0.
set -euo pipefail

SGLANG=/scratch/enroot-images/sglang-harbor.sqsh
AIPERF=/scratch/enroot-images/aiperf.sqsh
MODEL=/scratch/models/Qwen3-0.6B
PORT=30000
READY_SECS=180
ROOT=${SLURM_SUBMIT_DIR}
HOME_BIND=${ROOT}/home
MOUNTS="/scratch:/scratch,${HOME_BIND}:${HOME},${ROOT}:${ROOT}"
ART=${ROOT}/runs/${SLURM_JOB_ID}

echo "[$(date -Iseconds)] job=${SLURM_JOB_ID} user=$(whoami) host=$(hostname)"
echo "sglang=${SGLANG} aiperf=${AIPERF} model=${MODEL} port=${PORT}"
test -r "${SGLANG}"
test -r "${AIPERF}"
test -f "${MODEL}/config.json"
mkdir -p "${ART}"

srun --overlap -n1 --gres=gpu:1 \
  --container-image="${SGLANG}" --container-mounts="${MOUNTS}" \
  python3 -m sglang.launch_server \
    --model-path "${MODEL}" --served-model-name qwen \
    --host 0.0.0.0 --port "${PORT}" \
    --tp-size 1 --mem-fraction-static 0.4 \
    --context-length 4096 --trust-remote-code >&2 &

timeout "${READY_SECS}" curl -sf \
  --retry-connrefused --retry-all-errors \
  --retry-delay 2 --retry 100 --max-time 2 \
  -o /dev/null "http://127.0.0.1:${PORT}/health"
echo "READY after ${SECONDS}s"

srun --overlap -n1 \
  --container-image="${AIPERF}" --container-mounts="${MOUNTS}" \
  aiperf profile \
    --model qwen --tokenizer "${MODEL}" \
    --url "http://127.0.0.1:${PORT}" --endpoint-type chat --streaming \
    --request-count 2 --concurrency 1 --num-dataset-entries 2 \
    --synthetic-input-tokens-mean 16 --output-tokens-mean 8 \
    --artifact-dir "${ART}"
echo "AIPERF ok"
