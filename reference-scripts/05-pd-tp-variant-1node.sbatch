#!/bin/bash
#SBATCH -J p1-tp4-d1-tp2-1n
#SBATCH -p hpcai
#SBATCH -N 1
#SBATCH -t 00:15:00
#SBATCH --gres=gpu:6
#SBATCH --propagate=NONE
#SBATCH -o runs/%j.out
#SBATCH -e runs/%j.err
#
# See README.md in this folder for how this script fits into the task.
#
# One node: prefill TP=4 + decode TP=2 + router, then 2 aiperf requests.
# Pass = aiperf exits 0. Two GPU steps use --exact so they get different GPUs.
set -euo pipefail

SGLANG=/scratch/enroot-images/sglang-harbor.sqsh
AIPERF=/scratch/enroot-images/aiperf.sqsh
MODEL=/scratch/models/Qwen3-0.6B
P_TP=4
D_TP=2
P_PORT=30100
D_PORT=30200
BOOT=9000
R_PORT=8000
READY_SECS=300
ROOT=${SLURM_SUBMIT_DIR}
ART=${ROOT}/runs/${SLURM_JOB_ID}

mounts() {
  mkdir -p "${ROOT}/home/$1"
  echo "/scratch:/scratch,${ROOT}/home/$1:${HOME},${ROOT}:${ROOT}"
}

wait_http() {
  timeout "${READY_SECS}" curl -sf \
    --retry-connrefused --retry-all-errors \
    --retry-delay 2 --retry 100 --max-time 2 \
    -o /dev/null "$1"
}

echo "[$(date -Iseconds)] job=${SLURM_JOB_ID} user=$(whoami) host=$(hostname)"
echo "sglang=${SGLANG} model=${MODEL} tp=${P_TP}+${D_TP} UCX_POSIX_USE_PROC_LINK=n"
test -r "${SGLANG}" && test -r "${AIPERF}" && test -f "${MODEL}/config.json"
mkdir -p "${ART}"

cat >"${ART}/worker.sh" <<'EOF'
set -euo pipefail
role=$1
port=$2
model=$3
tp=$4
shift 4
UCX_IB=$(for d in /sys/class/infiniband/*; do
  [ "$(cat "$d/ports/1/link_layer" 2>/dev/null)" = InfiniBand ] || continue
  [ "$(cat "$d/ports/1/state" 2>/dev/null)" = "4: ACTIVE" ] || continue
  echo "$(basename "$d"):1"
done | paste -sd,)
[ -n "$UCX_IB" ] || { echo "FATAL: no active IB on $(hostname)"; exit 1; }
export UCX_NET_DEVICES=$UCX_IB
export UCX_POSIX_USE_PROC_LINK=n
export TRITON_CACHE_DIR=/tmp/sglang-$role/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/sglang-$role/inductor
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
echo "$role host=$(hostname) ib=$UCX_IB gpu=$CUDA_VISIBLE_DEVICES tp=$tp posix_proc_link=$UCX_POSIX_USE_PROC_LINK"
exec python3 -m sglang.launch_server \
  --model-path "$model" --served-model-name qwen --trust-remote-code \
  --context-length 4096 --tp-size "$tp" --mem-fraction-static 0.4 \
  --disaggregation-mode "$role" --disaggregation-transfer-backend nixl \
  "$@" --host 0.0.0.0 --port "$port"
EOF

srun --exact -n1 --gres=gpu:${P_TP} --cpus-per-task=32 \
  --container-image="${SGLANG}" --container-mounts="$(mounts prefill)" \
  bash "${ART}/worker.sh" prefill "${P_PORT}" "${MODEL}" "${P_TP}" \
    --disaggregation-bootstrap-port "${BOOT}" \
  >"${ART}/prefill.log" 2>&1 &
P_PID=$!

srun --exact -n1 --gres=gpu:${D_TP} --cpus-per-task=16 \
  --container-image="${SGLANG}" --container-mounts="$(mounts decode)" \
  bash "${ART}/worker.sh" decode "${D_PORT}" "${MODEL}" "${D_TP}" \
    --max-running-requests 256 \
  >"${ART}/decode.log" 2>&1 &
D_PID=$!

wait_http "http://127.0.0.1:${P_PORT}/health"
echo "prefill ready"
wait_http "http://127.0.0.1:${D_PORT}/health"
echo "decode ready"

srun --overlap -n1 --cpus-per-task=2 \
  --container-image="${SGLANG}" --container-mounts="$(mounts router)" \
  python3 -m sglang_router.launch_router --pd-disaggregation \
    --prefill "http://127.0.0.1:${P_PORT}" "${BOOT}" \
    --decode "http://127.0.0.1:${D_PORT}" \
    --host 0.0.0.0 --port "${R_PORT}" \
  >"${ART}/router.log" 2>&1 &
R_PID=$!

wait_http "http://127.0.0.1:${R_PORT}/health"
echo "READY after ${SECONDS}s"

srun --overlap -n1 \
  --container-image="${AIPERF}" --container-mounts="$(mounts aiperf)" \
  aiperf profile \
    --model qwen --tokenizer "${MODEL}" \
    --url "http://127.0.0.1:${R_PORT}" --endpoint-type chat --streaming \
    --request-count 2 --concurrency 1 --num-dataset-entries 2 \
    --synthetic-input-tokens-mean 16 --output-tokens-mean 8 \
    --artifact-dir "${ART}"
echo "AIPERF ok"

set +e
kill "${P_PID}" "${D_PID}" "${R_PID}"
wait "${P_PID}" "${D_PID}" "${R_PID}"
set -e
