#!/bin/bash
#SBATCH -J p1-tp1-d1-tp1-2n
#SBATCH -p hpcai
#SBATCH -N 2
#SBATCH -t 00:15:00
#SBATCH --gres=gpu:1
#SBATCH --propagate=NONE
#SBATCH -o runs/%j.out
#SBATCH -e runs/%j.err
#
# See README.md in this folder for how this script fits into the task.
#
# 1 prefill + 1 decode + router, then 2 aiperf requests. Pass = aiperf exits 0.
set -euo pipefail

SGLANG=/scratch/enroot-images/sglang-harbor.sqsh
AIPERF=/scratch/enroot-images/aiperf.sqsh
MODEL=/scratch/models/Qwen3-0.6B
P_PORT=30100
D_PORT=30200
BOOT=9000
R_PORT=8000
READY_SECS=180
ROOT=${SLURM_SUBMIT_DIR}
ART=${ROOT}/runs/${SLURM_JOB_ID}

NODES=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
HEAD=${NODES[0]}
TAIL=${NODES[1]}
ip_of() { getent ahostsv4 "$1" | awk '{print $1; exit}'; }
HEAD_IP=$(ip_of "${HEAD}")
TAIL_IP=$(ip_of "${TAIL}")

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

echo "[$(date -Iseconds)] job=${SLURM_JOB_ID} user=$(whoami)"
echo "head=${HEAD}(${HEAD_IP}) tail=${TAIL}(${TAIL_IP}) model=${MODEL}"
test -r "${SGLANG}" && test -r "${AIPERF}" && test -f "${MODEL}/config.json"
mkdir -p "${ART}" "${ROOT}/home/prefill" "${ROOT}/home/decode" "${ROOT}/home/router"

# IB device list must be computed on the worker node (the two nodes differ).
cat >"${ART}/worker.sh" <<'EOF'
set -euo pipefail
role=$1
port=$2
model=$3
shift 3
UCX_IB=$(for d in /sys/class/infiniband/*; do
  [ "$(cat "$d/ports/1/link_layer" 2>/dev/null)" = InfiniBand ] || continue
  [ "$(cat "$d/ports/1/state" 2>/dev/null)" = "4: ACTIVE" ] || continue
  echo "$(basename "$d"):1"
done | paste -sd,)
[ -n "$UCX_IB" ] || { echo "FATAL: no active IB on $(hostname)"; exit 1; }
export UCX_NET_DEVICES=$UCX_IB
echo "$role host=$(hostname) ib=$UCX_IB"
exec python3 -m sglang.launch_server \
  --model-path "$model" --served-model-name qwen --trust-remote-code \
  --context-length 4096 --tp-size 1 --mem-fraction-static 0.4 \
  --disaggregation-mode "$role" --disaggregation-transfer-backend nixl \
  "$@" --host 0.0.0.0 --port "$port"
EOF

srun --exact -N1 -n1 -w "${HEAD}" --gres=gpu:1 --cpus-per-task=8 \
  --container-image="${SGLANG}" --container-mounts="$(mounts prefill)" \
  bash "${ART}/worker.sh" prefill "${P_PORT}" "${MODEL}" --disaggregation-bootstrap-port "${BOOT}" \
  >"${ART}/prefill.log" 2>&1 &
P_PID=$!

srun --exact -N1 -n1 -w "${TAIL}" --gres=gpu:1 --cpus-per-task=8 \
  --container-image="${SGLANG}" --container-mounts="$(mounts decode)" \
  bash "${ART}/worker.sh" decode "${D_PORT}" "${MODEL}" --max-running-requests 256 \
  >"${ART}/decode.log" 2>&1 &
D_PID=$!

wait_http "http://${HEAD_IP}:${P_PORT}/health"
echo "prefill ready"
wait_http "http://${TAIL_IP}:${D_PORT}/health"
echo "decode ready"

srun --overlap -N1 -n1 -w "${HEAD}" --cpus-per-task=2 \
  --container-image="${SGLANG}" --container-mounts="$(mounts router)" \
  python3 -m sglang_router.launch_router --pd-disaggregation \
    --prefill "http://${HEAD_IP}:${P_PORT}" "${BOOT}" \
    --decode "http://${TAIL_IP}:${D_PORT}" \
    --host 0.0.0.0 --port "${R_PORT}" \
  >"${ART}/router.log" 2>&1 &
R_PID=$!

wait_http "http://${HEAD_IP}:${R_PORT}/health"
echo "READY after ${SECONDS}s"

srun --overlap -N1 -n1 -w "${HEAD}" \
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
