#!/bin/bash
#SBATCH -J p6-tp2-d1-tp4-moe
#SBATCH -p hpcai
#SBATCH -N 2
#SBATCH -t 00:17:00
#SBATCH --gres=gpu:8
#SBATCH --propagate=NONE
#SBATCH -o runs/%j.out
#SBATCH -e runs/%j.err
#
# See README.md in this folder for how this script fits into the task.
#
# Two-node PD split on a small MoE model (Qwen1.5-MoE-A2.7B), 6 prefill + 1 decode:
#   prefill TP=2 EP=1; decode TP=4 EP=4. Pass = aiperf exits 0.
# This demonstrates the MoE-specific --ep-size flag, which the dense-model examples
# in this folder (04, 05) do not need.
set -euo pipefail

SGLANG=/scratch/enroot-images/sglang-harbor.sqsh
AIPERF=/scratch/enroot-images/aiperf.sqsh
MODEL=/scratch/models/Qwen1.5-MoE-A2.7B
R_PORT=8000
READY_SECS=400
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
mkdir -p "${ART}"

cat >"${ART}/worker.sh" <<'EOF'
set -euo pipefail
name=$1
port=$2
model=$3
tp=$4
shift 4
role=${name%%-*}
UCX_IB=$(for d in /sys/class/infiniband/*; do
  [ "$(cat "$d/ports/1/link_layer" 2>/dev/null)" = InfiniBand ] || continue
  [ "$(cat "$d/ports/1/state" 2>/dev/null)" = "4: ACTIVE" ] || continue
  echo "$(basename "$d"):1"
done | paste -sd,)
[ -n "$UCX_IB" ] || { echo "FATAL: no active IB on $(hostname)"; exit 1; }
export UCX_NET_DEVICES=$UCX_IB
export UCX_POSIX_USE_PROC_LINK=n
export TRITON_CACHE_DIR=/tmp/sglang-$name/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/sglang-$name/inductor
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
echo "$name host=$(hostname) ib=$UCX_IB gpu=$CUDA_VISIBLE_DEVICES tp=$tp"
exec python3 -m sglang.launch_server \
  --model-path "$model" --served-model-name qwen --trust-remote-code \
  --context-length 4096 --tp-size "$tp" --mem-fraction-static 0.4 \
  --disaggregation-mode "$role" --disaggregation-transfer-backend nixl \
  "$@" --host 0.0.0.0 --port "$port"
EOF

srun --exact -N1 -n1 -w "${HEAD}" --gres=gpu:2 --cpus-per-task=16 \
  --container-image="${SGLANG}" --container-mounts="$(mounts prefill-0)" \
  bash "${ART}/worker.sh" prefill-0 30100 "${MODEL}" 2 --disaggregation-bootstrap-port 9000 \
  >"${ART}/prefill-0.log" 2>&1 &
P0_PID=$!

srun --exact -N1 -n1 -w "${HEAD}" --gres=gpu:2 --cpus-per-task=16 \
  --container-image="${SGLANG}" --container-mounts="$(mounts prefill-1)" \
  bash "${ART}/worker.sh" prefill-1 30101 "${MODEL}" 2 --disaggregation-bootstrap-port 9001 \
  >"${ART}/prefill-1.log" 2>&1 &
P1_PID=$!

srun --exact -N1 -n1 -w "${HEAD}" --gres=gpu:2 --cpus-per-task=16 \
  --container-image="${SGLANG}" --container-mounts="$(mounts prefill-2)" \
  bash "${ART}/worker.sh" prefill-2 30102 "${MODEL}" 2 --disaggregation-bootstrap-port 9002 \
  >"${ART}/prefill-2.log" 2>&1 &
P2_PID=$!

srun --exact -N1 -n1 -w "${HEAD}" --gres=gpu:2 --cpus-per-task=16 \
  --container-image="${SGLANG}" --container-mounts="$(mounts prefill-3)" \
  bash "${ART}/worker.sh" prefill-3 30103 "${MODEL}" 2 --disaggregation-bootstrap-port 9003 \
  >"${ART}/prefill-3.log" 2>&1 &
P3_PID=$!

srun --exact -N1 -n1 -w "${TAIL}" --gres=gpu:2 --cpus-per-task=16 \
  --container-image="${SGLANG}" --container-mounts="$(mounts prefill-4)" \
  bash "${ART}/worker.sh" prefill-4 30100 "${MODEL}" 2 --disaggregation-bootstrap-port 9000 \
  >"${ART}/prefill-4.log" 2>&1 &
P4_PID=$!

srun --exact -N1 -n1 -w "${TAIL}" --gres=gpu:2 --cpus-per-task=16 \
  --container-image="${SGLANG}" --container-mounts="$(mounts prefill-5)" \
  bash "${ART}/worker.sh" prefill-5 30101 "${MODEL}" 2 --disaggregation-bootstrap-port 9001 \
  >"${ART}/prefill-5.log" 2>&1 &
P5_PID=$!

srun --exact -N1 -n1 -w "${TAIL}" --gres=gpu:4 --cpus-per-task=32 \
  --container-image="${SGLANG}" --container-mounts="$(mounts decode-0)" \
  bash "${ART}/worker.sh" decode-0 30200 "${MODEL}" 4 --ep-size 4 --max-running-requests 256 \
  >"${ART}/decode-0.log" 2>&1 &
D_PID=$!

wait_http "http://${HEAD_IP}:30100/health"; echo "prefill-0 ready"
wait_http "http://${HEAD_IP}:30101/health"; echo "prefill-1 ready"
wait_http "http://${HEAD_IP}:30102/health"; echo "prefill-2 ready"
wait_http "http://${HEAD_IP}:30103/health"; echo "prefill-3 ready"
wait_http "http://${TAIL_IP}:30100/health"; echo "prefill-4 ready"
wait_http "http://${TAIL_IP}:30101/health"; echo "prefill-5 ready"
wait_http "http://${TAIL_IP}:30200/health"; echo "decode-0 ready"

srun --overlap -N1 -n1 -w "${HEAD}" --cpus-per-task=2 \
  --container-image="${SGLANG}" --container-mounts="$(mounts router)" \
  python3 -m sglang_router.launch_router --pd-disaggregation \
    --prefill "http://${HEAD_IP}:30100" 9000 \
    --prefill "http://${HEAD_IP}:30101" 9001 \
    --prefill "http://${HEAD_IP}:30102" 9002 \
    --prefill "http://${HEAD_IP}:30103" 9003 \
    --prefill "http://${TAIL_IP}:30100" 9000 \
    --prefill "http://${TAIL_IP}:30101" 9001 \
    --decode "http://${TAIL_IP}:30200" \
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
kill "${P0_PID}" "${P1_PID}" "${P2_PID}" "${P3_PID}" "${P4_PID}" "${P5_PID}" "${D_PID}" "${R_PID}"
wait "${P0_PID}" "${P1_PID}" "${P2_PID}" "${P3_PID}" "${P4_PID}" "${P5_PID}" "${D_PID}" "${R_PID}"
set -e
