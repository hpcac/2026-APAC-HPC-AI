#!/bin/bash
#SBATCH -J ref09-official
#SBATCH -p hpcai
#SBATCH -N 2
#SBATCH -t 00:12:00
#SBATCH --gres=gpu:8
#SBATCH --propagate=NONE
#SBATCH -o runs/%j.out
#SBATCH -e runs/%j.err
#
# Organisers' reference run of 3.1-firmus-scripts/09 (2 prefill TP4 + 2 decode TP4/EP4). The deployment part is line-for-line 09;
# only 09's short exploratory sweep is replaced by the official command in official-aiperf.sh (same folder), one N per job.
# sbatch --export=ALL,N=<concurrency> [-t <walltime>] firmus-09-official.sbatch
set -euo pipefail

SGLANG=/scratch/enroot-images/sglang-harbor.sqsh
AIPERF=/scratch/enroot-images/aiperf.sqsh
MODEL=/scratch/models/Qwen3-235B-A22B-Instruct-2507-FP8
R_PORT=8000
READY_SECS=600  # /health cap; typically 2-4 min, fail if still down at 10 min
: "${N:?concurrency}"
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
    --retry-delay 2 --retry 300 --max-time 2 \
    -o /dev/null "$1"
}

echo "[$(date -Iseconds)] job=${SLURM_JOB_ID} user=$(whoami)"
echo "head=${HEAD}(${HEAD_IP}) tail=${TAIL}(${TAIL_IP}) model=${MODEL} N=${N}"
test -r "${SGLANG}" && test -r "${AIPERF}" && test -f "${MODEL}/config.json"
test -r "${ROOT}/official-aiperf.sh"
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
  --context-length 8192 --tp-size "$tp" --mem-fraction-static 0.90 \
  --disaggregation-mode "$role" --disaggregation-transfer-backend nixl \
  "$@" --host 0.0.0.0 --port "$port"
EOF

srun --exact -N1 -n1 -w "${HEAD}" --gres=gpu:4 --cpus-per-task=32 \
  --container-image="${SGLANG}" --container-mounts="$(mounts prefill-0)" \
  bash "${ART}/worker.sh" prefill-0 30100 "${MODEL}" 4 --disaggregation-bootstrap-port 9000 \
  >"${ART}/prefill-0.log" 2>&1 &
P0_PID=$!

srun --exact -N1 -n1 -w "${HEAD}" --gres=gpu:4 --cpus-per-task=32 \
  --container-image="${SGLANG}" --container-mounts="$(mounts prefill-1)" \
  bash "${ART}/worker.sh" prefill-1 30101 "${MODEL}" 4 --disaggregation-bootstrap-port 9001 \
  >"${ART}/prefill-1.log" 2>&1 &
P1_PID=$!

srun --exact -N1 -n1 -w "${TAIL}" --gres=gpu:4 --cpus-per-task=32 \
  --container-image="${SGLANG}" --container-mounts="$(mounts decode-0)" \
  bash "${ART}/worker.sh" decode-0 30200 "${MODEL}" 4 --ep-size 4 --max-running-requests 256 \
  >"${ART}/decode-0.log" 2>&1 &
D0_PID=$!

srun --exact -N1 -n1 -w "${TAIL}" --gres=gpu:4 --cpus-per-task=32 \
  --container-image="${SGLANG}" --container-mounts="$(mounts decode-1)" \
  bash "${ART}/worker.sh" decode-1 30201 "${MODEL}" 4 --ep-size 4 --max-running-requests 256 \
  >"${ART}/decode-1.log" 2>&1 &
D1_PID=$!

wait_http "http://${HEAD_IP}:30100/health"; echo "prefill-0 ready"
wait_http "http://${HEAD_IP}:30101/health"; echo "prefill-1 ready"
wait_http "http://${TAIL_IP}:30200/health"; echo "decode-0 ready"
wait_http "http://${TAIL_IP}:30201/health"; echo "decode-1 ready"

srun --overlap -N1 -n1 -w "${HEAD}" --cpus-per-task=2 \
  --container-image="${SGLANG}" --container-mounts="$(mounts router)" \
  python3 -m sglang_router.launch_router --pd-disaggregation \
    --prefill "http://${HEAD_IP}:30100" 9000 \
    --prefill "http://${HEAD_IP}:30101" 9001 \
    --decode "http://${TAIL_IP}:30200" \
    --decode "http://${TAIL_IP}:30201" \
    --host 0.0.0.0 --port "${R_PORT}" \
  >"${ART}/router.log" 2>&1 &
R_PID=$!

wait_http "http://${HEAD_IP}:${R_PORT}/health"
echo "READY after ${SECONDS}s"

echo "OFFICIAL start after ${SECONDS}s"
srun --overlap -N1 -n1 -w "${HEAD}" \
  --container-image="${AIPERF}" --container-mounts="$(mounts aiperf)" \
  --container-workdir="${ART}" \
  env N="${N}" HOST="127.0.0.1:${R_PORT}" MODEL=qwen bash "${ROOT}/official-aiperf.sh"
echo "AIPERF ok after ${SECONDS}s"

set +e
kill "${P0_PID}" "${P1_PID}" "${D0_PID}" "${D1_PID}" "${R_PID}"
wait "${P0_PID}" "${P1_PID}" "${D0_PID}" "${D1_PID}" "${R_PID}"
set -e
