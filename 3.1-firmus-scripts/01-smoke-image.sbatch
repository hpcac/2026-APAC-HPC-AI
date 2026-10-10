#!/bin/bash
#SBATCH -J smoke-image
#SBATCH -p hpcai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:15:00
#SBATCH --gres=gpu:1
#SBATCH --propagate=NONE
#SBATCH -o runs/%j.out
#SBATCH -e runs/%j.err
#
# See README.md in this folder for how this script fits into the task.
#
# /scratch/enroot-images/sglang-harbor.sqsh
# Pass = container starts, GPU visible, sglang imports. Exit code is the verdict.
set -euo pipefail

IMAGE=/scratch/enroot-images/sglang-harbor.sqsh
ROOT=${SLURM_SUBMIT_DIR}
HOME_BIND=${ROOT}/home

echo "[$(date -Iseconds)] job=${SLURM_JOB_ID} user=$(whoami) host=$(hostname) image=${IMAGE}"
test -r "${IMAGE}"
ls -lh "${IMAGE}"

srun --container-image="${IMAGE}" \
  --container-mounts="/scratch:/scratch,${HOME_BIND}:${HOME}" \
  bash -c 'set -euo pipefail
    echo "in-container host=$(hostname) user=$(whoami)"
    nvidia-smi -L
    python3 -c "import sglang; print(\"sglang\", getattr(sglang, \"__version__\", \"imported\"))"
  '
