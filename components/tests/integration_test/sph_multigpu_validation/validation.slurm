#!/bin/bash
#SBATCH --job-name=sph-current-parity
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40:4
#SBATCH --time=01:00:00
set -euo pipefail
project=/home/tangzijie/ssd/paper-differentiated-asteroid-impact-accretion
source_tree="$project/.worktrees/granit-sph-current-integration"
receipt="$project/runs/sph-current-integration-v1/validation-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export PATH=/usr/local/cuda-12.1/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH:-}
mkdir -p "$receipt/bin"
date -u +%FT%TZ > "$receipt/start-utc.txt"
trap 'date -u +%FT%TZ > "$receipt/end-utc.txt"' EXIT
printf '%s\n' "${CUDA_VISIBLE_DEVICES:-unset}" > "$receipt/cuda-visible-devices.txt"
scontrol show job "$SLURM_JOB_ID" > "$receipt/slurm-job.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv > "$receipt/gpu-start.csv"
cp "$source_tree/build-integration/bin/granit_sph" "$source_tree/build-integration/bin/granit_multigpu" "$receipt/bin/"
git -C "$source_tree" rev-parse HEAD > "$receipt/source-base.txt"
git -C "$source_tree" diff --binary > "$receipt/source.patch"
cp "$source_tree/build-integration/CMakeCache.txt" "$receipt/CMakeCache.txt"
cp "$source_tree/components/tests/integration_test/sph_multigpu_validation/run_validation.py" "$receipt/validation-script.py"
sha256sum "$receipt/bin/"* > "$receipt/binaries.sha256"
python3 "$source_tree/components/tests/integration_test/sph_multigpu_validation/run_validation.py" --work-dir "$receipt/numerics" --single "$receipt/bin/granit_sph" --multi "$receipt/bin/granit_multigpu"
