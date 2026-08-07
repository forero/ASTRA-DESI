#!/bin/bash
#SBATCH --job-name=dr2_groups_100x
#SBATCH --account=desi
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --time=48:00:00
#SBATCH --mem=0
#SBATCH --array=0-7%8
#SBATCH --output=/pscratch/sd/v/vtorresg/cosmic-web/dr2/logs/groups_100x_%A_%a.out
#SBATCH --error=/pscratch/sd/v/vtorresg/cosmic-web/dr2/logs/groups_100x_%A_%a.err
#SBATCH --chdir=/global/homes/v/vtorresg/ASTRA-DESI

set -euo pipefail

module load python/3.12
source /global/homes/v/vtorresg/venvs/my-umap-env/bin/activate


export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/astra-desi-matplotlib
export XDG_CACHE_HOME=/tmp/astra-desi-cache

TRACERS=(BGS BGS ELG ELG LRG LRG QSO QSO)
ZONES=(NGC SGC NGC SGC NGC SGC NGC SGC)

TASK_ID=${SLURM_ARRAY_TASK_ID}
if (( TASK_ID < 0 || TASK_ID >= ${#TRACERS[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}" >&2
  exit 2
fi

TRACER=${TRACERS[TASK_ID]}
ZONE=${ZONES[TASK_ID]}

SCRATCH_ROOT=/pscratch/sd/v/vtorresg/cosmic-web/dr2
RAW_DIR=${SCRATCH_ROOT}/raw
OUTPUT_ROOT=${SCRATCH_ROOT}/groups_100x
MASK_CACHE=${SCRATCH_ROOT}/groups_100x_masks
LOG_DIR=${SCRATCH_ROOT}/logs

mkdir -p "${OUTPUT_ROOT}" "${MASK_CACHE}" "${LOG_DIR}"

# "auto" is deliberately memory-aware.  It uses all 100 workers when the
# catalogue fits, and reduces only the number of simultaneous Delaunay graphs
# for the larger BGS/ELG cases.  Set WORKERS=100 to force all realizations to
# coexist, or a smaller integer after inspecting the first run's max RSS.
WORKERS=${WORKERS:-auto}
MEMORY_FRACTION=${MEMORY_FRACTION:-0.85}
MEMORY_BYTES_PER_POINT=${MEMORY_BYTES_PER_POINT:-1100}

TIME_FILE=${LOG_DIR}/groups_100x_${TRACER}_${ZONE}_${SLURM_JOB_ID}.time

echo "[sbatch] tracer=${TRACER} zone=${ZONE} workers=${WORKERS}"
echo "[sbatch] output=${OUTPUT_ROOT}/${TRACER,,}/${ZONE,,}/iterNN"

srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} --cpu-bind=cores \
  /usr/bin/time -f $'%E elapsed\n%U user CPU\n%S sys CPU\n%P CPU%%\n%M KB max RSS\n' \
  -o "${TIME_FILE}" \
  python -u -m group_finder \
    --raw-dir "${RAW_DIR}" \
    --output-root "${OUTPUT_ROOT}" \
    --mask-cache "${MASK_CACHE}" \
    --tracers "${TRACER}" \
    --zones "${ZONE}" \
    --iterations 0-99 \
    --workers "${WORKERS}" \
    --memory-fraction "${MEMORY_FRACTION}" \
    --memory-bytes-per-point "${MEMORY_BYTES_PER_POINT}" \
    --output-layout iteration \
    --resume \
    --no-plot
