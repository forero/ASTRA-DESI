#!/bin/bash
#SBATCH --job-name=dr2_voids_3cosmo
#SBATCH --account=desi
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --time=48:00:00
#SBATCH --mem=0
#SBATCH --array=0-7%8
#SBATCH --output=/pscratch/sd/v/vtorresg/dr2_3cosmo_100x_%A_%a.out
#SBATCH --error=/pscratch/sd/v/vtorresg/dr2_3cosmo_100x_%A_%a.err
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

TRACERS=(BGS LRG ELG QSO)
CAPS=(NGC SGC)

TASK_ID=${SLURM_ARRAY_TASK_ID}
TRACER_INDEX=$((TASK_ID / 2))
CAP_INDEX=$((TASK_ID % 2))
TRACER=${TRACERS[TRACER_INDEX]}
CAP=${CAPS[CAP_INDEX]}
WORKERS=${WORKERS:-auto}
MEMORY_FRACTION=${MEMORY_FRACTION:-0.60}
MEMORY_BYTES_PER_POINT=${MEMORY_BYTES_PER_POINT:-1500}
FINAL_CATALOG_ROOT=${FINAL_CATALOG_ROOT:-/pscratch/sd/v/vtorresg/void_catalog_dr2_new}

echo "[sbatch] tracer=${TRACER} cap=${CAP} cosmologies=3 workers=${WORKERS}"
srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} --cpu-bind=cores \
  python -u catalog/run_dr2_voids_three_cosmologies.py "${TRACER}" \
    --caps "${CAP}" \
    --random-indices all \
    --iterations 0-99 \
    --workers "${WORKERS}" \
    --memory-fraction "${MEMORY_FRACTION}" \
    --memory-bytes-per-point "${MEMORY_BYTES_PER_POINT}" \
    --final-catalog-root "${FINAL_CATALOG_ROOT}" \
    --resume
