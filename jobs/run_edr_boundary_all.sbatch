#!/bin/bash
#SBATCH --job-name=edr_bound_all
#SBATCH --account=desi
#SBATCH --partition=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=edr_boundary_all_%j.out
#SBATCH --error=edr_boundary_all_%j.err
#SBATCH --chdir=/global/homes/v/vtorresg/ASTRA-DESI

set -euo pipefail

module load python/3.12

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

DATA_DIR=${DATA_DIR:-/global/cfs/cdirs/desi/public/edr/vac/edr/lss/v2.0/LSScats/clustering}
OUTPUT_ROOT=${OUTPUT_ROOT:-/pscratch/sd/v/${USER}/cosmic-web/edr_boundary_all}
RADIUS_DEG=${RADIUS_DEG:-1.45}
ZONE_WORKERS=${ZONE_WORKERS:-5}

mkdir -p "${OUTPUT_ROOT}"

srun --cpu-bind=cores \
  python src/utils/edr_all_rosettes_boundary.py \
    --base-dir "${DATA_DIR}" \
    --output-root "${OUTPUT_ROOT}" \
    --zones all \
    --radius-deg "${RADIUS_DEG}" \
    --fractions 1 0.75 0.5 0.25 \
    --n-random 100 \
    --n-random-files 18 \
    --total-cores "${SLURM_CPUS_PER_TASK}" \
    --zone-workers "${ZONE_WORKERS}" \
    --distance-bins 12
