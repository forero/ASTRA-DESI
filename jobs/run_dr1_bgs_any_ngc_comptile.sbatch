#!/bin/bash
# Reprocess only DR1 BGS_ANY NGC with fiber-completeness-matched randoms.

#SBATCH --job-name=astra_bgs_ngc_comp
#SBATCH --account=desi
#SBATCH -q regular
#SBATCH --time=04:00:00
#SBATCH --mem=0
#SBATCH -C cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --output=/pscratch/sd/v/vtorresg/cosmic-web/dr1/logs/bgs_any_ngc_comptile_%j.out
#SBATCH --error=/pscratch/sd/v/vtorresg/cosmic-web/dr1/logs/bgs_any_ngc_comptile_%j.err
#SBATCH --chdir=/global/homes/v/vtorresg/ASTRA-DESI

set -euo pipefail

module load python/3.12
source /global/homes/v/vtorresg/venvs/my-umap-env/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PAIR_NJOBS_CAP=${PAIR_NJOBS_CAP:-128}

export ASTRA_CLASS_SPLIT_ITER=0
export ASTRA_CLASS_SKIP_COMBINED=0
export ASTRA_PROB_SPLIT_ITER=0
export ASTRA_PROB_SKIP_COMBINED=0
export ASTRA_SKIP_GROUPS=1
export ASTRA_PROGRESS=1

PROJECT_ROOT=/global/homes/v/vtorresg/ASTRA-DESI
INPUT_ROOT=/global/cfs/cdirs/desi/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5pip
OUTPUT_ROOT=/pscratch/sd/v/vtorresg/cosmic-web/dr1_comptile
RAW_OUT=${OUTPUT_ROOT}/raw
GROUPS_OUT=${OUTPUT_ROOT}/groups
SPILL_DIR=${OUTPUT_ROOT}/pair_spill
LOG_DIR=${OUTPUT_ROOT}/logs

mkdir -p "${RAW_OUT}" "${GROUPS_OUT}" "${SPILL_DIR}" "${LOG_DIR}"

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=cores \
  python "${PROJECT_ROOT}/src/main.py" \
    --release DR1 \
    --zones NGC \
    --tracers BGS_ANY \
    --out-tag BGS_ANY \
    --base-dir "${INPUT_ROOT}" \
    --raw-out "${RAW_OUT}" \
    --class-out "${OUTPUT_ROOT}" \
    --groups-out "${GROUPS_OUT}" \
    --spill-dir "${SPILL_DIR}" \
    --n-random 100 \
    --n-random-files 18 \
    --r-lower -0.25 \
    --r-med 0.25 \
    --r-upper 0.65 \
    --dr1-completeness-randoms \
    --dr1-completeness-seed 20260912 \
    --skip-dr1-properties \
    --progress

echo "Corrected probability product:"
echo "${OUTPUT_ROOT}/probabilities/bgs/ngc/zone_NGC_BGS_ANY_probability.fits.gz"
