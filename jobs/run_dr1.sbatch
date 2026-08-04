#!/bin/bash
#SBATCH --job-name=astra_dr1
#SBATCH --account=desi
#SBATCH -q regular
#SBATCH --time=01:30:00
#SBATCH --mem=0
#SBATCH --array=0-1%1
#SBATCH -C cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --output=/pscratch/sd/v/vtorresg/cosmic-web/dr1/logs/zone_%A_%a.out
#SBATCH --error=/pscratch/sd/v/vtorresg/cosmic-web/dr1/logs/zone_%A_%a.err
#SBATCH --chdir=/global/homes/v/vtorresg/ASTRA-DESI

set -euo pipefail

module load python/3.12
source ~/venvs/my-umap-env/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export ASTRA_CLASS_SPLIT_ITER=0
export ASTRA_CLASS_SKIP_COMBINED=0
export ASTRA_PROB_SPLIT_ITER=0
export ASTRA_PROB_SKIP_COMBINED=0
export ASTRA_SKIP_GROUPS=1

R_LOWER=${R_LOWER:--0.25}
R_MED=${R_MED:-0.25}
R_UPPER=${R_UPPER:-0.65}
export PAIR_NJOBS_CAP=${PAIR_NJOBS_CAP:-128}

PROJECT_ROOT=$PWD
MAIN_SCRIPT="${PROJECT_ROOT}/src/main.py"
MASK_SCRIPT="${PROJECT_ROOT}/src/releases/dr1_mask.py"

CLUSTER_DIR="/global/cfs/cdirs/desi/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5pip"
SCRATCH_ROOT="/pscratch/sd/v/vtorresg/cosmic-web/dr1"

RAW_OUT="${SCRATCH_ROOT}/raw"
CLASS_ROOT="${SCRATCH_ROOT}"
GROUPS_OUT="${SCRATCH_ROOT}/groups"
PLOTS_OUT="${SCRATCH_ROOT}/figs"
LOGDIR="${SCRATCH_ROOT}/logs"
PAIR_SPILL="${SCRATCH_ROOT}/pair_spill"
MASK_DIR="${SCRATCH_ROOT}/masks/bright_dark"
MASK_NSIDE=${MASK_NSIDE:-128}
MASK_TILE_RADIUS_DEG=${MASK_TILE_RADIUS_DEG:-1.605}
MASK_MIN_TILES=${MASK_MIN_TILES:-4}
MASK_MIN_LSS_OBJECTS=${MASK_MIN_LSS_OBJECTS:-1}

mkdir -p "${RAW_OUT}" "${CLASS_ROOT}" "${GROUPS_OUT}" "${PLOTS_OUT}" "${LOGDIR}" "${PAIR_SPILL}" "${MASK_DIR}"
export ASTRA_DR1_MASK_DIR="${MASK_DIR}"

ZLABELS=("NGC" "SGC")
CONFIG_JSON=${CONFIG_JSON:-}

if (( SLURM_ARRAY_TASK_ID >= ${#ZLABELS[@]} )); then
  echo "error: SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID} exceeds ZLABELS" >&2
  exit 1
fi
ZONE_LABEL=${ZLABELS[SLURM_ARRAY_TASK_ID]}

TRACE_ARGS=()
if [[ -n ${CONFIG_JSON} ]]; then
  TRACE_ARGS+=(--config "${CONFIG_JSON}")
fi

LOG_PREFIX="${LOGDIR}/zone_${ZONE_LABEL}"

REQUIRED_MASKS=("${MASK_DIR}/dr1_mask_bright_nside${MASK_NSIDE}_ngc.fits"
                "${MASK_DIR}/dr1_mask_bright_nside${MASK_NSIDE}_sgc.fits"
                "${MASK_DIR}/dr1_mask_dark_nside${MASK_NSIDE}_ngc.fits"
                "${MASK_DIR}/dr1_mask_dark_nside${MASK_NSIDE}_sgc.fits")
NEED_MASKS=0
for mask_path in "${REQUIRED_MASKS[@]}"; do
  if [[ ! -s "${mask_path}" ]]; then
    NEED_MASKS=1
    break
  fi
done

if (( NEED_MASKS == 1 )); then
  echo "--> generating DR1 bright/dark NGC/SGC masks in ${MASK_DIR}"
  python "${MASK_SCRIPT}" \
    --out-dir "${MASK_DIR}" \
    --nside "${MASK_NSIDE}" \
    --tile-radius-deg "${MASK_TILE_RADIUS_DEG}" \
    --min-tiles "${MASK_MIN_TILES}" \
    --min-lss-objects "${MASK_MIN_LSS_OBJECTS}"
fi

srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} --cpu-bind=cores \
     /usr/bin/time -f $'%E elapsed\n%U user CPU\n%S sys CPU\n%P CPU%%\n%M KB max RSS\n' \
     -o "${LOG_PREFIX}.time" \
     python "${MAIN_SCRIPT}" \
       --release DR1 \
       --zones "${ZONE_LABEL}" \
       ${TRACE_ARGS[@]} \
       --base-dir  "${CLUSTER_DIR}" \
       --raw-out   "${RAW_OUT}" \
       --class-out "${CLASS_ROOT}" \
       --groups-out "${GROUPS_OUT}" \
       --plot-output "${PLOTS_OUT}" \
       --spill-dir "${PAIR_SPILL}" \
       --r-lower "${R_LOWER}" \
       --r-med "${R_MED}" \
       --r-upper "${R_UPPER}" \
       --n-random 100