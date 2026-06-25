#!/bin/bash
#SBATCH --job-name=dr1_mr20_voids
#SBATCH --account=desi
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=0
#SBATCH --array=0-99%10
#SBATCH --hint=nomultithread
#SBATCH --output=/global/homes/v/vtorresg/ASTRA-DESI/logs/dr1_mr20_voids_%A_%a.out
#SBATCH --error=/global/homes/v/vtorresg/ASTRA-DESI/logs/dr1_mr20_voids_%A_%a.err
#SBATCH --chdir=/global/homes/v/vtorresg/ASTRA-DESI

set -euo pipefail

module load python/3.12

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PROJECT_ROOT=$PWD
SCRIPT="${PROJECT_ROOT}/run_dr1_mag.py"

OUTPUT_DIR=${OUTPUT_DIR:-/pscratch/sd/v/vtorresg/cosmic-web/dr1/void-cat-mr20-z024-100iter-group-finder-randomzmatch}
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

CAPS=${CAPS:-"NGC SGC"}
read -r -a CAP_ARGS <<< "${CAPS}"
CAP_SPLIT=${CAP_SPLIT:-mask}

R_THRESHOLD=${R_THRESHOLD:--0.25}
MIN_GROUP_SIZE=${MIN_GROUP_SIZE:-4}
MIN_RAND_FOR_SHAPE=${MIN_RAND_FOR_SHAPE:-3}
SEED_BASE=${SEED_BASE:-12345}
RANDOM_Z_MODE=${RANDOM_Z_MODE:-data-z}

EXTRA_ARGS=()
if [[ -n ${RANDOM_INDEX:-} ]]; then
  EXTRA_ARGS+=(--random-index "${RANDOM_INDEX}")
fi
if [[ ${OVERWRITE:-0} == "1" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi
if [[ ${QUIET:-0} == "1" ]]; then
  EXTRA_ARGS+=(--quiet)
fi

ITER=${SLURM_ARRAY_TASK_ID}

srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} --cpu-bind=cores \
     /usr/bin/time -f $'%E elapsed\n%U user CPU\n%S sys CPU\n%P CPU%%\n%M KB max RSS\n' \
     -o "${LOG_DIR}/iter_${ITER}.time" \
     python "${SCRIPT}" \
       --iteration "${ITER}" \
       --seed-base "${SEED_BASE}" \
       --caps "${CAP_ARGS[@]}" \
       --cap-split "${CAP_SPLIT}" \
       --output-dir "${OUTPUT_DIR}" \
       --log-dir "${LOG_DIR}" \
       --random-z-mode "${RANDOM_Z_MODE}" \
       --r-threshold "${R_THRESHOLD}" \
       --min-group-size "${MIN_GROUP_SIZE}" \
       --min-rand-for-shape "${MIN_RAND_FOR_SHAPE}" \
       "${EXTRA_ARGS[@]}"