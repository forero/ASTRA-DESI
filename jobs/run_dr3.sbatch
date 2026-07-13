#!/bin/bash
#SBATCH --job-name=elg_ngc
#SBATCH --account=desi
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=03:00:00
#SBATCH --mem=0
#SBATCH --hint=nomultithread
#SBATCH --output=/pscratch/sd/v/vtorresg/cosmic-web/dr3_%j.out
#SBATCH --error=/pscratch/sd/v/vtorresg/cosmic-web/dr3_%j.err
#SBATCH --chdir=/global/homes/v/vtorresg/ASTRA-DESI

set -euo pipefail

module load python/3.12

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export PAIR_NJOBS_CAP=${PAIR_NJOBS_CAP:-32}
export DR3_ITER_NJOBS_CAP=${DR3_ITER_NJOBS_CAP:-${PAIR_NJOBS_CAP}}

R_LOWER=${R_LOWER:--0.25}
R_MED=${R_MED:-0.25}
R_UPPER=${R_UPPER:-0.65}
N_RANDOM=${N_RANDOM:-100}
N_RANDOM_FILES=${N_RANDOM_FILES:-18}
MODE=${MODE:-iter}

TRACER_LABEL="ELG"
ZONE_LABEL="NGC"

PROJECT_ROOT=$PWD
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

ITER_SCRIPT="${PROJECT_ROOT}/src/utils/run_dr3_iterations.py"
PROB_SCRIPT="${PROJECT_ROOT}/src/utils/build_dr3_probabilities.py"

DEFAULT_CLUSTER_DIR="${CLUSTER_DIR:-/global/cfs/cdirs/desi/survey/catalogs/DA3/LSS/matterhorn-v2/LSScats/test/nonKP/}"
BGS_BRIGHT_M2135_CLUSTER_DIR="${BGS_BRIGHT_M2135_CLUSTER_DIR:-/global/cfs/cdirs/desi/survey/catalogs/DA3/analysis/matterhorn-v2/LSScats/test/nonKP/}"
CLUSTER_DIR="${DEFAULT_CLUSTER_DIR}"
SCRATCH_ROOT="/pscratch/sd/v/vtorresg/cosmic-web/dr3"

RAW_OUT="${SCRATCH_ROOT}/raw"
CLASS_ROOT="${SCRATCH_ROOT}"
LOGDIR="${SCRATCH_ROOT}/logs"
PAIR_SPILL="${SCRATCH_ROOT}/pair_spill"
PROB_SPILL="${SCRATCH_ROOT}/prob_spill"

mkdir -p "${RAW_OUT}" "${CLASS_ROOT}" "${LOGDIR}" "${PAIR_SPILL}" "${PROB_SPILL}"

TRACER_KEY=${TRACER_LABEL,,}
TRACER_KEY=${TRACER_KEY// /_}
case "${TRACER_KEY}" in
  qso) TRACER_LABEL="QSO" ;;
  elg|elg_lopnotqso) TRACER_LABEL="ELG_LOPnotqso" ;;
  lrg) TRACER_LABEL="LRG" ;;
  bgs|bgs_bright) TRACER_LABEL="BGS_BRIGHT" ;;
  bgs-21.35|bgs_21.35|bgs_m21.35|bgs_-21.35|bgs_bright-21.35|bgs_bright_21.35|bgs_bright_m21.35|bgs_bright_-21.35) TRACER_LABEL="BGS_BRIGHT-21.35" ;;
  *)
    echo "error: TRACER must be one of QSO, ELG_LOPnotqso, LRG, BGS_BRIGHT, BGS_BRIGHT-21.35 (got '${TRACER_LABEL}')" >&2
    exit 1
    ;;
esac

if [[ "${TRACER_LABEL}" == "BGS_BRIGHT-21.35" ]]; then
  CLUSTER_DIR="${BGS_BRIGHT_M2135_CLUSTER_DIR}"
fi

ZONE_LABEL=${ZONE_LABEL^^}
case "${ZONE_LABEL}" in
  NGC|SGC) ;;
  *)
    echo "error: ZONE must be NGC or SGC (got '${ZONE_LABEL}')" >&2
    exit 1
    ;;
esac

FORCE_ARGS=()
if [[ ${FORCE:-0} == "1" ]]; then
  FORCE_ARGS+=(--force)
fi

ITER_ARGS=()
if [[ -n ${ITERATIONS:-} ]]; then
  read -r -a ITER_TOKENS <<< "${ITERATIONS}"
  ITER_ARGS+=(--iterations "${ITER_TOKENS[@]}")
fi

WORKER_ARGS=()
if [[ -n ${ITER_WORKERS:-} ]]; then
  WORKER_ARGS+=(--iteration-workers "${ITER_WORKERS}")
fi

LOG_PREFIX="${LOGDIR}/${MODE}_${ZONE_LABEL}_${TRACER_LABEL}"
REAL_CATALOG="${CLUSTER_DIR}/${TRACER_LABEL}_${ZONE_LABEL}_clustering.dat.fits"
if [[ ! -s "${REAL_CATALOG}" ]]; then
  echo "[sbatch] missing DR3 clustering catalogue: ${REAL_CATALOG}"
  echo "[sbatch] skipping tracer=${TRACER_LABEL} zone=${ZONE_LABEL}"
  exit 0
fi

echo "[sbatch] mode=${MODE} tracer=${TRACER_LABEL} zone=${ZONE_LABEL}"
echo "[sbatch] cluster_dir=${CLUSTER_DIR}"
echo "[sbatch] cpus=${SLURM_CPUS_PER_TASK} pair_njobs_cap=${PAIR_NJOBS_CAP} dr3_iter_njobs_cap=${DR3_ITER_NJOBS_CAP}"

case "${MODE}" in
  iter)
    if [[ ! -r "${ITER_SCRIPT}" ]]; then
      echo "[sbatch] missing iteration runner: ${ITER_SCRIPT}" >&2
      ls -ld "${PROJECT_ROOT}" "${PROJECT_ROOT}/src" "${PROJECT_ROOT}/src/utils" >&2 || true
      exit 2
    fi
    srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} --cpu-bind=cores \
         /usr/bin/time -f $'%E elapsed\n%U user CPU\n%S sys CPU\n%P CPU%%\n%M KB max RSS\n' \
         -o "${LOG_PREFIX}.time" \
         python "${ITER_SCRIPT}" \
           --base-dir "${CLUSTER_DIR}" \
           --raw-out "${RAW_OUT}" \
           --class-out "${CLASS_ROOT}" \
           --zone "${ZONE_LABEL}" \
           --tracer "${TRACER_LABEL}" \
           --n-random "${N_RANDOM}" \
           --n-random-files "${N_RANDOM_FILES}" \
           --spill-dir "${PAIR_SPILL}" \
           --progress \
           "${WORKER_ARGS[@]}" \
           "${FORCE_ARGS[@]}" \
           "${ITER_ARGS[@]}"
    ;;
  prob)
    if [[ ! -r "${PROB_SCRIPT}" ]]; then
      echo "[sbatch] missing probability runner: ${PROB_SCRIPT}" >&2
      ls -ld "${PROJECT_ROOT}" "${PROJECT_ROOT}/src" "${PROJECT_ROOT}/src/utils" >&2 || true
      exit 2
    fi
    srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} --cpu-bind=cores \
         /usr/bin/time -f $'%E elapsed\n%U user CPU\n%S sys CPU\n%P CPU%%\n%M KB max RSS\n' \
         -o "${LOG_PREFIX}.time" \
         python "${PROB_SCRIPT}" \
           --raw-out "${RAW_OUT}" \
           --class-out "${CLASS_ROOT}" \
           --zone "${ZONE_LABEL}" \
           --tracer "${TRACER_LABEL}" \
           --n-random "${N_RANDOM}" \
           --spill-dir "${PROB_SPILL}" \
           --r-lower "${R_LOWER}" \
           --r-med "${R_MED}" \
           --r-upper "${R_UPPER}" \
           "${FORCE_ARGS[@]}"
    ;;
  *)
    echo "error: MODE must be 'iter' or 'prob' (got '${MODE}')" >&2
    exit 1
    ;;
esac
