#!/bin/bash
#SBATCH --job-name=qso_ngc
#SBATCH --account=desi
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=0
#SBATCH --array=0%1
#SBATCH --hint=nomultithread
#SBATCH --output=/pscratch/sd/v/vtorresg/cosmic-web/dr2_res/logs/zone_%A_%a.out
#SBATCH --error=/pscratch/sd/v/vtorresg/cosmic-web/dr2_res/logs/zone_%A_%a.err
#SBATCH --chdir=/global/homes/v/vtorresg/ASTRA-DESI

set -euo pipefail

module load python/3.12

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export PAIR_NJOBS_CAP=1

R_LOWER=${R_LOWER:--0.9}
R_UPPER=${R_UPPER:-0.9}

PROJECT_ROOT=$PWD
MAIN_SCRIPT="${PROJECT_ROOT}/src/main.py"

CLUSTER_DIR="/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v2/PIP/"
SCRATCH_ROOT="/pscratch/sd/v/vtorresg/cosmic-web/dr2_res"

RAW_OUT="${SCRATCH_ROOT}/raw"
CLASS_ROOT="${SCRATCH_ROOT}"
GROUPS_OUT="${SCRATCH_ROOT}/groups"
PLOTS_OUT="${SCRATCH_ROOT}/figs"
LOGDIR="${SCRATCH_ROOT}/logs"

mkdir -p "${RAW_OUT}" "${CLASS_ROOT}" "${GROUPS_OUT}" "${PLOTS_OUT}" "${LOGDIR}"

ZLABELS=("NGC")
TRACERS_BY_ZONE=("QSO")

if (( SLURM_ARRAY_TASK_ID >= ${#ZLABELS[@]} )); then
  echo "error: SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID} exceeds ZLABELS" >&2
  exit 1
fi
ZONE_LABEL=${ZLABELS[SLURM_ARRAY_TASK_ID]}

TRACER_LABEL=""
if (( SLURM_ARRAY_TASK_ID < ${#TRACERS_BY_ZONE[@]} )); then
  TRACER_LABEL=${TRACERS_BY_ZONE[SLURM_ARRAY_TASK_ID]}
elif [[ -n ${TRACER_OVERRIDE:-} ]]; then
  TRACER_LABEL=${TRACER_OVERRIDE}
fi

TRACE_ARGS=()
if [[ -n ${TRACER_LABEL} ]]; then
  TRACE_ARGS+=(--tracers ${TRACER_LABEL} --out-tag "${TRACER_LABEL// /_}")
fi

TRACER_SAFE=${TRACER_LABEL// /_}
LOG_PREFIX="${LOGDIR}/zone_${ZONE_LABEL}_${TRACER_SAFE:-all}"

srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} --cpu-bind=cores \
     /usr/bin/time -f $'%E elapsed\n%U user CPU\n%S sys CPU\n%P CPU%%\n%M KB max RSS\n' \
     -o "${LOG_PREFIX}.time" \
     python "${MAIN_SCRIPT}" \
       --release DR2 \
       --zones "${ZONE_LABEL}" \
       ${TRACE_ARGS[@]} \
       --base-dir  "${CLUSTER_DIR}" \
       --raw-out   "${RAW_OUT}" \
       --class-out "${CLASS_ROOT}" \
       --groups-out "${GROUPS_OUT}" \
       --plot-output "${PLOTS_OUT}" \
       --r-lower "${R_LOWER}" \
       --r-upper "${R_UPPER}" \
       --webtype filament \
       --source data \
       --n-random 100 \