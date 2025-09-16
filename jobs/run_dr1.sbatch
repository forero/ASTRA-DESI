#!/bin/bash
#SBATCH --job-name=cosmic_web_astra_dr1
#SBATCH --account=desi
#SBATCH --partition=regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --time=00:30:00
#SBATCH --mem=0
#SBATCH --array=0%1
#SBATCH --hint=nomultithread
#SBATCH --output=/pscratch/sd/v/vtorresg/cosmic-web/dr1/logs/zone_%A_%a.out
#SBATCH --error=/pscratch/sd/v/vtorresg/cosmic-web/dr1/logs/zone_%A_%a.err
#SBATCH --chdir=/global/homes/v/vtorresg/ASTRA-DESI

module load python/3.12

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

export OMP_PLACES=cores
export OMP_PROC_BIND=close

export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export PAIR_NJOBS_CAP=${PAIR_NJOBS_CAP:-$(( SLURM_CPUS_PER_TASK<24 ? SLURM_CPUS_PER_TASK : 24 ))}

export OMP_DISPLAY_ENV=VERBOSE

PROJECT_ROOT="$PWD"
MAIN_SCRIPT="${PROJECT_ROOT}/src/main.py"

CLUSTER_DIR="/global/cfs/cdirs/desi/public/dr1/vac/dr1/lss/guadalupe/v1.0/LSScats/clustering"
SCRATCH_BASE="/pscratch/sd/v/vtorresg/cosmic-web"

RAW_OUT="${SCRATCH_BASE}/dr1/raw"
CLASS_OUT="${SCRATCH_BASE}/dr1/class"
GROUPS_OUT="${SCRATCH_BASE}/dr1/groups"
FIGS_OUT="${SCRATCH_BASE}/dr1/figs"
LOGDIR="${SCRATCH_BASE}/dr1/logs"

mkdir -p "${RAW_OUT}" "${CLASS_OUT}" "${GROUPS_OUT}" "${FIGS_OUT}" "${LOGDIR}"

ZLABELS=("NGC1")
ZONE_LABEL="${ZLABELS[$SLURM_ARRAY_TASK_ID]}"

TRACERS_BY_ZONE=("BGS_BRIGHT")

CONFIG_JSON=""

if [ ${#TRACERS_BY_ZONE[@]} -gt 0 ]; then
  if [ -n "${TRACERS_BY_ZONE[$SLURM_ARRAY_TASK_ID]+x}" ] && [ -n "${TRACERS_BY_ZONE[$SLURM_ARRAY_TASK_ID]}" ]; then
    TRACER_LABEL="${TRACERS_BY_ZONE[$SLURM_ARRAY_TASK_ID]}"
  else
    TRACER_LABEL="${TRACERS_BY_ZONE[0]}"
  fi
elif [ -n "${TRACER}" ]; then
  TRACER_LABEL="${TRACER}"
else
  TRACER_LABEL=""
fi

if [ -z "${TRACER_LABEL}" ]; then
  exit 2
fi

TRACE_OPT=(--tracers "${TRACER_LABEL}" --out-tag "${TRACER_LABEL}")

TRACER_SAFE="${TRACER_LABEL// /_}"

srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} --cpu-bind=cores \
  --output "${LOGDIR}/zone_${ZONE_LABEL}_${TRACER_SAFE}.out" \
  --error  "${LOGDIR}/zone_${ZONE_LABEL}_${TRACER_SAFE}.err" \
  /usr/bin/time -f $'%E elapsed\n%U user CPU\n%S sys CPU\n%P CPU%%\n%M KB max RSS\n' \
  -o "${LOGDIR}/zone_${ZONE_LABEL}_${TRACER_SAFE}.time" \
  python "${MAIN_SCRIPT}" \
    --release DR1 \
    --zones "${ZONE_LABEL}" \
    "${TRACE_OPT[@]}" \
    ${CONFIG_JSON:+--config "${CONFIG_JSON}"} \
    --base-dir  "${CLUSTER_DIR}" \
    --raw-out   "${RAW_OUT}" \
    --class-out "${CLASS_OUT}" \
    --groups-out "${GROUPS_OUT}" \
    --plot-output "${FIGS_OUT}" \
    --webtype filament \
    --source data \
    --n-random 100
