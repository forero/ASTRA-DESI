#!/bin/bash
#SBATCH --job-name=astra_edr
#SBATCH --account=desi
#SBATCH --partition=regular
#SBATCH --time=00:30:00
#SBATCH --mem=0
#SBATCH --array=0-19%5
#SBATCH -C cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --output=/pscratch/sd/v/vtorresg/cosmic-web/edr/logs/zone_%a.out
#SBATCH --error=/pscratch/sd/v/vtorresg/cosmic-web/edr/logs/zone_%a.err
#SBATCH --chdir=/global/homes/v/vtorresg/ASTRA-DESI

set -euo pipefail

module load python/3.12

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

R_LOWER=${R_LOWER:--0.9}
R_UPPER=${R_UPPER:-0.9}
export PAIR_NJOBS_CAP=${PAIR_NJOBS_CAP:-128}

PROJECT_ROOT=$PWD
MAIN_SCRIPT="${PROJECT_ROOT}/src/main.py"

CLUSTER_DIR="/global/cfs/cdirs/desi/public/edr/vac/edr/lss/v2.0/LSScats/clustering"
SCRATCH_ROOT="/pscratch/sd/v/vtorresg/cosmic-web/edr"

RAW_OUT="${SCRATCH_ROOT}/raw"
RELEASE_ROOT="${SCRATCH_ROOT}"
GROUPS_OUT="${SCRATCH_ROOT}/groups"
PLOTS_OUT="${SCRATCH_ROOT}/figs"
LOGDIR="${SCRATCH_ROOT}/logs"

mkdir -p "${RAW_OUT}" "${RELEASE_ROOT}" "${GROUPS_OUT}" "${PLOTS_OUT}" "${LOGDIR}"

ZONE="${SLURM_ARRAY_TASK_ID}"

srun --cpu-bind=cores \
     /usr/bin/time -f $'%E elapsed\n%U user CPU\n%S sys CPU\n%P CPU%%\n%M KB max RSS\n' \
     -o "${LOGDIR}/zone_${ZONE}.time" \
     python "${MAIN_SCRIPT}" \
       --release EDR \
       --zone "${ZONE}" \
       --base-dir "${CLUSTER_DIR}" \
       --raw-out "${RAW_OUT}" \
       --class-out "${RELEASE_ROOT}" \
       --groups-out "${GROUPS_OUT}" \
       --plot-output "${PLOTS_OUT}" \
       --r-lower "${R_LOWER}" \
       --r-upper "${R_UPPER}" \
       --n-random 100 \
       --plot