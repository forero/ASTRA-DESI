#!/bin/bash
#SBATCH --job-name=cosmic_web_astra_dr1
#SBATCH --account=desi
#SBATCH --partition=regular_milan_ss11
#SBATCH --time=00:30:00
#SBATCH --mem=0
#SBATCH --cpus-per-task=64
#SBATCH --array=0-1%1
#SBATCH --ntasks=1
#SBATCH -C cpu
#SBATCH --output=/pscratch/sd/v/vtorresg/cosmic-web/dr1/logs/zone_%A_%a.out
#SBATCH --error=/pscratch/sd/v/vtorresg/cosmic-web/dr1/logs/zone_%A_%a.err
#SBATCH --chdir=/global/homes/v/vtorresg/ASTRA-DESI

module load python/3.12

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK}

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

ZLABELS=("NGC1" "NGC2")
ZONE_LABEL="${ZLABELS[$SLURM_ARRAY_TASK_ID]}"

CONFIG_JSON=""

srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} --cpu-bind=cores /usr/bin/time -f \
    $'%E elapsed\n%U user CPU\n%S sys CPU\n%P CPU%%\n%M KB max RSS\n' \
    -o "${LOGDIR}/zone_${ZONE_LABEL}.time" \
    python "${MAIN_SCRIPT}" \
      --release DR1 \
      --region N \
      --zones "${ZONE_LABEL}" \
      ${CONFIG_JSON:+--config "${CONFIG_JSON}"} \
      --base-dir  "${CLUSTER_DIR}" \
      --raw-out   "${RAW_OUT}" \
      --class-out "${CLASS_OUT}" \
      --groups-out "${GROUPS_OUT}" \
      --plot-output "${FIGS_OUT}" \
      --webtype filament \
      --source data \
      --n-random 1