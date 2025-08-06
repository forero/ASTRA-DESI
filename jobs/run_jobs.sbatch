#!/bin/bash
#SBATCH --job-name=cosmic_web_astra
#SBATCH --account=desi
#SBATCH --partition=regular_milan_ss11
#SBATCH --time=00:30:00
#SBATCH --mem=20G
#SBATCH --array=START-END%5
#SBATCH -C cpu
#SBATCH --output=/pscratch/sd/v/vtorresg/cosmic-web/logs/zone_%a.out
#SBATCH --error=/pscratch/sd/v/vtorresg/cosmic-web/logs/zone_%a.err

module load python/3.12

PROJECT_ROOT="${SLURM_SUBMIT_DIR}/.."
MAIN_SCRIPT="${PROJECT_ROOT}/src/main.py"

CLUSTER_DIR="/global/cfs/cdirs/desi/public/edr/vac/edr/lss/v2.0/LSScats/clustering"
SCRATCH_BASE="/pscratch/sd/v/vtorresg/cosmic-web"

RAW_OUT="${SCRATCH_BASE}/edr/raw"
CLASS_OUT="${SCRATCH_BASE}/edr/class"
LOGDIR="${SCRATCH_BASE}/edr/logs"

mkdir -p "${RAW_OUT}" "${CLASS_OUT}" "${LOGDIR}"

ZONE="${SLURM_ARRAY_TASK_ID}"
echo "[$(date +%H:%M:%S)] Procesando zona ${ZONE}"

srun /usr/bin/time -f \
    $'%E elapsed\n%U user CPU\n%S sys CPU\n%P CPU%%\n%M KB max RSS\n' \
    -o "${LOGDIR}/zone_${ZONE}.time" \
    python "${MAIN_SCRIPT}" \
      --zone      "${ZONE}" \
      --base-dir  "${CLUSTER_DIR}" \
      --raw-out   "${RAW_OUT}" \
      --class-out "${CLASS_OUT}"
