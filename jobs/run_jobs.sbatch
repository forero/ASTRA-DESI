#!/usr/bin/env bash
#SBATCH --job-name=cosmicweb
#SBATCH --account=desi
#SBATCH --partition=regular
#SBATCH --array=0-19
#SBATCH --output=/pscratch/sd/v/vtorresg/cosmic-web/logs/zone_%A_%a.out
#SBATCH --error=/pscratch/sd/v/vtorresg/cosmic-web/logs/zone_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

module load python/3.12

# Project root (adjust if needed)
PROJECT_ROOT="$HOME/ASTRA-DEV"        # or wherever your repo lives
WRAPPER="$PROJECT_ROOT/jobs/run_jobs.sh"

BASE_DIR="/global/cfs/cdirs/desi/public/edr/vac/edr/lss/v2.0/LSScats/clustering"
SCRATCH="/pscratch/sd/v/vtorresg/cosmic-web"

mkdir -p "$SCRATCH/logs"

ZONE=$SLURM_ARRAY_TASK_ID

echo "[$(date +%H:%M:%S)] Starting zone $ZONE"
srun bash "$WRAPPER" "$ZONE" "$BASE_DIR" \
     "$SCRATCH/01_create_raw" "$SCRATCH/02_astra_classification"
