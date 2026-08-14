#!/bin/bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 MANIFEST" >&2
  echo "Configure the task range with START_TASK and TASK_COUNT." >&2
  exit 2
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This launcher must run inside a Slurm allocation." >&2
  exit 2
fi

MANIFEST_SOURCE=$(realpath "$1")
if [[ ! -f "${MANIFEST_SOURCE}" ]]; then
  echo "Manifest not found: ${MANIFEST_SOURCE}" >&2
  exit 2
fi

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUNNER="${PROJECT_ROOT}/src/run_quijote.py"

# Safe full-node defaults measured for ~310k-halo Quijote catalogues:
# 5 simulations x 50 Delaunay workers = 250 of 256 CPU cores and ~285 GiB.
SIMS_PARALLEL=${SIMS_PARALLEL:-5}
CPUS_PER_SIM=${CPUS_PER_SIM:-50}
ITER_WORKERS=${ITER_WORKERS:-50}
N_ITERATIONS=${N_ITERATIONS:-100}
START_TASK=${START_TASK:-0}
TASK_COUNT=${TASK_COUNT:-700}

INPUT_ROOT=${INPUT_ROOT:-/pscratch/sd/v/vtorresg/quijotes/Halos/FoF}
OUTPUT_ROOT=${OUTPUT_ROOT:-/pscratch/sd/v/vtorresg/quijotes/ASTRA/FoF}
READFOF_PATH=${READFOF_PATH:-/global/homes/v/vtorresg/venvs/pylians/lib64/python3.6/site-packages/readfof.py}
LOG_ROOT=${LOG_ROOT:-${PROJECT_ROOT}/logs/quijote-interactive-${SLURM_JOB_ID}}

for value in "${SIMS_PARALLEL}" "${CPUS_PER_SIM}" "${ITER_WORKERS}" \
             "${N_ITERATIONS}" "${TASK_COUNT}"; do
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Parallelism, iteration, and task-count values must be positive integers." >&2
    exit 2
  fi
done
if ! [[ "${START_TASK}" =~ ^[0-9]+$ ]]; then
  echo "START_TASK must be a non-negative integer." >&2
  exit 2
fi
if (( ITER_WORKERS > CPUS_PER_SIM )); then
  echo "ITER_WORKERS cannot exceed CPUS_PER_SIM." >&2
  exit 2
fi

available_text=${SLURM_CPUS_ON_NODE:-${SLURM_JOB_CPUS_PER_NODE:-256}}
if [[ "${available_text}" =~ ^([0-9]+) ]]; then
  AVAILABLE_CPUS=${BASH_REMATCH[1]}
else
  echo "Cannot parse allocated CPU count: ${available_text}" >&2
  exit 2
fi
REQUESTED_CPUS=$((SIMS_PARALLEL * CPUS_PER_SIM))
if (( REQUESTED_CPUS > AVAILABLE_CPUS )); then
  echo "Requested ${REQUESTED_CPUS} concurrent CPUs but only ${AVAILABLE_CPUS} are allocated." >&2
  exit 2
fi

TOTAL_TASKS=$(awk 'NF >= 2 && $1 !~ /^#/ {count++} END {print count+0}' "${MANIFEST_SOURCE}")
if (( TOTAL_TASKS == 0 || START_TASK >= TOTAL_TASKS )); then
  echo "No manifest tasks available from START_TASK=${START_TASK}." >&2
  exit 2
fi
STOP_TASK=$((START_TASK + TASK_COUNT - 1))
if (( STOP_TASK >= TOTAL_TASKS )); then
  STOP_TASK=$((TOTAL_TASKS - 1))
fi

module load python/3.12
if [[ -z "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN=$(command -v python3.12 || true)
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
  shopt -s nullglob
  nersc_pythons=(/global/common/software/nersc/pe/conda-envs/*/python-3.12/nersc-python/bin/python3.12)
  shopt -u nullglob
  if (( ${#nersc_pythons[@]} > 0 )); then
    PYTHON_BIN=${nersc_pythons[${#nersc_pythons[@]}-1]}
  fi
fi
if [[ -z "${PYTHON_BIN:-}" ]] || \
   ! "${PYTHON_BIN}" -c 'import sys; assert sys.version_info[:2] == (3, 12); import numpy, scipy, fitsio' >/dev/null; then
  echo "Python must provide the current NumPy, SciPy, and fitsio stack." >&2
  exit 2
fi
if [[ ! -f "${READFOF_PATH}" ]]; then
  echo "readfof.py not found: ${READFOF_PATH}" >&2
  exit 2
fi
if ! command -v parallel >/dev/null; then
  echo "GNU parallel is required but was not found." >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "${LOG_ROOT}"
JOBLOG="${LOG_ROOT}/parallel.tsv"
MANIFEST="${LOG_ROOT}/manifest.snapshot"
cp "${MANIFEST_SOURCE}" "${MANIFEST}"

echo "quijote-interactive --> allocation=${SLURM_JOB_ID} CPUs=${AVAILABLE_CPUS}"
echo "quijote-interactive --> tasks=${START_TASK}..${STOP_TASK} simulations=${SIMS_PARALLEL} workers/simulation=${ITER_WORKERS}"
echo "quijote-interactive --> iterations=${N_ITERATIONS} logs=${LOG_ROOT}"
echo "quijote-interactive --> manifest snapshot=${MANIFEST}"

seq "${START_TASK}" "${STOP_TASK}" | parallel \
  --jobs "${SIMS_PARALLEL}" \
  --line-buffer \
  --tag \
  --halt soon,fail=1 \
  --joblog "${JOBLOG}" \
  srun --exclusive --exact --nodes=1 --ntasks=1 \
    --cpus-per-task="${CPUS_PER_SIM}" --cpu-bind=cores \
    "${PYTHON_BIN}" "${RUNNER}" \
      --input-root "${INPUT_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --readfof-path "${READFOF_PATH}" \
      --manifest "${MANIFEST}" \
      --task-index {} \
      --n-iterations "${N_ITERATIONS}" \
      --iteration-workers "${ITER_WORKERS}" \

echo "quijote-interactive --> completed tasks ${START_TASK}..${STOP_TASK}"
