#!/bin/bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 MANIFEST" >&2
  echo "Configure the range with START_TASK and TASK_COUNT." >&2
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
RUNNER="${PROJECT_ROOT}/src/power_quijote.py"

# Many smaller FFT teams give substantially better node throughput than a few
# 32-thread teams. At the measured ~6.25 GiB peak per N=512 simulation, this
# layout uses about 300 GiB and leaves ~200 GiB free on a 500-GiB node.
SIMS_PARALLEL=${SIMS_PARALLEL:-48}
CPUS_PER_SIM=${CPUS_PER_SIM:-5}
PK_THREADS=${PK_THREADS:-5}
START_TASK=${START_TASK:-0}
TASK_COUNT=${TASK_COUNT:-2500}
GRID=${GRID:-512}
MAS=${MAS:-CIC}
AXIS=${AXIS:-2}
BIN_WIDTH_KF=${BIN_WIDTH_KF:-2}
KMAX=${KMAX:-0.5}
MAKE_PLOTS=${MAKE_PLOTS:-0}

INPUT_ROOT=${INPUT_ROOT:-/pscratch/sd/v/vtorresg/quijotes/Halos/FoF}
ASTRA_ROOT=${ASTRA_ROOT:-/pscratch/sd/v/vtorresg/quijotes/ASTRA/FoF}
OUTPUT_ROOT=${OUTPUT_ROOT:-/pscratch/sd/v/vtorresg/quijotes/PowerSpectrum/FoF}
PYTHON_BIN=${PYTHON_BIN:-/global/homes/v/vtorresg/venvs/pylians/bin/python}
LOG_ROOT=${LOG_ROOT:-${PROJECT_ROOT}/logs/power-quijote-${SLURM_JOB_ID}}

for value in "${SIMS_PARALLEL}" "${CPUS_PER_SIM}" "${PK_THREADS}" \
             "${TASK_COUNT}" "${GRID}" "${BIN_WIDTH_KF}"; do
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Parallelism, task count, grid, and bin width must be positive integers." >&2
    exit 2
  fi
done
if ! [[ "${START_TASK}" =~ ^[0-9]+$ ]]; then
  echo "START_TASK must be a non-negative integer." >&2
  exit 2
fi
if [[ "${MAKE_PLOTS}" != 0 && "${MAKE_PLOTS}" != 1 ]]; then
  echo "MAKE_PLOTS must be 0 or 1." >&2
  exit 2
fi
if (( PK_THREADS > CPUS_PER_SIM )); then
  echo "PK_THREADS cannot exceed CPUS_PER_SIM." >&2
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
  echo "Requested ${REQUESTED_CPUS} CPUs but only ${AVAILABLE_CPUS} are allocated." >&2
  exit 2
fi

TOTAL_TASKS=$(awk 'NF >= 4 && $1 !~ /^#/ {count++} END {print count+0}' "${MANIFEST_SOURCE}")
if (( TOTAL_TASKS == 0 || START_TASK >= TOTAL_TASKS )); then
  echo "No manifest tasks available from START_TASK=${START_TASK}." >&2
  exit 2
fi
STOP_TASK=$((START_TASK + TASK_COUNT - 1))
if (( STOP_TASK >= TOTAL_TASKS )); then
  STOP_TASK=$((TOTAL_TASKS - 1))
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Pylians Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if ! "${PYTHON_BIN}" -c 'import fitsio, MAS_library, numpy, Pk_library, readfof, redshift_space_library' >/dev/null; then
  echo "Pylians Python cannot import the packages needed by power_quijote.py." >&2
  exit 2
fi
if ! command -v parallel >/dev/null; then
  echo "GNU parallel is required but was not found." >&2
  exit 2
fi

export OMP_NUM_THREADS="${PK_THREADS}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "${LOG_ROOT}"
JOBLOG="${LOG_ROOT}/parallel.tsv"
MANIFEST="${LOG_ROOT}/manifest.snapshot"
cp "${MANIFEST_SOURCE}" "${MANIFEST}"

extra_args=(--no-plot)
if (( MAKE_PLOTS == 1 )); then
  module load python/3.12
  extra_args=()
fi

echo "power-quijote --> allocation=${SLURM_JOB_ID} CPUs=${AVAILABLE_CPUS}"
echo "power-quijote --> tasks=${START_TASK}..${STOP_TASK} simulations=${SIMS_PARALLEL} threads/simulation=${PK_THREADS}"
echo "power-quijote --> grid=${GRID} MAS=${MAS} delta_k=${BIN_WIDTH_KF}kf kmax=${KMAX} plots=${MAKE_PLOTS}"
echo "power-quijote --> logs=${LOG_ROOT} manifest=${MANIFEST}"

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
      --astra-root "${ASTRA_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --manifest "${MANIFEST}" \
      --task-index {} \
      --grid "${GRID}" \
      --mas "${MAS}" \
      --threads "${PK_THREADS}" \
      --axis "${AXIS}" \
      --bin-width-kf "${BIN_WIDTH_KF}" \
      --kmax "${KMAX}" \
      "${extra_args[@]}"

echo "power-quijote --> completed tasks ${START_TASK}..${STOP_TASK}"
