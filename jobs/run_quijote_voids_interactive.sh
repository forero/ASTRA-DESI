#!/bin/bash
# Run all Quijote FoF void random iterations on one interactive CPU node.
#
# Typical allocation:
#   salloc -A desi -C cpu -q interactive -N 1 -t 04:00:00
#   srun -N 1 -n 1 -c 256 --cpu-bind=cores --pty bash
#   bash jobs/run_quijote_voids_interactive.sh
#
# Re-run the same command in a later allocation to continue.  Successful
# realizations are recorded in the GNU Parallel joblog, while --resume skips
# random iterations already completed inside an interrupted realization.

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: this launcher must run inside a Slurm interactive allocation." >&2
  exit 2
fi

module load python/3.12
source /global/homes/v/vtorresg/venvs/my-umap-env/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/astra-desi-matplotlib
export XDG_CACHE_HOME=/tmp/astra-desi-cache

PROJECT_ROOT=${PROJECT_ROOT:-/global/homes/v/vtorresg/ASTRA-DESI}
INPUT_ROOT=${INPUT_ROOT:-/pscratch/sd/v/vtorresg/quijotes/Halos/FoF}
OUTPUT_ROOT=${OUTPUT_ROOT:-/pscratch/sd/v/vtorresg/quijotes/void_finder/FoF}
RUNNER=${RUNNER:-${PROJECT_ROOT}/catalog/run_quijote_voids.py}
PYTHON_BIN=${PYTHON_BIN:-python}

ITERATION_START=${ITERATION_START:-0}
ITERATION_STOP=${ITERATION_STOP:-99}
BASE_SEED=${BASE_SEED:-12345}
RANDOM_FACTOR=${RANDOM_FACTOR:-1.0}
R_THRESHOLD=${R_THRESHOLD:--0.25}
MIN_MEMBERS=${MIN_MEMBERS:-4}
MIN_HALO_PARTICLES=${MIN_HALO_PARTICLES:-0}
MIN_HALO_MASS=${MIN_HALO_MASS:-}
MAX_HALO_MASS=${MAX_HALO_MASS:-}
BOUNDARY_BUFFER=${BOUNDARY_BUFFER:-auto}
INCLUDE_MEMBERSHIP=${INCLUDE_MEMBERSHIP:-0}

# Optional filters, for example:
#   PARAMETERS="fiducial Om_m Om_p"
#   REALIZATION_MIN=0 REALIZATION_MAX=99
PARAMETERS=${PARAMETERS:-}
REALIZATION_MIN=${REALIZATION_MIN:-0}
REALIZATION_MAX=${REALIZATION_MAX:-2147483647}

# Automatic concurrency uses all allocated cores when the RAM estimate allows
# it.  A full ~310k-halo run has ~620k Delaunay vertices.  The default reserves
# 32 GiB for the OS/filesystem and budgets 1.75 GiB per concurrent process.
MEMORY_PER_WORKER_GB=${MEMORY_PER_WORKER_GB:-1.75}
MEMORY_RESERVE_GB=${MEMORY_RESERVE_GB:-32}
TOTAL_CPUS=${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-$(nproc)}}
TOTAL_CPUS=${TOTAL_CPUS%%(*}
TOTAL_MEMORY_GB=$(awk '/MemTotal/ {printf "%d\n", $2 / 1024 / 1024}' /proc/meminfo)
MAX_MEMORY_WORKERS=$(awk \
  -v total="${TOTAL_MEMORY_GB}" \
  -v reserve="${MEMORY_RESERVE_GB}" \
  -v per_worker="${MEMORY_PER_WORKER_GB}" \
  'BEGIN {value=int((total-reserve)/per_worker); print (value > 0 ? value : 1)}')

if [[ -n "${WORKERS:-}" ]]; then
  ACTIVE_WORKERS=${WORKERS}
else
  if (( TOTAL_CPUS < MAX_MEMORY_WORKERS )); then
    ACTIVE_WORKERS=${TOTAL_CPUS}
  else
    ACTIVE_WORKERS=${MAX_MEMORY_WORKERS}
  fi
fi

if ! [[ "${ACTIVE_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: WORKERS must be a positive integer; got ${ACTIVE_WORKERS}." >&2
  exit 2
fi
if (( ACTIVE_WORKERS > TOTAL_CPUS )); then
  echo "ERROR: WORKERS=${ACTIVE_WORKERS} exceeds allocated CPUs=${TOTAL_CPUS}." >&2
  exit 2
fi
if (( ITERATION_START < 0 || ITERATION_STOP < ITERATION_START )); then
  echo "ERROR: require 0 <= ITERATION_START <= ITERATION_STOP." >&2
  exit 2
fi
if [[ ! -f "${RUNNER}" ]]; then
  echo "ERROR: runner not found: ${RUNNER}" >&2
  exit 2
fi
if ! command -v parallel >/dev/null 2>&1; then
  echo "ERROR: GNU Parallel is not available in PATH." >&2
  exit 2
fi

STATE_ROOT=${STATE_ROOT:-${OUTPUT_ROOT}/run_state}
LOG_ROOT=${LOG_ROOT:-${STATE_ROOT}/logs}
MANIFEST=${MANIFEST:-${STATE_ROOT}/quijote_fof_manifest.txt}
DEFAULT_RUN_TAG=seed${BASE_SEED}_iter${ITERATION_START}-${ITERATION_STOP}_rf${RANDOM_FACTOR}_r${R_THRESHOLD}_m${MIN_MEMBERS}_p${MIN_HALO_PARTICLES}_b${BOUNDARY_BUFFER}_mass${MIN_HALO_MASS:-none}-${MAX_HALO_MASS:-none}_membership${INCLUDE_MEMBERSHIP}
RUN_TAG=${RUN_TAG:-${DEFAULT_RUN_TAG}}
JOBLOG=${JOBLOG:-${STATE_ROOT}/parallel_${RUN_TAG}.tsv}
mkdir -p "${STATE_ROOT}" "${LOG_ROOT}"

parameter_enabled() {
  local candidate=$1
  local wanted
  if [[ -z "${PARAMETERS}" ]]; then
    return 0
  fi
  for wanted in ${PARAMETERS}; do
    if [[ "${candidate}" == "${wanted}" ]]; then
      return 0
    fi
  done
  return 1
}

manifest_tmp=${MANIFEST}.tmp.$$
: > "${manifest_tmp}"
while IFS= read -r first_file; do
  relative=${first_file#${INPUT_ROOT}/}
  parameter=${relative%%/*}
  remainder=${relative#*/}
  realization=${remainder%%/*}
  if ! [[ "${realization}" =~ ^[0-9]+$ ]]; then
    continue
  fi
  if ! parameter_enabled "${parameter}"; then
    continue
  fi
  if (( realization < REALIZATION_MIN || realization > REALIZATION_MAX )); then
    continue
  fi
  printf '%s %d\n' "${parameter}" "${realization}" >> "${manifest_tmp}"
done < <(find "${INPUT_ROOT}" -mindepth 4 -maxdepth 4 -type f \
         -name 'group_tab_003.0' -print | sort)
sort -k1,1 -k2,2n "${manifest_tmp}" -o "${manifest_tmp}"
mv "${manifest_tmp}" "${MANIFEST}"

N_REALIZATIONS=$(wc -l < "${MANIFEST}")
N_ITERATIONS=$((ITERATION_STOP - ITERATION_START + 1))
N_TASKS=$((N_REALIZATIONS * N_ITERATIONS))
if (( N_REALIZATIONS == 0 )); then
  echo "ERROR: no matching Quijote FoF catalogues found under ${INPUT_ROOT}." >&2
  exit 2
fi

echo "[quijote-voids] Slurm job:       ${SLURM_JOB_ID}"
echo "[quijote-voids] CPUs visible:    ${TOTAL_CPUS}"
echo "[quijote-voids] RAM visible:     ${TOTAL_MEMORY_GB} GiB"
echo "[quijote-voids] workers:         ${ACTIVE_WORKERS}"
echo "[quijote-voids] RAM/worker:      ${MEMORY_PER_WORKER_GB} GiB"
echo "[quijote-voids] realizations:    ${N_REALIZATIONS}"
echo "[quijote-voids] iterations:      ${ITERATION_START}-${ITERATION_STOP}"
echo "[quijote-voids] total ASTRA runs: ${N_TASKS}"
echo "[quijote-voids] output:          ${OUTPUT_ROOT}"
echo "[quijote-voids] joblog:          ${JOBLOG}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[quijote-voids] DRY_RUN=1; manifest created, no void jobs launched."
  exit 0
fi

run_one_realization() {
  local parameter=$1
  local realization=$2
  local log_dir=${LOG_ROOT}/${parameter}
  local log_path=${log_dir}/${realization}.log
  local -a command
  mkdir -p "${log_dir}"
  printf '[%s] START parameter=%s realization=%s iterations=%s-%s\n' \
    "$(date -u +%FT%TZ)" "${parameter}" "${realization}" \
    "${ITERATION_START}" "${ITERATION_STOP}" >> "${log_path}"

  command=("${PYTHON_BIN}" -u "${RUNNER}" "${parameter}" "${realization}"
           --input-root "${INPUT_ROOT}"
           --output-root "${OUTPUT_ROOT}"
           --random-seed "${BASE_SEED}"
           --random-iteration "${ITERATION_START}-${ITERATION_STOP}"
           --random-factor "${RANDOM_FACTOR}"
           --r-threshold "${R_THRESHOLD}"
           --min-members "${MIN_MEMBERS}"
           --min-halo-particles "${MIN_HALO_PARTICLES}"
           --boundary-buffer "${BOUNDARY_BUFFER}"
           --resume
           --repair-incomplete)
  if [[ -n "${MIN_HALO_MASS}" ]]; then
    command+=(--min-halo-mass "${MIN_HALO_MASS}")
  fi
  if [[ -n "${MAX_HALO_MASS}" ]]; then
    command+=(--max-halo-mass "${MAX_HALO_MASS}")
  fi
  if [[ "${INCLUDE_MEMBERSHIP}" == "1" ]]; then
    command+=(--include-membership)
  fi
  if ! "${command[@]}" >> "${log_path}" 2>&1; then
    printf '[%s] FAILED parameter=%s realization=%s iteration_range=%s-%s\n' \
      "$(date -u +%FT%TZ)" "${parameter}" "${realization}" \
      "${ITERATION_START}" "${ITERATION_STOP}" >> "${log_path}"
    return 1
  fi
  printf '[%s] DONE parameter=%s realization=%s\n' \
    "$(date -u +%FT%TZ)" "${parameter}" "${realization}" >> "${log_path}"
}

export -f run_one_realization
export PYTHON_BIN RUNNER INPUT_ROOT OUTPUT_ROOT LOG_ROOT
export ITERATION_START ITERATION_STOP BASE_SEED RANDOM_FACTOR
export R_THRESHOLD MIN_MEMBERS MIN_HALO_PARTICLES MIN_HALO_MASS MAX_HALO_MASS
export BOUNDARY_BUFFER INCLUDE_MEMBERSHIP

resume_options=()
if [[ -s "${JOBLOG}" ]]; then
  resume_options=(--resume-failed)
  echo "[quijote-voids] resuming failed/unfinished realization jobs from joblog."
fi

parallel --jobs "${ACTIVE_WORKERS}" \
  --colsep '[[:space:]]+' \
  --joblog "${JOBLOG}" \
  "${resume_options[@]}" \
  --eta \
  run_one_realization {1} {2} :::: "${MANIFEST}"

echo "[quijote-voids] all selected realization jobs completed."
