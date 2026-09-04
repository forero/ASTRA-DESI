#!/bin/bash
# Build Quijote consensus catalogues only for realizations whose requested
# random-iteration completion products are all present.
#
# Run inside a full interactive CPU-node step, for example:
#   salloc -A desi -C cpu -q interactive -N 1 -t 04:00:00
#   srun -N 1 -n 1 -c 256 --cpu-bind=cores --pty bash
#   bash jobs/run_quijote_consensus_ready_interactive.sh

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: run this launcher inside a Slurm interactive allocation." >&2
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
INPUT_ROOT=${INPUT_ROOT:-/pscratch/sd/v/vtorresg/quijotes/void_finder/FoF}
RUNNER=${RUNNER:-${PROJECT_ROOT}/catalog/run_quijote_consensus.py}
PYTHON_BIN=${PYTHON_BIN:-python}
SNAPSHOT=${SNAPSHOT:-3}
ITERATION_START=${ITERATION_START:-0}
ITERATION_STOP=${ITERATION_STOP:-99}
VOL_FRAC=${VOL_FRAC:-0.5}
V_CUT=${V_CUT:-0.5}
QUERY_WORKERS=${QUERY_WORKERS:-1}
QUERY_BATCH_SIZE=${QUERY_BATCH_SIZE:-4096}
KEEP_ALL=${KEEP_ALL:-0}

# Optional selection examples:
#   PARAMETERS="fiducial Om_m Om_p"
#   REALIZATION_MIN=0 REALIZATION_MAX=499
PARAMETERS=${PARAMETERS:-}
REALIZATION_MIN=${REALIZATION_MIN:-0}
REALIZATION_MAX=${REALIZATION_MAX:-2147483647}

# One consensus process uses one CPU worker by default.  The memory estimate
# bounds concurrency while allowing a 503-GiB CPU node to use all 256 cores.
MEMORY_PER_WORKER_GB=${MEMORY_PER_WORKER_GB:-1.5}
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
elif (( TOTAL_CPUS < MAX_MEMORY_WORKERS )); then
  ACTIVE_WORKERS=${TOTAL_CPUS}
else
  ACTIVE_WORKERS=${MAX_MEMORY_WORKERS}
fi

if ! [[ "${ACTIVE_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: WORKERS must be a positive integer." >&2
  exit 2
fi
if (( ACTIVE_WORKERS > TOTAL_CPUS )); then
  echo "ERROR: WORKERS=${ACTIVE_WORKERS} exceeds allocated CPUs=${TOTAL_CPUS}." >&2
  exit 2
fi
if (( ITERATION_START < 0 || ITERATION_STOP < ITERATION_START || ITERATION_STOP >= 1000 )); then
  echo "ERROR: require 0 <= ITERATION_START <= ITERATION_STOP < 1000." >&2
  exit 2
fi
if [[ ! -f "${RUNNER}" ]]; then
  echo "ERROR: consensus runner not found: ${RUNNER}" >&2
  exit 2
fi
if ! command -v parallel >/dev/null 2>&1; then
  echo "ERROR: GNU Parallel is not available in PATH." >&2
  exit 2
fi

STATE_ROOT=${STATE_ROOT:-${INPUT_ROOT}/consensus_run_state}
LOG_ROOT=${LOG_ROOT:-${STATE_ROOT}/logs}
READY_MANIFEST=${READY_MANIFEST:-${STATE_ROOT}/ready_manifest.txt}
SKIPPED_MANIFEST=${SKIPPED_MANIFEST:-${STATE_ROOT}/skipped_incomplete.txt}
N_ITERATIONS=$((ITERATION_STOP - ITERATION_START + 1))
DEFAULT_RUN_TAG=n${N_ITERATIONS}_iter${ITERATION_START}-${ITERATION_STOP}_vf${VOL_FRAC}_vc${V_CUT}_all${KEEP_ALL}
RUN_TAG=${RUN_TAG:-${DEFAULT_RUN_TAG}}
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)_$$
JOBLOG=${JOBLOG:-${STATE_ROOT}/parallel_${RUN_TAG}_${SLURM_JOB_ID}_${RUN_STAMP}.tsv}
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

ready_tmp=${READY_MANIFEST}.tmp.$$
skipped_tmp=${SKIPPED_MANIFEST}.tmp.$$
candidates_tmp=${STATE_ROOT}/candidates.tmp.$$
scan_tmp=${STATE_ROOT}/readiness_scan.tmp.$$
: > "${ready_tmp}"
: > "${skipped_tmp}"
: > "${candidates_tmp}"
: > "${scan_tmp}"
printf -v snapshot_label '%03d' "${SNAPSHOT}"

for parameter_root in "${INPUT_ROOT}"/*; do
  [[ -d "${parameter_root}" ]] || continue
  parameter=${parameter_root##*/}
  if ! parameter_enabled "${parameter}"; then
    continue
  fi

  for realization_root in "${parameter_root}"/*; do
    [[ -d "${realization_root}" ]] || continue
    realization=${realization_root##*/}
    if ! [[ "${realization}" =~ ^[0-9]+$ ]]; then
      continue
    fi
    if (( realization < REALIZATION_MIN || realization > REALIZATION_MAX )); then
      continue
    fi
    printf '%s %d\n' "${parameter}" "${realization}" >> "${candidates_tmp}"
  done
done

check_case_readiness() {
  local parameter=$1
  local realization=$2
  local case_root=${INPUT_ROOT}/${parameter}/${realization}/groups_${snapshot_label}
  local iteration iteration_label product
  local -A products=()

  # Read each random_* directory once.  Testing every path with `-s` causes
  # hundreds of slow, individual Lustre metadata lookups per realization.
  # summary.json is written last by the void runner, so its presence together
  # with voids_all.fits is the completion marker.  The Python runner performs
  # the full FITS/JSON validation before building the consensus catalogue.
  while IFS= read -r product; do
    products["${product}"]=1
  done < <(find "${case_root}" -mindepth 2 -maxdepth 2 \
             \( -name summary.json -o -name voids_all.fits \) \
             -printf '%P\n' 2>/dev/null)

  for ((iteration=ITERATION_START; iteration<=ITERATION_STOP; iteration++)); do
    printf -v iteration_label '%03d' "${iteration}"
    if [[ -z "${products[random_${iteration_label}/summary.json]+x}" ||
          -z "${products[random_${iteration_label}/voids_all.fits]+x}" ]]; then
      printf 'SKIP\t%s\t%d\tfirst_missing=%d\n' \
        "${parameter}" "${realization}" "${iteration}"
      return 0
    fi
  done
  printf 'READY\t%s\t%d\n' "${parameter}" "${realization}"
}

export -f check_case_readiness
export INPUT_ROOT snapshot_label ITERATION_START ITERATION_STOP

N_CANDIDATES=$(wc -l < "${candidates_tmp}")
echo "[quijote-consensus] scanning ${N_CANDIDATES} candidate realizations with ${ACTIVE_WORKERS} workers..."
if (( N_CANDIDATES > 0 )); then
  parallel --jobs "${ACTIVE_WORKERS}" \
    --colsep '[[:space:]]+' \
    check_case_readiness {1} {2} :::: "${candidates_tmp}" > "${scan_tmp}"
fi

awk -F '\t' '$1 == "READY" {print $2, $3}' "${scan_tmp}" > "${ready_tmp}"
awk -F '\t' '$1 == "SKIP" {print $2, $3, $4}' "${scan_tmp}" > "${skipped_tmp}"
rm -f "${candidates_tmp}" "${scan_tmp}"

sort -k1,1 -k2,2n "${ready_tmp}" -o "${ready_tmp}"
sort -k1,1 -k2,2n "${skipped_tmp}" -o "${skipped_tmp}"
mv "${ready_tmp}" "${READY_MANIFEST}"
mv "${skipped_tmp}" "${SKIPPED_MANIFEST}"

N_READY=$(wc -l < "${READY_MANIFEST}")
N_SKIPPED=$(wc -l < "${SKIPPED_MANIFEST}")
echo "[quijote-consensus] Slurm job:          ${SLURM_JOB_ID}"
echo "[quijote-consensus] CPUs visible:       ${TOTAL_CPUS}"
echo "[quijote-consensus] RAM visible:        ${TOTAL_MEMORY_GB} GiB"
echo "[quijote-consensus] workers:            ${ACTIVE_WORKERS}"
echo "[quijote-consensus] iterations:         ${ITERATION_START}-${ITERATION_STOP}"
echo "[quijote-consensus] ready realizations: ${N_READY}"
echo "[quijote-consensus] skipped incomplete: ${N_SKIPPED}"
echo "[quijote-consensus] ready manifest:     ${READY_MANIFEST}"
echo "[quijote-consensus] skipped manifest:   ${SKIPPED_MANIFEST}"
echo "[quijote-consensus] joblog:             ${JOBLOG}"

if (( N_READY == 0 )); then
  echo "[quijote-consensus] nothing ready; exiting successfully."
  exit 0
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[quijote-consensus] DRY_RUN=1; no consensus jobs launched."
  exit 0
fi

run_one_consensus() {
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
           --snapshot "${SNAPSHOT}"
           --iterations "${ITERATION_START}-${ITERATION_STOP}"
           --vol-frac "${VOL_FRAC}"
           --v-cut "${V_CUT}"
           --query-workers "${QUERY_WORKERS}"
           --query-batch-size "${QUERY_BATCH_SIZE}"
           --resume
           --repair-incomplete)
  if [[ "${KEEP_ALL}" == "1" ]]; then
    command+=(--keep-all)
  fi
  if ! "${command[@]}" >> "${log_path}" 2>&1; then
    printf '[%s] FAILED parameter=%s realization=%s\n' \
      "$(date -u +%FT%TZ)" "${parameter}" "${realization}" >> "${log_path}"
    return 1
  fi
  printf '[%s] DONE parameter=%s realization=%s\n' \
    "$(date -u +%FT%TZ)" "${parameter}" "${realization}" >> "${log_path}"
}

export -f run_one_consensus
export PYTHON_BIN RUNNER INPUT_ROOT LOG_ROOT SNAPSHOT
export ITERATION_START ITERATION_STOP VOL_FRAC V_CUT
export QUERY_WORKERS QUERY_BATCH_SIZE KEEP_ALL

parallel --jobs "${ACTIVE_WORKERS}" \
  --colsep '[[:space:]]+' \
  --joblog "${JOBLOG}" \
  --eta \
  run_one_consensus {1} {2} :::: "${READY_MANIFEST}"

echo "[quijote-consensus] all ready consensus jobs completed."
