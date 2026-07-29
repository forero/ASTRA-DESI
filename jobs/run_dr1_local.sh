#!/usr/bin/env bash
set -euo pipefail

#   bash jobs/run_dr1_local.sh all
#   bash jobs/run_dr1_local.sh LRG
#
#   DATA_DIR=/temp/data
#   OUTPUT_ROOT=/ruta/a/resultados
#   N_RANDOM=100
#   N_RANDOM_FILES=18
#   PAIR_NJOBS_CAP=8

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

DATA_DIR=${DATA_DIR:-/temp/data}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${PROJECT_ROOT}/outputs/dr1_local"}
N_RANDOM=${N_RANDOM:-100}
N_RANDOM_FILES=${N_RANDOM_FILES:-18}
PAIR_NJOBS_CAP=${PAIR_NJOBS_CAP:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)}
SELECTION=${1:-all}
SELECTION=${SELECTION%_}

if command -v python3 >/dev/null 2>&1; then
  PYTHON=${PYTHON:-python3}
else
  PYTHON=${PYTHON:-python}
fi

TRACERS=(
  "BGS_ANY"
  "BGS_BRIGHT-21.5"
  "LRG"
  "ELG_LOPnotqso"
  "QSO"
)
ZONES=("NGC" "SGC")

case "${SELECTION}" in
  all)
    SELECTED_TRACERS=("${TRACERS[@]}")
    ;;
  BGS_ANY|BGS_BRIGHT-21.5|LRG|ELG_LOPnotqso|QSO)
    SELECTED_TRACERS=("${SELECTION}")
    ;;
  *)
    echo "Uso: $0 {all|BGS_ANY|BGS_BRIGHT-21.5|LRG|ELG_LOPnotqso|QSO}" >&2
    exit 2
    ;;
esac

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "No existe el directorio de entrada: ${DATA_DIR}" >&2
  exit 1
fi

RAW_OUT="${OUTPUT_ROOT}/raw"
CLASS_ROOT="${OUTPUT_ROOT}"
GROUPS_OUT="${OUTPUT_ROOT}/groups"
SPILL_DIR="${OUTPUT_ROOT}/spill"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${RAW_OUT}" "${GROUPS_OUT}" "${SPILL_DIR}" "${LOG_DIR}"

export PAIR_NJOBS_CAP
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export ASTRA_CLASS_SPLIT_ITER=0
export ASTRA_CLASS_SKIP_COMBINED=0
export ASTRA_PROB_SPLIT_ITER=0
export ASTRA_PROB_SKIP_COMBINED=0
export ASTRA_SKIP_GROUPS=${ASTRA_SKIP_GROUPS:-0}

for tracer in "${SELECTED_TRACERS[@]}"; do
  for zone in "${ZONES[@]}"; do
    real_file="${DATA_DIR}/${tracer}_${zone}_clustering.dat.fits"
    if [[ ! -f "${real_file}" ]]; then
      echo "Falta el catálogo de datos: ${real_file}" >&2
      exit 1
    fi

    for ((idx = 0; idx < N_RANDOM_FILES; idx++)); do
      random_file="${DATA_DIR}/${tracer}_${zone}_${idx}_clustering.ran.fits"
      if [[ ! -f "${random_file}" ]]; then
        echo "Falta el catálogo random: ${random_file}" >&2
        exit 1
      fi
    done
  done

  echo "==> Proc DR1 ${tracer}: NGC and SGC"
  "${PYTHON}" "${PROJECT_ROOT}/src/main.py" \
    --release DR1 \
    --local-zone-files \
    --base-dir "${DATA_DIR}" \
    --zones NGC SGC \
    --tracers "${tracer}" \
    --out-tag "${tracer}" \
    --raw-out "${RAW_OUT}" \
    --class-out "${CLASS_ROOT}" \
    --groups-out "${GROUPS_OUT}" \
    --spill-dir "${SPILL_DIR}" \
    --n-random "${N_RANDOM}" \
    --n-random-files "${N_RANDOM_FILES}" \
    --progress \
    2>&1 | tee "${LOG_DIR}/${tracer}.log"
done

echo "Pipeline terminado. Resultados en: ${OUTPUT_ROOT}"
