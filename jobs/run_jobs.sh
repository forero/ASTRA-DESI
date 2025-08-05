#!/usr/bin/env bash
set -euo pipefail

module load python/3.12

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAIN_SCRIPT="$PROJECT_ROOT/src/main.py"

CLUSTER_DIR="/global/cfs/cdirs/desi/public/edr/vac/edr/lss/v2.0/LSScats/clustering"
SCRATCH_BASE="/pscratch/sd/v/vtorresg/cosmic-web"

RAW_OUT="$SCRATCH_BASE/edr/raw"
CLASS_OUT="$SCRATCH_BASE/edr/class"
LOGDIR="$SCRATCH_BASE/edr/logs"

mkdir -p "$RAW_OUT" "$CLASS_OUT" "$LOGDIR"

ZONE_ARG="${1:-all}"

run_zone(){
  local zone=$1
  local ts=$(date +%Y%m%d_%H%M%S)
  echo "[$(date +%H:%M:%S)] -> Zone $zone"

  local tlog="$LOGDIR/zone_${zone}_${ts}.time"
  # elapsed, user CPU, sys CPU, CPU%, max RSS in KB
  local fmt=$'%E elapsed\n%U user CPU\n%S sys CPU\n%P CPU%%\n%M KB max RSS\n'

  /usr/bin/time -f "$fmt" -o "$tlog" \
    python3 "$MAIN_SCRIPT" \
      --zone      "$zone" \
      --base-dir  "$CLUSTER_DIR" \
      --raw-out   "$RAW_OUT" \
      --class-out "$CLASS_OUT" \
      # 1>>"$LOGDIR/zone_${zone}_${ts}.out" \
      2>>"$LOGDIR/zone_${zone}_${ts}.err"

  local kb=$(awk '/KB max RSS/ {print $1}' "$tlog")
  local gb=$(awk "BEGIN {printf \"%.2f\", $kb/1024/1024}")
  echo "$gb GB max RSS" >> "$tlog"
}

if [[ "$ZONE_ARG" == "all" ]]; then
  for z in $(seq 0 19); do
    run_zone "$z"
  done
else
  run_zone "$ZONE_ARG"
fi