set -euo pipefail
module load python/3.12

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

RELEASE_ROOT=${RELEASE_ROOT:-/pscratch/sd/v/vtorresg/cosmic-web/dr1}

EXTRA_FLAGS=()
if [[ "${INCLUDE_FIGS:-0}" == "1" ]]; then
  EXTRA_FLAGS+=(--include-figs)
fi

PSCRATCH_DIR=${PSCRATCH_DIR:-/pscratch/sd/v/vtorresg/cosmic-web}

RELEASE_VERSION=${RELEASE_VERSION:-v1.0}
TITLE="ASTRA-DESI DR1 Release ${RELEASE_VERSION}"
DESCRIPTION_FILE=${DESCRIPTION_FILE:-${SCRIPT_DIR}/descriptions/dr1.html}
MEMBERS_JSON=${MEMBERS_JSON:-${SCRIPT_DIR}/json/members.json}
KEYWORDS=("ASTRA" "DESI" "cosmic-web" "LSS" "DR1")

KEYWORD_ARGS=()
if ((${#KEYWORDS[@]} > 0)); then
  KEYWORD_ARGS=(--keywords "${KEYWORDS[@]}")
fi

python "${SCRIPT_DIR}/zenodo_push.py" \
  --release-root "$RELEASE_ROOT" \
  --pscratch-dir "$PSCRATCH_DIR" \
  --keep-tree \
  --title "$TITLE" \
  --description-file "$DESCRIPTION_FILE" \
  --creators-json "$MEMBERS_JSON" \
  "${KEYWORD_ARGS[@]}" \
  --token-file ~/.zenodo_token \
# --publish \
  "${EXTRA_FLAGS[@]}"