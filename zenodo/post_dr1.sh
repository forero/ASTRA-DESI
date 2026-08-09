#!/usr/bin/env bash
set -euo pipefail

if ! python -c 'import sys, requests; raise SystemExit(sys.version_info < (3, 8))' >/dev/null 2>&1; then
  if command -v module >/dev/null 2>&1; then
    module load python/3.12 >/dev/null 2>&1
  fi
fi

if ! python -c 'import sys, requests; raise SystemExit(sys.version_info < (3, 8))' >/dev/null 2>&1; then
  echo 'ERROR: Python >=3.8 with requests is required.' >&2
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

DATA_DIR=${DATA_DIR:-/pscratch/sd/v/vtorresg/cosmic-web/dr1/zenodo}
EXISTING_DEPOSITION_ID=${EXISTING_DEPOSITION_ID:-19687221}
RESUME_DRAFT_ID=${RESUME_DRAFT_ID:-}

TITLE=${TITLE:-ASTRA-DESI DR1 Release v0.13}
ZENODO_VERSION=${ZENODO_VERSION:-v13}
DESCRIPTION_FILE=${DESCRIPTION_FILE:-${SCRIPT_DIR}/json/desc.md}
MEMBERS_JSON=${MEMBERS_JSON:-${SCRIPT_DIR}/json/members.json}
TOKEN_ENV=${TOKEN_ENV:-ZENODO_TOKEN}

# Zenodo documents a standard limit of 50 GB per record. Set this to 1 only
# after allocating enough additional quota to the draft for this 67.14 GB upload.
ZENODO_ALLOW_LARGE_UPLOAD=${ZENODO_ALLOW_LARGE_UPLOAD:-0}
ZENODO_MAX_TOTAL_SIZE_GB=${ZENODO_MAX_TOTAL_SIZE_GB:-50}

# Publishing is intentionally opt-in. The default leaves a draft for review.
PUBLISH=${PUBLISH:-0}
DRY_RUN=${DRY_RUN:-0}
CREATE_DRAFT_ONLY=${CREATE_DRAFT_ONLY:-0}

shopt -s nullglob
ARCHIVES=("${DATA_DIR}"/*.tar.gz)
shopt -u nullglob

if ((${#ARCHIVES[@]} == 0)); then
  echo "ERROR: no .tar.gz files found in ${DATA_DIR}" >&2
  exit 1
fi

EXTRA_FLAGS=()

if [[ -n "${RESUME_DRAFT_ID}" ]]; then
  EXTRA_FLAGS+=(--resume-draft-id "${RESUME_DRAFT_ID}")
else
  EXTRA_FLAGS+=(--existing-deposition-id "${EXISTING_DEPOSITION_ID}")
fi

case "${ZENODO_ALLOW_LARGE_UPLOAD}" in
  0) ;;
  1) EXTRA_FLAGS+=(--allow-large-upload) ;;
  *) echo 'ERROR: ZENODO_ALLOW_LARGE_UPLOAD must be 0 or 1.' >&2; exit 2 ;;
esac

case "${PUBLISH}" in
  0) ;;
  1) EXTRA_FLAGS+=(--publish) ;;
  *) echo 'ERROR: PUBLISH must be 0 or 1.' >&2; exit 2 ;;
esac

case "${DRY_RUN}" in
  0) ;;
  1) EXTRA_FLAGS+=(--dry-run) ;;
  *) echo 'ERROR: DRY_RUN must be 0 or 1.' >&2; exit 2 ;;
esac

case "${CREATE_DRAFT_ONLY}" in
  0) ;;
  1) EXTRA_FLAGS+=(--create-draft-only) ;;
  *) echo 'ERROR: CREATE_DRAFT_ONLY must be 0 or 1.' >&2; exit 2 ;;
esac

python "${SCRIPT_DIR}/zenodo_push.py" \
  --direct-upload \
  --paths "${ARCHIVES[@]}" \
  --title "${TITLE}" \
  --version "${ZENODO_VERSION}" \
  --description-file "${DESCRIPTION_FILE}" \
  --creators-json "${MEMBERS_JSON}" \
  --keywords ASTRA DESI cosmic-web LSS DR1 \
  --token-env "${TOKEN_ENV}" \
  --max-total-size-gb "${ZENODO_MAX_TOTAL_SIZE_GB}" \
  "${EXTRA_FLAGS[@]}"