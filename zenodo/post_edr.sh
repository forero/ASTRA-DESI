set -euo pipefail
module load python/3.12

PATHS=("/pscratch/sd/v/vtorresg/cosmic-web/edr/raw"
       "/pscratch/sd/v/vtorresg/cosmic-web/edr/class"
       "/pscratch/sd/v/vtorresg/cosmic-web/edr/groups")

PSCRATCH_DIR="/pscratch/sd/v/vtorresg/cosmic-web"

TITLE="ASTRA-DESI EDR Release v0.2"
DESCRIPTION="Early Data Release products for ASTRA-DESI (raw, class, groups)."
MEMBERS_JSON='./json/members.json'
KEYWORDS=("ASTRA" "DESI" "cosmic-web" "LSS")

python zenodo_push.py \
  --paths "${PATHS[@]}" \
  --pscratch-dir "$PSCRATCH_DIR" \
  --keep-tree \
  --title "$TITLE" \
  --description "$DESCRIPTION" \
  --creators-json "$MEMBERS_JSON" \
  --keywords "${KEYWORDS[@]}" \
  --token-file ~/.zenodo_token
  # --publish \
#   --dry-run \ #for testing!!!
#   --sandbox \