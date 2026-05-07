#!/usr/bin/env bash
# Walk a set of UE-render output trees and emit a flat list of valid location dirs
# (one per line) into a .txt file consumable by train.sh / infer.sh.
#
# Self-contained: works from any CWD. Override DATA_FOLDERS / OUTPUT_FILE via env.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# ── Defaults (override via env) ─────────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-/work/nvme/beab/rluo2/viewpoint-transfer/data}"
# Space-separated list of subfolder names under DATA_ROOT to scan.
DATA_FOLDERS="${DATA_FOLDERS:-outputs_arranged outputs_non_arranged}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/split_files}"

# Build absolute paths
data_roots=()
for f in ${DATA_FOLDERS}; do
    data_roots+=("${DATA_ROOT}/${f}")
done

cd "${PROJECT_ROOT}"

echo "==================================================================="
echo "  DATA_FOLDERS = ${DATA_FOLDERS}"
echo "  OUTPUT_DIR  = ${OUTPUT_DIR}"
echo "==================================================================="

python -m view_transfer_via_query.prepare_data.gather_locations \
    --data_roots "${data_roots[@]}" \
    --output     "${OUTPUT_DIR}"
