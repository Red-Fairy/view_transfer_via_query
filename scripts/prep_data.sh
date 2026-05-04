#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

data_root=/work/nvme/beab/rluo2/viewpoint-transfer/data
data_folders=(outputs_non_arranged_24fps outputs_non_arranged_16fps)

# Prefix each folder with data_root, expand as separate args
data_roots=()
for f in "${data_folders[@]}"; do
    data_roots+=("${data_root}/${f}")
done

source /work/nvme/beab/rluo2/anaconda3/etc/profile.d/conda.sh
conda activate wan

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

python -m view_transfer_via_query.prepare_data.gather_locations \
    --data_roots "${data_roots[@]}" \
    --output     "../data/train_locations.txt"
