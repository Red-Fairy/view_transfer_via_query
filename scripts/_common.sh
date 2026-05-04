# Shared path setup for all view_transfer_via_query scripts.
#
# Usage (top of every other script):
#     source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
#
# Defines:
#     PROJECT_ROOT   = view_transfer_via_query/                    (the submodule root)
#     DIFFSYNTH_ROOT = view_transfer_via_query/..  = DiffSynth-Studio/  (provides `diffsynth` lib + pretrained models)
#
# Side-effects:
#     - Exports PYTHONPATH so both `view_transfer_via_query.X` and `diffsynth.X` resolve from any CWD.
#     - Does NOT cd anywhere — caller decides whether to `cd "${PROJECT_ROOT}"` for output convention.

# Refuse to run as a standalone script — must be sourced (we export and may cd).
if ! (return 0 2>/dev/null); then
    echo "ERROR: $0 must be sourced, not executed."
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIFFSYNTH_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

export PROJECT_ROOT DIFFSYNTH_ROOT
export PYTHONPATH="${DIFFSYNTH_ROOT}:${PYTHONPATH:-}"
