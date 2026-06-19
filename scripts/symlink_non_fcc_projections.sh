#!/usr/bin/env bash
# Symlink all non-FCC files from commondata_projections_fccee_30MW_4IP_60lumi
# (which serves as the reference for which non-FCC datasets to include) into a
# target directory, using relative paths pointing into commondata_projections_L0.
# Usage: scripts/symlink_non_fcc_projections.sh <target_dir>

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$REPO_ROOT/${1:?Usage: $0 <target_dir>}"
REF="$REPO_ROOT/commondata_projections_fccee_30MW_4IP_60lumi"

if [ ! -d "$TARGET" ]; then
    echo "Error: target directory '$TARGET' does not exist." >&2
    exit 1
fi

for f in "$REF"/*.yaml; do
    fname="$(basename "$f")"
    if [[ "$fname" != FCCee_* && "$fname" != fccee_* ]]; then
        # Read the symlink target from the reference dir and replicate it
        link_target="$(readlink "$f")"
        ln -sf "$link_target" "$TARGET/$fname"
    fi
done

echo "Symlinks created in $TARGET"
