#!/bin/bash
# Build the clustering-graphs SIF. Needs sudo.
#
# The SIF holds Python + uv-managed venv only. Source code (src/main.py etc.)
# is bind-mounted at runtime via container/run.sh -- no rebuild needed when
# code changes. Rebuild only when requirements.txt or the .def changes.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

# Singularity peaks at ~8-10 GB of scratch for layer extraction. Route that
# to a location with enough free space. Default keeps it on the same disk
# but in /var/tmp where systemd usually doesn't auto-clean.
TMPDIR_BUILD="${SINGULARITY_TMPDIR:-/var/tmp/singularity-build-$USER}"
mkdir -p "$TMPDIR_BUILD"
export SINGULARITY_TMPDIR="$TMPDIR_BUILD"

# Pre-flight disk check -- 12 GB free is the minimum we want to see.
AVAIL_GB=$(df -BG --output=avail "$TMPDIR_BUILD" | tail -1 | tr -dc '0-9')
echo "[build.sh] free space in $TMPDIR_BUILD: ${AVAIL_GB} GB"
if (( AVAIL_GB < 12 )); then
    echo "[build.sh] WARN: <12 GB free; the build may fail at squashfs step." >&2
    echo "[build.sh] Free space (singularity cache lives at ~/.singularity, can be cleaned with 'singularity cache clean')." >&2
fi

echo "[build.sh] $(date +%H:%M:%S) starting build (10-20 minutes typical)"
# --fakeroot uses /etc/subuid + /etc/subgid mappings -- no sudo needed.
# Fall back to sudo only if fakeroot isn't configured for the calling user.
if grep -q "^$USER:" /etc/subuid 2>/dev/null; then
    singularity build --fakeroot clustgraphs.sif clustering_graphs.def
else
    echo "[build.sh] $USER not in /etc/subuid -- falling back to sudo build"
    sudo -E singularity build clustgraphs.sif clustering_graphs.def
fi
echo "[build.sh] $(date +%H:%M:%S) build done"

ls -lh clustgraphs.sif

echo "[build.sh] post-build smoke test:"
singularity exec clustgraphs.sif /opt/venv/bin/python -c "
import mteb, torch, pyamg, sklearn
print('SIF OK')
print('  mteb', mteb.__version__)
print('  torch', torch.__version__)
print('  pyamg', pyamg.__version__)
print('  sklearn', sklearn.__version__)
"
