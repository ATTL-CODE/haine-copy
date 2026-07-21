#!/bin/bash
###############################################################################
# HAINE TELEMAC-2D stall diagnostic - single-shot
#
# Runs everything needed to pinpoint why the Kochi tidal run stalls at
# ITERATION 0, and dumps all results into diagnosis.txt for upload.
#
# Usage (from ~/haine_telemac/, inside the 'telemac' conda env):
#     bash diagnose.sh
#
# Expected total runtime: about 3-6 minutes.
#   - Mesh quality analysis (Python): ~30-60 s
#   - Rest-case TELEMAC run with a 120 s hard timeout: up to ~2.5 min
#   - Boundary/liquid-file checks: a few seconds
# The rest-run is force-killed after 120 s so the script can never hang.
###############################################################################

OUT="diagnosis.txt"
PY="/root/miniforge3/envs/telemac/opentelemac/scripts/python3"

echo "HAINE TELEMAC DIAGNOSIS - $(date)" > "$OUT"
echo "=================================================================" >> "$OUT"

###############################################################################
# SECTION 1: Environment sanity
###############################################################################
{
echo ""
echo "### SECTION 1: ENVIRONMENT ###"
echo "which telemac2d.py : $(which telemac2d.py 2>&1)"
echo "python             : $(which python 2>&1)"
echo "conda env          : $CONDA_PREFIX"
echo "working dir        : $(pwd)"
echo "files present:"
ls -la kochi.slf kochi.cli kochi_liq.txt 2>&1 | sed 's/^/    /'
} >> "$OUT"

###############################################################################
# SECTION 2: Mesh quality analysis
#   Looks for the usual culprits that stall a solver at step 0:
#   degenerate/zero-area triangles, slivers (extreme aspect ratio),
#   duplicate nodes, inverted (negative-area) elements, and how the
#   connectivity/orientation looks.
###############################################################################
echo "" >> "$OUT"
echo "### SECTION 2: MESH QUALITY ###" >> "$OUT"

python - <<'PYEOF' >> "$OUT" 2>&1
import sys
sys.path.insert(0, '/root/miniforge3/envs/telemac/opentelemac/scripts/python3')
import numpy as np
from data_manip.formats.selafin import Selafin

slf = Selafin('kochi.slf')
x, y = slf.meshx, slf.meshy
ikle = slf.ikle2
npoin, nelem = slf.npoin2, slf.nelem2
print(f"nodes={npoin}, triangles={nelem}")

# signed area of every triangle (0.5 * cross product)
x1, y1 = x[ikle[:,0]], y[ikle[:,0]]
x2, y2 = x[ikle[:,1]], y[ikle[:,1]]
x3, y3 = x[ikle[:,2]], y[ikle[:,2]]
signed_area = 0.5 * ((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
area = np.abs(signed_area)

print(f"signed area: min={signed_area.min():.3e}, max={signed_area.max():.3e}")
print(f"  negative-area (inverted/CW) triangles: {(signed_area < 0).sum()}")
print(f"  positive-area (CCW) triangles        : {(signed_area > 0).sum()}")
print(f"  EXACTLY zero-area (degenerate)        : {(signed_area == 0).sum()}")
print(f"area magnitude: min={area.min():.3e}, max={area.max():.3e}, mean={area.mean():.3e}")
print(f"  near-zero area (<1e-12) triangles     : {(area < 1e-12).sum()}")
print(f"  tiny area (<1e-9) triangles           : {(area < 1e-9).sum()}")

# edge lengths & aspect ratio (longest edge / shortest edge) per triangle
e1 = np.hypot(x2-x1, y2-y1)
e2 = np.hypot(x3-x2, y3-y2)
e3 = np.hypot(x1-x3, y1-y3)
edges = np.vstack([e1, e2, e3])
emin = edges.min(axis=0)
emax = edges.max(axis=0)
# guard against zero-length edges
safe_emin = np.where(emin > 0, emin, np.nan)
aspect = emax / safe_emin
print(f"edge length: global min={edges.min():.3e}, max={edges.max():.3e}")
print(f"  zero-length edges (duplicate nodes in a tri): {(edges == 0).sum()}")
print(f"aspect ratio (longest/shortest edge):")
print(f"  max aspect ratio  : {np.nanmax(aspect):.1f}")
print(f"  # tris aspect>100 : {(aspect > 100).sum()}")
print(f"  # tris aspect>1000: {(aspect > 1000).sum()}")

# worst offenders
worst = np.argsort(-np.nan_to_num(aspect, nan=0))[:10]
print("worst 10 sliver triangles (id, aspect, area):")
for t in worst:
    print(f"    tri {t}: aspect={aspect[t]:.1f}, area={area[t]:.3e}, nodes={ikle[t].tolist()}")

# duplicate node coordinates
coords = np.column_stack([x, y])
uniq, counts = np.unique(coords, axis=0, return_counts=True)
dups = (counts > 1).sum()
print(f"duplicate node coordinates: {dups} positions shared by >1 node")

# nodes not referenced by any triangle
used = np.unique(ikle)
unused = npoin - len(used)
print(f"unused nodes (in no triangle): {unused}")

# node index range sanity
print(f"ikle index range: min={ikle.min()}, max={ikle.max()} (should be 0..{npoin-1})")
PYEOF

###############################################################################
# SECTION 3: Boundary condition file consistency
###############################################################################
echo "" >> "$OUT"
echo "### SECTION 3: BOUNDARY FILE (.cli) ###" >> "$OUT"
{
echo "line count: $(wc -l < kochi.cli)"
echo "first 3 lines:"
head -3 kochi.cli | sed 's/^/    /'
echo "last 3 lines:"
tail -3 kochi.cli | sed 's/^/    /'
echo "count of open (starts with 5) : $(grep -c '^5 ' kochi.cli)"
echo "count of wall (starts with 2) : $(grep -c '^2 ' kochi.cli)"
echo "liquid boundary file first 4 lines:"
head -4 kochi_liq.txt | sed 's/^/    /'
} >> "$OUT" 2>&1

###############################################################################
# SECTION 4: Rest-case run (no tidal forcing) with hard timeout
#   The decisive test. If a flat-water rest run also stalls at ITER 0,
#   the mesh is the cause. If it completes, the tide forcing is the cause.
###############################################################################
echo "" >> "$OUT"
echo "### SECTION 4: REST-CASE RUN (120s hard timeout) ###" >> "$OUT"

# clean any old diag temp dirs / result
rm -rf kochi_diag_rest.cas_2026-* kochi_diag_r.slf 2>/dev/null

export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

echo "launching rest run at $(date +%H:%M:%S) ..." >> "$OUT"
timeout 120 telemac2d.py kochi_diag_rest.cas > diag_rest_run.log 2>&1
RC=$?
echo "rest run exit code: $RC  (124 = hit the 120s timeout = STALLED)" >> "$OUT"

{
echo "--- last 30 lines of rest run log ---"
tail -30 diag_rest_run.log | sed 's/^/    /'
echo "--- iteration/error signature ---"
grep -iE "iteration|correct end|plante|sub-iteration|positive depth|error" diag_rest_run.log | tail -15 | sed 's/^/    /'
NEWDIR=$(ls -dt kochi_diag_rest.cas_* 2>/dev/null | head -1)
if [ -n "$NEWDIR" ]; then
  echo "rest T2DRES size: $(ls -la "$NEWDIR"/T2DRES 2>/dev/null | awk '{print $5}') bytes  (1956924 = initial frame only = stalled)"
fi
} >> "$OUT" 2>&1

###############################################################################
# SECTION 5: TELEMAC's own mesh checker (if available)
###############################################################################
echo "" >> "$OUT"
echo "### SECTION 5: TELEMAC checkmesh (if present) ###" >> "$OUT"
CHECKMESH=$(find /root/miniforge3/envs/telemac/opentelemac -iname "checkmesh*.py" 2>/dev/null | head -1)
if [ -n "$CHECKMESH" ]; then
  echo "found: $CHECKMESH" >> "$OUT"
  timeout 60 python "$CHECKMESH" kochi.slf >> "$OUT" 2>&1
else
  echo "no checkmesh utility found in install (skipping)" >> "$OUT"
fi

###############################################################################
echo "" >> "$OUT"
echo "=================================================================" >> "$OUT"
echo "DIAGNOSIS COMPLETE - $(date)" >> "$OUT"
echo "Upload diagnosis.txt back to the chat." >> "$OUT"

echo ""
echo "=============================================="
echo "  DONE. Results written to: diagnosis.txt"
echo "  Upload that file back to the chat."
echo "=============================================="
