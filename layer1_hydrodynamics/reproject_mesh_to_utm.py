"""
Fix the coordinate-units problem: the mesh in kochi.slf is stored in
geographic degrees (lon/lat, EPSG:4326), but TELEMAC's shallow-water
solver requires Cartesian METERS. Running physics on degree-coordinates
makes every timestep CFL-unstable -> the solver stalls at ITERATION 0.

This script reprojects the node coordinates to UTM Zone 43N (EPSG:32643,
the correct zone for Kochi ~76E) in metres and rewrites kochi.slf.
Mesh connectivity (IKLE), boundary numbering (IPOBO), and bathymetry are
all preserved unchanged - only X/Y coordinates change - so kochi.cli and
kochi_liq.txt remain valid without modification.

Run inside the 'telemac' conda env, from ~/haine_telemac/:
    python reproject_mesh_to_utm.py
"""
import sys
sys.path.insert(0, '/root/miniforge3/envs/telemac/opentelemac/scripts/python3')

import numpy as np
from data_manip.formats.selafin import Selafin
from pyproj import Transformer

SLF_FILE = "kochi.slf"

# ---------------------------------------------------------------
# 1. Load current (degree-coordinate) mesh
# ---------------------------------------------------------------
print("Loading kochi.slf ...")
slf = Selafin(SLF_FILE)
lon = np.asarray(slf.meshx, dtype=np.float64)
lat = np.asarray(slf.meshy, dtype=np.float64)
bottom = slf.get_values(0)[0].copy()
ikle2 = slf.ikle2
ipob2 = slf.ipob2
npoin2 = slf.npoin2
nelem2 = slf.nelem2

print(f"  nodes={npoin2}, triangles={nelem2}")
print(f"  current X range (deg): {lon.min():.4f} to {lon.max():.4f}")
print(f"  current Y range (deg): {lat.min():.4f} to {lat.max():.4f}")

# ---------------------------------------------------------------
# 2. Reproject EPSG:4326 (lon/lat deg) -> EPSG:32643 (UTM 43N metres)
# ---------------------------------------------------------------
print("Reprojecting to UTM Zone 43N (EPSG:32643) ...")
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
xm, ym = transformer.transform(lon, lat)
xm = np.asarray(xm, dtype=np.float64)
ym = np.asarray(ym, dtype=np.float64)

print(f"  new X range (m): {xm.min():.1f} to {xm.max():.1f}  (span {xm.max()-xm.min():.0f} m)")
print(f"  new Y range (m): {ym.min():.1f} to {ym.max():.1f}  (span {ym.max()-ym.min():.0f} m)")

# Shift origin to a local datum near the domain so coordinates are small
# positive numbers (avoids single-precision rounding loss on ~700000 m UTM
# eastings when the .slf stores coords as 4-byte floats).
x0 = np.floor(xm.min())
y0 = np.floor(ym.min())
xm_local = xm - x0
ym_local = ym - y0
print(f"  applied local origin shift: X0={x0:.0f}, Y0={y0:.0f}")
print(f"  local X range (m): {xm_local.min():.1f} to {xm_local.max():.1f}")
print(f"  local Y range (m): {ym_local.min():.1f} to {ym_local.max():.1f}")

# sanity: typical triangle edge should now be ~50-150 m, not ~0.001
x1, y1 = xm_local[ikle2[:,0]], ym_local[ikle2[:,0]]
x2, y2 = xm_local[ikle2[:,1]], ym_local[ikle2[:,1]]
edge = np.hypot(x2-x1, y2-y1)
print(f"  sample edge length now: mean={edge.mean():.1f} m (was ~0.001 deg)")

# ---------------------------------------------------------------
# 3. Rewrite kochi.slf with projected coordinates (everything else same)
# ---------------------------------------------------------------
print("Rewriting kochi.slf in metres ...")
out = Selafin('')
out.title = 'KOCHI HAINE MESH (UTM43N m)'.ljust(80)[:80]
out.nbv1 = 1
out.nbv2 = 0
out.nvar = 1
out.varindex = range(1)
out.varnames = ['BOTTOM          ']
out.varunits = ['M               ']
out.npoin2 = npoin2
out.ndp2 = 3
out.nelem2 = nelem2
out.meshx = xm_local
out.meshy = ym_local
out.ikle2 = ikle2
out.ndp3 = 3
out.nplan = 1
out.nelem3 = nelem2
out.npoin3 = npoin2
out.ikle3 = ikle2
out.iparam = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
out.ipob2 = ipob2
out.ipob3 = ipob2

out.fole = {}
out.fole.update({'hook': open(SLF_FILE, 'wb')})
out.fole.update({'name': SLF_FILE})
out.fole.update({'endian': '>'})
out.fole.update({'float': ('f', 4)})
out.append_header_slf()
out.append_core_time_slf(0.0)
out.append_core_vars_slf([bottom])
out.fole['hook'].close()

print("\nDone. kochi.slf rewritten in metres (UTM 43N, local origin).")
print(f"Remember the origin shift if you ever map results back to lon/lat:")
print(f"  true_UTM_easting  = local_x + {x0:.0f}")
print(f"  true_UTM_northing = local_y + {y0:.0f}")
