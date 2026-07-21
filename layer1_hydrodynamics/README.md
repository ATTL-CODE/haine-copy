# Layer 1 — Hydrodynamic Engine (TELEMAC-2D)

Produces the tidal flow field whose bed shear stress gates cohesive deposition.

- `reproject_mesh_to_utm.py` — reproject the mesh from geographic degrees to **UTM 43N (EPSG:32643)**. **Critical:** solving in degrees causes CFL instability every timestep; the model must be in metres.
- `09_fes2014_tidal_extraction.py` — extract FES2014 tidal constituents for the open-sea boundary forcing.
- `telemac_build/` — scripts to build openTELEMAC from source (`build.sh`, `build*.bat`) and `diagnose.sh`.

**Continuity notes:** the solver mesh is stored in a LOCAL frame — add offset `x0=629505, y0=1079516` to recover true UTM before sampling against the SPM grid. Boundary rank in the `.cli` file is line order / IPOBO, not the trailing colour tag. Use `KENT=5` (prescribed elevation) for the tidal open boundary. The pilot is tidal-only; wave (TOMAWAC) and solver-based river forcing are roadmap items.
