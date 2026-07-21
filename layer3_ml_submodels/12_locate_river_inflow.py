"""
HAINE Layer 1 extension - locate the Periyar river inflow on the mesh and
classify the open-boundary structure, to decide how to add monsoon discharge.

READ-ONLY. Reads kochi.cli (boundary codes) + kochi.slf (coords, bathymetry),
recovers true UTM (same +offset the coupling uses), and reports:
  - open (KENT=5) vs wall (KLOG=2) boundary node counts
  - where the sea boundary sits vs the port AOI
  - the nearest boundary node to the Periyar inlet and its code
  - whether open/wall nodes exist at the river inlet (Route A) or not (Route B)
and saves a boundary map to Downloads.

Run: conda activate telemac && python 12_locate_river_inflow.py
"""
import os
import glob
import numpy as np
import rasterio

TEL      = os.path.expanduser("~/haine_telemac")
MESH     = os.path.join(TEL, "kochi.slf")     # UTM-metres mesh (local frame), has BOTTOM
GEO      = os.path.join(TEL, "kochi_r.slf")   # degree mesh, same node order (for offset)
CLI      = os.path.join(TEL, "kochi.cli")     # boundary conditions (Conlim), 13 columns
CUBE_DIR = os.path.expanduser("~/haine_layer2/spm_fused_port")

# Periyar main discharge into the Cochin backwater (Varapuzha / Eloor branch)
PERIYAR_LONLAT = (76.28, 10.07)
# secondary candidate: the Chitrapuzha / southern branch near the harbour back-reach
ALT_LONLAT     = (76.30, 9.95)


def utm_offset():
    from data_manip.formats.selafin import Selafin
    from pyproj import Transformer
    geo = Selafin(GEO)
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
    gx, gy = tr.transform(np.asarray(geo.meshx), np.asarray(geo.meshy))
    return float(np.floor(gx.min())), float(np.floor(gy.min())), tr


def main():
    from data_manip.formats.selafin import Selafin
    m = Selafin(MESH)
    x0, y0, tr = utm_offset()
    mx = np.asarray(m.meshx) + x0
    my = np.asarray(m.meshy) + y0
    print(f"UTM offset applied: x0={x0:.0f} y0={y0:.0f}")
    print(f"mesh box (true UTM): x {mx.min():.0f}..{mx.max():.0f}  "
          f"y {my.min():.0f}..{my.max():.0f}")

    # bathymetry (BOTTOM), if the geometry file carries it
    names = [(n.decode() if isinstance(n, bytes) else n).strip().upper()
             for n in m.varnames]
    bottom = None
    for i, n in enumerate(names):
        if n.startswith("BOTTOM"):
            try:
                bottom = np.asarray(m.get_variables_at(0, [i])[0])
            except Exception as e:
                print("  (could not read BOTTOM:", e, ")")
            break

    # parse the .cli (Conlim): col0 = LIHBOR, last two ints = N (global node), K
    lih, nodes = [], []
    with open(CLI) as f:
        for line in f:
            t = line.split()
            if len(t) < 13:
                continue
            try:
                lih.append(int(t[0]))
                nodes.append(int(t[-2]))
            except ValueError:
                continue
    lih = np.array(lih)
    nodes = np.array(nodes)
    bx, by = mx[nodes - 1], my[nodes - 1]
    openm = (lih == 5)   # KENT prescribed elevation (the sea boundary)
    wallm = (lih == 2)   # KLOG wall
    print(f"\nboundary nodes parsed: {len(nodes)}  "
          f"open KENT=5: {int(openm.sum())}  wall KLOG=2: {int(wallm.sum())}  "
          f"other: {int((~openm & ~wallm).sum())}")
    print(f"open-boundary box:  x {bx[openm].min():.0f}..{bx[openm].max():.0f}  "
          f"y {by[openm].min():.0f}..{by[openm].max():.0f}")

    # port AOI box
    t = sorted(glob.glob(os.path.join(CUBE_DIR, "port_spm_*.tif")))[0]
    b = rasterio.open(t).bounds
    print(f"port AOI box:       x {b.left:.0f}..{b.right:.0f}  "
          f"y {b.bottom:.0f}..{b.top:.0f}")

    # test each candidate river inlet
    for label, (lon, lat) in [("Periyar (N branch)", PERIYAR_LONLAT),
                              ("alt (S back-reach)", ALT_LONLAT)]:
        px, py = tr.transform(lon, lat)
        inside = (mx.min() <= px <= mx.max()) and (my.min() <= py <= my.max())
        d = np.hypot(bx - px, by - py)
        j = int(np.argmin(d))
        code = "OPEN" if openm[j] else ("WALL" if wallm[j] else f"LIH={lih[j]}")
        # boundary nodes within 2 km of this inlet, split by code
        near = d < 2000.0
        print(f"\n{label}: UTM x={px:.0f} y={py:.0f}  inside mesh: {inside}")
        print(f"  nearest boundary node: {d[j]:.0f} m away, code {code}, "
              f"global node #{nodes[j]}")
        print(f"  boundary nodes within 2 km: {int(near.sum())}  "
              f"(open {int((near & openm).sum())}, wall {int((near & wallm).sum())})")
        if inside and (near & wallm).sum() >= 3 and (near & openm).sum() == 0:
            print("  -> ROUTE A candidate: walled river edge here to convert to a "
                  "flowrate boundary.")
        elif inside and near.sum() == 0:
            print("  -> inlet is INTERIOR (no boundary nearby) -> ROUTE B: inject via "
                  "SOURCES at the nearest interior node.")
        elif not inside:
            print("  -> inlet is OUTSIDE the mesh -> mesh does not reach this river; "
                  "use the in-mesh back-reach or SOURCES at the domain edge.")

    # map
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        plt.figure(figsize=(9, 11))
        if bottom is not None:
            sc = plt.scatter(mx, my, c=bottom, s=1, cmap="terrain", alpha=0.5)
            plt.colorbar(sc, label="BOTTOM (m)")
        else:
            plt.scatter(mx, my, s=1, c="lightgray", alpha=0.4)
        plt.scatter(bx[wallm], by[wallm], s=5, c="dimgray", label="wall KLOG=2")
        plt.scatter(bx[openm], by[openm], s=8, c="red", label="open KENT=5 (sea)")
        for label, (lon, lat), col in [("Periyar", PERIYAR_LONLAT, "cyan"),
                                       ("alt", ALT_LONLAT, "magenta")]:
            px, py = tr.transform(lon, lat)
            plt.scatter([px], [py], s=220, marker="*", c=col,
                        edgecolor="k", zorder=5, label=f"{label} inlet")
        plt.gca().add_patch(Rectangle((b.left, b.bottom), b.right - b.left,
                            b.top - b.bottom, fill=False, edgecolor="lime",
                            lw=2, label="port AOI"))
        plt.legend(loc="upper right")
        plt.gca().set_aspect("equal")
        plt.title("Mesh boundary codes + river inlets (true UTM)")
        out = os.path.expanduser("~/Downloads/mesh_boundary_river.png")
        plt.savefig(out, dpi=130, bbox_inches="tight")
        print(f"\nsaved boundary map -> {out}")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
