"""
HAINE Layer 1 extension - isolate the Periyar wall segment to convert into a
prescribed-flowrate liquid boundary, and predict the liquid-boundary ordering.

READ-ONLY (reads kochi.slf, kochi_r.slf, kochi.cli; writes only a PNG).

It:
  - recovers true UTM (same +offset the coupling uses),
  - parses kochi.cli IN BOUNDARY ORDER (K rank), classifying open (KENT=5) vs
    wall (KLOG=2),
  - lists the current open segments = existing liquid boundaries + their K-ranges,
  - finds the contiguous wall run at the Periyar inlet and PROPOSES a channel-width
    open span centred on the deepest node,
  - estimates the implied mean inflow velocity for a monsoon Q (sanity check),
  - predicts whether the new river boundary will be liquid boundary #1 or #2,
  - saves a zoomed boundary map with the proposed span highlighted.

Run: conda activate telemac && python 13_periyar_segment.py
"""
import os
import numpy as np

TEL  = os.path.expanduser("~/haine_telemac")
MESH = os.path.join(TEL, "kochi.slf")
GEO  = os.path.join(TEL, "kochi_r.slf")
CLI  = os.path.join(TEL, "kochi.cli")

PERIYAR_LONLAT = (76.28, 10.07)   # Periyar / Varapuzha inlet
TARGET_HALF_WIDTH_M = 300.0       # propose ~600 m total open span (channel width)
Q_MONSOON = 500.0                 # m3/s, for the implied-velocity sanity check
DATUM_NOTE = "depth ~= max(0, -BOTTOM), assuming BOTTOM is bed elevation below MSL=0"


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

    names = [(n.decode() if isinstance(n, bytes) else n).strip().upper()
             for n in m.varnames]
    bottom = None
    for i, n in enumerate(names):
        if n.startswith("BOTTOM"):
            try:
                bottom = np.asarray(m.get_variables_at(0, [i])[0])
            except Exception:
                pass
            break

    # parse .cli in order; keep (K, N, LIHBOR)
    K, N, LIH = [], [], []
    with open(CLI) as f:
        for line in f:
            t = line.split()
            if len(t) < 13:
                continue
            try:
                LIH.append(int(t[0])); N.append(int(t[-2])); K.append(int(t[-1]))
            except ValueError:
                continue
    K = np.array(K); N = np.array(N); LIH = np.array(LIH)
    order = np.argsort(K)                    # ensure boundary order
    K, N, LIH = K[order], N[order], LIH[order]
    bx, by = mx[N - 1], my[N - 1]
    bz = bottom[N - 1] if bottom is not None else np.full(len(N), np.nan)
    depth = np.maximum(0.0, -bz)             # see DATUM_NOTE

    is_open = (LIH == 5)
    is_wall = (LIH == 2)
    print(f"boundary points: {len(K)}  open(KENT=5): {int(is_open.sum())}  "
          f"wall(KLOG=2): {int(is_wall.sum())}")

    # contiguous runs in K order
    def runs(mask):
        out, i, n = [], 0, len(mask)
        while i < n:
            if mask[i]:
                j = i
                while j + 1 < n and mask[j + 1]:
                    j += 1
                out.append((i, j))
                i = j + 1
            else:
                i += 1
        return out

    open_runs = runs(is_open)
    print(f"\nexisting OPEN segments (= current liquid boundaries): {len(open_runs)}")
    for r0, r1 in open_runs:
        print(f"  K {K[r0]}..{K[r1]}  ({r1 - r0 + 1} nodes)  "
              f"x {bx[r0:r1+1].min():.0f}..{bx[r0:r1+1].max():.0f}  "
              f"y {by[r0:r1+1].min():.0f}..{by[r0:r1+1].max():.0f}")

    # nearest wall node to the Periyar inlet
    px, py = tr.transform(*PERIYAR_LONLAT)
    dist = np.hypot(bx - px, by - py)
    dist[~is_wall] = np.inf
    c = int(np.argmin(dist))
    print(f"\nPeriyar inlet UTM: x={px:.0f} y={py:.0f}")
    print(f"nearest wall node: K={K[c]} node#{N[c]}  {dist[c]:.0f} m away  "
          f"depth~{depth[c]:.1f} m")

    # the contiguous wall run containing c
    wall_runs = runs(is_wall)
    run = next(rr for rr in wall_runs if rr[0] <= c <= rr[1])
    print(f"contiguous wall run at inlet: K {K[run[0]]}..{K[run[1]]}  "
          f"({run[1]-run[0]+1} nodes)")

    # propose a channel-width span centred on the deepest node near c (within run)
    lo, hi = run
    win = slice(max(lo, c - 40), min(hi, c + 40) + 1)
    local = np.arange(win.start, win.stop)
    deep = local[int(np.argmax(depth[win]))]           # deepest node near inlet
    # grow from 'deep' along the run until ~TARGET_HALF_WIDTH each side
    def along(a, b):
        return np.hypot(bx[a] - bx[b], by[a] - by[b])
    L = R = deep
    accL = accR = 0.0
    while L - 1 >= lo and accL < TARGET_HALF_WIDTH_M:
        accL += along(L, L - 1); L -= 1
    while R + 1 <= hi and accR < TARGET_HALF_WIDTH_M:
        accR += along(R, R + 1); R += 1
    span = np.arange(L, R + 1)
    width = accL + accR
    mean_depth = float(np.nanmean(depth[span]))
    area = max(1.0, width * mean_depth)
    v_imp = Q_MONSOON / area

    print("\n--- PROPOSED open segment (Periyar flowrate boundary) ---")
    print(f"  K range     : {K[L]}..{K[R]}   ({len(span)} nodes)")
    print(f"  global nodes: {list(N[span])}")
    print(f"  along-bnd width : {width:.0f} m   mean depth ~ {mean_depth:.1f} m   "
          f"({DATUM_NOTE})")
    print(f"  cross-section ~ {area:.0f} m2 -> implied inflow speed for "
          f"Q={Q_MONSOON:.0f} m3/s: {v_imp:.2f} m/s")
    if v_imp > 1.5:
        print("  ! implied speed high -> widen the span (raise TARGET_HALF_WIDTH_M)")
    elif v_imp < 0.05:
        print("  ! implied speed very low -> narrow the span (lower TARGET_HALF_WIDTH_M)")
    else:
        print("  implied speed is physically reasonable.")

    # predict liquid-boundary ordering: TELEMAC numbers by ascending start-K
    sea_start = min(K[r0] for r0, _ in open_runs)
    river_start = K[L]
    if river_start < sea_start:
        print(f"\nPredicted ordering: RIVER = liquid boundary #1 (start K {river_start}), "
              f"SEA = #2 (start K {sea_start}).")
        print("  -> boundary-file columns order: Q(1) then SL(2); "
              "PRESCRIBED keywords list river first.")
    else:
        print(f"\nPredicted ordering: SEA = liquid boundary #1 (start K {sea_start}), "
              f"RIVER = #2 (start K {river_start}).")
        print("  -> boundary-file columns order: SL(1) then Q(2); "
              "PRESCRIBED keywords list sea first.")
    print("  (confirm against TELEMAC's run listing, which prints the liquid-boundary "
          "list; we set values to match.)")

    # zoomed map
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pad = 3000
        xmin, xmax = px - pad, px + pad
        ymin, ymax = py - pad, py + pad
        sel = (bx > xmin) & (bx < xmax) & (by > ymin) & (by < ymax)
        plt.figure(figsize=(9, 9))
        m_in = (mx > xmin) & (mx < xmax) & (my > ymin) & (my < ymax)
        if bottom is not None:
            plt.scatter(mx[m_in], my[m_in], c=bottom[m_in], s=3, cmap="terrain",
                        alpha=0.5)
            plt.colorbar(label="BOTTOM (m)")
        plt.scatter(bx[sel & is_wall], by[sel & is_wall], s=12, c="dimgray",
                    label="wall")
        plt.scatter(bx[sel & is_open], by[sel & is_open], s=14, c="red",
                    label="open (sea)")
        plt.scatter(bx[span], by[span], s=60, c="yellow", edgecolor="k",
                    label="PROPOSED river open span")
        plt.scatter([px], [py], s=260, marker="*", c="cyan", edgecolor="k",
                    label="Periyar inlet")
        plt.legend(loc="upper right"); plt.gca().set_aspect("equal")
        plt.title("Periyar inlet - proposed flowrate boundary span")
        out = os.path.expanduser("~/Downloads/periyar_segment.png")
        plt.savefig(out, dpi=140, bbox_inches="tight")
        print(f"\nsaved zoom map -> {out}")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
