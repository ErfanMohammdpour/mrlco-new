"""2-D Pareto front and hypervolume. Minimization. paper_result=false."""

from __future__ import annotations


def dominates(p, q, tol=1e-12):
    """p dominates q iff p <= q in both coords and < in at least one."""
    px, py = float(p[0]), float(p[1])
    qx, qy = float(q[0]), float(q[1])
    le = (px <= qx + tol) and (py <= qy + tol)
    lt = (px + tol < qx) or (py + tol < qy)
    return bool(le and lt)


def nondominated(points, tol=1e-12):
    pts = [(float(p[0]), float(p[1])) for p in points]
    out = []
    for i, p in enumerate(pts):
        if any(dominates(q, p, tol=tol) for j, q in enumerate(pts) if j != i):
            continue
        if p not in out:
            out.append(p)
    return out


def hypervolume_2d(points, ref, tol=1e-12):
    """Minimization hypervolume vs reference (worse in both coords).

    Staircase: sort by T, width to next T (or ref_T) × height (ref_E − E).
    """
    rx, ry = float(ref[0]), float(ref[1])
    front = []
    for t, e in nondominated(points, tol=tol):
        if t + tol >= rx or e + tol >= ry:
            continue
        front.append((t, e))
    if not front:
        return 0.0
    front.sort(key=lambda p: (p[0], p[1]))
    hv = 0.0
    ts = [p[0] for p in front] + [rx]
    for i, (_t, e) in enumerate(front):
        width = ts[i + 1] - ts[i]
        height = ry - e
        if width > 0.0 and height > 0.0:
            hv += width * height
    return float(hv)
