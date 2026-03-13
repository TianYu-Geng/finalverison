import numpy as np
from scipy.ndimage import distance_transform_edt


def upsample_occupancy(occupancy, scale):
    return np.kron(
        occupancy.astype(np.uint8),
        np.ones((scale, scale), dtype=np.uint8)
    ).astype(bool)


def build_highres_clearance(occupancy, scale):
    occ_hr = upsample_occupancy(occupancy, scale)
    free_hr = ~occ_hr
    clearance_hr = distance_transform_edt(free_hr).astype(np.float32) / float(scale)
    clearance_hr[occ_hr] = 0.0
    return occ_hr, free_hr, clearance_hr


def point_to_segment_distance(px, py, x1, y1, x2, y2):
    vx = x2 - x1
    vy = y2 - y1
    wx = px - x1
    wy = py - y1

    c1 = vx * wx + vy * wy
    if c1 <= 0.0:
        return float(np.hypot(px - x1, py - y1))

    c2 = vx * vx + vy * vy
    if c2 <= c1:
        return float(np.hypot(px - x2, py - y2))

    t = c1 / c2
    proj_x = x1 + t * vx
    proj_y = y1 + t * vy
    return float(np.hypot(px - proj_x, py - proj_y))


def point_to_polyline_distance(px, py, polyline_xy):
    if len(polyline_xy) == 1:
        x, y = polyline_xy[0]
        return float(np.hypot(px - x, py - y))

    dmin = np.inf
    for a, b in zip(polyline_xy[:-1], polyline_xy[1:]):
        d = point_to_segment_distance(px, py, a[0], a[1], b[0], b[1])
        if d < dmin:
            dmin = d
    return float(dmin)