"""Per-frame measurements of one cashew and the view (TOP / BACK / NECK / OTHER) they point to.

The numbers are shown live so the owner can tune the view limits; no limit is built in.
Outline features alone separate only "kidney shaped" from "narrow" views, so features from
inside the mask (the hollow under the neck, symmetry, texture) are measured too.
"""
import cv2
import numpy as np

VIEWS = ("NECK", "BACK", "TOP")         # checked in this order; anything else is OTHER
OTHER = "OTHER"

# name -> (short label on the picture, description)
FEATURES = {
    "aspect":     ("A", "length / width of the outline"),
    "solidity":   ("S", "outline area / convex hull area (1 = no bay)"),
    "defect":     ("D", "deepest bay of the outline / width"),
    "hollow":     ("H", "share of the mask clearly darker or redder than the kernel"),
    "hollow_off": ("O", "distance of that darker part from the long axis / width"),
    "symmetry":   ("Y", "overlap of the mask with its mirror image about the long axis"),
    "edge":       ("E", "texture inside the mask (mean Laplacian of L)"),
}
DARK_L_MAX = 30.0       # L* (0-100) below which a pixel counts as "dark" for dark_pct (logged only)

_ERODE = np.ones((5, 5), np.uint8)


def measure(crop_bgr, seg):
    """Dict of features for one cashew crop and its Segment. Lengths in pixels of the crop."""
    (_, _), (w, h), _ = cv2.minAreaRect(seg.contour)
    length, width = max(w, h), max(1.0, min(w, h))
    hull_area = cv2.contourArea(cv2.convexHull(seg.contour))
    depth = 0.0
    if len(seg.contour) > 3:
        try:
            defects = cv2.convexityDefects(seg.contour, cv2.convexHull(seg.contour, returnPoints=False))
            if defects is not None:
                depth = float(defects.reshape(-1, 4)[:, 3].max()) / 256.0
        except cv2.error:
            pass

    inner = cv2.erode(seg.mask, _ERODE)
    if cv2.countNonZero(inner) < 50:
        inner = seg.mask
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[..., 0] * (100.0 / 255.0), lab[..., 1] - 128.0, lab[..., 2] - 128.0
    inside = inner > 0
    n = float(inside.sum())
    med_L, med_a, med_b = float(np.median(L[inside])), float(np.median(a[inside])), float(np.median(b[inside]))

    darker = (L < 0.75 * med_L) & inside
    redder = (a > med_a + 6.0) & inside
    hollow_mask = cv2.morphologyEx((darker | redder).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    hollow = float(hollow_mask.sum()) / n

    # Long axis of the outline, for "how far from the centre line" and for the mirror test.
    pts = seg.contour.reshape(-1, 2).astype(np.float32)
    mean, vecs = cv2.PCACompute(pts, mean=None)[:2]
    axis, normal = vecs[0], np.array([-vecs[0][1], vecs[0][0]], np.float32)
    hollow_off = 0.0
    if hollow > 0.01:
        ys, xs = np.nonzero(hollow_mask)
        hollow_off = abs(float(np.dot(np.array([xs.mean(), ys.mean()], np.float32) - mean[0], normal))) / width

    ys, xs = np.nonzero(seg.mask)
    rel = np.stack([xs, ys], 1).astype(np.float32) - mean[0]
    along, across = rel @ axis, rel @ normal
    side = int(np.ceil(max(np.abs(along).max(), np.abs(across).max()))) + 2
    grid = np.zeros((2 * side + 1, 2 * side + 1), bool)
    grid[np.round(across).astype(int) + side, np.round(along).astype(int) + side] = True
    mirrored = grid[::-1, :]
    union = float((grid | mirrored).sum())
    symmetry = float((grid & mirrored).sum()) / union if union else 0.0

    lap = np.abs(cv2.Laplacian(L, cv2.CV_32F, ksize=3))
    spots = 0
    count, _, stats, _ = cv2.connectedComponentsWithStats(darker.astype(np.uint8), connectivity=8)
    for i in range(1, count):
        if 0.0005 * n <= stats[i, cv2.CC_STAT_AREA] <= 0.03 * n:
            spots += 1

    return {
        "aspect": float(length / width), "solidity": float(seg.area / max(hull_area, 1.0)), "defect": float(depth / width),
        "hollow": hollow, "hollow_off": hollow_off, "symmetry": symmetry, "edge": float(lap[inside].mean()),
        "L": med_L, "a": med_a, "b": med_b,
        "dark_pct": float(((L < DARK_L_MAX) & inside).sum()) * 100.0 / n,
        "red_pct": float(redder.sum()) * 100.0 / n,
        "spots": spots,
        "length_px": float(length), "width_px": float(width), "area_px": float(seg.area),
    }


def classify(features, rules):
    """View from the owner's limits: rules = {"NECK": {"hollow": [min, max], ...}, "BACK": {...}, "TOP": {...}}.
    A view without limits is never chosen, so with empty rules every frame is OTHER."""
    for view in VIEWS:
        limits = (rules or {}).get(view) or {}
        used = [(name, lim) for name, lim in limits.items() if name in FEATURES and lim and (lim[0] is not None or lim[1] is not None)]
        if not used:
            continue
        if all((lim[0] is None or features[name] >= lim[0]) and (lim[1] is None or features[name] <= lim[1]) for name, lim in used):
            return view
    return OTHER


def short_text(features):
    """One line for the live picture, e.g. 'A1.52 S0.85 D0.35 H0.12 O0.20 Y0.80 E3.1'."""
    return " ".join(f"{label}{features[name]:.2f}" for name, (label, _) in FEATURES.items())
