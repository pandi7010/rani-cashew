"""Cashew mask by blue-background removal.

The belt is blue and the gaps between the rollers are dark; a cashew is neither. Everything that
measures a cashew (centroid, view, colour, size, dark / skin area) uses only this mask.
The CNN crop is NOT made here (see preprocess.py), so the model keeps seeing what it was trained on.
"""
from dataclasses import dataclass

import cv2
import numpy as np

# Start values: the band that separated 1,300+ crops of the old dataset and the four rolling samples.
# TEACH BELT COLOUR replaces the three blue values from an empty-belt frame.
DEFAULTS = {
    "blue_h_min": 95, "blue_h_max": 135,    # OpenCV hue (0-179) of the belt
    "blue_s_min": 60,                       # belt pixels are at least this saturated
    "dark_v_max": 45,                       # roller gaps / shadows are darker than this
    "min_area_px": 200,                     # smaller blobs are not a cashew
    "edge_margin_px": 2,                    # a box this close to the ROI edge is a cut-off cashew
}

_OPEN = np.ones((5, 5), np.uint8)
_CLOSE = np.ones((9, 9), np.uint8)


@dataclass
class Segment:
    mask: np.ndarray        # uint8, 255 inside the cashew, same size as the crop
    contour: np.ndarray
    area: float
    cx: float               # mask centroid, crop coordinates
    cy: float
    rect: tuple             # (x, y, w, h), crop coordinates


def segment(crop_bgr, params=None):
    """Largest not-blue, not-dark blob of the crop, or None when there is none (e.g. a black piece)."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    p = {**DEFAULTS, **(params or {})}
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (int(p["blue_h_min"]), int(p["blue_s_min"]), 0), (int(p["blue_h_max"]), 255, 255))
    dark = cv2.inRange(hsv, (0, 0, 0), (180, 255, int(p["dark_v_max"])))
    raw = cv2.bitwise_not(cv2.bitwise_or(blue, dark))
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, _OPEN)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, _CLOSE)
    contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < float(p["min_area_px"]):
        return None
    mask = np.zeros(raw.shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)      # filled: holes inside the cashew belong to it
    m = cv2.moments(contour)
    if m["m00"] <= 0:
        return None
    return Segment(mask, contour, float(area), m["m10"] / m["m00"], m["m01"] / m["m00"], cv2.boundingRect(contour))


def touches_edge(box, roi, margin):
    """True when a box (x1, y1, x2, y2) reaches the edge of its ROI: the cashew is cut off there."""
    return (box[0] <= roi[0] + margin or box[1] <= roi[1] + margin or
            box[2] >= roi[2] - margin or box[3] >= roi[3] - margin)


def teach_belt(belt_bgr, current=None):
    """Blue band of an EMPTY belt image. Returns new params, or None when the image is not mostly blue."""
    p = {**DEFAULTS, **(current or {})}
    hsv = cv2.cvtColor(belt_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lit = (v > int(p["dark_v_max"])) & (s > 30)
    if lit.mean() < 0.2:
        return None
    hues, sats = h[lit], s[lit]
    lo, hi = np.percentile(hues, [1, 99])
    if hi - lo > 60:        # not one colour: something other than the belt is in the picture
        return None
    p["blue_h_min"] = int(max(0, lo - 5))
    p["blue_h_max"] = int(min(179, hi + 5))
    p["blue_s_min"] = int(max(30, np.percentile(sats, 1) - 15))
    return p
