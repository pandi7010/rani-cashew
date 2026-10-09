"""Inspection engines and fixed size rules (pure functions).

Engine switch, as on the 5-track machine: OFF = not measured, SHADOW = measured and logged as "would reject",
never fires, ACTIVE = may turn an accepted (GOOD) cashew into a reject. An engine can never accept a reject.
Limits are the defaults of the 5-track manual (B3, B4); sizes in mm need a calibrated px per mm.
"""
from statistics import median
from typing import List, Optional

OFF, SHADOW, ACTIVE = "OFF", "SHADOW", "ACTIVE"
MODES = (OFF, SHADOW, ACTIVE)

# name -> (title on the page, what it looks for, measured already?)
ENGINES = {
    "view":   ("View (rolled / NECK seen)", "GOOD only when the NECK view was seen in M frames", True),
    "shape":  ("Contour & Shape", "broken, too small, split pieces", True),
    "colour": ("OpenCV Colour", "dark, brown, reddish spots and dark area inside the mask", False),
    "sw_ww":  ("SW / WW colour", "scorched (SW) or white (WW) from the colour inside the mask", True),
    "size":   ("Size grade", "length / width / area outside every grade", True),
}
ENGINE_DEFAULTS = {name: SHADOW for name in ENGINES}
SHAPE_DEFAULTS = {"min_solidity": 0.75, "min_area_mm2": 100.0, "min_length_mm": 16.0, "min_length_width": 1.15}
DOUBLE_DEFAULTS = {"enabled": True, "max_single_length_mm": 45.0, "min_frames": 2, "min_fraction": 0.5}
RULE_DEFAULTS = {"max_lost_ms": 300.0}

SHAPE_BROKEN, SHAPE_TOO_SMALL, SHAPE_SPLIT = "SHAPE_BROKEN", "SHAPE_TOO_SMALL", "SHAPE_SPLIT"


def shape_check(features: List[dict], px_per_mm: float, limits: Optional[dict] = None) -> Optional[str]:
    """Reason why the shape engine would reject this cashew, or None. Uses the middle value of its views."""
    if not features or px_per_mm <= 0:
        return None
    lim = {**SHAPE_DEFAULTS, **(limits or {})}
    mid = lambda key: median(f[key] for f in features)
    solidity, length, width, area = mid("solidity"), mid("length_px") / px_per_mm, mid("width_px") / px_per_mm, mid("area_px") / px_per_mm ** 2
    if solidity < lim["min_solidity"]:
        return f"{SHAPE_BROKEN} solidity {solidity:.2f}"
    if area < lim["min_area_mm2"]:
        return f"{SHAPE_TOO_SMALL} {area:.0f} mm2"
    if length < lim["min_length_mm"] or length / max(width, 0.1) < lim["min_length_width"]:
        return f"{SHAPE_SPLIT} {length:.1f} x {width:.1f} mm"
    return None


def is_double(lengths_px: List[float], px_per_mm: float, rule: Optional[dict] = None) -> bool:
    """Two touching cashews: the detection was longer than one cashew in enough of its frames."""
    r = {**DOUBLE_DEFAULTS, **(rule or {})}
    if not r["enabled"] or not lengths_px or px_per_mm <= 0:
        return False
    long_frames = sum(1 for length in lengths_px if length / px_per_mm > r["max_single_length_mm"])
    return long_frames >= r["min_frames"] and long_frames >= r["min_fraction"] * len(lengths_px)
