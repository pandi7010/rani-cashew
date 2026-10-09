"""Step (c2): SW / WW colour and size grade of a GOOD cashew (pure functions, no camera, no UI).

Colour: one number per cashew = median over its frames of a LAB value measured inside the cashew mask (the blue
belt is removed, the picture has FFC / colour gain applied). Metric "b" (b*, yellow), "a" (a*, red) or "L" (L*,
lightness). A threshold splits SW from WW; `sw_above` says on which side scorched cashews lie. TEACH suggests all
three from cashews the owner ran as "only WW" and "only SW".

Size: taught limits per grade (length, width, area in mm / mm2, mean -/+ k x std). WW uses W180 ... W450, SW uses
SW180 ... SW450, BUDDS is one row for both colours.
"""
from statistics import median

WW, SW = "WW", "SW"
METRICS = {"b": "b* (yellow)", "a": "a* (red)", "L": "L* (lightness)"}
COLOUR_DEFAULTS = {"metric": "b", "threshold": None, "sw_above": True}
SIZES = ("180", "210", "240", "320", "450")
WW_GRADES = tuple(f"W{s}" for s in SIZES)
SW_GRADES = tuple(f"SW{s}" for s in SIZES)
BUDDS = "BUDDS"
ALL_GRADES = WW_GRADES + SW_GRADES + (BUDDS,)
LIMIT_KEYS = ("l_min", "l_max", "w_min", "w_max", "a_min", "a_max")
OUT_OF_RANGE, NOT_TAUGHT, COLOUR_NOT_TAUGHT = "SIZE_OUT_OF_RANGE", "SIZE_NOT_TAUGHT", "COLOUR_NOT_TAUGHT"
MIN_COLOUR_SAMPLES = 20


# ------------------------------------------------------------------ colour
def track_colour(features):
    """{"L", "a", "b"}: median over the frames that have a mask, or None when no frame has one."""
    frames = [f for f in features if f]
    if not frames:
        return None
    return {key: float(median(f[key] for f in frames)) for key in ("L", "a", "b")}


def colour_of(values, cfg):
    """(SW or WW or None, value used). None while no threshold is taught or the cashew has no colour value."""
    cfg = {**COLOUR_DEFAULTS, **(cfg or {})}
    if not values or cfg["threshold"] is None:
        return None, (values or {}).get(cfg["metric"])
    value = values[cfg["metric"]]
    above = value > float(cfg["threshold"])
    return (SW if above == bool(cfg["sw_above"]) else WW), value


def _stats(values):
    values = [float(v) for v in values]
    mean = sum(values) / len(values)
    return mean, (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


def suggest(ww_samples, sw_samples):
    """From taught cashews ([{"L", "a", "b"}, ...] each): for every metric the threshold between the two groups,
    the side of SW, and how many of the taught cashews it would put in the wrong group. Returns
    {"best": metric, "metrics": {metric: {"threshold", "sw_above", "wrong_pct", "separation", ...}}} or None
    with fewer than MIN_COLOUR_SAMPLES in a group."""
    if len(ww_samples) < MIN_COLOUR_SAMPLES or len(sw_samples) < MIN_COLOUR_SAMPLES:
        return None
    out = {}
    for metric in METRICS:
        ww, sw = [s[metric] for s in ww_samples], [s[metric] for s in sw_samples]
        (m_ww, s_ww), (m_sw, s_sw) = _stats(ww), _stats(sw)
        s_ww, s_sw = max(s_ww, 1e-6), max(s_sw, 1e-6)
        # the point as many standard deviations from the one mean as from the other
        threshold = (m_ww * s_sw + m_sw * s_ww) / (s_ww + s_sw)
        sw_above = m_sw > m_ww
        wrong = sum((v > threshold) == sw_above for v in ww) + sum((v > threshold) != sw_above for v in sw)
        out[metric] = {"threshold": round(threshold, 2), "sw_above": sw_above,
                       "wrong_pct": round(100.0 * wrong / (len(ww) + len(sw)), 1),
                       "separation": round(abs(m_sw - m_ww) / ((s_ww ** 2 + s_sw ** 2) / 2) ** 0.5, 2),
                       "ww_mean": round(m_ww, 2), "ww_std": round(s_ww, 2), "sw_mean": round(m_sw, 2), "sw_std": round(s_sw, 2)}
    best = min(out, key=lambda m: (out[m]["wrong_pct"], -out[m]["separation"]))
    return {"best": best, "metrics": out, "ww": len(ww_samples), "sw": len(sw_samples)}


# ------------------------------------------------------------------ size
def grades_of(colour):
    return SW_GRADES if colour == SW else WW_GRADES


def _fits(limits, length, width, area):
    """True when every taught limit of the grade holds (a missing limit is not checked; length is required)."""
    if limits.get("l_min") is None or limits.get("l_max") is None:
        return False
    for value, low, high in ((length, "l_min", "l_max"), (width, "w_min", "w_max"), (area, "a_min", "a_max")):
        if limits.get(low) is not None and value < limits[low]:
            return False
        if limits.get(high) is not None and value > limits[high]:
            return False
    return True


def _distance(limits, length, width, area):
    """How far from the middle of the grade, in half-widths of its ranges (for cashews that fit two grades)."""
    total, count = 0.0, 0
    for value, low, high in ((length, "l_min", "l_max"), (width, "w_min", "w_max"), (area, "a_min", "a_max")):
        if limits.get(low) is not None and limits.get(high) is not None and limits[high] > limits[low]:
            total += abs(value - (limits[low] + limits[high]) / 2.0) / ((limits[high] - limits[low]) / 2.0)
            count += 1
    return total / max(1, count)


def grade_of(colour, length_mm, width_mm, area_mm2, grades):
    """(grade name or None, reason). Grades of the cashew's colour plus BUDDS; the best fitting one when the
    taught ranges overlap. reason: "" | SIZE_NOT_TAUGHT (no grade of this colour has limits) | SIZE_OUT_OF_RANGE."""
    grades = grades or {}
    names = [n for n in grades_of(colour) + (BUDDS,) if grades.get(n, {}).get("l_min") is not None]
    if not [n for n in names if n != BUDDS]:
        return None, NOT_TAUGHT
    fitting = [n for n in names if _fits(grades[n], length_mm, width_mm, area_mm2)]
    if not fitting:
        return None, OUT_OF_RANGE
    return min(fitting, key=lambda n: _distance(grades[n], length_mm, width_mm, area_mm2)), ""


def mapping_name(colour, grade):
    """Name in the ejector mapping: W240, SW210, BUDDS for WW and SW_BUDDS for SW."""
    return "SW_BUDDS" if (grade == BUDDS and colour == SW) else grade


def copy_w_to_sw(grades):
    """New grades dict in which every SW grade has the limits of its W grade (taught note says so)."""
    out = {name: dict(row) for name, row in (grades or {}).items()}
    copied = []
    for size in SIZES:
        source = out.get(f"W{size}")
        if source and source.get("l_min") is not None:
            out[f"SW{size}"] = {**{k: source[k] for k in LIMIT_KEYS if k in source}, "taught": f"copied from W{size}"}
            copied.append(f"SW{size}")
    return out, copied


def would_be(colour, grade, reason):
    """Short text for the log and the results table: "SW210", "WW out of range", "SW? ..."."""
    if grade:
        return mapping_name(colour, grade)
    return f"{colour or '?'} {reason.replace('SIZE_', '').replace('_', ' ').lower()}".strip()
