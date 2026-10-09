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
COLOUR_DEFAULTS = {"metric": "b", "threshold": None, "sw_above": True, "enabled": False}     # enabled = COLOUR SPLIT switch
SIZES = ("180", "210", "240", "320", "450")
WW_GRADES = tuple(f"W{s}" for s in SIZES)
SW_GRADES = tuple(f"SW{s}" for s in SIZES)
BUDDS, SW_BUDDS = "BUDDS", "SW_BUDDS"
ALL_GRADES = WW_GRADES + (BUDDS,) + SW_GRADES + (SW_BUDDS,)
# Start values (owner, 2026-10-10, from the old length table), length in mm: min <= length < max; None = no limit.
START_LENGTHS = {"180": (30.0, None), "210": (28.0, 30.0), "240": (26.0, 28.0), "320": (24.5, 26.0), "450": (18.0, 24.5)}
BUDDS_BELOW = 18.0


def default_grades():
    """The size table before the owner changes it: W and SW grades with the same lengths, BUDDS below 18 mm."""
    out = {}
    for prefix in ("W", "SW"):
        for size, (low, high) in START_LENGTHS.items():
            out[f"{prefix}{size}"] = {"l_min": low, "l_max": high}
    out[BUDDS] = {"l_min": None, "l_max": BUDDS_BELOW}
    out[SW_BUDDS] = {"l_min": None, "l_max": BUDDS_BELOW}
    return out


def grades_or_default(grades):
    return grades if grades else default_grades()


def has_limits(row):
    return bool(row) and (row.get("l_min") is not None or row.get("l_max") is not None)
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
    """(SW or WW or None, value used). COLOUR SPLIT off: always WW (all GOOD cashews use the W grades).
    None while the split is on but no threshold is set or the cashew has no colour value."""
    cfg = {**COLOUR_DEFAULTS, **(cfg or {})}
    if not cfg.get("enabled"):
        return WW, (values or {}).get(cfg["metric"])
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
    """Grade names of a colour, with its BUDDS row."""
    return SW_GRADES + (SW_BUDDS,) if colour == SW else WW_GRADES + (BUDDS,)


def _fits(limits, length, width, area):
    """True when the cashew is inside the grade: min <= value < max for the length; width and area limits
    (set only by an older TEACH) are checked when present. A missing limit is an open end."""
    if not has_limits(limits):
        return False
    for value, low, high in ((length, "l_min", "l_max"), (width, "w_min", "w_max"), (area, "a_min", "a_max")):
        if limits.get(low) is not None and value < limits[low]:
            return False
        if limits.get(high) is not None and value >= limits[high]:
            return False
    return True


def _distance(limits, length):
    """How far the length is from the middle of the grade (for a cashew that fits two overlapping grades)."""
    low, high = limits.get("l_min"), limits.get("l_max")
    if low is not None and high is not None:
        return abs(length - (low + high) / 2.0) / max((high - low) / 2.0, 1e-6)
    return 1.0                      # an open-ended grade (>= 30 mm, < 18 mm): the closed one wins when both fit


def grade_of(colour, length_mm, width_mm, area_mm2, grades):
    """(grade name or None, reason). Grades of the cashew's colour (with its BUDDS row); the best fitting one when
    ranges overlap. reason: "" | SIZE_NOT_TAUGHT (no grade of this colour has limits) | SIZE_OUT_OF_RANGE."""
    grades = grades_or_default(grades)
    names = [n for n in grades_of(colour) if has_limits(grades.get(n))]
    if not [n for n in names if n not in (BUDDS, SW_BUDDS)]:
        return None, NOT_TAUGHT
    if not length_mm or length_mm <= 0:
        return None, OUT_OF_RANGE
    fitting = [n for n in names if _fits(grades[n], length_mm, width_mm, area_mm2)]
    if not fitting:
        return None, OUT_OF_RANGE
    return min(fitting, key=lambda n: _distance(grades[n], length_mm)), ""


def mapping_name(colour, grade):
    """Name in the ejector mapping (the grade names are the mapping names: W240, SW210, BUDDS, SW_BUDDS)."""
    return grade


def copy_w_to_sw(grades):
    """New grades dict in which every SW grade (and SW_BUDDS) has the limits of its W grade."""
    out = {name: dict(row) for name, row in grades_or_default(grades).items()}
    copied = []
    for source_name, target in [(f"W{size}", f"SW{size}") for size in SIZES] + [(BUDDS, SW_BUDDS)]:
        source = out.get(source_name)
        if has_limits(source):
            out[target] = {**{k: source.get(k) for k in LIMIT_KEYS if k in source}, "taught": f"copied from {source_name}"}
            copied.append(target)
    return out, copied


def text_of(row):
    """Short text of a grade's length range: ">= 30", "28 - 30", "< 18", "-"."""
    low, high = (row or {}).get("l_min"), (row or {}).get("l_max")
    if low is None and high is None:
        return "-"
    if high is None:
        return f">= {low:g}"
    if low is None:
        return f"< {high:g}"
    return f"{low:g} - {high:g}"


def changes(old, new, prefix=""):
    """["key: old -> new", ...] for two (nested) settings dicts; only what differs."""
    out = []
    for key in sorted(set(old or {}) | set(new or {}), key=str):
        a, b = (old or {}).get(key), (new or {}).get(key)
        if isinstance(a, dict) or isinstance(b, dict):
            out += changes(a if isinstance(a, dict) else {}, b if isinstance(b, dict) else {}, f"{prefix}{key}.")
        elif a != b:
            out.append(f"{prefix}{key}: {'-' if a is None else a} -> {'-' if b is None else b}")
    return out


def would_be(colour, grade, reason):
    """Short text for the log and the results table: "SW210", "WW out of range", "SW? ..."."""
    if grade:
        return mapping_name(colour, grade)
    return f"{colour or '?'} {reason.replace('SIZE_', '').replace('_', ' ').lower()}".strip()
