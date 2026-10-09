"""Teach a size grade from measured cashews (5-track "TEACH FROM CAMERA"): limits = mean -/+ k x std.

Pure functions. A sample is {"l": length mm, "w": width mm, "a": area mm2}. The limits are STORED with the grade
(size_config["grades"]); which grade a GOOD cashew gets is still decided by the old length table until step (c2).
"""
MIN_SAMPLES = 30        # 5-track: needs at least 30 cashews
TARGET_SAMPLES = 50
K_DEFAULT = 3.0
GRADES = ("W180", "W210", "W240", "W320", "W450", "BUDDS")
KEYS = (("l", "LENGTH", "mm"), ("w", "WIDTH", "mm"), ("a", "AREA", "mm2"))


def mean_std(values):
    values = [float(v) for v in values]
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    return mean, (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


def usable(samples):
    """Samples that can teach a size: measured (length, width and area above 0) and not a double."""
    return [s for s in samples if s.get("l", 0) > 0 and s.get("w", 0) > 0 and s.get("a", 0) > 0 and not s.get("double")]


def limits(samples, k=K_DEFAULT):
    """{l_min, l_max, w_min, w_max, a_min, a_max, n, k} from the samples, or None with fewer than MIN_SAMPLES."""
    good = usable(samples)
    if len(good) < MIN_SAMPLES:
        return None
    out = {"n": len(good), "k": float(k)}
    for key, _, _ in KEYS:
        mean, std = mean_std(s[key] for s in good)
        out[f"{key}_min"], out[f"{key}_max"] = round(max(0.0, mean - k * std), 2), round(mean + k * std, 2)
    return out
