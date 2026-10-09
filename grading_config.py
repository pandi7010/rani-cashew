"""The "grading" block of pneumatic_config.json: defaults and merging with what is stored."""
import copy

import decision
import engines
import lane_tracker
import segmentation
import view_features


def defaults() -> dict:
    return {
        "voting": dict(decision.VOTING_DEFAULTS),
        "priority": list(decision.PRIORITY_DEFAULT),
        "good_classes": list(decision.GOOD_CLASSES_DEFAULT),
        "skin_classes": list(decision.SKIN_CLASSES_DEFAULT),
        # View limits are the owner's: {"NECK": {"hollow": [min, max]}, ...}. Empty = every frame is OTHER.
        "views": {"rules": {view: {} for view in view_features.VIEWS}, "show_values": True},
        "segmentation": dict(segmentation.DEFAULTS),
        "tracker": dict(lane_tracker.DEFAULTS),
        "frame_log": {"enabled": True, "keep_days": 7},
        # Inspection engines (OFF / SHADOW / ACTIVE) and the fixed rules of the 5-track machine.
        "engines": dict(engines.ENGINE_DEFAULTS),
        "shape": dict(engines.SHAPE_DEFAULTS),
        "double": dict(engines.DOUBLE_DEFAULTS),
        "rules": dict(engines.RULE_DEFAULTS),
        "avg_cashew_weight_g": 1.8,         # for kg/hr only (5-track default); sorting is not affected
    }


def merged(stored) -> dict:
    """Defaults overlaid with the stored values (stored keys win; unknown keys are kept)."""
    out = defaults()
    for key, value in (stored or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = {**out[key], **copy.deepcopy(value)}
        else:
            out[key] = copy.deepcopy(value)
    rules = out["views"].setdefault("rules", {})
    for view in view_features.VIEWS:
        rules.setdefault(view, {})
    return out
