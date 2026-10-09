"""One decision per cashew from the votes of all its frames (no GUI, no camera: plain rules).

Order of the rules (PLAN.md 3.4):
  1. double kernel                                   -> DOUBLE KERNEL (reject)
  2. REJECTION seen (YOLO class, or CNN once it has one) -> REJECTION (reject)
  3. owner's combination rules (priority_rules)      -> their final class
  4. UPW seen, or skin together with another defect  -> UPW
  5. highest class of the priority list that was seen
  6. no defect, enough GOOD frames, NECK seen        -> GOOD
  7. no defect, enough GOOD frames, NECK not seen    -> UNDEFINED (not_rolled)
  8. anything else                                   -> UNDEFINED
"Seen" = enough frames at or above the confidence limit. A TESTA / TESTA SMALL vote from a NECK
frame counts as `neck_weight` frames, because skin usually hides under the neck.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from class_registry import normalize_name

GOOD, UNDEFINED, REJECTION, DOUBLE, UPW = "GOOD", "UNDEFINED", "REJECTION", "DOUBLE KERNEL", "UPW"
NECK = "NECK"

# Reasons for UNDEFINED
NOT_ROLLED, NO_FRAMES, TOO_FEW_FRAMES, LOW_CONFIDENCE = "not_rolled", "no_frames", "too_few_frames", "low_confidence"
SINGLE_DEFECT_FRAME, NOT_IN_PRIORITY, NOT_MAPPED, MISSED_LINE = "single_defect_frame", "class_not_in_priority_list", "not_mapped", "first_seen_after_line"

# Start values given by the owner (FINAL_PROMPT section 4 and the answers of 2026-10-09).
VOTING_DEFAULTS = {
    "defect_conf": 0.50, "defect_min_frames": 2,
    "good_conf": 0.80, "good_min_frames": 3,
    "neck_min_frames": 1,               # M: GOOD needs the NECK view in at least this many frames (0 = not required)
    "neck_weight": 2,                   # a skin vote from a NECK frame counts as this many frames
    "rejection_conf": 0.50, "rejection_min_frames": 2,
    "single_defect_blocks_good": True,  # "no defect in any frame": one defect frame is not a defect, but not GOOD either
}
PRIORITY_DEFAULT = ["UPW", "PKW2", "PKW1", "OIL", "BROWN", "DW2", "DW1", "TESTA", "TESTA SMALL", "WHITE_DW"]
GOOD_CLASSES_DEFAULT = ["GOOD_CASHEW", "GOOD", "SW"]       # "no defect" votes; SW / WW is decided by colour
SKIN_CLASSES_DEFAULT = ["TESTA", "TESTA SMALL"]
# Owner, 2026-10-09: BACK_CUT_OIL / OIL_BACKCUT are graded (and trained) as OIL. EDGECUT stays its own class;
# it is not in the priority list, so it goes to the reject relay until a relay is given for it.
CLASS_ALIASES = {"back cut oil": "OIL", "oil backcut": "OIL", "backcut oil": "OIL", "oil back cut": "OIL"}


def canonical(name):
    """The class name the grading rules use for a model's class name."""
    return CLASS_ALIASES.get(normalize_name(name), name) if name else name


@dataclass
class FrameVote:
    t_ms: float
    cnn_class: Optional[str] = None
    cnn_conf: float = 0.0
    yolo_class: Optional[str] = None
    yolo_conf: float = 0.0
    view: str = "OTHER"
    usable: bool = True         # False for a cut-off crop at the ROI edge: it does not vote

    @property
    def cls(self):
        """The class of the frame: the CNN's, or YOLO's when the CNN gave none."""
        return (self.cnn_class, self.cnn_conf) if self.cnn_class else (self.yolo_class, self.yolo_conf)


@dataclass
class Decision:
    result: str                 # GOOD, UNDEFINED, REJECTION, DOUBLE KERNEL or a defect class name
    reason: str
    reject: bool                # True -> reject relay
    frames_total: int = 0
    frames_used: int = 0
    good_frames: int = 0
    neck_frames: int = 0
    seen: Dict[str, float] = field(default_factory=dict)        # class -> weighted qualifying frames
    votes: Dict[str, int] = field(default_factory=dict)         # class -> frames of any confidence
    views: Dict[str, int] = field(default_factory=dict)


def _blank(name) -> bool:
    return normalize_name(name) in ("", "none", "-", "any")


def decide(votes: List[FrameVote], voting: Optional[dict] = None, priority: Optional[List[str]] = None,
           good_classes=None, skin_classes=None, combination_rules=None, double: bool = False) -> Decision:
    v = {**VOTING_DEFAULTS, **(voting or {})}
    priority = [normalize_name(c) for c in (priority if priority is not None else PRIORITY_DEFAULT)]
    display = {normalize_name(c): c for c in (PRIORITY_DEFAULT + GOOD_CLASSES_DEFAULT)}
    good_set = {normalize_name(c) for c in (good_classes if good_classes is not None else GOOD_CLASSES_DEFAULT)}
    skin_set = {normalize_name(c) for c in (skin_classes if skin_classes is not None else SKIN_CLASSES_DEFAULT)}
    rejection, upw = normalize_name(REJECTION), normalize_name(UPW)

    used = [f for f in votes if f.usable]
    d = Decision(UNDEFINED, NO_FRAMES, True, frames_total=len(votes), frames_used=len(used))
    rejection_frames = 0
    weighted: Dict[str, float] = {}     # defect class -> weighted frames at / above defect_conf
    for f in used:
        d.views[f.view] = d.views.get(f.view, 0) + 1
        if f.view == NECK:
            d.neck_frames += 1
        name, conf = f.cls
        name = canonical(name)
        key = normalize_name(name) if name else ""
        if name:
            display.setdefault(key, name)
            d.votes[display[key]] = d.votes.get(display[key], 0) + 1
        if ((f.yolo_class and normalize_name(f.yolo_class) == rejection and f.yolo_conf >= v["rejection_conf"]) or
                (f.cnn_class and normalize_name(f.cnn_class) == rejection and f.cnn_conf >= v["rejection_conf"])):
            rejection_frames += 1
        if not key or key == rejection:
            continue
        if key in good_set:
            if conf >= v["good_conf"]:
                d.good_frames += 1
        elif conf >= v["defect_conf"]:
            weight = v["neck_weight"] if (f.view == NECK and key in skin_set) else 1
            weighted[key] = weighted.get(key, 0) + weight
    d.seen = {display[k]: n for k, n in weighted.items()}
    if rejection_frames:
        d.seen[REJECTION] = rejection_frames

    def done(result, reason, reject):
        d.result, d.reason, d.reject = result, reason, reject
        return d

    if double:
        return done(DOUBLE, "double", True)
    if rejection_frames >= v["rejection_min_frames"]:
        return done(REJECTION, "rejection_seen", True)

    seen = {k for k, n in weighted.items() if n >= v["defect_min_frames"]}

    for rule in combination_rules or []:        # owner's "these classes together -> that class" table
        wanted = [normalize_name(c) for c in rule.get("conditions", []) if not _blank(c)]
        final = rule.get("final")
        if wanted and not _blank(final) and all(c in seen for c in wanted):
            is_reject = normalize_name(final) in (rejection, normalize_name(UNDEFINED))
            return done(str(final), "combination_rule", is_reject)

    if seen:
        others = seen - skin_set - {upw}
        if upw in seen or (seen & skin_set and others):
            return done(UPW, "upw_seen" if upw in seen else "skin_with_other_defect", False)
        for cls in priority:
            if cls in seen:
                return done(display.get(cls, cls.upper()), "defect_seen", False)
        return done(UNDEFINED, NOT_IN_PRIORITY, True)

    if not used:
        return done(UNDEFINED, NO_FRAMES, True)
    if weighted and v["single_defect_blocks_good"]:
        return done(UNDEFINED, SINGLE_DEFECT_FRAME, True)
    if d.good_frames >= v["good_min_frames"]:
        if d.neck_frames >= v["neck_min_frames"]:
            return done(GOOD, "good", False)
        return done(UNDEFINED, NOT_ROLLED, True)
    return done(UNDEFINED, TOO_FEW_FRAMES if len(used) < v["good_min_frames"] else LOW_CONFIDENCE, True)
