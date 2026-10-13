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
SINGLE_DEFECT_FRAME, NOT_IN_PRIORITY, NOT_MAPPED = "single_defect_frame", "class_not_in_priority_list", "not_mapped"
# Always-reject reasons of the 5-track machine (manual B14). They come before every class rule.
APPEARED_AFTER_LINE, STUCK, DOUBLE_OR_TOUCHING = "APPEARED_AFTER_LINE", "STUCK", "DOUBLE_OR_TOUCHING"
MISSED_LINE = APPEARED_AFTER_LINE
# A cashew that disappeared before its line: decision LOST, nothing is fired, it is counted.
LOST, SEEN_ONLY_ONCE, LOST_BEFORE_LINE, LEFT_ROI_UNDECIDED = "LOST", "SEEN_ONLY_ONCE", "LOST_BEFORE_LINE", "LEFT_ROI_UNDECIDED"
OFF, SHADOW, ACTIVE = "OFF", "SHADOW", "ACTIVE"        # inspection engine switch
VIEW_NOT_ROLLED = "VIEW_NOT_ROLLED"

# Start values given by the owner (FINAL_PROMPT section 4 and the answers of 2026-10-09).
VOTING_DEFAULTS = {
    "defect_conf": 0.60, "defect_min_frames": 2,
    "good_conf": 0.70, "good_min_frames": 3,
    "neck_min_frames": 0,               # M: GOOD needs the NECK view in at least this many frames. 0 = rule on hold
                                        # (owner, 2026-10-09); a cashew whose NECK was never seen is still noted.
    "neck_weight": 2,                   # a skin vote from a NECK frame counts as this many frames
    "rejection_conf": 0.75, "rejection_min_frames": 3,
    "single_defect_blocks_good": False, # True: one defect frame (below the frame limit) makes the cashew Undefined
}
PRIORITY_DEFAULT = ["UPW", "PKW2", "PKW1", "OIL", "BROWN", "DW2", "DW1", "TESTA", "TESTA SMALL", "WHITE_DW"]
GOOD_CLASSES_DEFAULT = ["GOOD_CASHEW", "GOOD", "SW"]       # "no defect" votes; SW / WW is decided by colour
SKIN_CLASSES_DEFAULT = ["TESTA", "TESTA SMALL"]
# Owner, 2026-10-14: "if you found PKW + TESTA it should go to PKW only". A piece with skin on it shows the skin in
# most frames and the broken side in one, so next to skin ONE piece frame (at the defect confidence) is enough.
PIECE_CLASSES_DEFAULT = ["PKW2", "PKW1"]
# Owner, 2026-10-09: BACK_CUT_OIL / OIL_BACKCUT are graded (and trained) as OIL. EDGECUT stays its own class;
# it is not in the priority list, so it goes to the reject relay until a relay is given for it.
CLASS_ALIASES = {"back cut oil": "OIL", "oil backcut": "OIL", "backcut oil": "OIL", "oil back cut": "OIL",
                 "double kernels": "DOUBLE", "double kernel": "DOUBLE", "double cashews": "DOUBLE"}
DOUBLE_CLASS = "DOUBLE"         # CNN class "two cashews stuck together": always the reject ejector (besides the size rule)
REJECT_CLASSES_DEFAULT = ["EDGECUT"]    # classes that always go to the reject ejector under their own name (owner, 2026-10-09)
NOISE = "NOISE"                 # a tiny short-lived detection that was not a cashew: logged, never counted or fired
MERGED = "MERGED"               # a fragment of another tracked cashew (broken ID, second box): logged, never decided or fired
BACKGROUND = "BACKGROUND"       # a detection that never moved with the belt (roller, mark on the frame): logged, never decided or fired
IGNORED_PREFIX = "IGNORED: "    # counter keys of MERGED / NOISE / BACKGROUND (not part of TOTAL)
SIZE_NOT_READY = "SIZE_NOT_READY"       # GOOD while px per mm and the size grades are not taught yet: reject ejector


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
        """The class of the frame: the CNN's, and only the CNN's. Owner, 2026-10-13: YOLO finds the cashew in the frame;
        it has no say in what the cashew is. Its class is still stored (yolo_class) for the log and the screen."""
        return (self.cnn_class, self.cnn_conf)


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
    shadow: List[str] = field(default_factory=list)             # "would reject" notes of engines in SHADOW


ALWAYS_PASS = ("GOOD_CASHEW", "GOOD")
ALWAYS_FIRE = (REJECTION, DOUBLE_CLASS, DOUBLE, UNDEFINED)      # plus the reject classes of the settings (EDGECUT ...)


def can_change_pass(name, reject_classes=()) -> str:
    """"" when the PASS / FIRE chip of this class may be changed, otherwise the reason why not."""
    key = normalize_name(name)
    if key in {normalize_name(n) for n in ALWAYS_PASS}:
        return f"{name} is always PASS."
    if key in {normalize_name(n) for n in tuple(ALWAYS_FIRE) + tuple(reject_classes or ())}:
        return f"{name} always goes to the reject ejector and cannot be set to PASS."
    return ""


def passed_class(votes, good_classes) -> str:
    """For a cashew decided GOOD: the PASS class it was really seen as (e.g. "TESTA SMALL"), or "" when that is
    GOOD_CASHEW itself. votes = {class: frames}; the PASS class with the most frames."""
    always = {normalize_name(n) for n in ALWAYS_PASS}
    good = {normalize_name(c) for c in (good_classes or ())}
    counted = {name: n for name, n in (votes or {}).items() if normalize_name(name) in good}
    if not counted:
        return ""
    top = max(counted, key=lambda name: (counted[name], normalize_name(name) in always))
    return "" if normalize_name(top) in always else str(top)


def _blank(name) -> bool:
    return normalize_name(name) in ("", "none", "-", "any")


def decide(votes: List[FrameVote], voting: Optional[dict] = None, priority: Optional[List[str]] = None,
           good_classes=None, skin_classes=None, combination_rules=None, double: bool = False,
           appeared_after_line: bool = False, stuck: bool = False, lost_ms: float = 0.0, max_lost_ms: float = 300.0,
           view_engine: str = ACTIVE, reject_classes=None) -> Decision:
    """view_engine: ACTIVE = GOOD needs the NECK view; SHADOW = it is only noted in `shadow`; OFF = not checked."""
    v = {**VOTING_DEFAULTS, **(voting or {})}
    priority = [normalize_name(c) for c in (priority if priority is not None else PRIORITY_DEFAULT)]
    display = {normalize_name(c): c for c in (PRIORITY_DEFAULT + GOOD_CLASSES_DEFAULT)}
    good_set = {normalize_name(c) for c in (good_classes if good_classes is not None else GOOD_CLASSES_DEFAULT)}
    # REJECTION, DOUBLE and the reject classes can never count as "no defect", whatever the PASS list says.
    good_set -= {normalize_name(c) for c in tuple(ALWAYS_FIRE) + tuple(reject_classes if reject_classes is not None else REJECT_CLASSES_DEFAULT)}
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
        if f.cnn_class and normalize_name(f.cnn_class) == rejection and f.cnn_conf >= v["rejection_conf"]:
            rejection_frames += 1       # the CNN's REJECTION only; YOLO's class does not count (owner, 2026-10-13)
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

    if appeared_after_line:
        return done(UNDEFINED, APPEARED_AFTER_LINE, True)
    if double:
        return done(DOUBLE, DOUBLE_OR_TOUCHING, True)
    if stuck:
        return done(UNDEFINED, STUCK, True)
    if lost_ms > max_lost_ms:
        return done(UNDEFINED, f"LOST_{lost_ms:.0f}MS", True)
    if rejection_frames >= v["rejection_min_frames"]:
        return done(REJECTION, "rejection_seen", True)

    seen = {k for k, n in weighted.items() if n >= v["defect_min_frames"]}
    if normalize_name(DOUBLE_CLASS) in seen:
        return done(DOUBLE, "double_class_seen", True)
    for name in (reject_classes if reject_classes is not None else REJECT_CLASSES_DEFAULT):
        if normalize_name(name) in seen:
            return done(display.get(normalize_name(name), str(name)), "reject_class_seen", True)

    for rule in combination_rules or []:        # owner's "these classes together -> that class" table
        wanted = [normalize_name(c) for c in rule.get("conditions", []) if not _blank(c)]
        final = rule.get("final")
        if wanted and not _blank(final) and all(c in seen for c in wanted):
            is_reject = normalize_name(final) in (rejection, normalize_name(UNDEFINED))
            return done(str(final), "combination_rule", is_reject)

    if seen:
        if upw in seen:
            return done(UPW, "upw_seen", False)
        for cls in priority:
            if cls in seen:
                if cls in skin_set:         # skin would win: a piece class seen in even one frame takes it (PKW + TESTA = PKW)
                    pieces = [normalize_name(c) for c in PIECE_CLASSES_DEFAULT if weighted.get(normalize_name(c), 0) > 0]
                    if pieces:
                        piece = max(pieces, key=lambda c: weighted[c])      # most frames; equal: the first of the list
                        return done(display.get(piece, piece.upper()), "piece_with_skin", False)
                return done(display.get(cls, cls.upper()), "defect_seen", False)
        return done(UNDEFINED, NOT_IN_PRIORITY, True)

    if not used:
        return done(UNDEFINED, NO_FRAMES, True)
    if weighted and v["single_defect_blocks_good"]:
        return done(UNDEFINED, SINGLE_DEFECT_FRAME, True)
    if d.good_frames >= v["good_min_frames"]:
        if view_engine == OFF or d.neck_frames >= max(1, v["neck_min_frames"]):
            return done(GOOD, "good", False)
        # NECK not seen often enough. It rejects only with M above 0 and the View engine ACTIVE;
        # otherwise the cashew stays GOOD and "would be not rolled" is noted for the log.
        if v["neck_min_frames"] > 0 and view_engine == ACTIVE:
            return done(UNDEFINED, NOT_ROLLED, True)
        d.shadow.append(VIEW_NOT_ROLLED)
        return done(GOOD, "good", False)
    return done(UNDEFINED, TOO_FEW_FRAMES if len(used) < v["good_min_frames"] else LOW_CONFIDENCE, True)
