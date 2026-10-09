"""Class / grade -> ejector mapping (pure rules, no GUI). The grading logic uses ONLY this mapping:
no class name is tied to an ejector number in the code, so any combination works without code changes.

Rules (owner, 2026-10-09):
  * One ejector may carry several classes / grades (DW1 + DW2, W180 + SW180, PKW1 + PKW2 ...).
  * One class / grade may be on only ONE ejector per ROI. Ticking it on a second ejector means "move it".
  * Ejectors 1 (ROI 1) and 21 (ROI 2) are reject only: nothing can be assigned to them.
  * A class or grade without an ejector goes to the reject ejector of its ROI.
  * An ejector without a class is free and is never fired by grading.
  * ROI 2 mirrors ROI 1: ejector r in ROI 1 and r + 20 in ROI 2 (when the mirror option is used).
"""
from typing import Dict, Iterable, List, Optional, Tuple

from class_registry import normalize_name

RELAYS_PER_ROI = 20
REJECT_RELAY = {1: 1, 2: 21}
ROI_RELAYS = {1: range(2, 21), 2: range(22, 41)}        # the ejectors a class can be assigned to
ALL_RELAYS = range(1, 41)

GRADES_WW = ["W180", "W210", "W240", "W320", "W450", "BUDDS"]
GRADES_SW = ["SW180", "SW210", "SW240", "SW320", "SW450", "SW_BUDDS"]
GROUPS = ["GOOD_CASHEW", "WW", "SW"]                    # whole-group names a grade falls back to


def roi_of(relay: int) -> int:
    return 1 if relay <= RELAYS_PER_ROI else 2


def mirror(relay: int) -> int:
    """The same ejector position in the other ROI."""
    return relay + RELAYS_PER_ROI if relay <= RELAYS_PER_ROI else relay - RELAYS_PER_ROI


def is_reject(relay: int) -> bool:
    return relay in REJECT_RELAY.values()


def clean(mapping) -> Dict[int, List[str]]:
    """{relay 1-40: [names]} from whatever the config holds; a name appears once per ejector."""
    out = {r: [] for r in ALL_RELAYS}
    for key, names in (mapping or {}).items():
        try:
            relay = int(key)
        except (TypeError, ValueError):
            continue
        if relay not in out:
            continue
        seen = set()
        for name in names or []:
            norm = normalize_name(name)
            if norm and norm not in seen:
                seen.add(norm)
                out[relay].append(str(name))
    return out


def duplicates(mapping) -> Dict[int, Dict[str, List[int]]]:
    """{roi: {class: [ejectors]}} for every class that sits on more than one ejector of a ROI."""
    m, out = clean(mapping), {}
    for roi, relays in ROI_RELAYS.items():
        where: Dict[str, Tuple[str, List[int]]] = {}
        for relay in relays:
            for name in m[relay]:
                where.setdefault(normalize_name(name), (name, []))[1].append(relay)
        found = {name: places for name, places in where.values() if len(places) > 1}
        if found:
            out[roi] = found
    return out


def wrongly_on_reject(mapping) -> Dict[int, List[str]]:
    """Classes stored on a reject ejector (1 / 21): not allowed, they would be ignored."""
    m = clean(mapping)
    return {relay: m[relay] for relay in REJECT_RELAY.values() if m[relay]}


def owner(mapping, roi: int, name: str) -> Optional[int]:
    """The ejector of a class in a ROI (lowest number if a broken config has it twice), or None."""
    m, norm = clean(mapping), normalize_name(name)
    for relay in ROI_RELAYS[roi]:
        if any(normalize_name(n) == norm for n in m[relay]):
            return relay
    return None


def conflicts(mapping, relay: int, names: Iterable[str]) -> Dict[str, int]:
    """{class: the other ejector it is on now} for a new selection of one ejector."""
    out = {}
    for name in names:
        other = owner(mapping, roi_of(relay), name)
        if other is not None and other != relay:
            out[name] = other
    return out


def assign(mapping, relay: int, names: Iterable[str], move: bool = False, mirror_roi: bool = True) -> Dict[int, List[str]]:
    """New mapping with `names` as the classes of `relay` (an empty list frees the ejector).
    A class that is on another ejector of the ROI is moved here when move=True; otherwise ValueError.
    mirror_roi=True does the same on the mirror ejector of the other ROI."""
    if relay not in ALL_RELAYS:
        raise ValueError(f"ejector {relay} does not exist (1-40)")
    if is_reject(relay):
        raise ValueError(f"ejector {relay} is the reject ejector: no class can be assigned to it")
    names = [n for n in clean({relay: list(names)})[relay]]
    m = clean(mapping)
    for target in ([relay, mirror(relay)] if mirror_roi else [relay]):
        clash = conflicts(m, target, names)
        if clash and not move:
            text = ", ".join(f"{name} is on ejector {other}" for name, other in clash.items())
            raise ValueError(f"already assigned in ROI {roi_of(target)}: {text}")
        norms = {normalize_name(n) for n in names}
        for other in ROI_RELAYS[roi_of(target)]:
            if other != target:
                m[other] = [n for n in m[other] if normalize_name(n) not in norms]
        m[target] = list(names)
    return m


def lookup(mapping) -> Dict[int, Dict[str, int]]:
    """{roi: {normalised class name: ejector}}. Reject ejectors carry nothing; a duplicate uses the lowest ejector."""
    m, out = clean(mapping), {1: {}, 2: {}}
    for roi, relays in ROI_RELAYS.items():
        for relay in relays:
            for name in m[relay]:
                out[roi].setdefault(normalize_name(name), relay)
    return out


def resolve(table: Dict[str, int], candidates: Iterable[str]) -> Optional[int]:
    """The ejector of the first candidate name that has one (table = lookup(mapping)[roi]), or None."""
    for name in candidates:
        relay = table.get(normalize_name(name))
        if relay is not None:
            return relay
    return None


def grade_name(colour: str, size_label: str) -> str:
    """Name of a size grade in the mapping: W180 for WW, SW180 for SW; BUDDS / SW_BUDDS."""
    size_label = str(size_label).upper()
    if str(colour).upper() != "SW":
        return size_label
    return f"SW{size_label[1:]}" if size_label.startswith("W") and size_label[1:].isdigit() else f"SW_{size_label}"


def grade_candidates(colour: str, size_label: str) -> List[str]:
    """Names tried for a GOOD cashew, most specific first: its grade, its colour group, then GOOD_CASHEW."""
    return [grade_name(colour, size_label), "SW" if str(colour).upper() == "SW" else "WW", "GOOD_CASHEW", "GOOD"]


def target_of(table: Dict[str, int], roi: int, candidates: Iterable[str]) -> Tuple[int, bool]:
    """(ejector, assigned?) for a decided class: its ejector, or the reject ejector of the ROI when it has none."""
    relay = resolve(table, candidates)
    return (relay, True) if relay is not None else (REJECT_RELAY[roi], False)


def unassigned(mapping, classes: Iterable[str], grades: Iterable[str] = (), roi: int = 1) -> List[str]:
    """Classes and grades that have no ejector in this ROI (they go to the reject ejector)."""
    table = lookup(mapping)[roi]
    out = [name for name in classes if resolve(table, [name]) is None]
    for grade in grades:
        colour = "SW" if grade in GRADES_SW else "WW"
        if resolve(table, [grade, colour, "GOOD_CASHEW", "GOOD"]) is None:
            out.append(grade)
    return out


def free_relays(mapping, roi: int) -> List[int]:
    m = clean(mapping)
    return [relay for relay in ROI_RELAYS[roi] if not m[relay]]


def label(mapping, relay: int) -> str:
    """Text of an ejector row."""
    if is_reject(relay):
        return "REJECT - FIXED"
    names = clean(mapping)[relay]
    return ", ".join(names) if names else "(free)"


def summary(mapping, classes: Iterable[str], grades: Iterable[str] = ()) -> List[Tuple[str, str, str]]:
    """Rows (class / grade, ROI 1 target, ROI 2 target) for the table shown before saving."""
    tables = lookup(mapping)
    rows = []

    def cell(roi, candidates):
        relay, assigned = target_of(tables[roi], roi, candidates)
        if assigned and normalize_name(candidates[0]) not in tables[roi]:
            via = next(c for c in candidates if normalize_name(c) in tables[roi])
            return f"A{relay:02d} (via {via})"
        return f"A{relay:02d}" if assigned else f"REJECT A{relay:02d}"
    for name in classes:
        rows.append((name, cell(1, [name]), cell(2, [name])))
    for grade in grades:
        colour = "SW" if grade in GRADES_SW else "WW"
        chain = [grade, colour, "GOOD_CASHEW", "GOOD"]
        rows.append((grade, cell(1, chain), cell(2, chain)))
    return rows
