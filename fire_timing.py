"""Valve timing (pure functions, no hardware).

Time mode, as in the 5-track manual (B1): the cashew must be in front of the nozzle in the MIDDLE of the pulse:

    delay = distance / speed - pulse / 2

`distance` is from the Decision line to the nozzle. The config stores, per relay, the distance from the
relay before it (the first relay of a ROI: from the Decision line), so distances are added up along the ROI.
"""
from typing import Dict, Iterable, List

ROI_RELAYS = {1: range(1, 21), 2: range(21, 41)}


def delay_ms(distance_mm: float, speed_mm_s: float, pulse_ms: float) -> int:
    if speed_mm_s <= 0:
        raise ValueError("conveyor speed must be above 0")
    return max(0, int(round(distance_mm / speed_mm_s * 1000.0 - pulse_ms / 2.0)))


def cumulative_distances(spacing_mm: Dict[int, float], relays: Iterable[int]) -> Dict[int, float]:
    """Distance of each relay from the Decision line, from the stored relay-to-relay spacings."""
    total, out = 0.0, {}
    for relay in relays:
        total += float(spacing_mm.get(relay, 0.0))
        out[relay] = total
    return out


def delays_from_distances(spacing_mm: Dict[int, float], speed_mm_s: float, pulse_ms: float) -> Dict[int, int]:
    """Delay of every relay 1-40 from the distances, the conveyor speed and the pulse."""
    out = {}
    for relays in ROI_RELAYS.values():
        for relay, distance in cumulative_distances(spacing_mm, relays).items():
            out[relay] = delay_ms(distance, speed_mm_s, pulse_ms)
    return out


def inconsistent_delays(delays: Dict[int, int]) -> List[int]:
    """Relays whose delay is not larger than the delay of the relay before them in the same ROI."""
    bad = []
    for relays in ROI_RELAYS.values():
        relays = list(relays)
        bad += [r for prev, r in zip(relays, relays[1:]) if delays.get(r, 0) <= delays.get(prev, 0)]
    return bad
