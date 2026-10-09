"""Ejector -> PLC coil table (which M coil drives which ejector). Pure rules and the change log.

The table belongs to the machine (inovance_config.json, written by the PLC controller's own update_config()).
Rules: ejectors are relays 1-40; each has its own address; M140 (conveyor) and M141 (feeder) and the
addresses configured for relays 41 / 42 can never be given to an ejector.
"""
import os
import time
from typing import Dict, List

EJECTORS = range(1, 41)
PROTECTED = {140: "M140 conveyor", 141: "M141 feeder"}
DEFAULT_BASE = 100
LOG_NAME = "mapping_changes.log"


def default_table() -> Dict[int, int]:
    return {relay: DEFAULT_BASE + relay - 1 for relay in EJECTORS}


def problems(table: Dict[int, int], extra_protected=()) -> List[str]:
    """Texts of everything that makes a table unusable (empty list = it may be saved)."""
    out, seen = [], {}
    protected = dict(PROTECTED)
    for address in extra_protected:
        protected.setdefault(int(address), f"M{int(address)} (conveyor / feeder)")
    for relay in EJECTORS:
        address = table.get(relay)
        if not isinstance(address, int) or isinstance(address, bool) or not 0 <= address <= 65535:
            out.append(f"Ejector {relay}: '{address}' is not an address (0-65535)")
            continue
        if address in protected:
            out.append(f"Ejector {relay}: {protected[address]} can never be an ejector")
        seen.setdefault(address, []).append(relay)
    for address, relays in sorted(seen.items()):
        if len(relays) > 1:
            out.append(f"M{address} is given to ejectors {', '.join(str(r) for r in relays)} (each ejector needs its own coil)")
    return out


def changes(old: Dict[int, int], new: Dict[int, int]) -> List[str]:
    return [f"ejector {relay}: M{old.get(relay)} -> M{new.get(relay)}" for relay in EJECTORS if old.get(relay) != new.get(relay)]


def log_changes(log_dir, old: Dict[int, int], new: Dict[int, int]) -> List[str]:
    """Write every changed ejector to logs/mapping_changes.log (time, old -> new). Returns the lines."""
    lines = changes(old, new)
    if lines and log_dir:
        os.makedirs(str(log_dir), exist_ok=True)
        with open(os.path.join(str(log_dir), LOG_NAME), "a", encoding="utf-8") as f:
            for line in lines:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {line}\n")
    return lines
