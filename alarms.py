"""Alarms of the Home page and the SAFETY pill (pure function; the UI only shows what this returns).

Texts follow the 5-track manual (A6), with PLC instead of TCM32.
"""
from typing import List, Optional

MIN_FREE_GB = 20.0


def min_safe_fps(lane_speed_px_s: float, max_step_px: float) -> float:
    """Below this AI rate a cashew moves further between two inspected pictures than the tracker allows."""
    return lane_speed_px_s / max_step_px if max_step_px > 0 else 0.0


def collect(camera_connected: bool, camera_state: str, plc_connected: bool, running: bool, ai_fps: float,
            min_fps: float, free_gb: Optional[float], missed_rejects: List[str], dry_run: bool) -> List[str]:
    out = []
    if not camera_connected:
        out.append(f"CAMERA {camera_state} - no pictures from the camera")
    if not plc_connected:
        out.append("PLC NOT CONNECTED - rejects cannot be ejected" + (" (DRY-RUN is on: nothing would be sent anyway)" if dry_run else ""))
    if missed_rejects:
        out.append("CONVEYOR STOPPED WITH PENDING REJECTS: " + ", ".join(missed_rejects[-6:]) +
                   "\nThese crossed the line but were not ejected reliably. Remove the cashews between the line and the valves.")
    if running and 0 < ai_fps < min_fps:
        out.append(f"AI FPS {ai_fps:.1f} IS TOO LOW FOR SAFE TRACKING (need {min_fps:.0f})")
    if free_gb is not None and free_gb < MIN_FREE_GB:
        out.append(f"DISK SPACE LOW ({free_gb:.0f} GB free)")
    return out
