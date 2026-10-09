"""One tracker per ROI (lane): one cashew = one ID, although it rolls and its box changes every frame.

What it relies on: the belt moves one way at a nearly constant speed and the camera is fixed. So
  * the position is the mask centroid (the caller's choice), not the box,
  * the speed is kept in px/ms along the flow and predicts where a cashew will be,
  * all tracks and detections are paired at once (Hungarian), not greedily,
  * a pairing that moves backwards, jumps too far forward or changes roller channel is not allowed,
  * a cashew that is missed for a short time keeps its ID when it reappears where it was predicted.
"""
from collections import deque
from dataclasses import dataclass, field
from statistics import median
from typing import Any, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

import flow_geometry as fg

# max_step_px comes from the old tracker (160 px). The others are first values of mine and must be
# tuned on a recording; all are in the config under grading -> tracker.
DEFAULTS = {
    "max_step_px": 160.0,       # forward limit per frame while the lane speed is not known yet
    "gate_extra_px": 80.0,      # allowed error around the predicted position once the speed is known
    "speed_factor": 1.5,        # ... plus this much of the predicted step (speed changes, rolling)
    "back_tol_px": 25.0,        # centroid jitter of a rolling cashew; more than this backwards = another cashew
    "max_lateral_px": 100.0,    # sideways limit (same roller channel)
    "hold_ms": 250.0,           # how long a missed cashew keeps its ID
}
_FAR = 1e6


@dataclass
class Detection:
    cx: float
    cy: float
    box: list
    data: Any = None            # whatever the caller wants back (class, view, features, ...)
    track: Any = None           # set by LaneTracker.update(): the track this detection belongs to


@dataclass
class Track:
    id: str
    roi: int
    cx: float
    cy: float
    box: list
    born_ms: float
    seen_ms: float              # last time a detection was matched
    now_ms: float               # time of the (possibly predicted) position
    v: float = 0.0              # px/ms along the flow
    hits: int = 1
    missed: bool = False        # True while the position is only predicted
    decided: bool = False
    counted: bool = False
    frames: list = field(default_factory=list)      # per-frame records, filled by the caller
    path: deque = field(default_factory=lambda: deque(maxlen=400))      # (t_ms, cx, cy) of real detections
    info: dict = field(default_factory=dict)        # decision result, label, relay ... (caller's)


def _axes(flow):
    dx, dy = fg._unit(flow)
    return (dx, dy), (abs(dy), abs(dx))     # along the flow (signed), across it


class LaneTracker:
    def __init__(self, roi_id: int, params: Optional[dict] = None):
        self.roi_id = roi_id
        self.params = {**DEFAULTS, **(params or {})}
        self.tracks: List[Track] = []
        self._seq = 0
        self._speeds = deque(maxlen=30)     # recent speeds of well-tracked cashews

    def clear(self):
        """Forget the tracks (stop, new model). The ID counter goes on, so IDs stay unique in the logs."""
        self.tracks = []
        self._speeds.clear()

    @property
    def lane_speed(self) -> float:
        """Typical speed of this lane in px/ms (0 until some cashews have been tracked)."""
        return median(self._speeds) if self._speeds else 0.0

    def update(self, detections: List[Detection], t_ms: float, flow, roi=None):
        """Pair detections with tracks. Returns the tracks alive after this frame; `track.missed`
        tells which ones were not seen now. `roi` (x1, y1, x2, y2) lets predicted tracks end when they leave."""
        p = self.params
        (ax, ay), (lx, ly) = _axes(flow)
        along = lambda x, y: x * ax + y * ay
        lateral = lambda x, y: x * lx + y * ly

        cost = np.full((len(self.tracks), len(detections)), _FAR)
        for i, trk in enumerate(self.tracks):
            dt = max(0.0, t_ms - trk.now_ms)
            v = trk.v if trk.hits > 1 else self.lane_speed
            pred = along(trk.cx, trk.cy) + v * dt
            forward_limit = (v * dt * p["speed_factor"] + p["gate_extra_px"]) if v > 0 else p["max_step_px"]
            for j, det in enumerate(detections):
                step = along(det.cx, det.cy) - along(trk.cx, trk.cy)
                side = abs(lateral(det.cx, det.cy) - lateral(trk.cx, trk.cy))
                if step < -p["back_tol_px"] or step > forward_limit or side > p["max_lateral_px"]:
                    continue
                cost[i, j] = float(np.hypot(along(det.cx, det.cy) - pred, side))

        matched_tracks, matched_dets = set(), set()
        if cost.size:
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] >= _FAR:
                    continue
                trk, det = self.tracks[i], detections[j]
                dt = t_ms - trk.seen_ms
                if dt > 0:
                    v_now = max(0.0, (along(det.cx, det.cy) - along(*trk.path[-1][1:])) / dt)
                    trk.v = v_now if trk.hits == 1 else 0.7 * trk.v + 0.3 * v_now
                trk.cx, trk.cy, trk.box = det.cx, det.cy, list(det.box)
                trk.seen_ms = trk.now_ms = t_ms
                trk.hits += 1
                trk.missed = False
                trk.path.append((t_ms, det.cx, det.cy))
                if trk.hits >= 3 and trk.v > 0:
                    self._speeds.append(trk.v)
                matched_tracks.add(i)
                matched_dets.add(j)
                det.track = trk

        alive = []
        for i, trk in enumerate(self.tracks):
            if i in matched_tracks:
                alive.append(trk)
                continue
            if t_ms - trk.seen_ms > p["hold_ms"]:
                continue
            # Not seen now: move it along at its speed so it can be found again (and still be decided).
            dt = max(0.0, t_ms - trk.now_ms)
            step = (trk.v if trk.hits > 1 else self.lane_speed) * dt
            trk.cx += ax * step
            trk.cy += ay * step
            trk.box = [trk.box[0] + ax * step, trk.box[1] + ay * step, trk.box[2] + ax * step, trk.box[3] + ay * step]
            trk.now_ms = t_ms
            trk.missed = True
            if roi is not None and not (roi[0] <= trk.cx <= roi[2] and roi[1] <= trk.cy <= roi[3]):
                continue
            alive.append(trk)

        for j, det in enumerate(detections):
            if j in matched_dets:
                continue
            self._seq += 1
            trk = Track(id=f"R{self.roi_id}-{self._seq:06d}", roi=self.roi_id, cx=det.cx, cy=det.cy, box=list(det.box),
                        born_ms=t_ms, seen_ms=t_ms, now_ms=t_ms)
            trk.path.append((t_ms, det.cx, det.cy))
            det.track = trk
            alive.append(trk)

        self.tracks = alive
        return alive


def crossing_time(path, flow, line_pos):
    """Time (ms) at which the centroid crossed the line, interpolated between the last detection before
    it and the first one after it. None when the path does not span the line."""
    before = None
    for t, cx, cy in path:
        d = fg.progress(flow, line_pos, cx, cy)
        if d < 0:
            before = (t, d)
        elif before is not None:
            t1, d1 = before
            return t1 + (t - t1) * (-d1) / (d - d1) if d != d1 else t
        else:
            return None
    return None


def track_speed(path, flow):
    """Speed over the whole track in px/ms (first to last real detection), or None."""
    if len(path) < 2:
        return None
    (t1, x1, y1), (t2, x2, y2) = path[0], path[-1]
    if t2 <= t1:
        return None
    (ax, ay), _ = _axes(flow)
    return ((x2 - x1) * ax + (y2 - y1) * ay) / (t2 - t1)
