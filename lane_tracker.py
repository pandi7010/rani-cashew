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
    "hold_ms": 600.0,           # how long a missed cashew keeps its ID (longer than max_lost_ms, so a gap can be judged)
    "stuck_speed_fraction": 0.30,   # slower than this share of the lane speed ...
    "stuck_time_ms": 700.0,         # ... for this long = STUCK (values of the 5-track machine)
    "stopped_speed_fraction": 0.10, # all cashews slower than this share of the lane speed = conveyor stopped
    "young_hits": 4,            # a track seen fewer times than this is predicted with the lane speed: while a cashew
                                # is only partly inside the ROI its box centre moves at about half the real speed
    "speed_tolerance": 0.5,     # allowed error of the predicted position, as a share of the distance travelled
    "min_travel_px": 150.0,     # only a track that travelled this far tells the lane speed (a mark that hardly moves does not)
    "size_ratio_min": 0.4,      # a box much narrower / wider ACROSS the lane than the cashew's last box is something else
                                # (along the lane the box grows while the cashew enters the ROI, so that side is not compared)
    "duplicate_overlap": 0.60,  # a new box lying this much inside a tracked cashew's box (or the other way) is the same cashew
}
_FAR = 1e6


@dataclass
class Detection:
    cx: float
    cy: float
    box: list
    data: Any = None            # whatever the caller wants back (class, view, features, ...)
    track: Any = None           # set by LaneTracker.update(): the track this detection belongs to (None: ignored)
    weak: bool = False          # low YOLO confidence: may continue a known cashew, never starts a new ID
    ignored: str = ""           # why it got no track: "weak" or "duplicate of <id>"


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
    misses: int = 0             # frames in a row without a detection
    max_gap_ms: float = 0.0     # longest time without a detection (LOST rule)
    stuck: bool = False         # much slower than the lane for stuck_time_ms
    slow_since: Optional[float] = None
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
        self.dropped: List[Track] = []      # tracks that ended in the last update()
        self.stopped = False                # True while every tracked cashew stands (almost) still

    def clear(self):
        """Forget the tracks (stop, new model). The ID counter goes on, so IDs stay unique in the logs."""
        self.tracks = []
        self.dropped = []
        self.stopped = False
        # The lane speed is kept: the belt runs as fast after the next START, and the first cashew needs it.

    @property
    def lane_speed(self) -> float:
        """Typical speed of this lane in px/ms (0 until some cashews have been tracked)."""
        return median(self._speeds) if self._speeds else 0.0

    def _speed_of(self, trk):
        """(speed used to predict this track, highest speed it may have) in px/ms."""
        lane = self.lane_speed
        own = trk.v if trk.hits > 1 else 0.0
        if trk.hits < self.params["young_hits"]:
            if lane > 0:
                return lane, max(own, lane)     # entering the ROI: its own speed is not reliable yet
            if own > 0:                         # no lane speed yet (first cashew): the real speed is between the
                return own * 1.5, own * 2.2     # measured one and twice that (box centre of a partly visible cashew)
        return (own if own > 0 else lane), max(own, lane)

    def _covering(self, det, tracks):
        """The track (seen in this frame) whose box the detection overlaps enough to be the same cashew, or None."""
        need = self.params["duplicate_overlap"]
        ax1, ay1, ax2, ay2 = det.box
        area = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        for trk in tracks:
            if trk.missed:
                continue
            bx1, by1, bx2, by2 = trk.box
            inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
            if inter / area >= need or inter / max(1.0, (bx2 - bx1) * (by2 - by1)) >= need:
                return trk
        return None

    def companion(self, trk, flow):
        """Another live cashew at (almost) the same place as this track: the two are one cashew. Used to merge a
        fragment (a short track) into the track that carries on, so that it never gets its own decision."""
        (ax, ay), (lx, ly) = _axes(flow)
        length = abs((trk.box[2] - trk.box[0]) * ax + (trk.box[3] - trk.box[1]) * ay) or 1.0
        best, best_gap = None, None
        for other in self.tracks:
            if other is trk:
                continue
            other_length = abs((other.box[2] - other.box[0]) * ax + (other.box[3] - other.box[1]) * ay) or 1.0
            gap = abs((other.cx - trk.cx) * ax + (other.cy - trk.cy) * ay)
            side = abs((other.cx - trk.cx) * lx + (other.cy - trk.cy) * ly)
            if gap <= 0.6 * max(length, other_length) and side <= self.params["max_lateral_px"] and (best is None or gap < best_gap):
                best, best_gap = other, gap
        return best

    def update(self, detections: List[Detection], t_ms: float, flow, roi=None):
        """Pair detections with tracks. Returns the tracks alive after this frame; `track.missed`
        tells which ones were not seen now. `roi` (x1, y1, x2, y2) lets predicted tracks end when they leave."""
        p = self.params
        (ax, ay), (lx, ly) = _axes(flow)
        along = lambda x, y: x * ax + y * ay
        lateral = lambda x, y: x * lx + y * ly

        # A detection belongs to a track when it lies near the place the cashew must have reached since it was LAST
        # SEEN (not since the last frame): a cashew that was missed for some frames is found again under the same ID.
        cost = np.full((len(self.tracks), len(detections)), _FAR)
        lane = self.lane_speed
        for i, trk in enumerate(self.tracks):
            _, sx, sy = trk.path[-1]
            dt = max(0.0, t_ms - trk.seen_ms)
            v, v_high = self._speed_of(trk)
            start = along(sx, sy)
            pred = start + v * dt
            if v_high > 0:
                # one frame at the usual step gives the old limit: step <= v * dt * speed_factor + gate_extra_px
                reach = p["gate_extra_px"] + max(p["speed_factor"] - 1.0, p["speed_tolerance"]) * v_high * dt
                low, high = pred - reach, pred + reach
            else:                           # no speed known yet: a fixed step per frame
                low, high = start, start + p["max_step_px"] * (trk.misses + 1)
            across = max(1.0, abs((trk.box[2] - trk.box[0]) * lx + (trk.box[3] - trk.box[1]) * ly))
            for j, det in enumerate(detections):
                pos = along(det.cx, det.cy)
                side = abs(lateral(det.cx, det.cy) - lateral(sx, sy))
                if pos - start < -p["back_tol_px"] or pos > high or pos < low - p["back_tol_px"] or side > p["max_lateral_px"]:
                    continue
                ratio = max(1.0, abs((det.box[2] - det.box[0]) * lx + (det.box[3] - det.box[1]) * ly)) / across
                if ratio < p["size_ratio_min"] or ratio > 1.0 / p["size_ratio_min"]:
                    continue
                cost[i, j] = float(np.hypot(pos - pred, side)) + 40.0 * abs(ratio - 1.0)

        matched_tracks, matched_dets = set(), set()
        if cost.size:
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] >= _FAR:
                    continue
                trk, det = self.tracks[i], detections[j]
                dt = t_ms - trk.seen_ms
                trk.max_gap_ms = max(trk.max_gap_ms, dt)
                if dt > 0:
                    v_now = max(0.0, (along(det.cx, det.cy) - along(*trk.path[-1][1:])) / dt)
                    trk.v = v_now if trk.hits == 1 else 0.7 * trk.v + 0.3 * v_now
                trk.cx, trk.cy, trk.box = det.cx, det.cy, list(det.box)
                trk.seen_ms = trk.now_ms = t_ms
                trk.hits += 1
                trk.missed = False
                trk.misses = 0
                trk.path.append((t_ms, det.cx, det.cy))
                lane = self.lane_speed
                # A stuck or standing cashew must not pull the lane speed down.
                travelled = abs(along(det.cx, det.cy) - along(*trk.path[0][1:]))
                if trk.hits >= 3 and trk.v > 0 and travelled >= p["min_travel_px"] \
                        and (len(self._speeds) < 5 or trk.v >= p["stuck_speed_fraction"] * lane):
                    self._speeds.append(trk.v)
                if lane > 0 and trk.hits >= 3 and trk.v < p["stuck_speed_fraction"] * lane:
                    if trk.slow_since is None:
                        trk.slow_since = t_ms
                    elif t_ms - trk.slow_since > p["stuck_time_ms"]:
                        trk.stuck = True
                else:
                    trk.slow_since = None
                matched_tracks.add(i)
                matched_dets.add(j)
                det.track = trk

        alive, self.dropped = [], []
        for i, trk in enumerate(self.tracks):
            if i in matched_tracks:
                alive.append(trk)
                continue
            if t_ms - trk.seen_ms > p["hold_ms"]:
                self.dropped.append(trk)
                continue
            # Not seen now: move it along at its speed so it can be found again (and still be decided).
            dt = max(0.0, t_ms - trk.now_ms)
            step = self._speed_of(trk)[0] * dt
            trk.misses += 1
            trk.cx += ax * step
            trk.cy += ay * step
            trk.box = [trk.box[0] + ax * step, trk.box[1] + ay * step, trk.box[2] + ax * step, trk.box[3] + ay * step]
            trk.now_ms = t_ms
            trk.missed = True
            if roi is not None and not (roi[0] <= trk.cx <= roi[2] and roi[1] <= trk.cy <= roi[3]):
                trk.info["left_roi"] = True
                self.dropped.append(trk)
                continue
            alive.append(trk)

        for j, det in enumerate(detections):
            if j in matched_dets:
                continue
            if det.weak:                    # a faint detection alone is not a cashew
                det.ignored = "weak"
                continue
            twin = self._covering(det, alive)
            if twin is not None:            # a second box on a cashew that already has its ID: no second ID
                det.ignored = f"duplicate of {twin.id}"
                continue
            self._seq += 1
            trk = Track(id=f"R{self.roi_id}-{self._seq:06d}", roi=self.roi_id, cx=det.cx, cy=det.cy, box=list(det.box),
                        born_ms=t_ms, seen_ms=t_ms, now_ms=t_ms)
            trk.path.append((t_ms, det.cx, det.cy))
            det.track = trk
            alive.append(trk)

        self.tracks = alive
        seen = [t for t in alive if not t.missed and t.hits >= 5]
        lane = self.lane_speed
        self.stopped = bool(seen) and lane > 0 and all(t.v < p["stopped_speed_fraction"] * lane for t in seen)
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
