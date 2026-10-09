"""Replay a real run through the tracker, from the frames log (no camera, no PLC, nothing is fired).

    python3 tools/replay_tracks.py logs/frames_YYYYMMDD.jsonl [HH:MM:SS HH:MM:SS]

The frames log holds every detection of every track with its time and box (from 2026-10-10 on; older logs have no
positions and cannot be replayed). All boxes of the chosen time are fed again, frame by frame, to the tracker as it
is now. Output: how many IDs the program gave in that run, and how many the tracker gives now.
"""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import flow_geometry as fg
import lane_tracker


def load(path, start="", end=""):
    """{roi: {time_ms: [(logged id, box, weak)]}}, logged results per id."""
    frames, results = collections.defaultdict(lambda: collections.defaultdict(list)), {}
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        clock = row.get("time", "")[11:19]
        if (start and clock < start) or (end and clock > end):
            continue
        roi = int(str(row["id"])[1]) if str(row["id"])[:1] == "R" else 1
        results[row["id"]] = row.get("result", "")
        for f in row.get("frames", []):
            if "at" in f and f.get("box"):
                frames[roi][f["at"]].append((row["id"], f["box"], False))
    return frames, results


def replay(frames, flow=fg.RIGHT_TO_LEFT, rois=None):
    """{roi: {logged id: set of new ids}}."""
    out = {}
    for roi, by_time in frames.items():
        tracker, ids = lane_tracker.LaneTracker(roi), collections.defaultdict(set)
        for t in sorted(by_time):
            dets = [(owner, lane_tracker.Detection((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0, list(b))) for owner, b, _ in by_time[t]]
            tracker.update([d for _, d in dets], t, flow, (rois or {}).get(roi))
            for owner, d in dets:
                if d.track is not None:
                    ids[owner].add(d.track.id)
        out[roi] = ids
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    frames, results = load(sys.argv[1], *(sys.argv[2:4] + ["", ""])[:2])
    if not frames:
        sys.exit("No positions in this log for that time (logs written before 2026-10-10 03:40 have none).")
    for roi, ids in sorted(replay(frames).items()):
        logged = len(ids)
        now = len({i for v in ids.values() for i in v})
        kinds = collections.Counter(results.get(i, "?") for i in ids)
        print(f"ROI {roi}: the program logged {logged} IDs ({dict(kinds)}); the tracker now gives {now} IDs for the same boxes")
