"""A run of N cashews through one ROI, built from what the real run of 2026-10-10 03:15 showed in the log
(no camera recording of that run exists): belt 2.46 px/ms, frames 26-53 ms apart, boxes about 160 x 250 px,
cashews entering at the ROI edge partly visible (box centre at half speed), detections missed for 1-3 frames right
after the entry and again while rolling, sideways drift, faint detections, and tiny slow boxes that are not cashews.

detections(seed) -> [(t_ms, [(owner, cx, cy, box, weak)])]; owner is the cashew number or -1 for a ghost box.
"""
import random

EDGE, LENGTH, HEIGHT, SPEED = 2447.0, 165.0, 250.0, 2.46
ROI = (25, 1454, 2447, 2047)


def detections(n=20, seed=1, use_weak=True):
    rng = random.Random(seed)
    starts, t = [], 300.0
    for _ in range(n):
        starts.append(t)
        t += rng.uniform(330, 620)                      # 20 cashews in about 9 s, as in the run
    lanes = [rng.choice((1700.0, 1760.0, 1880.0, 1930.0)) for _ in range(n)]
    drift = [rng.uniform(-0.09, 0.09) for _ in range(n)]            # px/ms sideways while rolling
    # frames (counted per cashew from its first detection) in which YOLO does not find it
    missed = []
    for k in range(n):
        gone = set()
        if rng.random() < 0.55:                         # right after the entry: seen twice, then missed
            gone |= set(range(2, 2 + rng.randint(1, 3)))
        for _ in range(rng.randint(0, 2)):              # while rolling
            start = rng.randint(5, 20)
            gone |= set(range(start, start + rng.randint(1, 3)))
        missed.append(gone)
    ghosts = [(rng.uniform(0, t), rng.uniform(600, 2300), rng.uniform(1700, 1790)) for _ in range(6)]
    frames, now, seen = [], 0.0, [0] * n
    while now < t + 1300:
        now += rng.uniform(26, 53)
        dets = []
        for k in range(n):
            lead = EDGE - SPEED * (now - starts[k])
            if now < starts[k] or lead < ROI[0] - LENGTH:
                continue
            trail = min(EDGE, lead + LENGTH)
            lead = max(float(ROI[0]), lead)
            if trail - lead < 25:
                continue
            index, seen[k] = seen[k], seen[k] + 1
            weak = False
            if index in missed[k]:
                if not use_weak or rng.random() < 0.5:
                    continue
                weak = True                             # found, but below the normal confidence
            cy = lanes[k] + drift[k] * (now - starts[k]) + rng.uniform(-9, 9)
            cx = (lead + trail) / 2 + rng.uniform(-7, 7)
            dets.append((k, cx, cy, [lead, cy - HEIGHT / 2, trail, cy + HEIGHT / 2], weak))
        for g0, gx, gy in ghosts:                       # 22 x 55 px, hardly moving, 3-4 frames
            if g0 <= now < g0 + 115:
                x = gx - 0.2 * (now - g0)
                dets.append((-1, x, gy, [x - 11, gy - 27, x + 11, gy + 27], False))
        frames.append((now, dets))
    return frames


def run(tracker_module, frames, flow, weak_supported=True):
    """Feed the frames to a LaneTracker of the given module. Returns ({cashew: set of IDs}, ghost IDs, all IDs)."""
    tracker = tracker_module.LaneTracker(2)
    ids, ghost_ids = {}, set()
    for t, dets in frames:
        objects = []
        for owner, cx, cy, box, weak in dets:
            if weak and not weak_supported:
                continue                                # the old program never saw boxes below the confidence limit
            d = tracker_module.Detection(cx, cy, list(box))
            if weak_supported:
                d.weak = weak
            objects.append((owner, d))
        tracker.update([d for _, d in objects], t, flow, ROI)
        for owner, d in objects:
            if d.track is None:
                continue
            (ghost_ids if owner < 0 else ids.setdefault(owner, set())).add(d.track.id)
    return ids, ghost_ids, {i for v in ids.values() for i in v} | ghost_ids
