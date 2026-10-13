"""Time spent per stage of the grading loop (YOLO, masks, CNN, tracker, drawing ...), averaged per frame.

    stages = Stages()
    t = stages.start()
    ...                         # the work
    t = stages.lap("yolo", t)   # adds the time since t to "yolo", returns now
    stages.frame()              # once per frame
    stages.report()             # {"frames": n, "fps": .., "ms": {stage: ms per frame}, "total_ms": ..}

Costs well under 0.01 ms per call; always on.
"""
import threading
import time


class Stages:
    def __init__(self, clock=time.perf_counter):
        self._clock = clock
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with getattr(self, "_lock", threading.Lock()):
            self._sums, self._frames, self._since = {}, 0, self._clock()

    def start(self):
        return self._clock()

    def lap(self, name, since):
        now = self._clock()
        with self._lock:
            self._sums[name] = self._sums.get(name, 0.0) + (now - since)
        return now

    def add(self, name, seconds):
        with self._lock:
            self._sums[name] = self._sums.get(name, 0.0) + seconds

    def frame(self):
        with self._lock:
            self._frames += 1

    def report(self, reset=False):
        with self._lock:
            frames, elapsed = self._frames, max(1e-9, self._clock() - self._since)
            ms = {name: 1000.0 * total / max(1, frames) for name, total in self._sums.items()}
            if reset:
                self._sums, self._frames, self._since = {}, 0, self._clock()
        return {"frames": frames, "fps": frames / elapsed, "ms": ms, "total_ms": sum(ms.values())}

    @staticmethod
    def text(report):
        parts = "  ".join(f"{name} {value:.1f}" for name, value in sorted(report["ms"].items(), key=lambda item: -item[1]))
        return f"{report['fps']:.1f} fps, {report['total_ms']:.1f} ms per frame:  {parts}"
