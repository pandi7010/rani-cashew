"""USB camera capture (cv2.VideoCapture) with automatic reconnect, and a test source.

The thread puts `(frame_bgr, width, height)` on the output queue, or `("ERROR", message)`
while no frame is available. A test source (a video file, an image, or a folder of images)
can stand in for the camera so the grader can be tried without hardware.
"""
import glob
import logging
import os
import platform
import queue
import threading
import time
from dataclasses import dataclass, asdict

import cv2

log = logging.getLogger("camera")

try:    # OpenCV prints two warnings per failed open; this thread reports a missing camera itself, once.
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
# Tried from the largest down when looking for what a camera supports.
PROBE_RESOLUTIONS = [(3840, 2160), (2592, 1944), (2560, 1440), (1920, 1080), (1600, 1200),
                     (1280, 960), (1280, 720), (1024, 768), (800, 600), (640, 480)]
PROBE_FPS = [120, 90, 60, 30, 25, 15]
FRAME_TIMEOUT_S = 2.0       # no frame for this long -> reconnect
RETRY_INTERVAL_S = 1.0


@dataclass
class CameraSettings:
    source: str = "usb"             # "usb" or "test"
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    mjpg: bool = True               # MJPG gives the high resolutions / frame rates on most USB cameras
    auto_exposure: bool = True
    exposure: float = 100.0         # driver units (V4L2: usually 100 µs steps); used when auto_exposure is off
    test_path: str = ""             # video file, image file or folder of images
    test_loop: bool = True

    @classmethod
    def from_dict(cls, data):
        settings = cls()
        for key, value in (data or {}).items():
            if hasattr(settings, key):
                try:
                    setattr(settings, key, type(getattr(settings, key))(value))
                except (TypeError, ValueError):
                    log.warning("Camera setting %s=%r ignored (wrong type)", key, value)
        if settings.source not in ("usb", "test"):
            settings.source = "usb"
        return settings

    def to_dict(self):
        return asdict(self)


def _backend():
    system = platform.system()
    if system == "Linux":
        return cv2.CAP_V4L2
    if system == "Windows":
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def list_test_images(path):
    if os.path.isdir(path):
        files = []
        for ext in IMAGE_EXTENSIONS:
            files += glob.glob(os.path.join(path, f"*{ext}")) + glob.glob(os.path.join(path, f"*{ext.upper()}"))
        return sorted(set(files))
    if os.path.isfile(path) and path.lower().endswith(IMAGE_EXTENSIONS):
        return [path]
    return []


def probe_camera(index, mjpg=True):
    """Ask the camera which of the common resolutions / frame rates it accepts.
    The camera must not be open elsewhere. Returns a list of (width, height, fps), largest first."""
    cap = cv2.VideoCapture(int(index), _backend())
    modes = []
    try:
        if not cap.isOpened():
            return modes
        if mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        for width, height in PROBE_RESOLUTIONS:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            if got != (width, height):
                continue
            best_fps = 0
            for fps in PROBE_FPS:
                cap.set(cv2.CAP_PROP_FPS, fps)
                if int(round(cap.get(cv2.CAP_PROP_FPS))) == fps:
                    best_fps = fps
                    break
            modes.append((width, height, best_fps or int(round(cap.get(cv2.CAP_PROP_FPS)))))
    finally:
        cap.release()
    return modes


class CameraThread(threading.Thread):
    """States: CONNECTING, ONLINE, RECONNECTING, OFFLINE (USB) and TEST (test source running)."""

    def __init__(self, output_queue: queue.Queue, stop_event: threading.Event, settings: CameraSettings):
        super().__init__(daemon=True, name="CameraThread")
        self.output_queue = output_queue
        self.stop_event = stop_event
        self._settings = settings
        self._lock = threading.Lock()
        self._restart = threading.Event()
        self._probe_request = threading.Event()
        self.is_connected = False
        self.state = "CONNECTING"
        self.fps = 0.0
        self.actual = {}            # what the camera really delivers: width, height, fps, exposure, auto_exposure
        self.last_error = ""
        self.probe_result = None    # list of modes after request_probe(), or an error text
        self.reconnect_count = 0
        self._was_online = False

    # ------------------------------------------------------------------ control (any thread)
    @property
    def settings(self) -> CameraSettings:
        with self._lock:
            return CameraSettings(**asdict(self._settings))

    def apply_settings(self, settings: CameraSettings):
        """New settings: the source is closed and opened again with them."""
        with self._lock:
            self._settings = CameraSettings(**asdict(settings))
        self._was_online = False
        self._restart.set()

    def request_reconnect(self):
        self._restart.set()

    def request_probe(self):
        """Close the camera, find the modes it supports (probe_result), then open it again."""
        self.probe_result = None
        self._probe_request.set()
        self._restart.set()

    # ------------------------------------------------------------------ helpers
    def _publish(self, item):
        try:
            self.output_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.output_queue.put_nowait(item)
        except queue.Full:
            pass

    def _fail(self, state, message):
        self.is_connected = False
        self.state = state
        self.fps = 0.0
        if message != self.last_error:
            log.warning("[CAM] %s", message)
            self.last_error = message
        self._publish(("ERROR", message))

    def _wait(self, seconds):
        """Sleep, but wake up at once for stop / new settings."""
        end = time.monotonic() + seconds
        while time.monotonic() < end and not self.stop_event.is_set() and not self._restart.is_set():
            time.sleep(0.05)

    def _should_leave(self):
        return self.stop_event.is_set() or self._restart.is_set()

    # ------------------------------------------------------------------ main loop
    def run(self):
        while not self.stop_event.is_set():
            self._restart.clear()
            if self._probe_request.is_set():
                self._probe_request.clear()
                self._run_probe()
            settings = self.settings
            try:
                if settings.source == "test":
                    self._run_test_source(settings)
                else:
                    self._run_usb(settings)
            except Exception as exc:
                log.exception("[CAM] Capture loop failed")
                self._fail("RECONNECTING", f"Camera error: {exc}")
            if not self.stop_event.is_set() and not self._restart.is_set():
                self._wait(RETRY_INTERVAL_S)
        self.is_connected = False
        self.state = "OFFLINE"

    def _run_probe(self):
        settings = self.settings
        try:
            modes = probe_camera(settings.index, settings.mjpg)
            self.probe_result = modes if modes else f"Camera {settings.index} could not be opened."
            log.info("[CAM] Supported modes of camera %s: %s", settings.index, self.probe_result)
        except Exception as exc:
            log.exception("[CAM] Probe failed")
            self.probe_result = f"Probe failed: {exc}"

    def _run_usb(self, s: CameraSettings):
        cap = cv2.VideoCapture(int(s.index), _backend())
        try:
            if not cap.isOpened():
                self._fail("RECONNECTING" if self._was_online else "OFFLINE", f"USB camera {s.index} not found. Retrying...")
                return
            if s.mjpg:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, s.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, s.height)
            cap.set(cv2.CAP_PROP_FPS, s.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)         # always the newest frame
            # V4L2: 3 = automatic, 1 = manual exposure. Other drivers may ignore this; `actual` shows the result.
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3 if s.auto_exposure else 1)
            if not s.auto_exposure:
                cap.set(cv2.CAP_PROP_EXPOSURE, float(s.exposure))
            self.actual = {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": float(cap.get(cv2.CAP_PROP_FPS)), "exposure": float(cap.get(cv2.CAP_PROP_EXPOSURE)),
                "auto_exposure": float(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)),
            }
            log.info("[CAM] USB camera %s opened. Asked %sx%s @ %s fps, got %s", s.index, s.width, s.height, s.fps, self.actual)

            last_frame_time = time.monotonic()
            frames, window_start = 0, time.monotonic()
            while not self._should_leave():
                ok, frame = cap.read()
                now = time.monotonic()
                if not ok or frame is None or frame.size == 0:
                    if now - last_frame_time > FRAME_TIMEOUT_S:
                        self.reconnect_count += 1
                        self._fail("RECONNECTING", f"USB camera {s.index} stopped sending frames. Reconnecting...")
                        return
                    time.sleep(0.005)
                    continue
                last_frame_time = now
                if not self.is_connected:
                    self.is_connected, self.state, self.last_error, self._was_online = True, "ONLINE", "", True
                    log.info("[CAM] USB camera %s is online.", s.index)
                frames += 1
                if now - window_start >= 1.0:
                    self.fps = frames / (now - window_start)
                    frames, window_start = 0, now
                h, w = frame.shape[:2]
                self._publish((frame, w, h))
        finally:
            cap.release()
            if self.state == "ONLINE":
                self.is_connected = False

    def _run_test_source(self, s: CameraSettings):
        path = s.test_path
        images = list_test_images(path)
        is_video = os.path.isfile(path) and path.lower().endswith(VIDEO_EXTENSIONS)
        if not images and not is_video:
            self._fail("OFFLINE", f"Test source not found (video, image or folder of images): {path or '(empty)'}")
            return
        interval = 1.0 / max(1, s.fps)
        cap = cv2.VideoCapture(path) if is_video else None
        try:
            if cap is not None and not cap.isOpened():
                self._fail("OFFLINE", f"Test video could not be opened: {path}")
                return
            log.info("[CAM] Test source: %s (%s)", path, "video" if is_video else f"{len(images)} image(s)")
            index, frames, window_start = 0, 0, time.monotonic()
            cache = {}
            while not self._should_leave():
                tick = time.monotonic()
                if cap is not None:
                    ok, frame = cap.read()
                    if not ok:
                        if not s.test_loop:
                            self._fail("OFFLINE", "Test video finished.")
                            self._wait(3600)
                            return
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                else:
                    if index >= len(images):
                        if not s.test_loop:
                            self._fail("OFFLINE", "Test images finished.")
                            self._wait(3600)
                            return
                        index = 0
                    file = images[index]
                    if file not in cache:
                        if len(cache) > 32:
                            cache.clear()
                        cache[file] = cv2.imread(file)
                    frame = cache[file]
                    index += 1
                    if frame is None:
                        log.warning("[CAM] Test image could not be read: %s", file)
                        continue
                    frame = frame.copy()
                self.is_connected, self.state, self.last_error = True, "TEST", ""
                h, w = frame.shape[:2]
                self.actual = {"width": w, "height": h, "fps": float(s.fps)}
                frames += 1
                if tick - window_start >= 1.0:
                    self.fps = frames / (tick - window_start)
                    frames, window_start = 0, tick
                self._publish((frame, w, h))
                remaining = interval - (time.monotonic() - tick)
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            if cap is not None:
                cap.release()
            self.is_connected = False
