"""Camera capture behind one interface, with automatic reconnect.

Backends:
  AravisBackend  the industrial camera (USB3 Vision / GigE) through Aravis
  FileBackend    a video file, an image, or a folder of images (testing without the camera)

`CameraThread` owns one backend, reopens it when frames stop, and puts
`(frame_bgr, width, height, capture_time)` on the output queue, or `("ERROR", message)`
while there is no picture. `capture_time` is in the `time.time()` clock.
"""
import glob
import logging
import os
import queue
import re
import shutil
import threading
import time
from dataclasses import dataclass, asdict
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("camera")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
FRAME_TIMEOUT_S = 2.0       # no good frame for this long -> reconnect
RETRY_INTERVAL_S = 1.0
SOURCES = ("aravis", "file")

# GenICam pixel format -> how to turn the buffer into BGR. OpenCV names its Bayer codes by the
# pixels at row 1, columns 1-2, GenICam by row 0, columns 0-1, so the names are shifted by one
# position (checked with synthetic mosaics in tests/test_camera_backends.py).
PIXEL_FORMATS = {
    "BayerGB8": ("bayer", cv2.COLOR_BayerGR2BGR),
    "BayerGR8": ("bayer", cv2.COLOR_BayerGB2BGR),
    "BayerRG8": ("bayer", cv2.COLOR_BayerBG2BGR),
    "BayerBG8": ("bayer", cv2.COLOR_BayerRG2BGR),
    "Mono8": ("mono", cv2.COLOR_GRAY2BGR),
    "RGB8": ("rgb", cv2.COLOR_RGB2BGR),
    "BGR8": ("bgr", None),
}


class CameraError(Exception):
    """The camera cannot be used as it is (not found, unsupported pixel format, ...)."""


@dataclass
class CameraSettings:
    source: str = "aravis"
    device_id: str = ""                     # empty = the first camera Aravis finds
    exposure_us: Optional[float] = None     # None = leave the camera's own value
    gain: Optional[float] = None            # camera (hardware) gain; None = leave the camera's own value
    fps_limit_enabled: bool = False         # False = free running (as fast as the camera goes)
    fps_limit: float = 20.0
    test_path: str = ""                     # video file, image file or folder of images
    test_loop: bool = True
    test_fps: float = 25.0                  # used when the test files carry no timestamps

    @classmethod
    def from_dict(cls, data):
        settings = cls()
        for key, value in (data or {}).items():
            if not hasattr(settings, key):
                continue
            try:
                if key in ("exposure_us", "gain"):
                    setattr(settings, key, None if value is None else float(value))
                else:
                    setattr(settings, key, type(getattr(settings, key))(value))
            except (TypeError, ValueError):
                log.warning("Camera setting %s=%r ignored (wrong type)", key, value)
        if settings.source not in SOURCES:
            settings.source = "aravis"
        return settings

    def to_dict(self):
        return asdict(self)

    def needs_reopen(self, other: "CameraSettings") -> bool:
        """True when changing from `self` to `other` needs the source to be closed and opened again."""
        keys = ("source", "device_id", "test_path", "test_loop", "test_fps")
        return any(getattr(self, k) != getattr(other, k) for k in keys)


@dataclass
class Frame:
    bgr: np.ndarray
    timestamp: float            # time.time() clock, when the frame reached the computer
    camera_timestamp_ns: int = 0


def convert_to_bgr(data, width, height, pixel_format, flat_field=None):
    """Raw camera buffer -> BGR image. Raises CameraError for a format or size this code does not handle."""
    if pixel_format not in PIXEL_FORMATS:
        raise CameraError(f"Pixel format {pixel_format} is not supported (supported: {', '.join(PIXEL_FORMATS)}).")
    kind, code = PIXEL_FORMATS[pixel_format]
    channels = 1 if kind in ("bayer", "mono") else 3
    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.size != width * height * channels:
        raise CameraError(f"Buffer has {arr.size} bytes, but {pixel_format} {width} x {height} needs {width * height * channels}.")
    if channels == 1:
        raw = arr.reshape((height, width))
        if flat_field is not None:
            raw = flat_field.apply(raw)
        return cv2.cvtColor(raw, code)
    image = arr.reshape((height, width, 3))
    return image.copy() if code is None else cv2.cvtColor(image, code)


class FlatField:
    """White calibration (flat-field correction): a smooth gain map that evens out the lighting.
    Taught from one frame of a plain white surface and applied to every raw frame before colour conversion."""

    def __init__(self, path):
        self.path = str(path)
        self.enabled = False
        self.gain_map = None            # uint8, 128 = gain 1.0
        self._request = threading.Event()
        if os.path.exists(self.path):
            self.gain_map = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
            self.enabled = self.gain_map is not None
            if self.enabled:
                log.info("[CAM] White calibration map loaded: %s", self.path)

    @property
    def available(self):
        return self.gain_map is not None

    def request_calibration(self):
        self._request.set()

    def calibrate_if_requested(self, raw):
        if not self._request.is_set():
            return False
        self._request.clear()
        h, w = raw.shape[:2]
        small = cv2.resize(raw, (max(1, w // 16), max(1, h // 16)))     # removes noise and texture
        smooth = cv2.resize(cv2.GaussianBlur(small, (15, 15), 0), (w, h)).astype(np.float32)
        gain = float(np.mean(smooth)) / (smooth + 1e-5)
        self.gain_map = np.clip(gain * 128, 0, 255).astype(np.uint8)
        self.enabled = True
        cv2.imwrite(self.path, self.gain_map)
        log.info("[CAM] White calibration taught and saved: %s", self.path)
        return True

    def apply(self, raw):
        self.calibrate_if_requested(raw)
        if self.enabled and self.gain_map is not None and self.gain_map.shape == raw.shape:
            return cv2.multiply(raw, self.gain_map, scale=1 / 128.0)
        return raw


# ---------------------------------------------------------------------------- backends
class AravisBackend:
    name = "aravis"

    def __init__(self, settings: CameraSettings, flat_field: Optional[FlatField] = None):
        self.settings = settings
        self.flat_field = flat_field
        self.camera = self.device = self.stream = None
        self.pixel_format = ""
        self.lost = False
        self.info = {}
        self.bad_buffers = 0

    @staticmethod
    def _aravis():
        import gi
        gi.require_version('Aravis', '0.8')
        from gi.repository import Aravis
        return Aravis

    def open(self):
        Aravis = self._aravis()
        Aravis.update_device_list()
        count = Aravis.get_n_devices()
        if count == 0:
            raise CameraError("No camera found.")
        ids = [Aravis.get_device_id(i) for i in range(count)]
        wanted = self.settings.device_id
        if wanted and wanted not in ids:
            raise CameraError(f"Camera {wanted} not found (found: {', '.join(ids)}).")
        try:
            self.camera = Aravis.Camera.new(wanted or ids[0])
        except Exception as exc:
            raise CameraError(f"Camera could not be opened (in use by another program?): {exc}")
        self.device = self.camera.get_device()
        try:
            self.device.connect("control-lost", self._on_control_lost)
        except Exception:
            log.debug("control-lost signal not available", exc_info=True)

        self.pixel_format = self.camera.get_pixel_format_as_string()
        if self.pixel_format not in PIXEL_FORMATS:
            raise CameraError(f"Camera pixel format is {self.pixel_format}; supported: {', '.join(PIXEL_FORMATS)}. "
                              "Set the camera to an 8-bit format.")
        self.apply_live(self.settings, first=True)

        self.stream = self.camera.create_stream(None, None)
        payload = self.camera.get_payload()
        for _ in range(10):
            self.stream.push_buffer(Aravis.Buffer.new_allocate(payload))
        self.camera.start_acquisition()
        self.info = self.read_info()
        return self.info

    def _on_control_lost(self, *_args):
        self.lost = True

    def _try(self, label, action):
        try:
            action()
            return True
        except Exception as exc:
            log.warning("[CAM] %s failed: %s", label, exc)
            return False

    def apply_live(self, settings: CameraSettings, first=False):
        """Exposure, gain and frame-rate limit; can be changed while the camera runs."""
        self.settings = settings
        cam, dev = self.camera, self.device
        if settings.exposure_us is not None:
            self._try("ExposureAuto Off", lambda: dev.set_string_feature_value("ExposureAuto", "Off"))
            if first:
                try:
                    dev.set_string_feature_value("ExposureMode", "Timed")
                except Exception:
                    pass
            if not self._try("Set exposure", lambda: cam.set_exposure_time(float(settings.exposure_us))):
                self._try("Set ExposureTime", lambda: dev.set_float_feature_value("ExposureTime", float(settings.exposure_us)))
        if settings.gain is not None:
            self._try("GainAuto Off", lambda: dev.set_string_feature_value("GainAuto", "Off"))
            self._try("Set gain", lambda: cam.set_gain(float(settings.gain)))
        # Frame-rate limit. Off = free running, which is how the camera was used before.
        if self._try("Frame rate enable", lambda: dev.set_boolean_feature_value("AcquisitionFrameRateEnable", bool(settings.fps_limit_enabled))):
            if settings.fps_limit_enabled:
                self._try("Set frame rate", lambda: dev.set_float_feature_value("AcquisitionFrameRate", float(settings.fps_limit)))
        if not first:
            self.info = self.read_info()

    def read_info(self):
        cam, dev = self.camera, self.device
        info = {"pixel_format": self.pixel_format}

        def read(key, getter):
            try:
                info[key] = getter()
            except Exception:
                pass

        read("vendor", cam.get_vendor_name)
        read("model", cam.get_model_name)
        read("serial", lambda: dev.get_string_feature_value("DeviceSerialNumber"))
        read("width", lambda: cam.get_region()[2])
        read("height", lambda: cam.get_region()[3])
        read("exposure_us", cam.get_exposure_time)
        read("exposure_bounds", lambda: tuple(cam.get_exposure_time_bounds()))
        read("gain", cam.get_gain)
        read("gain_bounds", lambda: tuple(cam.get_gain_bounds()))
        read("fps_limit_enabled", lambda: dev.get_boolean_feature_value("AcquisitionFrameRateEnable"))
        read("fps_limit", lambda: dev.get_float_feature_value("AcquisitionFrameRate"))
        read("exposure_auto", lambda: dev.get_string_feature_value("ExposureAuto"))
        read("gain_auto", lambda: dev.get_string_feature_value("GainAuto"))
        read("white_balance_auto", lambda: dev.get_string_feature_value("BalanceWhiteAuto"))
        return info

    def read(self, timeout_s=0.2) -> Optional[Frame]:
        Aravis = self._aravis()
        buf = self.stream.timeout_pop_buffer(int(timeout_s * 1e6))
        if buf is None:
            return None
        try:
            if buf.get_status() != Aravis.BufferStatus.SUCCESS:
                self.bad_buffers += 1
                return None
            width, height = buf.get_image_width(), buf.get_image_height()
            bgr = convert_to_bgr(buf.get_data(), width, height, self.pixel_format, self.flat_field)
            system_ns = buf.get_system_timestamp()
            timestamp = system_ns / 1e9 if system_ns > 0 else time.time()
            return Frame(bgr, timestamp, buf.get_timestamp())
        finally:
            self.stream.push_buffer(buf)

    def close(self):
        try:
            if self.camera is not None:
                self.camera.stop_acquisition()
        except Exception:
            pass
        self.stream = self.device = self.camera = None


def list_test_images(path):
    if os.path.isdir(path):
        files = []
        for ext in IMAGE_EXTENSIONS:
            files += glob.glob(os.path.join(path, f"*{ext}")) + glob.glob(os.path.join(path, f"*{ext.upper()}"))
        return sorted(set(files))
    if os.path.isfile(path) and path.lower().endswith(IMAGE_EXTENSIONS):
        return [path]
    return []


_TIMESTAMP_IN_NAME = re.compile(r"_t(\d{13})\.[A-Za-z]+$")


def timestamp_from_name(path):
    """Capture time in seconds from a recorded file name (..._t<milliseconds>.jpg), or None."""
    match = _TIMESTAMP_IN_NAME.search(os.path.basename(path))
    return int(match.group(1)) / 1000.0 if match else None


class FileBackend:
    """Plays a video file, one image, or a folder of images. Recordings made by the grader carry their
    capture time in the file name and are played back with the same timing."""
    name = "file"
    lost = False

    def __init__(self, settings: CameraSettings, flat_field=None):
        self.settings = settings
        self.images, self.cap, self.index = [], None, 0
        self.finished = False
        self.info = {}
        self._next_due = None
        self._recorded_times = []

    def open(self):
        path = self.settings.test_path
        self.images = list_test_images(path)
        is_video = os.path.isfile(path) and path.lower().endswith(VIDEO_EXTENSIONS)
        if not self.images and not is_video:
            raise CameraError(f"Test source not found (video, image or folder of images): {path or '(empty)'}")
        if is_video:
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise CameraError(f"Test video could not be opened: {path}")
        self._recorded_times = [timestamp_from_name(f) for f in self.images]
        if any(t is None for t in self._recorded_times):
            self._recorded_times = []
        self.info = {"pixel_format": "file", "model": os.path.basename(path.rstrip("/")) or path,
                     "frames": len(self.images) if self.images else int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                     "recorded_timing": bool(self._recorded_times)}
        return self.info

    def apply_live(self, settings, first=False):
        self.settings = settings

    def _interval_after(self, index):
        """Seconds to wait after showing frame `index`."""
        if self._recorded_times and index + 1 < len(self._recorded_times):
            gap = self._recorded_times[index + 1] - self._recorded_times[index]
            if 0 < gap < 1.0:
                return gap
        return 1.0 / max(1.0, float(self.settings.test_fps))

    def read(self, timeout_s=0.2) -> Optional[Frame]:
        if self.finished:
            time.sleep(min(timeout_s, 0.05))
            return None
        now = time.monotonic()
        if self._next_due is not None and now < self._next_due:
            time.sleep(min(self._next_due - now, timeout_s))
            if time.monotonic() < self._next_due:
                return None
        shown = self.index
        if self.cap is not None:
            ok, bgr = self.cap.read()
            if not ok:
                if not self.settings.test_loop:
                    self.finished = True
                    return None
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, bgr = self.cap.read()
                if not ok:
                    raise CameraError("Test video has no frames.")
        else:
            if self.index >= len(self.images):
                if not self.settings.test_loop:
                    self.finished = True
                    return None
                self.index = shown = 0
            bgr = cv2.imread(self.images[self.index])
            self.index += 1
            if bgr is None:
                log.warning("[CAM] Test image could not be read: %s", self.images[shown])
                return None
        self._next_due = time.monotonic() + self._interval_after(shown)
        return Frame(bgr, time.time(), 0)

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


BACKENDS = {"aravis": AravisBackend, "file": FileBackend}


# ---------------------------------------------------------------------------- recorder
class FrameRecorder:
    """Saves frames as JPEG for later replay through FileBackend. Its own thread, so capture never waits;
    frames that cannot be written in time are dropped and counted."""

    def __init__(self):
        self.active = False
        self.folder = ""
        self.saved = self.dropped = 0
        self._queue = queue.Queue(maxsize=8)
        self._until = 0.0
        self._min_interval = 0.0
        self._last_taken = 0.0
        self._thread = None
        self.message = ""

    def start(self, folder, seconds, fps, min_free_gb=2.0):
        if self.active:
            return False, "A recording is already running."
        os.makedirs(folder, exist_ok=True)
        free_gb = shutil.disk_usage(folder).free / 1e9
        if free_gb < min_free_gb:
            return False, f"Only {free_gb:.1f} GB free on the disk; at least {min_free_gb:.0f} GB is needed."
        self.folder, self.saved, self.dropped = folder, 0, 0
        self._until = time.monotonic() + float(seconds)
        self._min_interval = 1.0 / max(1.0, float(fps))
        self._last_taken = 0.0
        self.active = True
        self.message = "Recording..."
        self._thread = threading.Thread(target=self._writer, daemon=True, name="FrameRecorder")
        self._thread.start()
        log.info("[CAM] Recording %s s at up to %s fps into %s", seconds, fps, folder)
        return True, self.message

    def stop(self):
        self._until = 0.0

    def offer(self, frame: Frame):
        """Called by the camera thread for every frame."""
        if not self.active:
            return
        now = time.monotonic()
        if now >= self._until:
            self.active = False
            return
        if now - self._last_taken < self._min_interval:
            return
        self._last_taken = now
        try:
            self._queue.put_nowait((frame.bgr, frame.timestamp))
        except queue.Full:
            self.dropped += 1

    def _writer(self):
        while self.active or not self._queue.empty():
            try:
                bgr, timestamp = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            name = f"f{self.saved:06d}_t{int(timestamp * 1000):013d}.jpg"
            try:
                cv2.imwrite(os.path.join(self.folder, name), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                self.saved += 1
            except Exception:
                log.exception("[CAM] Could not write %s", name)
                self.dropped += 1
        self.message = f"Recorded {self.saved} frames ({self.dropped} dropped) in {self.folder}"
        log.info("[CAM] %s", self.message)


# ---------------------------------------------------------------------------- thread
class CameraThread(threading.Thread):
    """States: CONNECTING, ONLINE, TEST (file source running), FINISHED (file source ended), RECONNECTING, OFFLINE."""

    def __init__(self, output_queue: queue.Queue, stop_event: threading.Event, settings: CameraSettings,
                 flat_field: Optional[FlatField] = None, backends=None):
        super().__init__(daemon=True, name="CameraThread")
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.flat_field = flat_field
        self.recorder = FrameRecorder()
        self._backends = backends or BACKENDS
        self._settings = settings
        self._lock = threading.Lock()
        self._restart = threading.Event()
        self._live_change = threading.Event()
        self._was_online = False
        self.is_connected = False
        self.state = "CONNECTING"
        self.fps = 0.0
        self.info = {}              # what the source reports: model, pixel format, size, exposure, gain ...
        self.last_error = ""
        self.reconnect_count = 0
        self.last_frame_time = 0.0

    # ------------------------------------------------------------------ control (any thread)
    @property
    def settings(self) -> CameraSettings:
        with self._lock:
            return CameraSettings(**asdict(self._settings))

    def apply_settings(self, settings: CameraSettings):
        """Exposure / gain / frame-rate changes are sent to the running camera; a new source reopens it."""
        with self._lock:
            reopen = self._settings.needs_reopen(settings)
            self._settings = CameraSettings(**asdict(settings))
        if reopen:
            self._was_online = False
            self._restart.set()
        else:
            self._live_change.set()

    def request_reconnect(self):
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

    def _fail(self, message):
        self.is_connected = False
        self.state = "RECONNECTING" if self._was_online else "OFFLINE"
        self.fps = 0.0
        if message != self.last_error:
            log.warning("[CAM] %s", message)
            self.last_error = message
        self._publish(("ERROR", message))

    def _wait(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end and not self.stop_event.is_set() and not self._restart.is_set():
            time.sleep(0.05)

    # ------------------------------------------------------------------ main loop
    def run(self):
        while not self.stop_event.is_set():
            self._restart.clear()
            settings = self.settings
            backend = self._backends[settings.source](settings, self.flat_field)
            try:
                self.info = backend.open()
                self._live_change.clear()
                log.info("[CAM] %s source opened: %s", settings.source, self.info)
                self._capture(backend, settings)
            except CameraError as exc:
                self._fail(str(exc))
            except Exception as exc:
                log.exception("[CAM] Capture failed")
                self._fail(f"Camera error: {exc}")
            finally:
                try:
                    backend.close()
                except Exception:
                    log.exception("[CAM] Closing the source failed")
            if not self.stop_event.is_set() and not self._restart.is_set():
                self._wait(RETRY_INTERVAL_S)
        self.is_connected = False
        self.state = "OFFLINE"

    def _capture(self, backend, settings):
        online_state = "TEST" if settings.source == "file" else "ONLINE"
        last_good = time.monotonic()
        frames, window_start = 0, time.monotonic()
        while not self.stop_event.is_set() and not self._restart.is_set():
            if self._live_change.is_set():
                self._live_change.clear()
                backend.apply_live(self.settings)
                self.info = backend.info
                log.info("[CAM] Settings applied: %s", self.info)
            frame = backend.read(0.2)
            now = time.monotonic()
            if frame is None:
                if getattr(backend, "finished", False):
                    self._fail("Test source finished.")
                    self.state = "FINISHED"
                    self._wait(3600)
                    return
                if backend.lost or now - last_good > FRAME_TIMEOUT_S:
                    self.reconnect_count += 1 if self._was_online else 0
                    self._fail("Camera connection lost. Reconnecting..." if backend.lost
                               else "Camera stopped sending frames. Reconnecting...")
                    return
                continue
            last_good = now
            if not self.is_connected:
                self.is_connected, self.state, self.last_error, self._was_online = True, online_state, "", True
                log.info("[CAM] Source is %s.", online_state)
            frames += 1
            if now - window_start >= 1.0:
                self.fps = frames / (now - window_start)
                frames, window_start = 0, now
            self.last_frame_time = frame.timestamp
            self.recorder.offer(frame)
            h, w = frame.bgr.shape[:2]
            self._publish((frame.bgr, w, h, frame.timestamp))
        self.is_connected = False
