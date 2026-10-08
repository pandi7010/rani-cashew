import os
import queue
import sys
import tempfile
import threading
import time
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import camera_backends as cb
from camera_backends import CameraError, CameraSettings, CameraThread, FileBackend, FlatField, Frame, FrameRecorder


def wait_for(condition, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if condition():
            return True
        time.sleep(0.02)
    return False


def colour_image(h=64, w=64):
    """Strong blue, a green ramp across, a red ramp down: every channel is different."""
    yy, xx = np.mgrid[0:h, 0:w]
    return np.dstack([np.full((h, w), 200), 60 + xx, 20 + yy]).astype(np.uint8)


def mosaic(bgr, layout):
    """Bayer mosaic of a BGR image. layout = colours at (row0,col0), (row0,col1), (row1,col0), (row1,col1)."""
    plane = {"B": bgr[..., 0], "G": bgr[..., 1], "R": bgr[..., 2]}
    out = np.zeros(bgr.shape[:2], np.uint8)
    for (r, c), colour in zip(((0, 0), (0, 1), (1, 0), (1, 1)), layout):
        out[r::2, c::2] = plane[colour][r::2, c::2]
    return out


class PixelFormatTest(unittest.TestCase):
    LAYOUTS = {"BayerRG8": "RGGB", "BayerBG8": "BGGR", "BayerGB8": "GBRG", "BayerGR8": "GRBG"}

    def test_every_bayer_layout_gives_the_true_colours(self):
        bgr = colour_image()
        for name, layout in self.LAYOUTS.items():
            out = cb.convert_to_bgr(mosaic(bgr, layout).tobytes(), 64, 64, name)
            error = np.abs(out[4:-4, 4:-4].astype(int) - bgr[4:-4, 4:-4]).mean()
            self.assertLess(error, 1.0, name)

    def test_camera_format_bayer_gb8_is_not_mixed_up_with_the_others(self):
        """The machine's camera delivers BayerGB8: a wrong code would swap red and blue or break the image."""
        bgr = colour_image()
        raw = mosaic(bgr, self.LAYOUTS["BayerGB8"]).tobytes()
        for wrong in ("BayerGR8", "BayerRG8", "BayerBG8"):
            out = cb.convert_to_bgr(raw, 64, 64, wrong)
            self.assertGreater(np.abs(out[4:-4, 4:-4].astype(int) - bgr[4:-4, 4:-4]).mean(), 30, wrong)

    def test_other_formats(self):
        gray = np.arange(48, dtype=np.uint8).reshape(6, 8)
        self.assertTrue((cb.convert_to_bgr(gray.tobytes(), 8, 6, "Mono8")[..., 1] == gray).all())
        rgb = np.zeros((6, 8, 3), np.uint8); rgb[..., 0] = 250       # pure red in RGB order
        self.assertEqual(tuple(cb.convert_to_bgr(rgb.tobytes(), 8, 6, "RGB8")[0, 0]), (0, 0, 250))
        self.assertEqual(tuple(cb.convert_to_bgr(rgb.tobytes(), 8, 6, "BGR8")[0, 0]), (250, 0, 0))

    def test_unknown_format_or_wrong_size_is_an_error_not_a_wrong_picture(self):
        with self.assertRaises(CameraError):
            cb.convert_to_bgr(bytes(64), 8, 8, "BayerGB12p")
        with self.assertRaises(CameraError):
            cb.convert_to_bgr(bytes(60), 8, 8, "BayerGB8")


class SettingsTest(unittest.TestCase):
    def test_round_trip_and_bad_values(self):
        s = CameraSettings.from_dict({"source": "usb", "exposure_us": "800", "gain": None, "fps_limit": "x", "unknown": 1})
        self.assertEqual((s.source, s.exposure_us, s.gain), ("aravis", 800.0, None))
        self.assertEqual(s.fps_limit, CameraSettings().fps_limit)
        self.assertEqual(CameraSettings.from_dict(s.to_dict()), s)

    def test_only_a_new_source_needs_a_reopen(self):
        a = CameraSettings()
        self.assertFalse(a.needs_reopen(CameraSettings(exposure_us=500, gain=2, fps_limit_enabled=True)))
        self.assertTrue(a.needs_reopen(CameraSettings(source="file")))
        self.assertTrue(a.needs_reopen(CameraSettings(test_path="/x")))


class FlatFieldTest(unittest.TestCase):
    def test_teach_evens_out_the_lighting_and_is_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ffc.png")
            ff = FlatField(path)
            self.assertFalse(ff.available)
            raw = np.tile(np.linspace(80, 160, 320, dtype=np.float32), (256, 1)).astype(np.uint8)   # darker on the left
            self.assertTrue((ff.apply(raw) == raw).all())           # nothing taught: unchanged
            ff.request_calibration()
            corrected = ff.apply(raw)
            self.assertTrue(ff.enabled and os.path.exists(path))
            self.assertLess(float(corrected[:, 40:-40].std()), float(raw[:, 40:-40].std()) / 4)
            again = FlatField(path)                                   # loaded at the next start
            self.assertTrue(again.enabled)
            again.enabled = False
            self.assertTrue((again.apply(raw) == raw).all())


class FileBackendTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def read_frames(self, backend, n, timeout=5.0):
        out, end = [], time.time() + timeout
        while len(out) < n and time.time() < end and not backend.finished:
            frame = backend.read(0.1)
            if frame is not None:
                out.append(frame)
        return out

    def test_folder_in_order_then_finished(self):
        for i in range(3):
            cv2.imwrite(os.path.join(self.tmp.name, f"f{i}.png"), np.full((12, 16, 3), 50 * (i + 1), np.uint8))
        backend = FileBackend(CameraSettings(source="file", test_path=self.tmp.name, test_loop=False, test_fps=200))
        info = backend.open()
        self.assertEqual(info["frames"], 3)
        frames = self.read_frames(backend, 5)
        self.assertEqual([int(f.bgr[0, 0, 0]) for f in frames], [50, 100, 150])
        self.assertTrue(backend.finished)

    def test_recording_is_replayed_with_its_own_timing(self):
        for i, t_ms in enumerate((1000000000000, 1000000000100, 1000000000200)):       # 100 ms apart
            cv2.imwrite(os.path.join(self.tmp.name, f"f{i:06d}_t{t_ms:013d}.jpg"), np.zeros((12, 16, 3), np.uint8))
        backend = FileBackend(CameraSettings(source="file", test_path=self.tmp.name, test_loop=False, test_fps=200))
        self.assertTrue(backend.open()["recorded_timing"])
        start = time.time()
        self.assertEqual(len(self.read_frames(backend, 3)), 3)
        self.assertGreater(time.time() - start, 0.18)        # two gaps of 100 ms, not 2 x 5 ms

    def test_video_and_missing_source(self):
        path = os.path.join(self.tmp.name, "clip.avi")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 120))
        for i in range(4):
            writer.write(np.full((120, 160, 3), 60 * i, np.uint8))
        writer.release()
        backend = FileBackend(CameraSettings(source="file", test_path=path, test_fps=200))
        backend.open()
        self.assertEqual(len(self.read_frames(backend, 6)), 6)      # repeats
        backend.close()
        with self.assertRaises(CameraError):
            FileBackend(CameraSettings(source="file", test_path=os.path.join(self.tmp.name, "nothing"))).open()


class FakeCamera:
    """Stands in for the Aravis camera: it can be unplugged and plugged in again."""
    plugged = True
    opens = 0
    live_settings = []
    name = "aravis"

    def __init__(self, settings, flat_field=None):
        self.settings, self.lost, self.info = settings, False, {}

    def open(self):
        FakeCamera.opens += 1
        if not FakeCamera.plugged:
            raise CameraError("No camera found.")
        self.info = {"model": "FAKE", "pixel_format": "BayerGB8", "exposure_us": self.settings.exposure_us}
        return self.info

    def apply_live(self, settings, first=False):
        self.settings = settings
        self.info = dict(self.info, exposure_us=settings.exposure_us)
        FakeCamera.live_settings.append(settings.exposure_us)

    def read(self, timeout_s=0.2):
        if not FakeCamera.plugged:
            time.sleep(0.02)
            return None
        time.sleep(0.005)
        return Frame(np.zeros((48, 64, 3), np.uint8), time.time(), 0)

    def close(self):
        pass


class CameraThreadTest(unittest.TestCase):
    def setUp(self):
        self._real = (cb.FRAME_TIMEOUT_S, cb.RETRY_INTERVAL_S)
        cb.FRAME_TIMEOUT_S, cb.RETRY_INTERVAL_S = 0.3, 0.1
        FakeCamera.plugged, FakeCamera.opens, FakeCamera.live_settings = True, 0, []
        self.stop, self.q = threading.Event(), queue.Queue(maxsize=2)
        self.thread = CameraThread(self.q, self.stop, CameraSettings(exposure_us=800), backends={"aravis": FakeCamera, "file": FileBackend})

    def tearDown(self):
        self.stop.set()
        self.thread.join(timeout=3)
        cb.FRAME_TIMEOUT_S, cb.RETRY_INTERVAL_S = self._real

    def test_frames_carry_a_capture_time_and_camera_reconnects_after_unplug(self):
        self.thread.start()
        self.assertTrue(wait_for(lambda: self.thread.state == "ONLINE"))
        frame, w, h, capture_time = self.q.get(timeout=2)
        self.assertEqual((w, h), (64, 48))
        self.assertLess(abs(capture_time - time.time()), 2.0)

        FakeCamera.plugged = False                                  # cable pulled
        self.assertTrue(wait_for(lambda: self.thread.state == "RECONNECTING" and not self.thread.is_connected))
        opens = FakeCamera.opens
        self.assertTrue(wait_for(lambda: FakeCamera.opens > opens + 1))     # keeps trying
        self.assertTrue(self.thread.is_alive())

        FakeCamera.plugged = True                                   # plugged in again
        self.assertTrue(wait_for(lambda: self.thread.state == "ONLINE" and self.thread.is_connected))
        self.assertEqual(self.thread.reconnect_count, 1)

    def test_camera_missing_at_start_then_plugged_in(self):
        FakeCamera.plugged = False
        self.thread.start()
        self.assertTrue(wait_for(lambda: self.thread.state == "OFFLINE" and "No camera" in self.thread.last_error))
        self.assertEqual(self.q.get(timeout=2)[0], "ERROR")
        FakeCamera.plugged = True
        self.assertTrue(wait_for(lambda: self.thread.state == "ONLINE"))

    def test_exposure_change_goes_to_the_running_camera_without_reopening(self):
        self.thread.start()
        self.assertTrue(wait_for(lambda: self.thread.state == "ONLINE"))
        opens = FakeCamera.opens
        self.thread.apply_settings(CameraSettings(exposure_us=1200))
        self.assertTrue(wait_for(lambda: FakeCamera.live_settings == [1200]))
        self.assertTrue(wait_for(lambda: self.thread.info.get("exposure_us") == 1200))
        self.assertEqual(FakeCamera.opens, opens)
        self.assertEqual(self.thread.state, "ONLINE")

    def test_switching_to_a_file_source_and_recording(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.thread.start()
            self.assertTrue(wait_for(lambda: self.thread.state == "ONLINE"))
            ok, _ = self.thread.recorder.start(os.path.join(tmp, "rec"), seconds=0.5, fps=50, min_free_gb=0)
            self.assertTrue(ok)
            self.assertTrue(wait_for(lambda: not self.thread.recorder.active and self.thread.recorder.saved > 3 and "Recorded" in self.thread.recorder.message))
            files = sorted(os.listdir(os.path.join(tmp, "rec")))
            self.assertEqual(len(files), self.thread.recorder.saved)
            self.assertIsNotNone(cb.timestamp_from_name(files[0]))
            # the recording plays back as a test source
            self.thread.apply_settings(CameraSettings(source="file", test_path=os.path.join(tmp, "rec")))
            self.assertTrue(wait_for(lambda: self.thread.state == "TEST"))
            self.assertTrue(self.thread.info["recorded_timing"])
            self.stop.set()
            self.thread.join(timeout=3)          # before the temporary folder disappears


if __name__ == "__main__":
    unittest.main()
