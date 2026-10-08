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
import usb_camera
from usb_camera import CameraSettings, CameraThread


def wait_for(condition, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if condition():
            return True
        time.sleep(0.02)
    return False


class CameraSettingsTest(unittest.TestCase):
    def test_round_trip_and_bad_values(self):
        s = CameraSettings.from_dict({"index": "2", "width": 1920, "fps": "60", "source": "gige", "unknown": 1, "height": "x"})
        self.assertEqual((s.index, s.width, s.fps), (2, 1920, 60))
        self.assertEqual(s.source, "usb")           # unknown source falls back
        self.assertEqual(s.height, CameraSettings().height)     # unreadable value keeps the default
        self.assertEqual(CameraSettings.from_dict(s.to_dict()), s)
        self.assertEqual(CameraSettings.from_dict(None), CameraSettings())


class TestSourceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.stop = threading.Event()
        self.q = queue.Queue(maxsize=2)
        self.thread = None

    def tearDown(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=3)
        self.tmp.cleanup()

    def start(self, **kwargs):
        self.thread = CameraThread(self.q, self.stop, CameraSettings(source="test", fps=60, **kwargs))
        self.thread.start()

    def frames(self, n, timeout=5.0):
        out, end = [], time.time() + timeout
        while len(out) < n and time.time() < end:
            try:
                item = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            if not (len(item) == 2 and item[0] == "ERROR"):
                out.append(item)
        return out

    def write_images(self, count):
        for i in range(count):
            cv2.imwrite(os.path.join(self.tmp.name, f"f{i:02d}.jpg"), np.full((120, 160, 3), 40 * (i + 1), np.uint8))

    def test_image_folder_plays_in_order_and_repeats(self):
        self.write_images(3)
        self.start(test_path=self.tmp.name)
        got = self.frames(7)
        self.assertEqual(len(got), 7)
        frame, w, h = got[0]
        self.assertEqual((w, h, frame.shape), (160, 120, (120, 160, 3)))
        self.assertTrue(self.thread.is_connected)
        self.assertEqual(self.thread.state, "TEST")
        self.assertGreater(len({int(f[0][0, 0, 0]) // 20 for f in got}), 1)      # more than one different image

    def test_video_file(self):
        path = os.path.join(self.tmp.name, "clip.avi")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 120))
        for i in range(5):
            writer.write(np.full((120, 160, 3), 50 * i, np.uint8))
        writer.release()
        self.start(test_path=path)
        got = self.frames(8)            # more frames than the clip has: it repeats
        self.assertEqual(len(got), 8)
        self.assertEqual(got[0][1:], (160, 120))

    def test_missing_source_reports_an_error_and_recovers(self):
        self.start(test_path=os.path.join(self.tmp.name, "nothing_here"))
        self.assertTrue(wait_for(lambda: self.thread.state == "OFFLINE" and "not found" in self.thread.last_error))
        self.assertFalse(self.thread.is_connected)
        self.assertEqual(self.q.get(timeout=2)[0], "ERROR")
        # The source appears later (like a camera being plugged in): new settings bring the picture back.
        self.write_images(2)
        self.thread.apply_settings(CameraSettings(source="test", fps=60, test_path=self.tmp.name))
        self.assertTrue(wait_for(lambda: self.thread.is_connected))
        self.assertEqual(len(self.frames(2)), 2)


class UsbCameraTest(unittest.TestCase):
    """A fake cv2.VideoCapture stands in for the camera: it can be unplugged and plugged in again."""

    class FakeCapture:
        plugged = True
        opened_with = []

        def __init__(self, index, backend=None):
            self.ok = UsbCameraTest.FakeCapture.plugged
            self.props = {}
            UsbCameraTest.FakeCapture.opened_with.append(index)

        def isOpened(self):
            return self.ok

        def set(self, prop, value):
            self.props[prop] = value
            return True

        def get(self, prop):
            return self.props.get(prop, 0)

        def read(self):
            if not UsbCameraTest.FakeCapture.plugged:
                return False, None
            time.sleep(0.005)
            return True, np.zeros((int(self.props[cv2.CAP_PROP_FRAME_HEIGHT]), int(self.props[cv2.CAP_PROP_FRAME_WIDTH]), 3), np.uint8)

        def release(self):
            self.ok = False

    def setUp(self):
        self._real = (cv2.VideoCapture, usb_camera.FRAME_TIMEOUT_S, usb_camera.RETRY_INTERVAL_S)
        cv2.VideoCapture = UsbCameraTest.FakeCapture
        usb_camera.FRAME_TIMEOUT_S, usb_camera.RETRY_INTERVAL_S = 0.3, 0.1
        UsbCameraTest.FakeCapture.plugged = True
        UsbCameraTest.FakeCapture.opened_with = []
        self.stop = threading.Event()
        self.q = queue.Queue(maxsize=2)
        self.thread = CameraThread(self.q, self.stop, CameraSettings(index=3, width=640, height=480, fps=30,
                                                                     auto_exposure=False, exposure=77))

    def tearDown(self):
        self.stop.set()
        self.thread.join(timeout=3)
        cv2.VideoCapture, usb_camera.FRAME_TIMEOUT_S, usb_camera.RETRY_INTERVAL_S = self._real

    def test_settings_are_sent_and_camera_reconnects_after_unplug(self):
        self.thread.start()
        self.assertTrue(wait_for(lambda: self.thread.state == "ONLINE"))
        self.assertEqual(UsbCameraTest.FakeCapture.opened_with[0], 3)
        self.assertEqual((self.thread.actual["width"], self.thread.actual["height"]), (640, 480))
        self.assertEqual(self.thread.actual["exposure"], 77)
        self.assertEqual(self.thread.actual["auto_exposure"], 1)        # 1 = manual (V4L2)
        frame, w, h = self.q.get(timeout=2)
        self.assertEqual((w, h), (640, 480))

        UsbCameraTest.FakeCapture.plugged = False                       # cable pulled
        self.assertTrue(wait_for(lambda: self.thread.state == "RECONNECTING" and not self.thread.is_connected))
        opens_while_unplugged = len(UsbCameraTest.FakeCapture.opened_with)
        self.assertTrue(wait_for(lambda: len(UsbCameraTest.FakeCapture.opened_with) > opens_while_unplugged + 1))   # keeps retrying

        UsbCameraTest.FakeCapture.plugged = True                        # plugged in again
        self.assertTrue(wait_for(lambda: self.thread.state == "ONLINE" and self.thread.is_connected))
        self.assertGreaterEqual(self.thread.reconnect_count, 1)

    def test_camera_missing_at_start_is_offline_then_comes_online(self):
        UsbCameraTest.FakeCapture.plugged = False
        self.thread.start()
        self.assertTrue(wait_for(lambda: self.thread.state == "OFFLINE"))
        UsbCameraTest.FakeCapture.plugged = True
        self.assertTrue(wait_for(lambda: self.thread.state == "ONLINE"))


if __name__ == "__main__":
    unittest.main()
