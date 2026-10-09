#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Instant Fire Grader - Final Master Release
(Stable Zero-Mask Tracking, Multi-Frame Priority Logic, Individual Conf, MM Sizing)
"""
from datetime import datetime
import os

import sys
import shutil
import time
import json
import glob
import threading
import re
import queue
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from collections import deque

import cv2
cv2.setNumThreads(0)
import numpy as np
from ultralytics import YOLO

import app_paths
import preprocess
import flow_geometry as fg
from class_registry import ClassRegistry, normalize_name
from camera_backends import CameraThread, CameraSettings, FlatField
from relay_output import RelayOutput, FIRST_EJECTOR, LAST_EJECTOR, REJECT_RELAY, RESULT_TEXT
import copy
from collections import OrderedDict
import segmentation
import view_features
import ui_theme as ui
import engines
import alarms
import fire_timing
import ejector_check
import ejector_check_ui
import production_capture
import undefined_capture
import colour_size
import dataset_ops
import relay_mapping
import decision
import grading_config
from lane_tracker import LaneTracker, Detection, crossing_time, track_speed
from cashew_log import CashewLog

log = app_paths.setup_logging("grader")

if app_paths.EXTERNAL_TOOLS_DIR.is_dir():
    sys.path.append(str(app_paths.EXTERNAL_TOOLS_DIR))
try:
    import tcm_magic_wand
except ImportError:
    tcm_magic_wand = None
    log.warning("tcm_magic_wand not found in %s: magic wand annotation and auto-teach are disabled.", app_paths.EXTERNAL_TOOLS_DIR)


def magic_polygon(image_bgr, x, y):
    """Polygon of the object under (x, y) from the magic wand tool, or None when the tool is missing.
    The tool looks for FastSAM-s.pt in the current directory, so its model is loaded here from the project folder."""
    if tcm_magic_wand is None:
        return None
    if getattr(tcm_magic_wand, "_fast_sam_model", None) is None and app_paths.FASTSAM_PATH.exists():
        from ultralytics import FastSAM
        tcm_magic_wand._fast_sam_model = FastSAM(str(app_paths.FASTSAM_PATH))
    return tcm_magic_wand.get_magic_polygon(image_bgr, x, y)


try:
    import serial
    import serial.tools.list_ports
except Exception:
    serial = None
    print("WARNING: pySerial not installed. Serial fallback will run in Simulation mode.")

try:
    from innovance_plc import InovancePLCController, InovanceModbusDriver
except Exception as e:
    InovancePLCController = None
    InovanceModbusDriver = None
    print(f"[WARN] Inovance PLC module import failed: {e}")


try:
    import tkinter as tk
    from tkinter import ttk, messagebox, font, filedialog
    from PIL import Image, ImageTk
except Exception:
    tk = None
    print("FATAL: Tkinter and/or Pillow (PIL) not installed.", file=sys.stderr)
    sys.exit(1)

# CNN dependencies (PyTorch + torchvision)
_CNN_AVAILABLE = False
_torch = None
_F = None
try:
    import torch as _torch
    import torch.nn.functional as _F
    import torchvision  # noqa: F401  (preprocess.py needs it)
    _CNN_AVAILABLE = True
    log.info("[CNN] PyTorch + torchvision loaded OK")
except ImportError as e:
    log.warning("[CNN] PyTorch/torchvision not available - CNN disabled: %s", e)


# ---------------- CONFIG ----------------
# Every path comes from app_paths.py (relative to this file, not to the current directory).
app_paths.ensure_dirs()
CFG_PATH = str(app_paths.CFG_PATH)
MODEL_ROOT = app_paths.MODELS_DIR
CNN_MODEL_PATH = app_paths.CNN_MODEL_PATH
DATASET_DIR = str(app_paths.DATASET_DIR)
GOLDEN_DIR = str(app_paths.GOLDEN_DIR)
FULL_FRAME_DIR = str(app_paths.FULL_FRAME_DIR)
MODELS_DIR = str(app_paths.MODELS_DIR)
CLASS_MAPPING_PATH = str(app_paths.CLASS_MAPPING_PATH)
DEFAULT_YOLO_CONF = 0.30


class _Throttle:
    """Log the same recurring error at most once every few seconds (per key)."""

    def __init__(self, interval_s=5.0):
        self.interval_s, self._last, self._skipped = interval_s, {}, {}

    def exception(self, key, message):
        now = time.monotonic()
        if now - self._last.get(key, -1e9) >= self.interval_s:
            skipped = self._skipped.pop(key, 0)
            log.exception("%s%s", message, f" ({skipped} more since the last report)" if skipped else "")
            self._last[key] = now
        else:
            self._skipped[key] = self._skipped.get(key, 0) + 1


_throttle = _Throttle()

NUM_ACTUATORS = 40
DEFAULT_ON_MS = 25

VIDEO_PANE_W = 1000 
VIDEO_PANE_H = 800 

BG_COLOR = (13, 8, 5)        # letterbox around the camera picture (B, G, R) = the theme's darkest tone
TEXT_COLOR = (230, 230, 230)
LINE_COLOR = (0, 0, 255) 
AOI_COLOR1 = (0, 255, 255) 
AOI_COLOR2 = (255, 0, 255) 
FIRE_COLOR = (0, 0, 255) 
CURSOR_COLOR = (0, 255, 0)
ARROW_COLOR = (255, 255, 0)

CLASS_COLORS = [
    (0, 255, 0),    (255, 0, 0),    (0, 255, 255),  (255, 0, 255),
    (255, 165, 0),  (128, 0, 128),  (0, 165, 255),  (0, 128, 0)
]

SERIAL_BAUD = 115200
READY_PROMPT = "Ready?\r\n"
ARDUINO1_RESPONSE = "Yes!"
ARDUINO2_RESPONSE = "Yes2!"

# ---------------- utils ----------------
def find_best_weights(model_root: Path) -> Optional[Path]:
    runs = model_root / "runs"
    candidates = []
    if runs.exists():
        for p in runs.rglob("**/weights/best.pt"):
            try: candidates.append((p.stat().st_mtime, p))
            except Exception: pass
            
    for p in model_root.glob("model_*.pt"):
        try: candidates.append((p.stat().st_mtime, p))
        except Exception: pass
        
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
        
    for n in ("best.pt","yolov8n.pt", "yolo26n.pt"):
        p = model_root / n
        if p.exists(): return p
    for n in ("best.pt","yolov8n.pt", "yolo26n.pt"):
        p = app_paths.BASE_DIR / n
        if p.exists(): return p
    return None

def get_centroid(box):
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0

def centroid_distance(box1, box2):
    cx1, cy1 = get_centroid(box1)
    cx2, cy2 = get_centroid(box2)
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

def resize_letterbox(img, target_w, target_h, bg_color=BG_COLOR):
    if img is None:
        return np.full((target_h, target_w, 3), bg_color, dtype=np.uint8), 1.0, 0, 0
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.full((target_h, target_w, 3), bg_color, dtype=np.uint8), 1.0, 0, 0
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas, scale, x_off, y_off

def draw_rulers(frame):
    if frame is None: return frame
    h, w = frame.shape[:2]
    cv2.line(frame, (0, 15), (w, 15), (100, 100, 100), 20)
    for x in range(0, w, 100):
        cv2.line(frame, (x, 5), (x, 25), (255, 255, 255), 2)
        cv2.putText(frame, str(x), (x+5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    cv2.line(frame, (15, 0), (15, h), (100, 100, 100), 20)
    for y in range(0, h, 100):
        cv2.line(frame, (5, y), (25, y), (255, 255, 255), 2)
        cv2.putText(frame, str(y), (20, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    return frame

def merge_boxes_area_rule(boxes: List[List[int]], classes: List[int], confs: List[float], class_map: Dict[int, str] = None) -> Tuple[List[List[int]], List[int], List[float]]:
    if not boxes: return [], [], []
    if class_map is None: class_map = {}
    
    n = len(boxes)
    areas = [max(0, b[2]-b[0]) * max(0, b[3]-b[1]) for b in boxes]

    inter_matrix = np.zeros((n,n), dtype=float)
    any_pair_meets = False
    for i in range(n):
        for j in range(i+1, n):
            x1 = max(boxes[i][0], boxes[j][0])
            y1 = max(boxes[i][1], boxes[j][1])
            x2 = min(boxes[i][2], boxes[j][2])
            y2 = min(boxes[i][3], boxes[j][3])
            inter_area = max(0, x2 - x1) * max(0, y2 - y1)
            inter_matrix[i,j] = inter_area
            inter_matrix[j,i] = inter_area
            if areas[i] > 0 and areas[j] > 0:
                if (inter_area / areas[i] >= 0.2) and (inter_area / areas[j] >= 0.2):
                    any_pair_meets = True

    links = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if any_pair_meets:
                if areas[i] > 0 and areas[j] > 0 and (inter_matrix[i,j] / areas[i] >= 0.2) and (inter_matrix[i,j] / areas[j] >= 0.2):
                    links[i].append(j); links[j].append(i)
            else:
                if inter_matrix[i,j] > 0:
                    links[i].append(j); links[j].append(i)

    visited = [False]*n
    groups = []
    for i in range(n):
        if visited[i]: continue
        stack = [i]
        comp = []
        visited[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in links[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        groups.append(comp)

    mb, mc, mcf = [], [], []
    for comp in groups:
        if not comp: continue
        group_boxes = [boxes[k] for k in comp]
        group_classes = [classes[k] for k in comp]
        group_confs = [confs[k] for k in comp]
        x1 = min(b[0] for b in group_boxes)
        y1 = min(b[1] for b in group_boxes)
        x2 = max(b[2] for b in group_boxes)
        y2 = max(b[3] for b in group_boxes)
        
        best_idx = int(np.argmax(group_confs)) 
            
        mb.append([int(x1), int(y1), int(x2), int(y2)])
        mc.append(int(group_classes[best_idx]))
        mcf.append(float(group_confs[best_idx]))
        
    return mb, mc, mcf

# ---------------- ArduinoMegaController ----------------
class ArduinoMegaController:
    """Handles serial communication to a single Arduino Mega in a background thread."""
    def __init__(self):
        self.client = None
        self.connected = False
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.baudrate = SERIAL_BAUD
        self._thread = None
        self.cmd_queue = queue.Queue()
        if serial is None:
            print("[WARN] 'serial' module not found. Pneumatic controller is disabled.")
        
    def start(self):
        if serial is None: return
        self.stop_event.clear()
        self._thread = threading.Thread(target=self._connection_loop, daemon=True)
        self._thread.start()
        print("[PNEU] Controller started.")
        
    def stop(self):
        self.stop_event.set()
        if self._thread:
            try: self._thread.join(timeout=0.5)
            except: pass
        self._close_board()
        print("[PNEU] Controller stopped.")

    def all_off(self):
        pass

    def is_connected(self) -> bool:
        return self.connected
        
    def pulse_actuator(self, idx: int, delay_ms: int, pulse_duration_ms: int):
        if not (1 <= idx <= NUM_ACTUATORS): return
        
        if not self.connected:
            return
            
        if delay_ms > 0:
            def delayed_fire():
                import time
                time.sleep(delay_ms / 1000.0)
                if self.connected:
                    cmd_str = f"FIRE:{idx}:0:{pulse_duration_ms}\n"
                    print(f"[ARDUINO] Sending Command: {cmd_str.strip()}")
                    self.cmd_queue.put(cmd_str.encode('ascii'))
            import threading
            threading.Thread(target=delayed_fire, daemon=True).start()
        else:
            cmd_str = f"FIRE:{idx}:0:{pulse_duration_ms}\n"
            print(f"[ARDUINO] Sending Command: {cmd_str.strip()}")
            self.cmd_queue.put(cmd_str.encode('ascii'))

    def _close_board(self):
        with self.lock:
            try:
                if self.client: self.client.close()
            except: pass
            if self.connected:
                print("[ARDUINO] Closed Connection")
            self.client = None
            self.connected = False

    def _connection_loop(self):
        while not self.stop_event.is_set():
            if not self.connected or not self.client:
                if serial:
                    raw_ports = serial.tools.list_ports.comports()
                    ports = [p.device for p in raw_ports if any(x in str(p.description).upper() or x in str(p.device).upper() for x in ["ARDUINO", "CH340", "USB", "SERIAL", "ACM", "TTYUSB"])]
                    if ports:
                        try:
                            client = serial.Serial(ports[0], self.baudrate, timeout=0.2, write_timeout=0.2)
                            time.sleep(0.6)
                            with self.lock:
                                self.client = client
                                self.connected = True
                                print(f"[ARDUINO] Successfully Connected to {ports[0]} at {self.baudrate} baud.")
                        except:
                            print(f"[ARDUINO] Failed to connect to {ports[0]}.")
                            self._close_board()
            else:
                try:
                    # Send up to 5 commands per loop to prevent buffer overflow
                    cmds_sent = 0
                    while not self.cmd_queue.empty() and cmds_sent < 5:
                        cmd = self.cmd_queue.get_nowait()
                        with self.lock:
                            self.client.write(cmd)
                        cmds_sent += 1
                except Exception as e:
                    print(f"[ARDUINO] Write Error: {e}")
                    self.connected = False
                    self._close_board()
                    
                # Only check connection status occasionally, not every 10ms
                if serial and getattr(self, '_loop_counter', 0) % 200 == 0:
                    raw_ports = serial.tools.list_ports.comports()
                    available = [p.device for p in raw_ports]
                    with self.lock:
                        if self.client and self.client.port not in available:
                            self.connected = False
                            print("[ARDUINO] Disconnected - Port no longer available.")
                            try: self.client.close()
                            except: pass
                            self.client = None
                        elif self.client:
                            try:
                                _ = self.client.in_waiting
                            except:
                                self.connected = False
                                print("[ARDUINO] Connection lost - Error reading from port.")
                                try: self.client.close()
                                except: pass
                                self.client = None
                self._loop_counter = getattr(self, '_loop_counter', 0) + 1
            time.sleep(0.01)


# ---------------- SharedState ----------------
class SharedState:
    def __init__(self, all_classes):
        self.lock = threading.RLock()
        
        self.delays_ms = [0]*NUM_ACTUATORS
        self.distances_mm = [0.0]*NUM_ACTUATORS
        self.conveyor_speed_mm_s = 300.0
        self.common_pulse_ms = DEFAULT_ON_MS
        self.pulses_ms = [DEFAULT_ON_MS] * NUM_ACTUATORS       # pulse of each ejector (the "set all" value until a row is changed)
        
        self.mapping_by_act: Dict[int, List[str]] = {i: [] for i in range(1, NUM_ACTUATORS + 1)}
        self.mapping_t1: Dict[str, int] = {}
        self.mapping_t2: Dict[str, int] = {}
        self.priority_rules = [] 
        
        self.class_confs = {c.lower(): 50 for c in all_classes}
        self.defect_allowances = {c.lower(): 0.0 for c in all_classes}
        
        self.yolo_model_path = ""
        self.capture_classes = {str(i): f"Class_{i}" for i in range(1, 21)}
        self.active_capture_class_id = "1"
        self.size_config = {"px_per_mm": 10.0}
        for sz in ["W180", "W210", "W240", "W320", "W450", "BUDDS"]:
            self.size_config[f"{sz}_L"] = 15
            self.size_config[f"{sz}_W"] = 10
            self.size_config[f"T1_Act_{sz}"] = 0
            self.size_config[f"T2_Act_{sz}"] = 0

        self.is_golden_capture = False
        self.golden_target_count = 0
        self.golden_current_count = 0
        self.golden_capture_type = "cropped" # Options: "cropped", "full", "both"

        self.cam_gain, self.cam_r, self.cam_g, self.cam_b = 1.0, 1.0, 1.0, 1.0
        self.analog_gain, self.digital_gain = 1.0, 1.0
        self.camera_settings = CameraSettings()             # camera / test source (camera_backends.py)
        self.flow_direction = fg.RIGHT_TO_LEFT              # also fixes the line orientation (vertical / horizontal)
        self.line_y_full = None                             # position of a horizontal decision line
        # ROI 2 may have its own line. While link_lines is on (default) both ROIs use the ROI 1 line.
        self.link_lines = True
        self.line2_x_full, self.line2_y_full = None, None
        self.dragging_roi = 1
        # Tracking / views / voting settings (grading_config.py), counters and details of recent decisions
        self.grading = grading_config.defaults()
        self.result_counts: Dict[str, int] = {}
        self.recent_decisions = OrderedDict()               # cashew ID -> text with its frame-by-frame votes
        self.size_samples = deque(maxlen=400)               # measured size of every decided cashew (Colour Size page, teach)
        self.view_live = {1: "", 2: ""}                     # latest measured values per ROI (view tuning)
        self.view_thumb = {}                                # ROI -> small picture of that cashew (BGR)
        self.lane_speed = {1: 0.0, 2: 0.0}                  # px/ms, from the tracker
        self.teach_belt_requested = False
        self.teach_belt_message = ""
        self.decision_times = deque(maxlen=5000)            # for pcs/min on the Home page
        self.capture_cfg = production_capture.merged_config()    # automatic production capture into the inbox
        self.no_fire = list(relay_mapping.NO_FIRE_DEFAULT)        # grades that run to the end of the belt (no ejector, not a reject)
        self.good_times = deque(maxlen=5000)                # ... and kg/hr of the good output
        self.engine_counts: Dict[str, int] = {}             # engine -> "would reject" (SHADOW) or rejected (ACTIVE)
        self.belt_stopped = {1: False, 2: False}            # from the trackers
        self.missed_rejects: List[str] = []                 # fires that were on their way when the conveyor stopped
        self.last_result = {}                               # ROI -> (time, was GOOD), for the lane LEDs
        self.opencv_filter = {
            "sw_color_thresh": 45,
            "sw_defect_size_px": 20,
            "ww_color_thresh": 85,
            "ww_defect_size_px": 16,
            "max_defect_px": 1200
        }
        
        self.yolo_conf = DEFAULT_YOLO_CONF
        
        self.phase = "READY"
        self.frame = None
        self.roi1, self.roi2, self.line_x_full = None, None, None  
        self.temp_start_pt, self.temp_end_pt = None, None
        self.mouse_x, self.mouse_y = 0, 0
        self.dragging_line, self.capture_requested = False, False 
        self.is_conveyor_on, self.is_feeder_on = False, False
        self.recent_fires = deque(maxlen=500)
        self.total_detected_count = 0
        self.line_pixel_callback = None

        self.load_from_config(CFG_PATH)

    def get_timing(self, act_id: int) -> Tuple[int, int]:
        if not (1 <= act_id <= NUM_ACTUATORS): return (0, DEFAULT_ON_MS)
        with self.lock: return (self.delays_ms[act_id-1], self.pulses_ms[act_id-1])

    def set_class_conf(self, cname: str, val: int):
        with self.lock: self.class_confs[cname.lower()] = int(val)
        self.save_to_config(CFG_PATH)

    def reset_roi1(self):
        with self.lock: self.roi1 = None
    def reset_roi2(self):
        with self.lock: self.roi2 = None
    # ── decision line: x for a vertical line, y for a horizontal one ──
    @property
    def line_is_vertical(self) -> bool:
        return fg.line_orientation(self.flow_direction) == fg.VERTICAL

    @property
    def line_pos(self):
        with self.lock:
            return self.line_x_full if self.line_is_vertical else self.line_y_full

    def line_pos_for(self, roi: int):
        """Decision line of a ROI. ROI 2 uses the ROI 1 line while the lines are linked."""
        with self.lock:
            if roi == 2 and not self.link_lines:
                return self.line2_x_full if self.line_is_vertical else self.line2_y_full
            return self.line_x_full if self.line_is_vertical else self.line_y_full

    def set_line_pos(self, value, notify=True, roi=1):
        with self.lock:
            if roi == 2 and not self.link_lines:
                if self.line_is_vertical: self.line2_x_full = value
                else: self.line2_y_full = value
            elif self.line_is_vertical: self.line_x_full = value
            else: self.line_y_full = value
        if notify: self._notify_line()

    def set_link_lines(self, linked: bool):
        with self.lock:
            self.link_lines = bool(linked)
            if not self.link_lines and self.line_pos_for(2) is None:    # start the ROI 2 line where the shared one is
                self.line2_x_full, self.line2_y_full = self.line_x_full, self.line_y_full
        self._notify_line()
        self.save_to_config(CFG_PATH)

    def _line_roi(self, pt) -> int:
        """Which ROI's line a click belongs to (always 1 while the lines are linked)."""
        with self.lock:
            r2 = self.roi2
            if self.link_lines or not r2:
                return 1
            return 2 if (r2[0] <= pt[0] <= r2[2] and r2[1] <= pt[1] <= r2[3]) else 1

    def _notify_line(self):
        if self.line_pixel_callback:
            axis, pos = ("X" if self.line_is_vertical else "Y"), self.line_pos
            txt = f"Line {axis}: {pos} px" if pos is not None else f"Line {axis}: Not Set"
            if not self.link_lines:
                pos2 = self.line_pos_for(2)
                txt = f"ROI 1 {txt}   |   ROI 2 line {axis}: " + (f"{pos2} px" if pos2 is not None else "Not Set")
            self.line_pixel_callback(txt, pos)

    # ── grading: counters, decision details, belt colour teach ──
    def add_decision(self, track_id: str, label: str, frames_txt: str, relay: int, count_keys, detail: str, row=None):
        """row = (class, SW / WW, colour value, length mm, grade) for the RECENT RESULTS table."""
        with self.lock:
            shown = row or (label, "-", "-", "-", "-")
            self.recent_fires.appendleft((time.strftime("%H:%M:%S"), track_id) + tuple(shown) + (f"A{relay:02d}" if relay else "-",))
            for key in count_keys:
                self.result_counts[key] = self.result_counts.get(key, 0) + 1
            self.recent_decisions[track_id] = detail
            now = time.time()
            self.decision_times.append(now)
            if label.startswith("GOOD"):
                self.good_times.append(now)
            if track_id[:1] == "R" and track_id[1:2].isdigit():
                self.last_result[int(track_id[1])] = (now, label.startswith("GOOD"))
            while len(self.recent_decisions) > 300:
                self.recent_decisions.popitem(last=False)

    def decisions_last_minute(self) -> int:
        limit = time.time() - 60.0
        with self.lock:
            return sum(1 for t in self.decision_times if t >= limit)

    def good_last_minute(self) -> int:
        limit = time.time() - 60.0
        with self.lock:
            return sum(1 for t in self.good_times if t >= limit)

    def count_engine(self, name: str):
        with self.lock:
            self.engine_counts[name] = self.engine_counts.get(name, 0) + 1

    def count_result(self, key: str):
        with self.lock:
            self.result_counts[key] = self.result_counts.get(key, 0) + 1

    def get_result_counts(self) -> Dict[str, int]:
        with self.lock: return dict(self.result_counts)

    def reset_result_counts(self):
        with self.lock:
            self.result_counts = {}
            self.engine_counts = {}

    def set_grading(self, grading: dict):
        with self.lock: self.grading = grading_config.merged(grading)
        self.save_to_config(CFG_PATH)

    def take_teach_belt_request(self) -> bool:
        with self.lock:
            wanted, self.teach_belt_requested = self.teach_belt_requested, False
            return wanted

    def set_flow_direction(self, flow):
        with self.lock:
            self.flow_direction = fg.valid_flow(flow)
        self._notify_line()
        self.save_to_config(CFG_PATH)

    def reset_line(self):
        with self.lock:
            self.line2_x_full, self.line2_y_full = None, None
        self.set_line_pos(None)

    def set_mapping_for_actuator(self, act_id: int, class_list: List[str]):
        if not (1 <= act_id <= NUM_ACTUATORS): return
        with self.lock:
            self.mapping_by_act[act_id] = sorted(list(set(class_list)))
            self._rebuild_processing_map()

    def get_mapping_for_actuator(self, act_id: int) -> List[str]:
        with self.lock: return self.mapping_by_act.get(act_id, [])
            
    def set_delay(self, act_id: int, delay_ms: int):
        if not (1 <= act_id <= NUM_ACTUATORS): return
        with self.lock: self.delays_ms[act_id-1] = delay_ms
        self.save_to_config(CFG_PATH)

    def set_common_pulse(self, pulse_ms: int):
        with self.lock:
            self.common_pulse_ms = pulse_ms
            self.pulses_ms = [int(pulse_ms)] * NUM_ACTUATORS
        self.save_to_config(CFG_PATH)

    def set_pulse(self, act_id: int, pulse_ms: int):
        if 1 <= act_id <= NUM_ACTUATORS:
            with self.lock: self.pulses_ms[act_id-1] = int(pulse_ms)

    def get_assigned_classes_for_track(self, track_num: int) -> set:
        assigned = set()
        start, end = (1, 20) if track_num == 1 else (21, 40)
        with self.lock:
            for act_id, classes in self.mapping_by_act.items():
                if start <= act_id <= end: assigned.update(classes)
        return assigned

    def add_fire_event(self, track: int, label: str, conf: float, act_id: int):
        with self.lock:
            ts = time.strftime("%H:%M:%S")
            self.recent_fires.appendleft((ts, f"Trk {track}", label, "-", "-", "-", "-", f"A{act_id}"))

    def get_recent_fires(self):
        with self.lock: return list(self.recent_fires)

    def increment_detected_count(self, amount: int = 1):
        with self.lock:
            self.total_detected_count += amount

    def get_detected_count(self) -> int:
        with self.lock:
            return self.total_detected_count

    def reset_detected_count(self):
        with self.lock:
            self.total_detected_count = 0

    def trigger_capture(self):
        with self.lock: self.capture_requested = True

    def check_and_clear_capture(self) -> bool:
        with self.lock:
            if self.capture_requested:
                self.capture_requested = False; return True
            return False

    def trigger_golden_capture(self, target_count: int, capture_type="cropped"):
        with self.lock:
            self.is_golden_capture = True
            self.golden_target_count = target_count
            self.golden_current_count = 0
            self.golden_capture_type = capture_type

    def cancel_golden_capture(self):
        with self.lock:
            self.is_golden_capture = False


    def is_golden_capture_active(self) -> bool:
        with self.lock:
            return self.is_golden_capture

    def increment_golden_capture(self) -> bool:
        """Returns True if the capture should proceed, False if we finished."""
        with self.lock:
            if not self.is_golden_capture:
                return False
            self.golden_current_count += 1
            if self.golden_current_count >= self.golden_target_count:
                self.is_golden_capture = False
            return True


    def get_active_capture_folder(self):
        with self.lock:
            return self.capture_classes.get(self.active_capture_class_id, "Unknown_Class")

    def load_from_config(self, cfg_path):
        if not os.path.exists(cfg_path): return
        try:
            with open(cfg_path, 'r') as f:
                cfg = json.load(f)
            with self.lock:
                delays = cfg.get("delays_ms", {})
                for i in range(1, NUM_ACTUATORS + 1):
                    self.delays_ms[i-1] = int(delays.get(str(i), 0))
                
                self.conveyor_speed_mm_s = float(cfg.get("conveyor_speed_mm_s", 300.0))
                dists = cfg.get("distances_mm", {})
                for i in range(1, NUM_ACTUATORS + 1):
                    self.distances_mm[i-1] = float(dists.get(str(i), 0.0))
                
                self.common_pulse_ms = int(cfg.get("common_pulse_ms", DEFAULT_ON_MS))
                stored_pulses = cfg.get("pulses_ms", {})
                self.pulses_ms = [int(stored_pulses.get(str(i), self.common_pulse_ms)) for i in range(1, NUM_ACTUATORS + 1)]
                self.priority_rules = cfg.get("priority_rules", [])
                
                loaded_confs = cfg.get("class_confs", {})
                for k, v in loaded_confs.items():
                    if k in self.class_confs: self.class_confs[k] = int(v)
                    
                if "defect_allowances" in cfg:
                    for k,v in cfg["defect_allowances"].items(): self.defect_allowances[k.lower()] = float(v)
                
                self.mapping_by_act = {i: [] for i in range(1, NUM_ACTUATORS + 1)}
                for k, v in cfg.get("mapping", {}).items():
                    if 1 <= int(k) <= NUM_ACTUATORS: self.mapping_by_act[int(k)] = list(v)
                
                if "size_config" in cfg: self.size_config.update(cfg["size_config"])
                if "opencv_filter" in cfg: self.opencv_filter.update(cfg["opencv_filter"])
                
                self.yolo_model_path = cfg.get("yolo_model_path", "")
                self.yolo_conf = float(cfg.get("yolo_conf", DEFAULT_YOLO_CONF))
                if "roi1" in cfg and cfg["roi1"]: self.roi1 = tuple(cfg["roi1"])
                if "roi2" in cfg and cfg["roi2"]: self.roi2 = tuple(cfg["roi2"])
                self.line_x_full = cfg.get("line_x_full")
                self.line_y_full = cfg.get("line_y_full")
                self.link_lines = bool(cfg.get("link_lines", True))
                self.line2_x_full, self.line2_y_full = cfg.get("line2_x_full"), cfg.get("line2_y_full")
                self.grading = grading_config.merged(cfg.get("grading"))
                self.capture_cfg = production_capture.merged_config(cfg.get("production_capture"))
                self.no_fire = [str(n) for n in cfg.get("no_fire", relay_mapping.NO_FIRE_DEFAULT)]
                self.flow_direction = fg.valid_flow(cfg.get("flow_direction", fg.RIGHT_TO_LEFT))
                if "camera_source" in cfg:
                    self.camera_settings = CameraSettings.from_dict(cfg["camera_source"])
                else:   # config from before camera_backends.py: the exposure was stored with the colour gains
                    old_exposure = (cfg.get("camera") or {}).get("exposure")
                    self.camera_settings = CameraSettings(exposure_us=float(old_exposure) if old_exposure else None)
                loaded_classes = cfg.get("capture_classes", {})
                if loaded_classes:
                    for k, v in loaded_classes.items():
                        self.capture_classes[str(k)] = v
                self.active_capture_class_id = str(cfg.get("active_capture_class_id", "1"))
                
                if "camera" in cfg:
                    c = cfg["camera"]
                    self.cam_gain = c.get("gain", 1.0)
                    self.cam_r = c.get("r", 1.0)
                    self.cam_g = c.get("g", 1.0)
                    self.cam_b = c.get("b", 1.0)
                    self.analog_gain = c.get("a_gain", 1.0)
                    self.digital_gain = c.get("d_gain", 1.0)
                
                self._rebuild_processing_map()
        except Exception:
            log.exception("Could not read the config %s; defaults are used for what was not loaded.", cfg_path)

    def save_to_config(self, cfg_path):
        try:
            with self.lock:
                cfg = {
                    "conveyor_speed_mm_s": self.conveyor_speed_mm_s,
                    "distances_mm": {str(i+1): self.distances_mm[i] for i in range(NUM_ACTUATORS)},
                    "delays_ms": {str(i+1): self.delays_ms[i] for i in range(NUM_ACTUATORS)},
                    "common_pulse_ms": self.common_pulse_ms,
                    "pulses_ms": {str(i+1): self.pulses_ms[i] for i in range(NUM_ACTUATORS)},
                    "mapping": {str(k): v for k,v in self.mapping_by_act.items()},
                    "priority_rules": self.priority_rules,
                    "size_config": self.size_config,
                    "opencv_filter": self.opencv_filter,
                    "class_confs": self.class_confs,
                    "defect_allowances": self.defect_allowances,
                    "yolo_model_path": self.yolo_model_path,
                    "yolo_conf": self.yolo_conf,
                    "roi1": self.roi1,
                    "roi2": self.roi2,
                    "line_x_full": self.line_x_full,
                    "line_y_full": self.line_y_full,
                    "link_lines": self.link_lines,
                    "line2_x_full": self.line2_x_full,
                    "line2_y_full": self.line2_y_full,
                    "grading": self.grading,
                    "production_capture": self.capture_cfg,
                    "no_fire": self.no_fire,
                    "flow_direction": self.flow_direction,
                    "camera_source": self.camera_settings.to_dict(),
                    "capture_classes": self.capture_classes,
                    "active_capture_class_id": self.active_capture_class_id,
                    "camera": {
                        "gain": self.cam_gain, "r": self.cam_r, "g": self.cam_g, "b": self.cam_b,
                        "a_gain": self.analog_gain, "d_gain": self.digital_gain
                    }
                }
            # Written to a temporary file first, so a crash while saving cannot leave half a config.
            tmp_path = f"{cfg_path}.tmp"
            with open(tmp_path, 'w') as f: json.dump(cfg, f, indent=2)
            os.replace(tmp_path, cfg_path)
        except Exception:
            log.exception("Could not save the config %s", cfg_path)

    def _rebuild_processing_map(self):
        """Class -> ejector tables of both ROIs, from the mapping alone (relay_mapping.py). Reject ejectors carry nothing."""
        tables = relay_mapping.lookup(self.mapping_by_act)
        self.mapping_t1, self.mapping_t2 = tables[1], tables[2]

    def set_mapping(self, mapping):
        """Replace the whole class -> ejector mapping (already checked by relay_mapping.assign)."""
        with self.lock:
            self.mapping_by_act = relay_mapping.clean(mapping)
            self._rebuild_processing_map()

    def get_processing_state(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "phase": self.phase,
                "roi1": self.roi1, "roi2": self.roi2,
                "temp_start": self.temp_start_pt, "temp_end": self.temp_end_pt,
                "line_pos": self.line_x_full if self.line_is_vertical else self.line_y_full,
                "line_pos2": self.line_pos_for(2),
                "link_lines": self.link_lines,
                "grading": copy.deepcopy(self.grading),
                "flow": self.flow_direction,
                "mouse_x": self.mouse_x, "mouse_y": self.mouse_y,
                "mapping_t1": self.mapping_t1.copy(), "mapping_t2": self.mapping_t2.copy(), "no_fire": list(self.no_fire),
                "priority_rules": list(self.priority_rules),
                "size_config": self.size_config.copy(),
                "class_confs": self.class_confs.copy(),
                "defect_allowances": self.defect_allowances.copy(),
                "opencv_filter": self.opencv_filter.copy(),
                "target_conf": self.yolo_conf,
                "is_conveyor_on": self.is_conveyor_on,
                "gain": self.cam_gain, "r": self.cam_r, "g": self.cam_g, "b": self.cam_b,
                "a_gain": self.analog_gain, "d_gain": self.digital_gain,
                "exposure": self.camera_settings.exposure_us if self.camera_settings.exposure_us is not None else "camera default"
            }

    def handle_mouse_down(self, pt: Tuple[int, int]):
        with self.lock:
            along = pt[0] if self.line_is_vertical else pt[1]       # the coordinate the line is set by
            if self.phase == "ROI1_START" or self.phase == "ROI2_START":
                self.temp_start_pt = pt; self.temp_end_pt = pt
                self.phase = "ROI1_DRAW" if self.phase == "ROI1_START" else "ROI2_DRAW"
            elif self.phase == "LINE_SET_MODE":
                self.set_line_pos(along, roi=self._line_roi(pt))
                self.phase = "READY"
                self.save_to_config(CFG_PATH)
            else:
                roi = self._line_roi(pt)
                pos = self.line_pos_for(roi)
                if pos is not None and abs(along - pos) < 15:
                    self.dragging_line, self.dragging_roi = True, roi

    def handle_mouse_move(self, pt: Tuple[int, int]):
        with self.lock:
            self.mouse_x, self.mouse_y = pt
            if self.phase in ("ROI1_DRAW", "ROI2_DRAW"): 
                self.temp_end_pt = pt
            elif self.dragging_line:
                self.set_line_pos(pt[0] if self.line_is_vertical else pt[1], roi=self.dragging_roi)

    def handle_mouse_up(self, pt: Tuple[int, int]):
        with self.lock:
            was_dragging = self.dragging_line
            self.dragging_line = False
            if self.phase in ("ROI1_DRAW", "ROI2_DRAW"):
                self.temp_end_pt = pt
                if not self.temp_start_pt: return
                x1, y1 = min(self.temp_start_pt[0], pt[0]), min(self.temp_start_pt[1], pt[1])
                x2, y2 = max(self.temp_start_pt[0], pt[0]), max(self.temp_start_pt[1], pt[1])
                if x2 - x1 > 20 and y2 - y1 > 20:
                    if self.phase == "ROI1_DRAW": self.roi1 = (x1, y1, x2, y2)
                    else: self.roi2 = (x1, y1, x2, y2)
                self.phase = "READY"
                self.temp_start_pt = None; self.temp_end_pt = None
                self.save_to_config(CFG_PATH)
            elif was_dragging:
                self.save_to_config(CFG_PATH)

# ---------------- ProcessingThread ----------------
class ProcessingThread(threading.Thread):
    def __init__(self, model_path: Optional[Path], registry: ClassRegistry,
                 camera_queue: queue.Queue, display_queue: queue.Queue, 
                 stop_event: threading.Event, shared_state: SharedState,
                 controller: Any, fire_counts: Dict[int, int],
                 fire_lock: threading.Lock, model: Any = None):

        super().__init__(daemon=True, name="ProcessingThread")
        self.model_path = model_path
        self.registry = registry        # the one list of class names (YOLO IDs from model.names)
        self.camera_queue = camera_queue
        self.display_queue = display_queue
        self.stop_event = stop_event
        self.shared_state = shared_state
        self.controller = controller    # RelayOutput: the only way a relay is fired
        self.fire_counts = fire_counts 
        self.fire_lock = fire_lock      
        self.model = None
        self.model_lock = threading.Lock()
        self.trackers = {1: LaneTracker(1), 2: LaneTracker(2)}    # one tracker and one ID series per ROI
        self.capture = production_capture.ProductionCapture()     # inbox capture; its own saver thread
        # Every Undefined / lost / double cashew: all views + the full frame into captures/<undefined|double>/<date>/
        self.undefined_capture = undefined_capture.UndefinedCapture(app_paths.CAPTURES_DIR)
        self._box_areas = {1: deque(maxlen=40), 2: deque(maxlen=40)}      # box size of real cashews per ROI (noise filter)
        self._clean_frame = None
        self.frame_time = time.time()
        self.ai_fps, self._ai_count, self._ai_since = 0.0, 0, time.time()   # frames graded per second
        frame_log = shared_state.grading.get("frame_log", {})
        self.cashew_log = CashewLog(app_paths.LOGS_DIR, keep_days=int(frame_log.get("keep_days", 7)),
                                    frames_enabled=bool(frame_log.get("enabled", True))).start()
        self._tracking_reset = threading.Event()
        
        self.device = 0 if (_CNN_AVAILABLE and _torch.cuda.is_available()) else 'cpu'
        log.info("ProcessingThread: device %s, YOLO model %s", self.device, self.model_path)
        if model is not None:
            self._install_model(self.model_path, model)
        elif self.model_path:
            self.reload_model(self.model_path)

        # ── CNN Color Classifier ──────────────────────────────────────────────
        self.cnn_model = None
        self.cnn_classes = []
        self.cnn_device = None
        self.cnn_transform = None
        self.cnn_path = None
        if _CNN_AVAILABLE:
            self.cnn_transform = preprocess.build_transform(train=False)
            self.reload_cnn_model(active_cnn_path())
        else:
            log.warning("[CNN] PyTorch missing - YOLO classes are used.")

        log.info("Class names in use: %s", self.registry.snapshot())

    def reload_cnn_model(self, path) -> bool:
        """Load a CNN checkpoint (at start, or when a model is activated while STOPPED). The old model stays on a failure."""
        path = Path(path)
        if not path.exists():
            log.warning("[CNN] %s not found - YOLO classes are used.", path)
            return False
        try:
            device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
            model, classes = preprocess.load_cnn_checkpoint(path, device)
        except Exception:
            log.exception("[CNN] Failed to load %s", path)
            return False
        self.cnn_device, self.cnn_classes, self.cnn_model, self.cnn_path = device, classes, model, path
        self.registry.add_names(classes)     # names YOLO does not have get an ID after the YOLO IDs
        log.info("[CNN] Loaded %s (%d classes: %s)", path.name, len(classes), classes)
        return True

    def _cnn_predict(self, crop_bgr: np.ndarray):
        """Run CNN on a BGR crop already cut to the cashew. Returns (class_name, confidence) or (None, 0.0)."""
        if self.cnn_model is None or crop_bgr is None or crop_bgr.size == 0:
            return None, 0.0
        try:
            tensor = preprocess.crop_to_tensor(crop_bgr, self.cnn_transform, tight=False).unsqueeze(0).to(self.cnn_device)
            with _torch.no_grad():
                probs = _F.softmax(self.cnn_model(tensor), dim=1)[0].cpu().numpy()
            idx = int(np.argmax(probs))
            return self.cnn_classes[idx], float(probs[idx])
        except Exception:
            _throttle.exception("cnn_predict", "[CNN] Prediction failed")
            return None, 0.0

    def _calculate_defect_area(self, crop_bgr: np.ndarray, color_thresh: int = 110) -> int:
        """
        Uses OpenCV to find the area of dark defects (like testa, dust) in the cashew crop.
        Returns the number of dark pixels within the cashew mask.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return 999999
        try:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            # Cashew is bright, background (blue roller) is dark. Otsu thresholding creates the cashew mask.
            _, cashew_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Defects are darker spots inside the bright cashew.
            # We use a static intensity threshold (e.g. < 110) to identify dark pixels.
            _, dark_spots = cv2.threshold(gray, color_thresh, 255, cv2.THRESH_BINARY_INV)
            
            # Only count dark pixels that are actually inside the cashew contour.
            actual_defects = cv2.bitwise_and(dark_spots, cashew_mask)
            defect_area = cv2.countNonZero(actual_defects)
            return defect_area
        except Exception:
            _throttle.exception("defect_area", "[CV] Defect area calculation failed")
            return 999999

    def _get_cashew_contour_rect(self, crop_bgr: np.ndarray):
        """Returns (cashew_area, (x, y, w, h)) inside the crop to localize cashew vs empty roller."""
        try:
            return preprocess.find_cashew_rect(crop_bgr)
        except Exception:
            _throttle.exception("cashew_rect", "[CV] Cashew localisation failed")
            return 0, None
        
    def get_color_for_class(self, cid: int):
        return CLASS_COLORS[cid % len(CLASS_COLORS)]

    def _install_model(self, model_path, new_model):
        with self.model_lock:
            self.model_path = model_path
            self.model = new_model
            # YOLO class IDs come from the model itself; nothing may renumber them.
            self.registry.set_yolo_names({int(k): str(v) for k, v in new_model.names.items()})
        self._tracking_reset.set()      # IDs kept by running tracks may belong to the previous model
        
    def reload_model(self, model_path) -> bool:
        model_str = str(model_path)
        if not model_str.endswith(".pt"):
            log.error("Only .pt YOLO models are supported: %s", model_str)
            return False
        try:
            self._install_model(model_path, YOLO(str(app_paths.resolve(model_str))))
            log.info("YOLO model loaded: %s, classes %s", model_str, self.registry.snapshot())
            return True
        except Exception:
            log.exception("Failed to load YOLO model %s", model_str)
            return False

    def run(self):
        while not self.stop_event.is_set():
            try:
                data = self.camera_queue.get(timeout=0.1)
                if len(data) == 2 and data[0] == "ERROR":
                    # Create an error frame to show the camera disconnected
                    err_frame = np.zeros((800, 1000, 3), dtype=np.uint8)
                    cv2.putText(err_frame, f"CAMERA ERROR: {data[1]}", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    try: self.display_queue.get_nowait()
                    except queue.Empty: pass
                    self.display_queue.put((err_frame, {"scale": 1.0, "x_off": 0, "y_off": 0, "orig_w": 1000, "orig_h": 800}))
                    continue
                
                frame_full, w, h = data[0], data[1], data[2]
                self.frame_time = data[3] if len(data) > 3 else time.time()    # when the camera took the frame
                if frame_full is None: continue
                
                with self.shared_state.lock:
                    self.shared_state.frame = frame_full.copy()
                
                state = self.shared_state.get_processing_state()
                
                g = max(0.1, state["gain"] * state["a_gain"] * state["d_gain"])
                r, gr, b = max(0.1, state["r"]), max(0.1, state["g"]), max(0.1, state["b"])
                if g != 1.0 or r != 1.0 or gr != 1.0 or b != 1.0:
                    # Ultra-fast color correction using cv2.transform matrix operation
                    # This avoids slow channel splitting and clipping
                    m = np.array([
                        [g * b, 0, 0],
                        [0, g * gr, 0],
                        [0, 0, g * r]
                    ], dtype=np.float32)
                    frame_full = cv2.transform(frame_full, m)
                
                if self.shared_state.take_teach_belt_request():
                    self._teach_belt(frame_full, state)

                if self.shared_state.check_and_clear_capture():
                    folder_name = self.shared_state.get_active_capture_folder()
                    target_dir = os.path.join(DATASET_DIR, folder_name)
                    os.makedirs(target_dir, exist_ok=True)
                    filename = os.path.join(target_dir, f"cashew_{int(time.time()*1000)}.jpg")
                    
                    # Save a metadata file in the folder to record the camera settings for future reference
                    metadata_path = os.path.join(target_dir, "camera_settings.txt")
                    if not os.path.exists(metadata_path):
                        with open(metadata_path, "w") as f:
                            f.write("--- Camera Settings for this Dataset ---\n")
                            f.write(f"Gain: {state.get('gain', 1.0):.2f}\n")
                            f.write(f"Red (R): {state.get('r', 1.0):.2f}\n")
                            f.write(f"Green (G): {state.get('g', 1.0):.2f}\n")
                            f.write(f"Blue (B): {state.get('b', 1.0):.2f}\n")
                            f.write(f"Exposure: {state.get('exposure', 'N/A')}\n")
                            f.write("\nNote: Use these exact settings in the GUI when running the AI model to ensure consistent colors.\n")
                    
                    # Create a copy for saving so we don't modify the display frame
                    save_frame = frame_full.copy()
                    
                    r1 = state["roi1"]
                    r2 = state["roi2"]
                    if r1 or r2:
                        b1 = r1 if r1 else r2
                        b2 = r2 if r2 else r1
                        crop_x1 = min(b1[0], b2[0])
                        crop_y1 = min(b1[1], b2[1])
                        crop_x2 = max(b1[2], b2[2])
                        crop_y2 = max(b1[3], b2[3])
                        save_frame = save_frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    
                    cv2.imwrite(filename, save_frame)
                
                display_frame = None
                
                if state["phase"] == "RUNNING" and self.model and (state["roi1"] or state["roi2"]):
                    # Don't draw rulers here, just process it directly to save massive amounts of CPU on the 5MP frame
                    frame_drawn = self._process_frame(frame_full, state)
                    self._ai_count += 1
                    if time.time() - self._ai_since >= 1.0:
                        self.ai_fps = self._ai_count / (time.time() - self._ai_since)
                        self._ai_count, self._ai_since = 0, time.time()
                    
                    # Crop and stack only the ROIs for the GUI display!
                    crops_to_show = []
                    if state["roi1"]:
                        rx1, ry1, rx2, ry2 = state["roi1"]
                        crops_to_show.append(frame_drawn[ry1:ry2, rx1:rx2])
                    if state["roi2"]:
                        rx1, ry1, rx2, ry2 = state["roi2"]
                        crops_to_show.append(frame_drawn[ry1:ry2, rx1:rx2])
                        
                    if len(crops_to_show) > 0:
                        display_img, _ = fg.stack_crops(crops_to_show)
                    else:
                        display_img = frame_drawn
                        
                    display_frame, scale, x_off, y_off = resize_letterbox(display_img, VIDEO_PANE_W, VIDEO_PANE_H, bg_color=BG_COLOR)
                    geometry = {"scale": scale, "x_off": x_off, "y_off": y_off, "orig_w": w, "orig_h": h}
                else:
                    for tracker in self.trackers.values():
                        tracker.clear()
                    self.ai_fps = 0.0
                    self._tracking_reset.clear()
                    frame_drawn = draw_rulers(frame_full.copy())
                    self._draw_setup_overlays(frame_drawn, state)
                    
                    mx, my = state["mouse_x"], state["mouse_y"]
                    if mx > 0 and my > 0:
                        cv2.putText(frame_drawn, f"X:{mx} Y:{my}", (mx+15, my-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CURSOR_COLOR, 2)
                        cv2.circle(frame_drawn, (mx, my), 3, CURSOR_COLOR, -1)
                        
                    display_frame, scale, x_off, y_off = resize_letterbox(frame_drawn, VIDEO_PANE_W, VIDEO_PANE_H, bg_color=BG_COLOR)
                    geometry = {"scale": scale, "x_off": x_off, "y_off": y_off, "orig_w": w, "orig_h": h}

                try: self.display_queue.get_nowait() 
                except queue.Empty: pass
                
                if self.model is None and state["phase"] == "RUNNING":
                    cv2.putText(display_frame, "MODEL DISABLED - CAPTURE ONLY MODE", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                
                self.display_queue.put((display_frame, geometry))

                
            except queue.Empty: pass
            except Exception:
                _throttle.exception("processing_loop", "Frame processing failed (this frame was skipped, nothing was fired for it)")

    # FIXED: Small arrow in the top corner (approx 25mm length x 5mm height visually)
    def _draw_arrows(self, frame, state):
        """Small flow arrow in a corner of each ROI, pointing the way the cashews travel."""
        flow = state["flow"]
        length = 60
        for idx, roi in enumerate([state["roi1"], state["roi2"]]):
            if not roi:
                continue
            x1, y1, x2, y2 = roi
            if flow == fg.LEFT_TO_RIGHT:
                start_pt, end_pt = (x1 + 10, y1 + 20), (x1 + 10 + length, y1 + 20)
                text_pt = (end_pt[0] + 10, y1 + 25)
            elif flow == fg.RIGHT_TO_LEFT:
                start_pt, end_pt = (x2 - 10, y1 + 20), (x2 - 10 - length, y1 + 20)
                text_pt = (end_pt[0] - 80, y1 + 25)
            elif flow == fg.TOP_TO_BOTTOM:
                start_pt, end_pt = (x1 + 20, y1 + 10), (x1 + 20, y1 + 10 + length)
                text_pt = (x1 + 30, y1 + 25)
            else:
                start_pt, end_pt = (x1 + 20, y2 - 10), (x1 + 20, y2 - 10 - length)
                text_pt = (x1 + 30, y2 - 15)
            cv2.arrowedLine(frame, start_pt, end_pt, ARROW_COLOR, 2, tipLength=0.3)
            cv2.putText(frame, f"T{idx+1} FLOW", text_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, ARROW_COLOR, 2)

    def _draw_setup_overlays(self, frame, state):
        if state["temp_start"] and state["temp_end"]:
            cv2.rectangle(frame, state["temp_start"], state["temp_end"], (255,255,255), 2)
        if state["roi1"]:
            x1,y1,x2,y2 = state["roi1"]
            cv2.rectangle(frame, (x1,y1), (x2,y2), AOI_COLOR1, 3)
            cv2.putText(frame, "ROI 1", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, AOI_COLOR1, 2)
        if state["roi2"]:
            x1,y1,x2,y2 = state["roi2"]
            cv2.rectangle(frame, (x1,y1), (x2,y2), AOI_COLOR2, 3)
            cv2.putText(frame, "ROI 2", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, AOI_COLOR2, 2)
        self._draw_lines(frame, state, 3)
        
        ph = state["phase"]
        msg = ""
        if ph == "ROI1_START": msg = "Draw ROI 1"
        elif ph == "ROI2_START": msg = "Draw ROI 2"
        elif ph == "LINE_SET_MODE": msg = "Click to set Decision Line"
        if msg: cv2.putText(frame, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        self._draw_arrows(frame, state)

    def _teach_belt(self, frame, state):
        """TEACH BELT COLOUR: blue band from the ROIs of an empty belt (frame already has the colour gain)."""
        parts = [frame[r[1]:r[3]:4, r[0]:r[2]:4].reshape(-1, 1, 3) for r in (state["roi1"], state["roi2"]) if r]
        pixels = np.concatenate(parts) if parts else frame[::8, ::8].reshape(-1, 1, 3)
        current = state["grading"]["segmentation"]
        new = segmentation.teach_belt(pixels, current)
        if new is None:
            message = "Belt colour NOT taught: the picture is not an evenly coloured, lit, empty belt."
        else:
            grading = state["grading"]
            grading["segmentation"] = new
            self.shared_state.set_grading(grading)
            message = (f"Belt colour taught: hue {new['blue_h_min']}-{new['blue_h_max']}, saturation from {new['blue_s_min']} "
                       f"(was {current['blue_h_min']}-{current['blue_h_max']}, {current['blue_s_min']}).")
        log.info(message)
        with self.shared_state.lock:
            self.shared_state.teach_belt_message = message

    def _draw_lines(self, frame, state, thickness):
        """Decision line(s): one across the picture when linked, otherwise one inside each ROI."""
        flow = state["flow"]
        if state.get("link_lines", True):
            fg.draw_line(frame, flow, state["line_pos"], LINE_COLOR, thickness)
            return
        for roi, pos in ((state["roi1"], state["line_pos"]), (state["roi2"], state.get("line_pos2"))):
            if pos is None:
                continue
            if not roi:
                fg.draw_line(frame, flow, pos, LINE_COLOR, thickness)
            elif fg.line_orientation(flow) == fg.VERTICAL:
                cv2.line(frame, (int(pos), roi[1]), (int(pos), roi[3]), LINE_COLOR, thickness)
            else:
                cv2.line(frame, (roi[0], int(pos)), (roi[2], int(pos)), LINE_COLOR, thickness)

    def _process_frame(self, frame, state):
        """One frame: YOLO finds the cashews, the CNN classifies every crop, each ROI's tracker gives the
        cashew its ID, every frame adds a vote, and ONE decision is made when the centroid crosses the line."""
        t_ms = float(self.frame_time) * 1000.0          # when the camera took the frame
        if self._tracking_reset.is_set():               # a new YOLO model was applied
            self._tracking_reset.clear()
            for tracker in self.trackers.values():
                tracker.clear()

        clean_frame = frame.copy()                      # for inference and saved images (no overlay)
        self._clean_frame = clean_frame
        flow, grading = state["flow"], state["grading"]
        lines = {1: state["line_pos"], 2: state.get("line_pos2")}
        self._draw_lines(frame, state, 2)

        rois = [(rid, roi) for rid, roi in ((1, state.get("roi1")), (2, state.get("roi2"))) if roi]
        found = {rid: ([], [], []) for rid, _ in rois}  # ROI -> boxes, YOLO class IDs, YOLO confidences

        with self.model_lock:
            if self.model is not None:
                crops, owners = [], []
                for rid, (rx1, ry1, rx2, ry2) in rois:
                    roi_crop = clean_frame[ry1:ry2, rx1:rx2]
                    if roi_crop.size > 0:
                        crops.append(roi_crop)
                        owners.append((rid, rx1, ry1))
                if crops:
                    # One YOLO call for all ROIs: the crops are put into a single image.
                    combined_img, places = fg.stack_crops(crops)
                    # Boxes below the normal confidence are kept as "weak": the tracker lets them continue a cashew it
                    # already knows (a rolling cashew is often found with less confidence), never start a new ID.
                    results = self.model.predict(combined_img, conf=min(self.WEAK_YOLO_CONF, state.get("target_conf", 0.30)), verbose=False, device=self.device)[0]
                    for box in results.boxes:
                        xy = box.xyxy[0].cpu().numpy().tolist()
                        i = fg.place_index(places, (xy[0] + xy[2]) / 2.0, (xy[1] + xy[3]) / 2.0)
                        if i < 0:
                            continue
                        px, py = places[i][0], places[i][1]
                        rid, rx1, ry1 = owners[i]
                        found[rid][0].append([int(xy[0] - px) + rx1, int(xy[1] - py) + ry1, int(xy[2] - px) + rx1, int(xy[3] - py) + ry1])
                        found[rid][1].append(int(box.cls[0].cpu().numpy()))
                        found[rid][2].append(float(box.conf[0].cpu().numpy()))

        for rid, roi in rois:
            boxes, classes, confs = merge_boxes_area_rule(*found[rid])
            detections = [self._measure_detection(clean_frame, box, cid, conf, roi, state, t_ms)
                          for box, cid, conf in zip(boxes, classes, confs)]
            for det in detections:
                det.weak = det.data["yolo_conf"] < float(state.get("target_conf", 0.30))
            tracker = self.trackers[rid]
            tracker.params = {**tracker.params, **grading.get("tracker", {})}
            tracks = tracker.update(detections, t_ms, flow, roi)
            for det in detections:
                crop = det.data.pop("crop", None)
                if det.track is None:           # a faint box alone, or a second box on a cashew that has its ID
                    continue
                if crop is not None and not det.track.decided:
                    self.capture.note_view(det.track.id, crop, decision.canonical(det.data["cnn"]), det.data["cnn_conf"])
                if not det.track.decided:
                    det.track.frames.append(det.data)
                    bx1, by1, bx2, by2 = det.data["box"]
                    self.undefined_capture.note_view(det.track.id, clean_frame[max(0, by1):by2, max(0, bx1):bx2])
                det.track.info["last"] = det.data
            live = next((d.data for d in detections if d.track is not None and d.data.get("feat")), None)
            for lost in tracker.dropped:        # disappeared before its line: LOST, nothing is fired
                if not lost.decided:
                    self._log_lost(lost, rid)
            with self.shared_state.lock:
                self.shared_state.lane_speed[rid] = tracker.lane_speed
                self.shared_state.belt_stopped[rid] = tracker.stopped
                if live is not None:
                    self.shared_state.view_live[rid] = f"{live['view']}  {view_features.short_text(live['feat'])}"
                    bx1, by1, bx2, by2 = live["box"]
                    crop = clean_frame[by1:by2, bx1:bx2]
                    if crop.size:
                        scale = 110.0 / max(crop.shape[:2])
                        self.shared_state.view_thumb[rid] = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))))

            line = lines[rid]
            for trk in tracks:
                if not trk.counted and trk.hits >= 2:   # seen in two frames: a cashew, not noise
                    trk.counted = True
                    self.shared_state.increment_detected_count()
                progress = fg.progress(flow, line, trk.cx, trk.cy) if line is not None else None
                if progress is not None and progress < 0 and not trk.missed:
                    self._golden_capture(clean_frame, trk, state)
                if progress is not None and progress >= 0 and not trk.decided:
                    # One cashew = one decision = at most one fire: a fragment is closed without a decision.
                    if not self._close_fragment(trk, rid, flow, "AT_LINE"):
                        self._decide_track(trk, rid, line, state, t_ms)
                self._draw_track(frame, trk, grading)

        # ROI boxes while RUNNING. The Home view shows only the ROI crops, so the frame is drawn just inside the ROI
        # (a box on the ROI edge itself would be cut away) and the name sits inside it.
        for (rid, roi), colour in zip(((1, state.get("roi1")), (2, state.get("roi2"))), (AOI_COLOR1, AOI_COLOR2)):
            if roi:
                x1, y1, x2, y2 = roi
                cv2.rectangle(frame, (x1 + 2, y1 + 2), (x2 - 3, y2 - 3), colour, 3)
                cv2.putText(frame, f"ROI {rid}", (x1 + 12, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
        self._draw_arrows(frame, state)
        return frame

    def _measure_detection(self, clean_frame, box, yolo_cid, yolo_conf, roi, state, t_ms):
        """Everything known about one detection in one frame: YOLO class, CNN class, mask, view, features."""
        grading = state["grading"]
        seg_params = grading.get("segmentation", {})
        H, W = clean_frame.shape[:2]
        bx1, by1, bx2, by2 = max(0, int(box[0])), max(0, int(box[1])), min(W, int(box[2])), min(H, int(box[3]))
        edge_cut = segmentation.touches_edge(box, roi, float(seg_params.get("edge_margin_px", 2)))
        data = {"t": t_ms, "yolo": self.registry.name(yolo_cid), "yolo_conf": float(yolo_conf), "cnn": None, "cnn_conf": 0.0,
                "view": view_features.OTHER, "edge_cut": edge_cut, "feat": None, "box": [bx1, by1, bx2, by2]}
        cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
        if bx2 <= bx1 or by2 <= by1:
            return Detection(cx, cy, [bx1, by1, bx2, by2], data)

        # Mask by blue-background removal, on the YOLO box plus a small border.
        pad = 6
        sx1, sy1, sx2, sy2 = max(0, bx1 - pad), max(0, by1 - pad), min(W, bx2 + pad), min(H, by2 + pad)
        seg_crop = clean_frame[sy1:sy2, sx1:sx2]
        seg = None
        try:
            seg = segmentation.segment(seg_crop, seg_params)
        except Exception:
            _throttle.exception("segment", "[CV] Cashew mask failed (box centre is used)")
        if seg is not None:
            cx, cy = sx1 + seg.cx, sy1 + seg.cy     # the mask centroid does not jump when the cashew rolls
            if not edge_cut:
                try:
                    data["feat"] = view_features.measure(seg_crop, seg)
                    data["view"] = view_features.classify(data["feat"], grading.get("views", {}).get("rules"))
                except Exception:
                    _throttle.exception("view_features", "[CV] View measurement failed for a cashew")

        # CNN crop: made exactly as before (preprocess.py), so the model sees what it was trained on.
        crop = clean_frame[by1:by2, bx1:bx2]
        if self.cnn_model is not None:
            cashew_area, cashew_rect = self._get_cashew_contour_rect(crop)
            if cashew_rect is not None and cashew_area >= 150:
                rx, ry, rw, rh = cashew_rect
                margin = 4
                tx1, ty1 = max(0, bx1 + rx - margin), max(0, by1 + ry - margin)
                tx2, ty2 = min(W, bx1 + rx + rw + margin), min(H, by1 + ry + rh + margin)
                data["box"] = [tx1, ty1, tx2, ty2]
                crop = clean_frame[ty1:ty2, tx1:tx2]
            cnn_name, cnn_conf = self._cnn_predict(crop)
            if cnn_name is not None:
                cv_cfg = state.get("opencv_filter", {})
                low = cnn_name.lower().replace("_", " ")
                if cv_cfg.get("enabled", False) and low in ("sw", "white dw"):     # old dark-spot filter, default OFF
                    if low == "sw":
                        def_px, max_px = self._calculate_defect_area(crop, color_thresh=cv_cfg.get("sw_color_thresh", 45)), cv_cfg.get("sw_defect_size_px", 20)
                    else:
                        def_px, max_px = self._calculate_defect_area(crop, color_thresh=cv_cfg.get("ww_color_thresh", 85)), cv_cfg.get("ww_defect_size_px", 16)
                    if def_px < max_px:
                        cnn_name = "GOOD_CASHEW"
                self.registry.id_for(cnn_name, register=True)
                data["cnn"], data["cnn_conf"] = cnn_name, float(cnn_conf)
                if not edge_cut and self.capture.wanted():
                    data["crop"] = crop.copy()      # only while production capture is on; handed over right after tracking
        return Detection(cx, cy, data["box"], data)

    def _golden_capture(self, clean_frame, trk, state):
        """Dataset auto capture (Teaching pages): crop and / or full frame of a cashew before the line."""
        if not self.shared_state.is_golden_capture_active():
            return
        x1, y1, x2, y2 = [int(v) for v in trk.box]
        pad = 20
        g_crop = clean_frame[max(0, y1 - pad):min(clean_frame.shape[0], y2 + pad), max(0, x1 - pad):min(clean_frame.shape[1], x2 + pad)]
        if g_crop.size == 0 or not self.shared_state.increment_golden_capture():
            return
        timestamp_ms = int(time.time() * 1000)
        c_type = self.shared_state.golden_capture_type
        if c_type in ("cropped", "both"):
            cv2.imwrite(os.path.join(GOLDEN_DIR, f"crop_{timestamp_ms}_{trk.id}.jpg"), g_crop)
        if c_type in ("full", "both"):
            cv2.imwrite(os.path.join(FULL_FRAME_DIR, f"full_{timestamp_ms}_{trk.id}.jpg"), clean_frame)
        meta_file = os.path.join(GOLDEN_DIR, "camera_metadata.txt")
        if not os.path.exists(meta_file):
            with open(meta_file, "w") as f:
                f.write(f"Capture Date & Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                for key, label in (("exposure", "Exposure"), ("gain", "Gain"), ("r", "R"), ("g", "G"), ("b", "B")):
                    f.write(f"{label}: {state.get(key, 'N/A')}\n")

    def _decide_track(self, trk, rid, line, state, t_ms):
        """The ONE decision of a cashew, at its Decision line, and the ONE fire that follows from it."""
        trk.decided = True
        grading, flow = state["grading"], state["flow"]
        map_dict = state["mapping_t1"] if rid == 1 else state["mapping_t2"]

        engine_mode = {**engines.ENGINE_DEFAULTS, **grading.get("engines", {})}
        px_per_mm = float(state["size_config"].get("px_per_mm", 10.0)) or 10.0
        feats = [f["feat"] for f in trk.frames if f["feat"]]
        first = trk.path[0]
        past_at_first = fg.progress(flow, line, first[1], first[2])
        votes = [decision.FrameVote(f["t"], f["cnn"], f["cnn_conf"], f["yolo"], f["yolo_conf"], f["view"], not f["edge_cut"])
                 for f in trk.frames]
        d = decision.decide(votes, grading.get("voting"), grading.get("priority"), grading.get("good_classes"),
                            grading.get("skin_classes"), state.get("priority_rules"),
                            double=engines.is_double([x["length_px"] for x in feats], px_per_mm, grading.get("double")),
                            appeared_after_line=past_at_first >= 0, stuck=trk.stuck, lost_ms=trk.max_gap_ms,
                            max_lost_ms=float(grading.get("rules", {}).get("max_lost_ms", 300.0)),
                            view_engine=engine_mode["view"], reject_classes=grading.get("reject_classes"))
        # Time already gone since the line (only for a cashew first seen after it): taken off the delay.
        late_ms = (past_at_first / self.trackers[rid].lane_speed) if (past_at_first >= 0 and self.trackers[rid].lane_speed > 0) else 0.0

        # Engines can only ADD a reject to an accepted cashew. SHADOW: noted as "would reject", nothing changes.
        shadow = list(d.shadow)
        if d.result == decision.GOOD and engine_mode["shape"] != engines.OFF:
            why = engines.shape_check(feats, px_per_mm, grading.get("shape"))
            if why and engine_mode["shape"] == engines.ACTIVE:
                d.result, d.reason, d.reject = decision.UNDEFINED, why, True
                self.shared_state.count_engine("shape")
            elif why:
                shadow.append(why)
        for note in shadow:
            self.shared_state.count_engine("view" if note == decision.VIEW_NOT_ROLLED else "shape")
        if d.reason == decision.NOT_ROLLED:
            self.shared_state.count_engine("view")
        med = lambda key: float(np.median([x[key] for x in feats])) if feats else None
        top = lambda key: float(max(x[key] for x in feats)) if feats else None
        length_px, width_px = med("length_px"), top("width_px")
        sw_votes = sum(n for name, n in d.votes.items() if normalize_name(name) == "sw")

        # Step (c2). Size: median over the frames, in mm. Colour: median LAB value inside the mask over the frames.
        mm_length = (med("length_px") or 0.0) / px_per_mm
        mm_width = (med("width_px") or 0.0) / px_per_mm
        mm_area = (med("area_px") or 0.0) / px_per_mm ** 2
        size_cfg = state["size_config"]
        colour_values = colour_size.track_colour(feats)
        colour, colour_value = colour_size.colour_of(colour_values, size_cfg.get("colour"))
        relay, grade = None, ""
        if d.reject:
            relay = REJECT_RELAY[rid]
            label = d.result if d.result != decision.UNDEFINED else f"UNDEF {d.reason}"
            colour = None
        elif d.result == decision.GOOD:
            # GOOD -> SW or WW -> size grade of that colour -> the grade's ejector (from the mapping).
            colour_taught = colour is not None
            if not colour_taught:           # no threshold yet: what the CNN voted, only to show a "would be"
                colour = colour_size.SW if sw_votes * 2 > d.good_frames else colour_size.WW
            size_grade, why = colour_size.grade_of(colour, mm_length, mm_width, mm_area, size_cfg.get("grades"))
            grade = colour_size.would_be(colour, size_grade, why)
            if not size_cfg.get("size_ready", False):
                # SHADOW: colour and size are measured and logged ("would be SW210"); the cashew goes to the reject ejector.
                d.reason, d.reject = f"{decision.SIZE_NOT_READY}:{grade}", True
                relay, label, d.result = REJECT_RELAY[rid], f"UNDEF SIZE_NOT_READY (would be {grade})", decision.UNDEFINED
            elif not colour_taught:         # no threshold, or no frame of this cashew had a mask to measure the colour in
                why_colour = colour_size.COLOUR_NOT_TAUGHT if colour_values else "COLOUR_NOT_MEASURED"
                d.reason, d.reject = why_colour, True
                relay, label, d.result = REJECT_RELAY[rid], f"UNDEF {why_colour.replace('_', ' ').lower()}", decision.UNDEFINED
            elif size_grade is None:        # outside every grade (or no grade of this colour taught): reject ejector
                d.reason, d.reject = f"{why} {mm_length:.1f} x {mm_width:.1f} mm", True
                relay, label, d.result = REJECT_RELAY[rid], f"UNDEF {grade}", decision.UNDEFINED
                self.shared_state.count_engine("size")
            else:
                relay, assigned = relay_mapping.target_of(map_dict, rid, [grade, colour, "GOOD_CASHEW", "GOOD"], state.get("no_fire"))
                label = f"GOOD {grade}" + (" (no fire)" if relay is None else "")
                if not assigned:            # a grade without an ejector goes to the reject ejector
                    d.reason, d.reject = f"{decision.NOT_MAPPED}:{grade}", True
                    label, d.result = f"UNDEF no ejector ({grade})", decision.UNDEFINED
        else:
            relay, assigned = relay_mapping.target_of(map_dict, rid, [d.result])
            label = d.result
            if not assigned:        # a class without an ejector goes to the reject ejector
                d.reason, d.reject = f"{decision.NOT_MAPPED}:{d.result}", True
                label = f"UNDEF no ejector ({d.result})"
                d.result = decision.UNDEFINED

        fire_result = ""
        if relay is not None and 1 <= relay <= NUM_ACTUATORS:
            d_ms, p_ms = self.shared_state.get_timing(relay)
            fire_result = self.controller.fire(relay, max(0.0, d_ms - late_ms), p_ms, source=f"{trk.id} {label}")
            cid = self.registry.id_for(d.result)
            if cid is not None:
                with self.fire_lock:
                    self.fire_counts[int(cid)] = self.fire_counts.get(int(cid), 0) + 1

        # Production capture: uncertain cashews, rejects and a few good ones go to the inbox (never waits).
        self.capture.on_decision(trk.id, rid, "ACCEPT" if d.result == decision.GOOD else "REJECT", d.reason,
                                 {"would": "; ".join(shadow), "result": d.result, "relay": relay})
        speed = track_speed(trk.path, flow)
        t_cross = crossing_time(trk.path, flow, line)
        trk.info.update(label=label, relay=relay, result=d.result)
        why = d.reason.split(":")[0].split(" ")[0]
        why = "LOST_OVER_LIMIT" if why.startswith("LOST_") else why
        count_keys = [d.result] + ([f"UNDEFINED: {why}"] if d.result == decision.UNDEFINED else [])
        frame_rows = [{"t": round(f["t"] - trk.born_ms, 1), "cnn": f["cnn"], "cnn_conf": round(f["cnn_conf"], 3),
                       "yolo": f["yolo"], "yolo_conf": round(f["yolo_conf"], 3), "view": f["view"], "edge_cut": f["edge_cut"],
                       "at": round(f["t"], 1), "box": f["box"],
                       "feat": {k: round(v, 3) for k, v in f["feat"].items()} if f["feat"] else None} for f in trk.frames]
        detail = [f"{trk.id}   {label}   relay {relay if relay else '-'}   {fire_result}",
                  f"result {d.result}, reason {d.reason}; frames {d.frames_used} used of {d.frames_total}; "
                  f"GOOD frames {d.good_frames}, NECK frames {d.neck_frames}",
                  f"seen (weighted frames): {d.seen or '-'}", f"views: {d.views or '-'}",
                  f"engines in SHADOW would reject: {', '.join(shadow) or '-'}", ""]
        for r in frame_rows:
            feat_txt = view_features.short_text(r["feat"]) + f" dark%{r['feat']['dark_pct']:.1f} red%{r['feat']['red_pct']:.1f}" if r["feat"] else "(no mask)"
            detail.append(f"+{r['t']:6.0f} ms  CNN {str(r['cnn']):<12} {r['cnn_conf']:.2f}  YOLO {str(r['yolo']):<12} {r['yolo_conf']:.2f}  "
                          f"{r['view']:<5}{' EDGE' if r['edge_cut'] else ''}  {feat_txt}")
        shown_class = label.replace("UNDEF SIZE_NOT_READY ", "GOOD ") if d.reason.startswith(decision.SIZE_NOT_READY) else label
        self.shared_state.add_decision(trk.id, label, f"{d.frames_used}f", relay or 0, count_keys, "\n".join(detail), row=(
            shown_class, colour or "-", "-" if colour_value is None else f"{colour_value:.1f}", f"{mm_length:.1f}" if mm_length else "-", grade or "-"))
        if len(trk.frames) >= 5:
            self._box_areas[rid].append(abs(trk.box[2] - trk.box[0]) * abs(trk.box[3] - trk.box[1]))
        # Auto capture: Undefined (not the GOOD cashews held back only because the size grades are not ready) and double.
        if d.result == decision.DOUBLE:
            category = undefined_capture.DOUBLE
        elif d.result == decision.UNDEFINED and not str(d.reason).startswith(decision.SIZE_NOT_READY):
            category = undefined_capture.UNDEFINED
        else:
            category = None
        self.undefined_capture.on_result(trk.id, category, d.reason, {
            "roi": rid, "result": d.result, "label": label, "relay": relay or "", "fire": fire_result, "frames": d.frames_total,
            "frames_used": d.frames_used, "votes": d.votes, "views_seen": d.views, "box": [int(v) for v in trk.box]}, self._clean_frame)
        with self.shared_state.lock:
            self.shared_state.size_samples.append({
                "id": trk.id, "t": time.time(), "l": mm_length, "w": mm_width, "a": mm_area, "solidity": med("solidity") or 0.0,
                "double": d.result == decision.DOUBLE or decision.DOUBLE in str(d.reason), "label": label,
                "lab": colour_values, "colour": colour or "", "colour_value": colour_value, "grade": grade, "relay": relay or 0})
        self.cashew_log.write({
            "id": trk.id, "roi": rid, "frames": d.frames_total, "frames_used": d.frames_used, "result": d.result, "reason": d.reason,
            "reject": int(d.reject), "good_frames": d.good_frames, "neck_frames": d.neck_frames, "seen": d.seen, "votes": d.votes,
            "views": d.views, "cnn_sw_votes": sw_votes, "length_px": length_px, "width_px": width_px, "area_px": top("area_px"),
            "length_mm": mm_length or "", "width_mm": mm_width or "",
            "L": med("L"), "a": med("a"), "b": med("b"), "dark_pct_max": top("dark_pct"), "red_pct_max": top("red_pct"),
            "spots_max": top("spots"), "speed_px_ms": speed if speed is not None else "", "t_cross_ms": t_cross if t_cross is not None else "",
            "grade": grade, "relay": relay or "", "fire": fire_result, "late": "", "shadow": "; ".join(shadow),
        }, frames=frame_rows)

    WEAK_YOLO_CONF = 0.10               # faint YOLO boxes down to this confidence may continue a known cashew
    NOISE_MAX_HITS, NOISE_AREA_SHARE = 8, 0.25
    FRAGMENT_MAX_HITS = 4               # a track seen at most this often can be a fragment of another cashew

    def _usual_box_area(self, rid):
        """Middle box area of the real cashews seen lately (this ROI, or both when this one has too few); 0 = unknown."""
        own = list(self._box_areas[rid])
        areas = own if len(own) >= 5 else own + list(self._box_areas[3 - rid])
        return float(np.median(areas)) if len(areas) >= 5 else 0.0

    def _fragment_kind(self, trk, rid, flow):
        """(MERGED, the track it belongs to) / (NOISE, None) / (None, None) for a track that ends or reaches the line.
        MERGED: another live cashew is at the same place, so this track is a piece of that cashew (a broken ID or a
        second box). NOISE: a tiny box seen a few times (a mark on the belt, a reflection), not a cashew."""
        other = self.trackers[rid].companion(trk, flow)
        if other is not None and (trk.hits <= other.hits or trk.hits <= self.FRAGMENT_MAX_HITS):
            return decision.MERGED, other
        usual = self._usual_box_area(rid)
        box_area = abs(trk.box[2] - trk.box[0]) * abs(trk.box[3] - trk.box[1])
        if usual > 0 and trk.hits <= self.NOISE_MAX_HITS and box_area < self.NOISE_AREA_SHARE * usual:
            return decision.NOISE, None
        return None, None

    def _track_note(self, trk, rid):
        """Where a track was seen and what was next to it, for the log."""
        try:
            (t0, x0, y0), (t1, x1, y1) = trk.path[0], trk.path[-1]
            others = [o for o in self.trackers[rid].tracks if o is not trk]
            near = min(others, key=lambda o: abs(o.cx - trk.cx) + abs(o.cy - trk.cy)) if others else None
            return (f"seen {trk.hits}x in {t1 - t0:.0f} ms from ({x0:.0f},{y0:.0f}) to ({x1:.0f},{y1:.0f}); "
                    f"box {abs(trk.box[2] - trk.box[0]):.0f}x{abs(trk.box[3] - trk.box[1]):.0f} px; speed {trk.v:.2f} px/ms; longest gap {trk.max_gap_ms:.0f} ms; "
                    + (f"nearest other cashew {near.id} at {abs(near.cx - trk.cx):.0f},{abs(near.cy - trk.cy):.0f} px" if near else "no other cashew in the ROI"))
        except Exception:
            return ""

    def _frame_rows(self, trk):
        """Per-frame positions of a track for the frames log (so a run can be replayed through the tracker)."""
        return [{"t": round(f["t"] - trk.born_ms, 1), "at": round(f["t"], 1), "box": f["box"], "yolo": f["yolo"],
                 "yolo_conf": round(f["yolo_conf"], 3)} for f in trk.frames]

    def _close_fragment(self, trk, rid, flow, where) -> bool:
        """True when the track was a fragment or noise: it is logged (MERGED / NOISE), never counted, decided or fired."""
        kind, other = self._fragment_kind(trk, rid, flow)
        if kind is None:
            return False
        trk.decided = True
        trk.info["result"] = kind
        if other is not None:
            other.info.setdefault("merged", []).append(trk.id)
            if not other.decided:           # what was seen on the fragment counts for the cashew it belongs to
                other.frames = sorted(trk.frames + other.frames, key=lambda f: f["t"])
        self.shared_state.count_result(f"{decision.IGNORED_PREFIX}{kind}")
        self.cashew_log.write({"id": trk.id, "roi": rid, "frames": len(trk.frames), "result": kind,
                               "reason": f"{where}:{other.id}" if other is not None else where, "fire": "NOT_FIRED",
                               "shadow": self._track_note(trk, rid)}, frames=self._frame_rows(trk))
        self.undefined_capture.forget(trk.id)
        return True

    def _log_lost(self, trk, rid):
        """A cashew that disappeared before its line (5-track: decision LOST). Counted and logged, never fired.
        A fragment of another cashew (MERGED) or a tiny short-lived box (NOISE) is logged but is not a lost cashew."""
        flow = self.shared_state.flow_direction
        if self._close_fragment(trk, rid, flow, "ENDED"):
            return
        if trk.hits < 2:
            reason = decision.SEEN_ONLY_ONCE
        else:
            reason = decision.LEFT_ROI_UNDECIDED if trk.info.get("left_roi") else decision.LOST_BEFORE_LINE
        self.shared_state.count_result(decision.LOST)
        note = self._track_note(trk, rid)
        self.cashew_log.write({"id": trk.id, "roi": rid, "frames": len(trk.frames), "result": decision.LOST, "reason": reason,
                               "fire": "NOT_FIRED", "shadow": note}, frames=self._frame_rows(trk))
        self.undefined_capture.on_result(trk.id, undefined_capture.UNDEFINED, f"LOST_{reason}", {
            "roi": rid, "result": decision.LOST, "fire": "NOT_FIRED", "frames": len(trk.frames), "where": note}, self._clean_frame)

    def _draw_track(self, frame, trk, grading):
        x1, y1, x2, y2 = [int(v) for v in trk.box]
        cv2.circle(frame, (int(trk.cx), int(trk.cy)), 5, (255, 255, 255), -1)
        if trk.decided:
            relay = trk.info.get("relay")
            cv2.rectangle(frame, (x1, y1), (x2, y2), FIRE_COLOR, 3)
            cv2.putText(frame, f"{trk.id} {str(trk.info.get('label', ''))[:24]} {'R%d' % relay if relay else '(NO RELAY)'}",
                        (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, FIRE_COLOR, 2)
            return
        last = trk.info.get("last") or {}
        name = last.get("cnn") or last.get("yolo") or "?"
        conf = last.get("cnn_conf") if last.get("cnn") else last.get("yolo_conf", 0.0)
        cid = self.registry.id_for(name)
        colour = self.get_color_for_class(cid if cid is not None else 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 1 if trk.missed else 2)
        text = f"{trk.id} {str(name).upper()[:12]} {conf:.2f} {last.get('view', '')}" + (" (PRED)" if trk.missed else "")
        cv2.putText(frame, text, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
        if last.get("feat") and not trk.missed and grading.get("views", {}).get("show_values", True):
            cv2.putText(frame, view_features.short_text(last["feat"]), (x1, y2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _get_size_data(self, length: float, width: float, cfg: dict, track_id: int) -> Tuple[Optional[int], str]:
        prefix = "T1_" if track_id == 1 else "T2_"
        if length < cfg.get("BUDDS_L", 18) and width < cfg.get("BUDDS_W", 11): return cfg.get(f"{prefix}Act_BUDDS", 0), "BUDDS"
        elif length < cfg.get("W450_L", 22) and width < cfg.get("W450_W", 12): return cfg.get(f"{prefix}Act_W450", 0), "W450"
        elif length < cfg.get("W320_L", 25) and width < cfg.get("W320_W", 14): return cfg.get(f"{prefix}Act_W320", 0), "W320"
        elif length < cfg.get("W240_L", 27) and width < cfg.get("W240_W", 16): return cfg.get(f"{prefix}Act_W240", 0), "W240"
        elif length < cfg.get("W210_L", 28) and width < cfg.get("W210_W", 18): return cfg.get(f"{prefix}Act_W210", 0), "W210"
        else: return cfg.get(f"{prefix}Act_W180", 0), "W180"


# ---------------- ScrolledFrame (GUI Helper) ----------------
class ScrolledFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self, bg="#1e1e1e", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        cw = canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        def on_configure_canvas(e): canvas.itemconfig(cw, width=e.width)
        canvas.bind("<Configure>", on_configure_canvas)
        canvas.configure(yscrollcommand=scrollbar.set)
        def on_mousewheel(event): canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.bind_all("<MouseWheel>", on_mousewheel) 
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")


class NavButton(tk.Frame):
    def __init__(self, parent, text, icon_img, colors, command, **kwargs):
        super().__init__(parent, bg=colors["bg"], cursor="hand2")
        self.colors = colors
        self.command = command
        self.text = text
        if icon_img:
            self.icon_lbl = tk.Label(self, image=icon_img, bg=colors["bg"], cursor="hand2")
        else:
            self.icon_lbl = tk.Label(self, text="?", font=("Roboto", 24, "bold"), bg=colors["bg"], fg="white", cursor="hand2")
        self.icon_lbl.pack(pady=(15,2))
        self.text_lbl = tk.Label(self, text=text, font=("Roboto", 9, "bold"), bg=colors["bg"], fg=colors["text_muted"], cursor="hand2")
        self.text_lbl.pack(pady=(0,15))
        
        for w in (self, self.icon_lbl, self.text_lbl):
            w.bind("<Button-1>", self._on_click)
            w.bind("<Enter>", self._on_enter)
            w.bind("<Leave>", self._on_leave)
            
        self._active = False
        
    def _on_click(self, e):
        self.command()
        
    def _on_enter(self, e):
        if not self._active:
            self.config(bg=self.colors["surface_light"])
            self.icon_lbl.config(bg=self.colors["surface_light"])
            self.text_lbl.config(bg=self.colors["surface_light"])
            
    def _on_leave(self, e):
        if not self._active:
            self.config(bg=self.colors["bg"])
            self.icon_lbl.config(bg=self.colors["bg"])
            self.text_lbl.config(bg=self.colors["bg"])
            
    def set_active(self, is_active):
        self._active = is_active
        bg_col = self.colors.get("info", "#0EA5E9") if is_active else self.colors["bg"]
        fg_col = "white" if is_active else self.colors["text_muted"]
        self.config(bg=bg_col)
        self.icon_lbl.config(bg=bg_col)
        self.text_lbl.config(bg=bg_col, fg=fg_col)


def active_cnn_path() -> Path:
    """The CNN checkpoint in use: the one activated on AI Model Select (saved_models/active_models.json),
    otherwise the delivered one."""
    try:
        with open(app_paths.ACTIVE_MODELS_PATH, "r", encoding="utf-8") as f:
            path = Path(MODEL_ROOT) / str(json.load(f).get("cnn_model", ""))
        if path.is_file():
            return path
    except (OSError, ValueError):
        pass
    return CNN_MODEL_PATH


# ---------------- Main GUI Application ----------------

def _software_version() -> str:
    """Short git commit of this folder (what is really running), or 'unknown'."""
    try:
        import subprocess
        out = subprocess.run(["git", "-C", str(app_paths.BASE_DIR), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=2)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TCM CASHEW X AI")
        if os.environ.get("GSN_UI_OFFSCREEN"):
            # Self-screenshot tests (ui_debug.py): a fixed 1920 x 1080 window outside the visible screen, set before the
            # window is first shown so the window manager never maximises or moves it.
            self.overrideredirect(True)
            self.geometry(f"1920x1080+{self.winfo_screenwidth() + 2000}+0")
        else:
            self.geometry("1800x900+10+10")
            self.attributes("-zoomed", True)
        self.configure(bg=ui.BG)
        
        self.colors = {     # the palette of ui_theme.py under the names the older pages use
            "bg": ui.BG, "surface": ui.PANEL, "surface_light": ui.BORDER,
            "text": ui.TEXT, "text_muted": ui.MUTED,
            "success": ui.GOOD, "danger": ui.BAD, "warning": ui.WARN, "info": ui.ACCENT,
            "accent": ui.ACCENT
        }
        
        self._last_fires_state = []
        
        saved_model = ""
        try:
            if os.path.exists(CFG_PATH):
                with open(CFG_PATH, "r") as f:
                    saved_model = str(json.load(f).get("yolo_model_path", ""))
        except Exception:
            log.exception("Could not read yolo_model_path from %s", CFG_PATH)

        def _load_icon(path, size):
            try:
                if path.exists():
                    return ImageTk.PhotoImage(Image.open(path).resize((size, size), Image.Resampling.LANCZOS))
            except Exception:
                log.exception("Could not load icon %s", path)
            return None

        self.img_test = _load_icon(app_paths.ASSETS_DIR / "btn_test.png", 22)
        self.img_gear = _load_icon(app_paths.ASSETS_DIR / "btn_gear.png", 22)
        if self.img_test is None or self.img_gear is None:
            self.img_test, self.img_gear = None, None
            
        icon_names = {
            "Home": "Production.png",
            "Pneumatic": "Pneumatics.png",
            "Colour Size": "Colour_Size.png",
            "Vision ROI": "Vision_ROI.png",
            "Teaching CNN": "TCM_Studio.png",
            "Dataset Capture": "Dataset_Capture.png",
            "Teaching YOLO": "Teaching_Console.png",
            "AI Model Select": "AI_Model.png",
            "Test Console": "Test_Console.png",
            "Network": "Network.png"
        }
        self.nav_icons = {name: _load_icon(app_paths.NAV_ICONS_DIR / file, 32) for name, file in icon_names.items()}
        
        # YOLO model: the one saved in the config (stored relative to this folder), else the newest in saved_models.
        self.model_path = ""
        if saved_model:
            for candidate in (app_paths.resolve(saved_model), MODEL_ROOT / os.path.basename(saved_model)):
                if candidate.exists():
                    self.model_path = str(candidate)
                    break
        if not self.model_path:
            if saved_model:
                log.warning("YOLO model from the config not found: %s", saved_model)
            found = find_best_weights(MODEL_ROOT)
            self.model_path = str(found) if found else ""
        log.info("YOLO model path: %s", self.model_path or "(none)")

        # Class names: YOLO IDs from model.names. cashew_class_mapping.json only adds names
        # (classes the user manages); its numbers are never used as YOLO IDs.
        yolo_model, yolo_names = None, {}
        if self.model_path:
            try:
                yolo_model = YOLO(self.model_path)
                yolo_names = {int(k): str(v) for k, v in yolo_model.names.items()}
            except Exception:
                log.exception("Failed to load YOLO model %s", self.model_path)
                yolo_model, self.model_path = None, ""
        if not yolo_names:
            log.warning("No YOLO model. Starting in image capture only mode.")
            yolo_names = {0: "Capture_Only_Mode"}
        self.registry = ClassRegistry(yolo_names)

        if os.path.exists(CLASS_MAPPING_PATH):
            try:
                with open(CLASS_MAPPING_PATH, "r") as f:
                    json_map = {int(k): str(v) for k, v in json.load(f).items()}
                moved = {k: v for k, v in json_map.items() if self.registry.id_for(v) not in (None, k)}
                if moved:
                    log.warning("cashew_class_mapping.json numbers differ from the YOLO model for %s; the model's IDs are used.", moved)
                self.registry.add_names(json_map[k] for k in sorted(json_map))
            except Exception:
                log.exception("Could not read %s", CLASS_MAPPING_PATH)

        # GUI-side copy of the class list (the processing thread uses self.registry directly).
        self.class_map = self.registry.snapshot()
        self.all_class_names = list(self.class_map.values())

        # Add hybrid size classes to allow UI mapping for individual valves
        for sz in ["w180", "w210", "w240", "w320", "w450", "budds"]:
            self.all_class_names.extend([f"ww_{sz}", f"sw_{sz}"])
            
        self.shared_state = SharedState(self.all_class_names)
        
        # Sync loaded capture_classes with all_class_names
        for v in self.shared_state.capture_classes.values():
            if v and v not in self.all_class_names and not v.startswith("Class_"):
                self.all_class_names.append(v)
                
        if InovancePLCController is not None:
            self.controller = InovancePLCController()
            log.info("[CONTROLLER] Inovance EC320 PLC controller is the pneumatic driver.")
        else:
            self.controller = ArduinoMegaController()
            log.info("[CONTROLLER] Fallback to Arduino Mega controller.")
        
        self.fire_counts = {cid: 0 for cid in self.class_map.keys()}
        self.fire_lock = threading.Lock()
        
        self.stop_event = threading.Event()
        self.camera_queue = queue.Queue(maxsize=2)
        self.display_queue = queue.Queue(maxsize=2)
        
        self.flat_field = FlatField(app_paths.FFC_GAIN_MAP_PATH)
        self.camera_thread = CameraThread(self.camera_queue, self.stop_event, self.shared_state.camera_settings, self.flat_field)
        # Every relay pulse of the grader goes through this object (relays 1-40 only, dry-run, master switch).
        # RUNNING = grading fires the valves; STOPPED = only the manual buttons do (relay_output.py enforces both).
        self.relay_output = RelayOutput(self.controller, is_running=lambda: self.shared_state.phase == "RUNNING")
        self.relay_output.manual_log_dir = app_paths.LOGS_DIR     # every manual fire is logged
        # Automatic ejector check (tool window): only through relay_output, only when STOPPED with the conveyor OFF.
        self.ejector_check = ejector_check.EjectorCheck(self.relay_output, log_dir=app_paths.LOGS_DIR, safe_to_run=self._ejector_check_blocked)
        self.processing_thread = ProcessingThread(
            self.model_path, self.registry, self.camera_queue, self.display_queue, 
            self.stop_event, self.shared_state, self.relay_output, self.fire_counts, self.fire_lock,
            model=yolo_model
        )
        
        self.current_video_geometry = {}
        
        self.active_page = tk.StringVar(value="Home")
        self.pages = {}
        
        self._setup_styles()
        self._build_gui()
        
        self.controller.start()
        self.camera_thread.start()
        self.processing_thread.start()
        
        self._update_loop()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use('clam')
        
        bg = self.colors["bg"]
        surf = self.colors["surface"]
        text = self.colors["text"]
        
        style.configure(".", background=bg, foreground=text, font=("Roboto", 10))
        style.configure("TFrame", background=bg)
        style.configure("Surface.TFrame", background=surf)
        style.configure("TLabel", background=bg, foreground=text)
        style.configure("Surface.TLabel", background=surf, foreground=text)
        style.configure("Header.TLabel", background=bg, foreground=text, font=("Roboto", 12, "bold"))
        style.configure("Title.TLabel", background=bg, foreground=self.colors["info"], font=("Roboto", 14, "bold"))
        
        style.configure("TButton", background=self.colors["surface_light"], foreground=text, font=("Roboto", 10, "bold"), borderwidth=0, padding=6)
        style.map("TButton", background=[('active', self.colors["info"])])
        
        style.configure("Nav.TButton", background=bg, foreground=self.colors["text_muted"], font=("Roboto", 11, "bold"), borderwidth=0, padding=12)
        style.map("Nav.TButton", background=[('active', surf)], foreground=[('active', text)])
        
        style.configure("Primary.TButton", background=self.colors["info"], foreground=text, font=("Roboto", 10, "bold"), borderwidth=0, padding=6)
        style.map("Primary.TButton", background=[('active', '#0284C7')])
        
        style.configure("Success.TButton", background=self.colors["success"], foreground=text, font=("Roboto", 12, "bold"), padding=10)
        style.map("Success.TButton", background=[('active', '#059669')])
        
        style.configure("Danger.TButton", background=self.colors["danger"], foreground=text, font=("Roboto", 12, "bold"), padding=10)
        style.map("Danger.TButton", background=[('active', '#DC2626')])
        
        style.configure("Treeview", background=surf, foreground=text, fieldbackground=surf, borderwidth=0, rowheight=30)
        style.configure("Treeview.Heading", background=self.colors["surface_light"], foreground=self.colors["info"], font=("Roboto", 10, "bold"), borderwidth=0)
        style.layout("Treeview", [("Treeview.treearea", {"sticky": "nswe"})])      # no light frame around tables
        style.configure("Home.Treeview", rowheight=26, font=("Roboto", 10), background=ui.PANEL, fieldbackground=ui.PANEL, foreground=ui.TEXT, borderwidth=0)
        style.configure("Home.Treeview.Heading", font=("Roboto", 9, "bold"), background=ui.BORDER, foreground=ui.ACCENT, borderwidth=0)
        style.layout("Home.Treeview", [("Treeview.treearea", {"sticky": "nswe"})])
        style.map("Home.Treeview", background=[("selected", ui.mix(ui.PANEL, ui.ACCENT, 0.3))])
        for bar in ("Vertical.TScrollbar", "Horizontal.TScrollbar"):
            style.configure(bar, background=ui.BORDER, troughcolor=ui.FIELD, bordercolor=ui.FIELD, lightcolor=ui.BORDER,
                            darkcolor=ui.BORDER, arrowcolor=ui.MUTED, relief="flat", borderwidth=0)
            style.map(bar, background=[("active", ui.IDLE), ("disabled", ui.FIELD)])
        style.map("Treeview", background=[('selected', self.colors["surface_light"])])

        style.configure("TEntry", fieldbackground=self.colors["surface_light"], foreground=text, borderwidth=0, padding=5)
        style.configure("TCombobox", fieldbackground=self.colors["surface_light"], foreground="black", borderwidth=0, padding=5)
        self.option_add('*TCombobox*Listbox.foreground', 'black')

    def _build_gui(self):
        """Window frame in the style of the 5-track machine: top bar with status pills, left menu,
        camera in the centre, page column on the right."""
        F = ui.FONT
        self.configure(bg=ui.BG)

        # ── top bar ──
        top = tk.Frame(self, bg=ui.PANEL, height=ui.TOP_H)
        top.pack(side=tk.TOP, fill=tk.X)
        top.pack_propagate(False)
        tk.Frame(self, bg=ui.BORDER, height=1).pack(side=tk.TOP, fill=tk.X)

        brand = tk.Frame(top, bg=ui.PANEL, width=ui.NAV_W)      # customer logo goes here when it is given
        brand.pack(side=tk.LEFT, fill=tk.Y)
        brand.pack_propagate(False)
        tk.Label(brand, text="TCM", font=(F, ui.F_BODY, "bold"), fg=ui.ACCENT, bg=ui.PANEL).pack(anchor="w", padx=16, pady=(12, 0))
        tk.Label(brand, text="CASHEW X AI", font=(F, ui.F_SMALL, "bold"), fg=ui.TEXT, bg=ui.PANEL).pack(anchor="w", padx=16)

        right = tk.Frame(top, bg=ui.PANEL)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10))
        for kind, label, colour, command in (("power", "SHUTDOWN", ui.BAD, self._on_shutdown), ("exit", "EXIT", ui.MUTED, self._on_exit)):
            col = tk.Frame(right, bg=ui.PANEL)
            col.pack(side=tk.RIGHT, padx=6)
            ui.IconButton(col, kind, ui.PANEL, command).pack(pady=(3, 0))
            tk.Label(col, text=label, font=(F, -9, "bold"), fg=colour, bg=ui.PANEL).pack()
        clock = tk.Frame(right, bg=ui.PANEL, width=150)
        clock.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10))
        clock.pack_propagate(False)
        self.lbl_clock = tk.Label(clock, text="00:00:00", font=(F, ui.F_HEAD, "bold"), fg=ui.TEXT, bg=ui.PANEL)
        self.lbl_clock.pack(anchor="e", pady=(9, 0))
        self.lbl_date = tk.Label(clock, text="", font=(F, ui.F_SMALL), fg=ui.MUTED, bg=ui.PANEL)
        self.lbl_date.pack(anchor="e")

        prog = tk.Frame(top, bg=ui.PANEL, highlightthickness=1, highlightbackground=ui.BORDER)
        prog.pack(side=tk.LEFT, padx=(12, 8), pady=13)
        tk.Label(prog, text="PROGRAM", font=(F, ui.F_SMALL), fg=ui.MUTED, bg=ui.PANEL).pack(side=tk.LEFT, padx=(10, 6), pady=8)
        tk.Label(prog, text="Default", font=(F, ui.F_SMALL, "bold"), fg=ui.TEXT, bg=ui.PANEL).pack(side=tk.LEFT)
        tk.Frame(prog, bg=ui.PANEL, width=10).pack(side=tk.LEFT)      # NEW / COPY / DELETE come with the programs (step e3)

        self.pills = {}
        self._pill_bar = top
        for key, name, short in (("camera", "CAMERA", "CAM"), ("plc", "PLC", "PLC"), ("safety", "SAFETY", "SAFETY"),
                                 ("conveyor", "CONVEYOR", "CONV"), ("state", "STATE", "STATE"),
                                 ("check", "EJECTOR CHECK", "EJ CHECK")):
            pill = ui.Pill(top, name, "—", ui.MUTED, short=short)
            pill.pack(side=tk.LEFT, padx=3, pady=13)
            self.pills[key] = pill
        top.bind("<Configure>", lambda e: self._fit_pills())

        # The old layout wrote to these labels; the pills show the same facts now.
        self.cam_indicator = self.ard_indicator = self.lbl_state_badge = self.lbl_prod_count = ui.Sink()

        # ── body ──
        body_frm = tk.Frame(self, bg=ui.BG)
        body_frm.pack(fill=tk.BOTH, expand=True)

        nav_frm = tk.Frame(body_frm, bg=ui.PANEL, width=ui.NAV_W)
        nav_frm.pack(side=tk.LEFT, fill=tk.Y)
        nav_frm.pack_propagate(False)
        tk.Frame(body_frm, bg=ui.BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y)
        tk.Frame(nav_frm, bg=ui.PANEL, height=14).pack(fill=tk.X)
        nav_items = ["Home", "Pneumatic", "Sorting Logic", "Colour Size", "Vision ROI", "Teaching CNN", "Dataset Capture", "Teaching YOLO", "AI Model Select", "Test Console", "Network", "Info"]
        self.nav_buttons = {}
        for item in nav_items:
            btn = ui.NavItem(nav_frm, item, command=lambda i=item: self._switch_page(i))
            btn.pack(fill=tk.X)
            self.nav_buttons[item] = btn
        foot = tk.Frame(nav_frm, bg=ui.PANEL)
        foot.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 12))
        tk.Frame(foot, bg=ui.BORDER, height=1).pack(fill=tk.X, padx=10, pady=(0, 8))
        # Product logo scaled to the menu width (the file is used as it is), then the machine name and the running commit.
        self._logo_image = None
        try:
            logo = Image.open(app_paths.ASSETS_DIR / "cashew_x_ai_logo_clean.png").convert("RGB")
            width = ui.NAV_W - 12
            logo = logo.resize((width, max(1, round(logo.height * width / logo.width))), Image.LANCZOS)
            self._logo_image = ImageTk.PhotoImage(logo)
            tk.Label(foot, image=self._logo_image, bg=ui.PANEL, bd=0).pack(pady=(0, 6))
        except Exception:
            log.warning("Logo assets/cashew_x_ai_logo_clean.png could not be loaded; text is shown instead")
            tk.Label(foot, text="TCM CASHEW X AI", font=(F, ui.F_BODY, "bold"), fg=ui.TEXT, bg=ui.PANEL).pack(anchor="w", padx=14)
        tk.Label(foot, text="GSN V21", font=(F, -12, "bold"), fg=ui.TEXT, bg=ui.PANEL).pack(anchor="w", padx=14)
        tk.Label(foot, text=f"Software {_software_version()}", font=(F, -10), fg=ui.MUTED, bg=ui.PANEL).pack(anchor="w", padx=14, pady=(1, 0))

        self.center_frm = tk.Frame(body_frm, bg=ui.PANEL, highlightthickness=1, highlightbackground=ui.BORDER)
        self.center_frm.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=16, pady=16)

        # Lane LED strip under the camera: one LED per ROI, green / red for the last decision.
        led_bar = tk.Frame(self.center_frm, bg=ui.PANEL, height=ui.LANE_BAR_H)
        led_bar.pack(side=tk.BOTTOM, fill=tk.X)
        led_bar.pack_propagate(False)
        self.lane_leds = {}
        for rid in (1, 2):
            cell = tk.Frame(led_bar, bg=ui.PANEL)
            cell.pack(side=tk.LEFT, expand=True)
            led = ui.LedLabel(cell, ui.PANEL)
            led.pack(side=tk.LEFT, padx=(0, 8))
            tk.Label(cell, text=f"ROI {rid}", font=(F, -11), fg=ui.MUTED, bg=ui.PANEL).pack(side=tk.LEFT)
            self.lane_leds[rid] = led

        self.video_label = tk.Label(self.center_frm, bg="#05080D")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.video_label.bind("<Button-1>", self._on_mouse)
        self.video_label.bind("<B1-Motion>", self._on_mouse)
        self.video_label.bind("<ButtonRelease-1>", self._on_mouse)

        self.fps_lbl = tk.Label(self.video_label, text="CAM 0 fps", font=(F, ui.F_SMALL), bg=ui.PANEL, fg=ui.MUTED, padx=10, pady=4)
        self.fps_lbl.place(relx=0.01, rely=0.99, anchor="sw")

        self.right_frm = tk.Frame(body_frm, bg=ui.BG, width=ui.RIGHT_W)
        self.right_frm.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 16), pady=16)
        self.right_frm.pack_propagate(False)

        self.page_container = tk.Frame(self.right_frm, bg=ui.BG)
        self.page_container.pack(fill=tk.BOTH, expand=True)

        self._build_pages()
        self._build_teaching_console_page()
        self._build_colour_size_page()
        self._build_test_console_page()
        self._build_info_page()
        self._switch_page("Home")

    def _fit_pills(self):
        """Keep the status pills inside the top bar: tighter padding, then short names, then a smaller font."""
        bar = self._pill_bar
        if bar.winfo_width() < 400:         # the window has not been laid out yet
            return
        level = getattr(self, "_pill_level", 0)
        while True:     # values change while running (and get longer), so the fit is checked again and again; it only tightens
            bar.update_idletasks()
            if level >= 3 or sum(child.winfo_reqwidth() + 14 for child in bar.winfo_children()) <= bar.winfo_width():
                break
            level += 1
            for pill in self.pills.values():
                pill.set_level(level)
        self._pill_level = level

    # ------------------------------------------------------------------ Pneumatic page (5-track layout, 2 ROIs x 20 ejectors)
    def _build_pneumatic_page(self):
        F = ui.FONT
        page = tk.Frame(self.page_container, bg=ui.BG)
        self.pages["Pneumatic"] = page
        self.act_vars, self.dist_vars, self.pn_map_lbls, self.pn_leds, self.map_lbls = {}, {}, {}, {}, getattr(self, "map_lbls", {})
        self._pn_status, self._pn_fires, self.pulse_vars = {}, {}, {}
        self._recalc_active = False

        def card(parent, title, note="", **pack):
            box = tk.Frame(parent, bg=ui.PANEL, highlightthickness=1, highlightbackground=ui.BORDER)
            box.pack(**pack)
            head = tk.Frame(box, bg=ui.PANEL)
            head.pack(fill=tk.X, padx=14, pady=(10, 4))
            tk.Label(head, text=title, font=(F, ui.F_SMALL, "bold"), fg=ui.MUTED, bg=ui.PANEL).pack(side=tk.LEFT)
            if note:
                tk.Label(head, text=note, font=(F, -11), fg=ui.MUTED, bg=ui.PANEL).pack(side=tk.LEFT, padx=12)
            return box, head

        def field(parent, var, width=6, **kw):
            return tk.Entry(parent, textvariable=var, width=width, bg=ui.FIELD, fg=ui.TEXT, insertbackground=ui.TEXT, bd=0, justify="center",
                            font=(F, -13), highlightthickness=1, highlightbackground=ui.BORDER, highlightcolor=ui.ACCENT, **kw)

        # ── right column: tools ──
        tools, _ = card(page, "EJECTOR TOOLS", side=tk.RIGHT, fill=tk.Y, padx=(12, 0))
        tools.config(width=318)
        tools.pack_propagate(False)
        pad = dict(fill=tk.X, padx=14)
        ui.FlatButton(tools, "RUN EJECTOR CHECK 1-40", command=self._open_ejector_check, height=34).pack(pady=(2, 6), **pad)
        ui.FlatButton(tools, "OPEN PLC DIAGNOSTICS", command=self._open_plc_diagnostics, height=34).pack(pady=(0, 8), **pad)
        tk.Frame(tools, bg=ui.BORDER, height=1).pack(pady=4, **pad)

        def number_row(label, var, unit=""):
            row = tk.Frame(tools, bg=ui.PANEL)
            row.pack(pady=3, **pad)
            tk.Label(row, text=label, font=(F, -13), fg=ui.TEXT, bg=ui.PANEL).pack(side=tk.LEFT)
            tk.Label(row, text=unit, font=(F, -11), fg=ui.MUTED, bg=ui.PANEL, width=5, anchor="w").pack(side=tk.RIGHT)
            field(row, var, 7).pack(side=tk.RIGHT, padx=(0, 4), ipady=2)

        self.weight_var = tk.StringVar(value=f"{self.shared_state.grading.get('avg_cashew_weight_g', 1.8):g}")
        number_row("AVG CASHEW WEIGHT", self.weight_var, "g")
        self.common_pulse_var = tk.StringVar(value=str(self.shared_state.common_pulse_ms))
        number_row("Pulse: set all ejectors", self.common_pulse_var, "ms")
        self.common_pulse_var.trace_add("write", self._update_common_pulse)
        self.conveyor_speed_var = tk.StringVar(value=str(int(getattr(self.shared_state, 'conveyor_speed_mm_s', 300))))
        number_row("Conveyor speed", self.conveyor_speed_var, "mm/s")
        self.conveyor_speed_var.trace_add("write", self._update_conveyor_speed)
        ui.FlatButton(tools, "RECALCULATE DELAYS", command=self._recalculate_all_from_distances, kind="muted", height=30).pack(pady=(4, 2), **pad)
        tk.Label(tools, text="delay = distance / speed - pulse / 2", font=(F, -11), fg=ui.MUTED, bg=ui.PANEL).pack(**pad)
        tk.Frame(tools, bg=ui.BORDER, height=1).pack(pady=8, **pad)

        tk.Label(tools, text="START / STOP", font=(F, ui.F_SMALL, "bold"), fg=ui.MUTED, bg=ui.PANEL, anchor="w").pack(**pad)
        self.lbl_pn_state = tk.Label(tools, text="STOPPED", font=(F, ui.F_HEAD, "bold"), fg=ui.MUTED, bg=ui.PANEL, anchor="w")
        self.lbl_pn_state.pack(**pad)
        tk.Label(tools, text="RUNNING: grading fires the valves, the buttons here are locked.\nSTOPPED: nothing fires from grading, the buttons here fire the REAL valves.",
                 font=(F, -11), fg=ui.MUTED, bg=ui.PANEL, anchor="w", justify="left", wraplength=286).pack(**pad)
        tk.Frame(tools, bg=ui.BORDER, height=1).pack(pady=8, **pad)

        # MANUAL FIRE: one ejector, all at once, one after the other. They write to the real PLC, only when STOPPED.
        tk.Label(tools, text="MANUAL FIRE", font=(F, ui.F_SMALL, "bold"), fg=ui.MUTED, bg=ui.PANEL, anchor="w").pack(**pad)
        row = tk.Frame(tools, bg=ui.PANEL)
        row.pack(pady=4, **pad)
        tk.Label(row, text="Ejector", font=(F, -13), fg=ui.TEXT, bg=ui.PANEL).pack(side=tk.LEFT)
        self.test_relay_var = tk.StringVar(value="1")
        ttk.Combobox(row, textvariable=self.test_relay_var, values=[str(r) for r in range(FIRST_EJECTOR, LAST_EJECTOR + 1)], width=4,
                     state="readonly").pack(side=tk.LEFT, padx=(6, 10))
        tk.Label(row, text="Pulse", font=(F, -13), fg=ui.TEXT, bg=ui.PANEL).pack(side=tk.LEFT)
        self.test_pulse_var = tk.StringVar(value="100")
        field(row, self.test_pulse_var, 5).pack(side=tk.LEFT, padx=(6, 4), ipady=2)
        tk.Label(row, text="ms", font=(F, -11), fg=ui.MUTED, bg=ui.PANEL).pack(side=tk.LEFT)
        self.lbl_test_relay_addr = tk.Label(tools, text="", bg=ui.PANEL, fg=ui.ACCENT, font=(F, -12, "bold"), anchor="w", justify="left", wraplength=286)
        self.lbl_test_relay_addr.pack(**pad)

        def show_test_relay(*_):
            try:
                relay = int(self.test_relay_var.get())
            except (ValueError, tk.TclError):
                return
            role = "REJECT ejector" if relay_mapping.is_reject(relay) else relay_mapping.label(self.shared_state.mapping_by_act, relay)
            self.lbl_test_relay_addr.config(text=f"{self.relay_output.describe(relay)}\nROI {relay_mapping.roi_of(relay)},  {role}")
        self.test_relay_var.trace_add("write", show_test_relay)
        show_test_relay()

        def step_relay(step):
            try:
                relay = int(self.test_relay_var.get())
            except (ValueError, tk.TclError):
                relay = FIRST_EJECTOR
            self.test_relay_var.set(str(min(LAST_EJECTOR, max(FIRST_EJECTOR, relay + step))))
        nav = tk.Frame(tools, bg=ui.PANEL)
        nav.pack(pady=(4, 6), **pad)
        ui.FlatButton(nav, "◀", command=lambda: step_relay(-1), kind="muted", height=34, width=40).pack(side=tk.LEFT)
        ui.FlatButton(nav, "▶", command=lambda: step_relay(1), kind="muted", height=34, width=40).pack(side=tk.RIGHT)
        ui.FlatButton(nav, "TEST FIRE THIS EJECTOR", command=self._test_fire_selected_relay, kind="danger", height=34).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        both = tk.Frame(tools, bg=ui.PANEL)
        both.pack(**pad)
        ui.FlatButton(both, "FIRE ALL", command=self._fire_all, kind="danger", height=32, width=138).pack(side=tk.LEFT)
        ui.FlatButton(both, "SEQUENCE ALL", command=self._sequence_all, kind="danger", height=32, width=138).pack(side=tk.RIGHT)
        off = tk.Label(tools, text="■  ALL RELAYS OFF NOW", font=(F, ui.F_BODY, "bold"), fg="#FFFFFF", bg=ui.BAD, cursor="hand2", pady=7)
        off.pack(pady=(6, 0), **pad)
        off.bind("<Button-1>", lambda e: self._all_relays_off())
        tk.Label(tools, text="Only when STOPPED. FIRE ALL and SEQUENCE ALL ask first; a single pulse fires at once.", font=(F, -11), fg=ui.MUTED, bg=ui.PANEL,
                 anchor="w", justify="left", wraplength=286).pack(pady=(4, 0), **pad)
        self.lbl_test_relay_result = tk.Label(tools, text="", bg=ui.PANEL, fg=ui.MUTED, font=(F, -12), anchor="w", justify="left", wraplength=286)
        self.lbl_test_relay_result.pack(pady=(4, 0), **pad)

        ui.FlatButton(tools, "SAVE PNEUMATICS", command=self._save_pneumatics, kind="primary", height=44, font=(F, ui.F_BODY, "bold")).pack(side=tk.BOTTOM, pady=(6, 14), **pad)
        self.lbl_pn_saved = tk.Label(tools, text="", font=(F, -12, "bold"), fg=ui.MUTED, bg=ui.PANEL, anchor="w")
        self.lbl_pn_saved.pack(side=tk.BOTTOM, **pad)

        # ── left: the two ejector tables ──
        left = tk.Frame(page, bg=ui.BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tables = tk.Frame(left, bg=ui.BG)
        tables.pack(fill=tk.X)
        columns = (("", 2), ("EJ", 4), ("PLC", 5), ("CLASSES / GRADES", 27), ("DIST mm", 7), ("DELAY", 7), ("PULSE", 5), ("FIRE", 4), ("LAST FIRE", 15), ("FIRES", 5))
        for block, (title, relays) in enumerate((("EJECTORS  ROI 1", range(1, 21)), ("EJECTORS  ROI 2", range(21, 41)))):
            box, _ = card(tables, title, "click the classes of a row to change them" if block == 0 else "", side=tk.LEFT, fill=tk.BOTH, expand=True,
                          padx=(0, 6) if block == 0 else (6, 0))
            grid = tk.Frame(box, bg=ui.PANEL)
            grid.pack(fill=tk.X, padx=10, pady=(0, 8))
            for col, (text, width) in enumerate(columns):
                tk.Label(grid, text=text, font=(F, -10, "bold"), fg=ui.MUTED, bg=ui.PANEL, width=width, anchor="w").grid(row=0, column=col, sticky="w", pady=(0, 2))
            for r, i in enumerate(relays, start=1):
                reject = relay_mapping.is_reject(i)
                led = tk.Label(grid, text="●", font=(F, -11), fg=ui.BORDER, bg=ui.PANEL)
                led.grid(row=r, column=0)
                self.pn_leds[i] = led
                tk.Label(grid, text=f"A{i:02d}", font=(F, -13, "bold"), fg=ui.WARN if reject else ui.TEXT, bg=ui.PANEL, anchor="w").grid(row=r, column=1, sticky="w")
                address = self.relay_output.address_of(i)
                tk.Label(grid, text=f"M{address}" if address is not None else "?", font=(F, -12), fg=ui.MUTED, bg=ui.PANEL, anchor="w").grid(row=r, column=2, sticky="w")
                classes = ttk.Label(grid, text="", width=30, anchor="w", font=(F, -12, "bold"), background=ui.PANEL, cursor="arrow" if reject else "hand2")
                classes.grid(row=r, column=3, sticky="w")
                classes.bind("<Button-1>", lambda e, x=i: self._popup_mapper(x))
                self.pn_map_lbls[i] = classes
                dist = tk.StringVar(value=str(self.shared_state.distances_mm[i - 1]))
                field(grid, dist, 7).grid(row=r, column=4, sticky="w", pady=1)
                delay = tk.StringVar(value=str(self.shared_state.delays_ms[i - 1]))
                field(grid, delay, 7).grid(row=r, column=5, sticky="w", pady=1)
                self.dist_vars[i], self.act_vars[i] = dist, delay
                pulse = tk.StringVar(value=str(self.shared_state.pulses_ms[i - 1]))
                field(grid, pulse, 5).grid(row=r, column=6, sticky="w", pady=1)
                self.pulse_vars[i] = pulse
                pulse.trace_add("write", lambda *_, ii=i: self._update_row_pulse(ii))
                # lightning button: one click fires this ejector's PLC coil at once, with the pulse of this row
                ui.FlatButton(grid, "⚡", command=lambda x=i: self._pulse_and_flash(x), kind="danger", height=22, width=34, font=(F, -12, "bold")).grid(row=r, column=7, sticky="w", padx=(4, 4))
                status = tk.Label(grid, text="", font=(F, -11), fg=ui.MUTED, bg=ui.PANEL, anchor="w", width=17)
                status.grid(row=r, column=8, sticky="w")
                fires = tk.Label(grid, text="0", font=(F, -12, "bold"), fg=ui.TEXT, bg=ui.PANEL, anchor="e", width=5)
                fires.grid(row=r, column=9, sticky="e")
                self._pn_status[i], self._pn_fires[i] = status, fires
                dist.trace_add("write", lambda *_, ii=i: self._on_distance_changed(ii))
                delay.trace_add("write", lambda *_, ii=i: self._update_act_timing(ii))

        # ── class chips: PASS (counts as no defect) or FIRE (its ejector, or the reject ejector) ──
        chips_box, chips_head = card(left, "CLASSES", "click: PASS = no defect,  FIRE = a defect (to its ejector, or REJECT)", fill=tk.X, pady=(10, 0))
        self._chip_frame = tk.Frame(chips_box, bg=ui.PANEL)
        self._chip_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
        self._chip_signature = None

        low = tk.Frame(left, bg=ui.BG)
        low.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        # ── inspection engines ──
        eng_box, _ = card(low, "INSPECTION ENGINES", "SHADOW = measured, \"would reject\" is logged, never fires", side=tk.LEFT, fill=tk.BOTH, padx=(0, 6))
        eng_box.config(width=610)
        eng_box.pack_propagate(False)
        self._engine_switches, self._engine_counts = {}, {}
        for name, (title, looks_for, measuring) in engines.ENGINES.items():
            row = tk.Frame(eng_box, bg=ui.PANEL)
            row.pack(fill=tk.X, padx=14, pady=3)
            text = tk.Frame(row, bg=ui.PANEL)
            text.pack(side=tk.LEFT, fill=tk.X, expand=True)
            tk.Label(text, text=title, font=(F, -13, "bold"), fg=ui.TEXT, bg=ui.PANEL, anchor="w").pack(fill=tk.X)
            count = tk.Label(text, text=looks_for, font=(F, -11), fg=ui.WARN if measuring else ui.MUTED, bg=ui.PANEL, anchor="w", justify="left", wraplength=370)
            count.pack(fill=tk.X)
            mode = {**engines.ENGINE_DEFAULTS, **self.shared_state.grading.get("engines", {})}[name]
            switch = ui.Segmented(row, list(engines.MODES), mode, command=lambda value, n=name: self._set_engine(n, value),
                                  colours={engines.OFF: ui.IDLE, engines.SHADOW: ui.WARN, engines.ACTIVE: ui.GOOD})
            switch.pack(side=tk.RIGHT)
            self._engine_switches[name], self._engine_counts[name] = switch, count
        tk.Label(eng_box, text="Always reject, cannot be switched off: Double / touching (longer than 45 mm or CNN class DOUBLE), stuck, lost over 300 ms, seen after the line.",
                 font=(F, -11), fg=ui.MUTED, bg=ui.PANEL, anchor="w", justify="left", wraplength=580).pack(fill=tk.X, padx=14, pady=(4, 8))

        # ── recent ejector events + mapping notes ──
        ev_box, _ = card(low, "RECENT EJECTOR EVENTS", side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))
        style = ttk.Style(self)
        style.configure("Events.Treeview", rowheight=20, font=(F, -12), background=ui.PANEL, fieldbackground=ui.PANEL, foreground=ui.TEXT, borderwidth=0)
        style.configure("Events.Treeview.Heading", font=(F, -11, "bold"), background=ui.BORDER, foreground=ui.ACCENT, borderwidth=0)
        style.layout("Events.Treeview", [("Treeview.treearea", {"sticky": "nswe"})])
        self.lbl_map_problem = tk.Label(ev_box, text="", font=(F, -12, "bold"), fg=ui.BAD, bg=ui.PANEL, anchor="w", justify="left", wraplength=520)
        self.lbl_map_problem.pack(side=tk.BOTTOM, fill=tk.X, padx=14)
        self.lbl_map_unassigned = tk.Label(ev_box, text="", font=(F, -12, "bold"), fg=ui.WARN, bg=ui.PANEL, anchor="w", justify="left", wraplength=520)
        self.lbl_map_unassigned.pack(side=tk.BOTTOM, fill=tk.X, padx=14, pady=(4, 8))
        self.event_tree = ttk.Treeview(ev_box, columns=("time", "ej", "source", "delay", "pulse", "result"), show="headings", height=6, style="Events.Treeview")
        for col, text, width in (("time", "TIME", 92), ("ej", "EJ", 44), ("source", "CASHEW / SOURCE", 220), ("delay", "DELAY", 60), ("pulse", "PULSE", 54), ("result", "RESULT", 120)):
            self.event_tree.heading(col, text=text)
            self.event_tree.column(col, width=width, anchor="w")
        self.event_tree.tag_configure("sent", foreground=ui.GOOD)
        self.event_tree.tag_configure("not", foreground=ui.WARN)
        self.event_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 4))
        self._events_shown = None
        self._update_all_map_labels()
        self._refresh_pneumatic(force=True)

    def _set_engine(self, name, mode):
        """OFF / SHADOW / ACTIVE of one inspection engine. An engine that does not measure yet cannot be ACTIVE."""
        if name in ("sw_ww", "size"):
            messagebox.showinfo("Inspection engine", f"{engines.ENGINES[name][0]} is switched with SIZE GRADES READY on the Colour Size page:\n"
                                                     "off = SHADOW (the grade is only logged), on = ACTIVE (GOOD cashews go to their grade ejector).")
            self._engine_switches[name].set(self.shared_state.grading["engines"].get(name, engines.SHADOW))
            return
        if mode == engines.ACTIVE and not engines.ENGINES[name][2]:
            messagebox.showinfo("Inspection engine", f"{engines.ENGINES[name][0]} does not measure yet in this version: NOT AVAILABLE as ACTIVE.\nIt stays in SHADOW.")
            return
        if mode == engines.ACTIVE and not messagebox.askyesno("Inspection engine", f"Set {engines.ENGINES[name][0]} to ACTIVE?\n\nIt will then turn GOOD cashews into rejects. "
                                                             "Do this only after its \"would reject\" numbers were checked in SHADOW."):
            return
        g = copy.deepcopy(self.shared_state.grading)
        g.setdefault("engines", {})[name] = mode
        self.shared_state.set_grading(g)
        self._engine_switches[name].set(mode)
        log.info("Inspection engine %s -> %s", name, mode)

    def _toggle_class_pass(self, name):
        """A class chip was clicked: PASS (a "no defect" class) <-> FIRE (a defect class)."""
        g = copy.deepcopy(self.shared_state.grading)
        good = [c for c in g.get("good_classes", [])]
        if any(normalize_name(c) == normalize_name(name) for c in good):
            good = [c for c in good if normalize_name(c) != normalize_name(name)]
            if not any(normalize_name(c) == normalize_name(name) for c in g.get("priority", []) + g.get("reject_classes", [])):
                g["priority"] = list(g.get("priority", [])) + [name]
        else:
            if not messagebox.askyesno("PASS", f"Set {name} to PASS?\n\nA cashew of this class then counts as \"no defect\" and is graded as GOOD."):
                return
            good.append(name)
        g["good_classes"] = good
        self.shared_state.set_grading(g)
        self._refresh_pneumatic(force=True)

    def _refresh_pneumatic(self, force=False):
        """Chips, engine counts, last fire and fires per ejector, recent events (while the page is shown)."""
        if not force and not self.pages["Pneumatic"].winfo_ismapped():
            return
        F, ss, out = ui.FONT, self.shared_state, self.relay_output
        with ss.lock:
            g = copy.deepcopy(ss.grading)
            table = dict(ss.mapping_t1)
            engine_counts = dict(ss.engine_counts)
        good = {normalize_name(c) for c in g.get("good_classes", [])}
        fixed = [decision.REJECTION, decision.DOUBLE_CLASS] + list(g.get("reject_classes", []))
        names = list(g.get("priority", [])) + ["GOOD_CASHEW"] + [c for c in g.get("good_classes", []) if normalize_name(c) not in ("good cashew", "good")]
        seen, chips = set(), []
        for name in names + fixed:
            key = normalize_name(name)
            if key in seen:
                continue
            seen.add(key)
            if name in fixed:
                chips.append((name, "REJECT", ui.BAD, True))
            elif key in good:
                chips.append((name, "PASS", ui.GOOD, False))
            else:
                relay = table.get(key)
                chips.append((name, f"A{relay:02d}" if relay else "REJECT", ui.BAD if relay else ui.WARN, False))
        signature = tuple(chips)
        if signature != self._chip_signature:
            self._chip_signature = signature
            for child in self._chip_frame.winfo_children():
                child.destroy()
            for index, (name, tag, colour, is_fixed) in enumerate(chips):
                chip = tk.Frame(self._chip_frame, bg=ui.mix(ui.PANEL, colour, 0.14), highlightthickness=1, highlightbackground=ui.mix(ui.PANEL, colour, 0.5),
                                cursor="arrow" if is_fixed else "hand2")
                chip.grid(row=index // 7, column=index % 7, sticky="ew", padx=3, pady=3)
                self._chip_frame.columnconfigure(index % 7, weight=1, uniform="chip")
                label = tk.Label(chip, text=name, font=(F, -12, "bold"), fg=ui.TEXT, bg=chip.cget("bg"), anchor="w")
                label.pack(side=tk.LEFT, padx=(8, 4), pady=5)
                mark = tk.Label(chip, text=f" {tag} ", font=(F, -11, "bold"), fg="#06131A", bg=colour)
                mark.pack(side=tk.RIGHT, padx=(0, 6))
                if is_fixed:
                    tk.Label(chip, text="FIXED", font=(F, -8, "bold"), fg=ui.MUTED, bg=chip.cget("bg")).pack(side=tk.RIGHT, padx=(0, 4))
                else:
                    for widget in (chip, label, mark):
                        widget.bind("<Button-1>", lambda e, n=name: self._toggle_class_pass(n))
        modes = {**engines.ENGINE_DEFAULTS, **g.get("engines", {})}
        for name, (title, looks_for, measuring) in engines.ENGINES.items():
            self._engine_switches[name].set(modes[name])
            if not measuring:
                text = f"{looks_for}  -  not measuring yet: NOT AVAILABLE"
            elif modes[name] == engines.OFF:
                text = looks_for
            else:
                word = "would reject" if modes[name] == engines.SHADOW else "rejected"
                text = f"{word}: {engine_counts.get(name, 0)}" + ("   (shadow: logged only, nothing is fired)" if modes[name] == engines.SHADOW else "")
            self._engine_counts[name].config(text=text)

        wanted, sent, _ = out.counts()
        events = out.recent_events()
        last = {}
        for when, relay, delay_ms, pulse_ms, result, source in events:
            last.setdefault(relay, (when, result))
        texts = {"SENT": "OK", "PLC_OFFLINE": "PLC ERROR offline", "REFUSED": "REFUSED", "WRITE_FAIL": "PLC ERROR write", "READBACK_FAIL": "PLC ERROR read-back",
                 "CANCELLED": "cancelled", "NOT_STOPPED": "refused: RUNNING"}
        for relay in range(FIRST_EJECTOR, LAST_EJECTOR + 1):
            if relay in last:
                when, result = last[relay]
                self._pn_status[relay].config(text=f"{texts.get(result, result)}  {time.strftime('%H:%M:%S', time.localtime(when))}", fg=ui.GOOD if result == "SENT" else ui.WARN)
            else:
                self._pn_status[relay].config(text="(free)" if not relay_mapping.is_reject(relay) and not ss.mapping_by_act.get(relay) else "-", fg=ui.MUTED)
            real, asked = sent.get(relay, 0), wanted.get(relay, 0)
            self._pn_fires[relay].config(text=str(real) if real == asked else f"{real} ({asked})")
        key = (len(events), events[0][0] if events else 0)
        if key != self._events_shown:
            self._events_shown = key
            self.event_tree.delete(*self.event_tree.get_children())
            for when, relay, delay_ms, pulse_ms, result, source in events[:60]:
                stamp = time.strftime("%H:%M:%S", time.localtime(when)) + f".{int(when * 1000) % 1000:03d}"
                self.event_tree.insert("", "end", values=(stamp, f"A{relay:02d}" if isinstance(relay, int) else relay, source, f"{delay_ms:.0f}", f"{pulse_ms:.0f}",
                                                         texts.get(result, result)), tags=("sent" if result == "SENT" else "not",))
        running = ss.phase == "RUNNING"
        self.lbl_pn_state.config(text="RUNNING" if running else "STOPPED", fg=ui.GOOD if running else ui.MUTED)
        shown = getattr(self, "_manual_shown", None)        # result of the last manual pulse, next to the buttons
        if shown is not None and shown in out.last_result:
            when, result, error = out.last_result[shown]
            if when >= getattr(self, "_manual_at", 0.0):
                self._show_manual_result(self._result_text(shown, result) + (f"  ({error})" if error else ""), result == "SENT")
        dirty = getattr(self, "_mapping_dirty", False)
        self.lbl_pn_saved.config(text="●  mapping changed, not saved" if dirty else "All changes saved", fg=ui.WARN if dirty else ui.MUTED)

    # ------------------------------------------------------------------ Info page (only facts the software knows)
    def _build_info_page(self):
        F = ui.FONT
        page = tk.Frame(self.page_container, bg=ui.BG)
        self.pages["Info"] = page
        self._info_rows = {}

        def card(parent, title, side, width=None):
            box = tk.Frame(parent, bg=ui.PANEL, highlightthickness=1, highlightbackground=ui.BORDER, width=width or 1)
            box.pack(side=side, fill=tk.BOTH, expand=width is None, padx=(0, 0 if side == tk.RIGHT else 16), pady=(0, 16))
            if width:
                box.pack_propagate(False)
            tk.Label(box, text=title, font=(F, ui.F_SMALL, "bold"), fg=ui.MUTED, bg=ui.PANEL, anchor="w").pack(fill=tk.X, padx=20, pady=(16, 8))
            return box

        left = None

        def row(box, key, label):
            line = tk.Frame(box, bg=ui.PANEL)
            line.pack(fill=tk.X, padx=20, pady=5)
            tk.Label(line, text=label, font=(F, ui.F_SMALL), fg=ui.MUTED, bg=ui.PANEL, width=24, anchor="w").pack(side=tk.LEFT)
            value = tk.Label(line, text="-", font=(F, ui.F_BODY), fg=ui.TEXT, bg=ui.PANEL, anchor="w", justify="left",
                             wraplength=370 if box is left else 760)
            value.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self._info_rows[key] = value

        top = tk.Frame(page, bg=ui.BG)
        top.pack(fill=tk.BOTH, expand=True)
        left = card(top, "PRODUCT", tk.LEFT, width=620)
        self._info_logo = None
        try:
            logo = Image.open(app_paths.ASSETS_DIR / "cashew_x_ai_logo_clean.png").convert("RGB")
            logo = logo.resize((480, round(logo.height * 480 / logo.width)), Image.LANCZOS)
            self._info_logo = ImageTk.PhotoImage(logo)
            tk.Label(left, image=self._info_logo, bg=ui.PANEL, bd=0).pack(anchor="w", padx=20, pady=(0, 12))
        except Exception:
            pass
        tk.Label(left, text="TCM CASHEW X AI", font=(F, ui.F_KPI, "bold"), fg=ui.TEXT, bg=ui.PANEL, anchor="w").pack(fill=tk.X, padx=20)
        tk.Label(left, text="GSN V21  -  2 ROIs, 40 ejectors", font=(F, ui.F_BODY, "bold"), fg=ui.ACCENT, bg=ui.PANEL, anchor="w").pack(fill=tk.X, padx=20, pady=(0, 12))
        for key, label in (("version", "Software (git commit)"), ("program", "Active program"), ("mode", "Ejectors"),
                           ("auto", "AUTO LEARNING"), ("samples", "Confirmed samples"), ("capture", "Production capture")):
            row(left, key, label)

        right = tk.Frame(top, bg=ui.BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        hardware = card(right, "HARDWARE", tk.TOP)
        for key, label in (("camera", "Camera"), ("camera_state", "Camera state"), ("plc", "Controller"), ("plc_state", "Controller state"),
                           ("ejectors", "Ejectors"), ("timing", "Valve timing")):
            row(hardware, key, label)
        software = card(right, "SOFTWARE", tk.TOP)
        for key, label in (("yolo", "Detection model"), ("cnn", "Grading model"), ("engines", "Inspection engines"), ("tracking", "Tracking"), ("logs", "Logs")):
            row(software, key, label)
        today = card(right, "SINCE THE LAST RESET", tk.TOP)
        for key, label in (("counts", "Results"), ("check", "Ejector check")):
            row(today, key, label)

    def _refresh_info(self):
        if not self.pages["Info"].winfo_ismapped():
            return
        ss, cam, out = self.shared_state, self.camera_thread, self.relay_output
        info = cam.info or {}
        enabled, samples, target = self._auto_learning_status()
        with ss.lock:
            g = copy.deepcopy(ss.grading)
            counts = dict(ss.result_counts)
            capture_on = bool(ss.capture_cfg.get("enabled"))
        total = sum(n for k, n in counts.items() if not k.startswith(("UNDEFINED: ", decision.IGNORED_PREFIX)))
        values = {
            "version": _software_version(), "program": self.active_program_name,
            "mode": "fire from grading only while RUNNING; manual buttons only while STOPPED",
            "auto": ("ON" if enabled else "OFF") + "   (planned, not active in this version)",
            "samples": f"{ui.thousands(samples)} / {ui.thousands(target)}",
            "capture": ("ON" if capture_on else "OFF") + f"   ({self.processing_thread.capture.blocking_reason()})",
            "camera": f"{info.get('vendor', '')} {info.get('model', '-')}   serial {info.get('serial', '-')}   {info.get('pixel_format', '')}".strip(),
            "camera_state": f"{cam.state},  {cam.fps:.0f} fps",
            "plc": f"Inovance PLC, Modbus TCP {getattr(self.controller, 'plc_ip', '?')}:{getattr(self.controller, 'plc_port', '?')}, coils M100-M139",
            "plc_state": "ONLINE" if self.controller.is_connected() else "OFFLINE",
            "ejectors": "40 (ROI 1: 1-20, ROI 2: 21-40); 1 and 21 are the reject ejectors; M140 conveyor and M141 feeder are never used as ejectors",
            "timing": f"delay = distance / speed - pulse / 2;  speed {ss.conveyor_speed_mm_s:g} mm/s, pulse {ss.common_pulse_ms} ms (Pneumatic page)",
            "yolo": os.path.basename(str(ss.yolo_model_path or "-")),
            "cnn": f"{Path(self._active_cnn_path()).name}   ({', '.join(self.processing_thread.cnn_classes) or 'not loaded'})",
            "engines": ",  ".join(f"{engines.ENGINES[name][0]} {mode}" for name, mode in {**engines.ENGINE_DEFAULTS, **g.get('engines', {})}.items()),
            "tracking": "per ROI, cashew IDs R1-000001 ..., one decision at the Decision line",
            "logs": "logs/cashews_<date>.csv, frames_<date>.jsonl, manual_fire_<date>.csv, ejector_check_<date>.csv, admin_changes.log",
            "counts": f"decided {total}:   " + (",  ".join(f"{k} {n}" for k, n in sorted(counts.items()) if not k.startswith(("UNDEFINED: ", decision.IGNORED_PREFIX))) or "-"),
            "check": self.ejector_check.last_summary[0],
        }
        for key, text in values.items():
            if self._info_rows[key].cget("text") != text:
                self._info_rows[key].config(text=text)

    def _refresh_teaching(self):
        """Production capture settings to its thread; Teaching CNN page redrawn while it is shown."""
        try:
            self.processing_thread.capture.configure(self.shared_state.capture_cfg, self.active_program_name, app_paths.INBOX_DIR,
                                                     {"cnn": Path(self._active_cnn_path()).name,
                                                      "yolo": os.path.basename(str(self.shared_state.yolo_model_path or ""))})
        except Exception:
            _throttle.exception("capture_cfg", "Production capture could not be configured")
        if getattr(self, "model_page", None) is not None and self.pages["AI Model Select"].winfo_ismapped():
            self.model_page.refresh()
        self._refresh_info()
        page = getattr(self, "teach_page", None)
        if page is not None and self.pages["Teaching CNN"].winfo_ismapped():
            capturing = self.shared_state.is_golden_capture_active()
            self.tcm_auto_cap_btn.config(text="STOP AUTO CAPTURE" if capturing else "START AUTO CAPTURE")
            page.refresh()

    def _acknowledge_alarms(self):
        """ACKNOWLEDGE: the operator has removed the cashews that were between the line and the valves."""
        log.info("Alarm acknowledged: conveyor stopped with pending rejects %s", self.shared_state.missed_rejects)
        with self.shared_state.lock:
            self.shared_state.missed_rejects = []

    def _on_exit(self):
        if messagebox.askyesno("Exit", "Close the grader program?"):
            self._on_close()

    def _on_shutdown(self):
        if messagebox.askyesno("Shutdown", "Close the grader and SWITCH OFF this computer?"):
            import subprocess
            self._on_close(then=lambda: subprocess.Popen(["systemctl", "poweroff"]))

    def _refresh_status(self):
        """Top bar pills, Home tiles and lane LEDs (called a few times per second)."""
        ss, cam = self.shared_state, self.camera_thread
        cam_colour = ui.GOOD if cam.is_connected else (ui.WARN if cam.state in ("CONNECTING", "RECONNECTING") else ui.BAD)
        self.pills["camera"].set(str(cam.state), cam_colour)
        plc_ok = self.controller.is_connected()
        if plc_ok and getattr(self, "_plc_was_ok", None) is False:
            self.plc_reconnects += 1
        self._plc_was_ok = plc_ok
        self.pills["plc"].set("ONLINE" if plc_ok else "OFFLINE", ui.GOOD if plc_ok else ui.BAD)
        if getattr(self, "_was_running", False) and ss.phase != "RUNNING":      # STOP, however it happened: nothing fires from grading
            cancelled = self.relay_output.cancel_pending()
            if cancelled:
                log.info("STOP: %d waiting ejector pulse(s) cancelled", cancelled)
        self._was_running = ss.phase == "RUNNING"
        self.pills["conveyor"].set("ON" if ss.is_conveyor_on else "OFF", ui.GOOD if ss.is_conveyor_on else ui.MUTED)
        phase = ss.phase
        running = phase == "RUNNING"
        self.pills["state"].set("RUNNING" if running else ("STOPPED" if phase == "READY" else "SETUP"),
                                ui.GOOD if running else (ui.MUTED if phase == "READY" else ui.WARN))
        self.lbl_machine_state.config(text="RUNNING" if running else "STOPPED", fg=ui.GOOD if running else ui.BAD)
        check_text, check_level = self.ejector_check.last_summary
        if self.ejector_check.busy():
            check_text, check_level = f"RUNNING R{self.ejector_check.current_relay or '-'}", "warn"
        self.pills["check"].set(check_text, {"good": ui.GOOD, "warn": ui.WARN, "bad": ui.BAD}.get(check_level, ui.MUTED))

        counts = ss.get_result_counts()
        total = sum(n for key, n in counts.items() if not key.startswith(("UNDEFINED: ", decision.IGNORED_PREFIX)))
        good, undefined = counts.get(decision.GOOD, 0), counts.get(decision.UNDEFINED, 0)
        rejection = counts.get(decision.REJECTION, 0)
        self.kpi["total"].set(ui.thousands(total), f"detected {ui.thousands(ss.get_detected_count())}")
        weight_g = float(ss.grading.get("avg_cashew_weight_g", 1.8))
        kg_hr = lambda pieces_per_min: pieces_per_min * 60.0 * weight_g / 1000.0
        self.kpi["good"].set(ui.thousands(good), (f"{100.0 * good / total:.1f} %   " if total else "") + f"{kg_hr(ss.good_last_minute()):.1f} kg/hr")
        self.kpi["undefined"].set(ui.thousands(undefined), f"{100.0 * undefined / total:.1f} %" if total else "")
        self.kpi["rejection"].set(ui.thousands(rejection), f"{100.0 * rejection / total:.1f} %" if total else "")
        per_min = ss.decisions_last_minute()
        self.kpi["throughput"].set(f"{kg_hr(per_min):.1f}", f"kg/hr   •   {per_min} pcs/min")
        shown = (decision.GOOD, decision.UNDEFINED, decision.REJECTION)
        parts = [f"NOT ROLLED {counts.get('UNDEFINED: ' + decision.NOT_ROLLED, 0)}", f"DOUBLE {counts.get(decision.DOUBLE, 0)}", f"LOST {counts.get(decision.LOST, 0)}", "LATE: step (c3)", f"kg/hr at {weight_g:g} g per cashew"]
        captured = self.processing_thread.undefined_capture.status()
        parts.insert(3, f"CAPTURED {captured['today']}" + (f" ({captured['blocked']})" if captured["blocked"] else ""))
        with ss.lock:
            engine_counts = dict(ss.engine_counts)
        if engine_counts:
            parts.append("SHADOW would reject: " + ", ".join(f"{name} {n}" for name, n in sorted(engine_counts.items())))
        parts += [f"{key} {n}" for key, n in sorted(counts.items()) if key not in shown + (decision.DOUBLE, decision.LOST) and not key.startswith(("UNDEFINED: ", decision.IGNORED_PREFIX))]
        ignored = sum(n for key, n in counts.items() if key.startswith(decision.IGNORED_PREFIX))
        if ignored:
            parts.append(f"FRAGMENTS / NOISE IGNORED {ignored}")
        self.lbl_kpi_more.config(text="     ".join(parts))

        # Alarms and the SAFETY pill (alarms.py decides, this only shows).
        with ss.lock:
            stopped = any(ss.belt_stopped.values())
            lane_px_s = max(ss.lane_speed.values()) * 1000.0
        if running and stopped:
            for _, relay, source in self.relay_output.pending():
                text = f"{source.split(' ')[0]} (R{relay})"
                if text not in ss.missed_rejects:
                    ss.missed_rejects.append(text)
        try:
            free_gb = shutil.disk_usage(str(app_paths.BASE_DIR)).free / 1e9
        except OSError:
            free_gb = None
        max_step = float(ss.grading.get("tracker", {}).get("max_step_px", 160.0))
        active = alarms.collect(cam.is_connected, str(cam.state), plc_ok, running, self.processing_thread.ai_fps,
                                alarms.min_safe_fps(lane_px_s, max_step), free_gb, list(ss.missed_rejects))
        self.pills["safety"].set("FAULT" if active else "OK", ui.BAD if active else ui.GOOD)
        # Production capture: not a machine fault (the SAFETY pill is not changed), shown in the same box.
        capture_state = self.processing_thread.capture.blocking_reason()
        if capture_state == production_capture.INBOX_FULL:
            active = active + ["INBOX FULL - review or clear to continue (Teaching CNN page). Production capture is stopped; nothing was deleted."]
        elif capture_state == production_capture.DISK_LOW:
            active = active + ["DISK SPACE LOW - production capture is stopped."]
        if active != self._alarms_shown:
            self._alarms_shown = list(active)
            self.lbl_alarms.config(text="\n".join(active))
            if active:
                self.alarm_box.pack(fill=tk.X, pady=(ui.GRID, 0), before=self._tiles_frame)
            else:
                self.alarm_box.pack_forget()
        if bool(ss.missed_rejects) != self._ack_shown:
            self._ack_shown = bool(ss.missed_rejects)
            if self._ack_shown:
                self.btn_acknowledge.pack(anchor="e", padx=12, pady=(0, 8))
            else:
                self.btn_acknowledge.pack_forget()

        now = time.time()
        for rid, led in self.lane_leds.items():
            when, was_good = ss.last_result.get(rid, (0.0, True))
            led.set(("good" if was_good else "bad") if now - when < 0.6 else "idle")

    def _build_pages(self):
        # Shared Auto Capture State & Logic for multiple tabs
        self.golden_count_var = tk.StringVar(value="1000")
        self.capture_mode_var = tk.StringVar(value="cropped")
        
        def toggle_golden():
            if not self.shared_state.is_golden_capture_active():
                try:
                    count = int(self.golden_count_var.get())
                except (ValueError, tk.TclError):
                    count = 1000
                ctype = self.capture_mode_var.get()
                self.shared_state.trigger_golden_capture(count, capture_type=ctype)
                if hasattr(self, 'golden_prod_btn'):
                    self.golden_prod_btn.config(text="STOP AUTO CAPTURE", bg=self.colors["danger"])
                if hasattr(self, 'golden_pneumatic_btn'):
                    self.golden_pneumatic_btn.config(text="Stop Capture", bg=self.colors["danger"])
            else:
                self.shared_state.cancel_golden_capture()
                if hasattr(self, 'golden_prod_btn'):
                    self.golden_prod_btn.config(text="START AUTO CAPTURE", bg=self.colors["info"])
                if hasattr(self, 'golden_pneumatic_btn'):
                    self.golden_pneumatic_btn.config(text="Auto Capture", bg=self.colors["info"])

        # PAGE: Home (5-track layout: machine button, KPI tiles, recent results)
        F = ui.FONT
        self.toggle_golden = toggle_golden
        p_prod = tk.Frame(self.page_container, bg=ui.BG)
        self.pages["Home"] = p_prod

        machine = tk.Frame(p_prod, bg=ui.PANEL, highlightthickness=1, highlightbackground=ui.BORDER)
        machine.pack(fill=tk.X)
        self.start_btn = ui.EngineButton(machine, ui.PANEL, command=self._on_start_stop)
        self.start_btn.pack(side=tk.LEFT, padx=(18, 6), pady=2)
        ops = tk.Frame(machine, bg=ui.PANEL)
        ops.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 18), pady=18)
        self.lbl_machine_state = tk.Label(ops, text="STOPPED", font=(F, ui.F_HEAD, "bold"), fg=ui.BAD, bg=ui.PANEL, anchor="w")
        self.lbl_machine_state.pack(fill=tk.X)
        tk.Label(ops, text="RUNNING = grading fires the valves. STOPPED = nothing fires from grading. The program always opens STOPPED.", font=(F, -11), fg=ui.MUTED, bg=ui.PANEL,
                 anchor="w", justify="left", wraplength=330).pack(fill=tk.X, pady=(2, 10))
        row = tk.Frame(ops, bg=ui.PANEL)
        row.pack(fill=tk.X, pady=(0, 8))
        self.conv_btn = ui.FlatButton(row, "CONVEYOR: OFF", command=self._toggle_conv, kind="muted", height=36, width=165)
        self.conv_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.feed_btn = ui.FlatButton(row, "FEEDER: OFF", command=self._toggle_feed, kind="muted", height=36, width=165)
        self.feed_btn.pack(side=tk.LEFT)
        self.golden_pneumatic_btn = ui.FlatButton(ops, "Auto Capture", command=toggle_golden, height=32, width=165)
        self.golden_pneumatic_btn.pack(anchor="w")

        self.alarm_box = tk.Frame(p_prod, bg=ui.mix(ui.PANEL, ui.BAD, 0.18), highlightthickness=1, highlightbackground=ui.BAD)
        tk.Label(self.alarm_box, text="ALARMS", font=(F, ui.F_SMALL, "bold"), fg=ui.BAD, bg=self.alarm_box.cget("bg")).pack(anchor="w", padx=12, pady=(8, 0))
        self.lbl_alarms = tk.Label(self.alarm_box, text="", font=(F, ui.F_SMALL), fg=ui.TEXT, bg=self.alarm_box.cget("bg"),
                                   justify="left", anchor="w", wraplength=ui.RIGHT_W - 40)
        self.lbl_alarms.pack(fill=tk.X, padx=12, pady=(2, 8))
        self.btn_acknowledge = ui.FlatButton(self.alarm_box, "ACKNOWLEDGE", command=self._acknowledge_alarms, kind="danger", height=30)
        self._alarms_shown, self._ack_shown = None, False

        tiles = tk.Frame(p_prod, bg=ui.BG)
        self._tiles_frame = tiles
        tiles.pack(fill=tk.X, pady=(ui.GRID, 0))
        for col in range(3):
            tiles.columnconfigure(col, weight=1, uniform="kpi")
        self.kpi = {}
        for i, (key, title, colour) in enumerate((("total", "TOTAL", ui.TEXT), ("good", "GOOD", ui.GOOD), ("undefined", "UNDEFINED", ui.WARN),
                                                  ("rejection", "REJECTION", ui.BAD), ("throughput", "THROUGHPUT", ui.TEXT))):
            tile = ui.KpiTile(tiles, title, colour)
            tile.grid(row=i // 3, column=i % 3, sticky="nsew", padx=(0 if i % 3 == 0 else ui.GRID, 0), pady=(0, ui.GRID))
            self.kpi[key] = tile

        def reset_counters():
            if messagebox.askyesno("Reset counters", "Set all counters on this page to zero?"):
                self.shared_state.reset_result_counts()
                self.shared_state.reset_detected_count()
                self.relay_output.reset_counts()
        reset = tk.Frame(tiles, bg=ui.PANEL, highlightthickness=1, highlightbackground=ui.BORDER, height=ui.KPI_H, cursor="hand2")
        reset.grid(row=1, column=2, sticky="nsew", padx=(ui.GRID, 0), pady=(0, ui.GRID))
        reset_lbl = tk.Label(reset, text="RESET COUNTERS", font=(F, ui.F_BODY, "bold"), fg=ui.TEXT, bg=ui.PANEL, cursor="hand2")
        reset_lbl.place(relx=0.5, rely=0.5, anchor="center")
        for widget in (reset, reset_lbl):
            widget.bind("<Button-1>", lambda e: reset_counters())

        self.lbl_kpi_more = tk.Label(p_prod, text="", font=(F, ui.F_SMALL), fg=ui.MUTED, bg=ui.BG, anchor="w", justify="left", wraplength=ui.RIGHT_W - 8)
        self.lbl_kpi_more.pack(fill=tk.X, pady=(0, ui.GRID))

        def clear_history():
            with self.shared_state.lock:
                self.shared_state.recent_fires.clear()
            self._last_fires_state = []
            for item in self.fire_tree.get_children():
                self.fire_tree.delete(item)

        prod_hdr = tk.Frame(p_prod, bg=ui.BG)
        prod_hdr.pack(fill="x", pady=(0, 6))
        tk.Label(prod_hdr, text="RECENT RESULTS", font=(F, ui.F_SMALL, "bold"), fg=ui.TEXT, bg=ui.BG).pack(side="left")
        tk.Label(prod_hdr, text="double click a row for its frames", font=(F, -11), fg=ui.MUTED, bg=ui.BG).pack(side="left", padx=10)
        ui.FlatButton(prod_hdr, "CLEAR HISTORY", command=clear_history, kind="muted", height=26).pack(side="right")

        tree_frame = tk.Frame(p_prod, bg=ui.PANEL, highlightthickness=1, highlightbackground=ui.BORDER)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        cols = ("time", "track", "label", "colour", "value", "length", "grade", "act")
        self.fire_tree = ttk.Treeview(tree_frame, columns=cols, show='headings', height=8)
        for col, title, width, anchor in (("time", "TIME", 66, "center"), ("track", "ID", 84, "center"), ("label", "CLASS / RESULT", 170, "w"),
                                          ("colour", "SW/WW", 52, "center"), ("value", "COLOUR", 58, "center"), ("length", "L mm", 50, "center"),
                                          ("grade", "GRADE", 74, "center"), ("act", "EJECTOR", 64, "center")):
            self.fire_tree.heading(col, text=title)
            self.fire_tree.column(col, width=width, anchor=anchor, stretch=(col == "label"))
        self.fire_tree.bind("<Double-1>", self._show_decision_detail)     # frame-by-frame votes of that cashew

        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.fire_tree.yview)
        self.fire_tree.configure(yscrollcommand=tree_scroll.set)

        self.fire_tree.pack(side="left", fill=tk.BOTH, expand=True)
        tree_scroll.pack(side="right", fill="y")
        

        # PAGE: Dataset Capture (5-track layout: cards, flat buttons; same functions as before)
        F = ui.FONT
        p_cap = tk.Frame(self.page_container, bg=ui.BG)
        self.pages["Dataset Capture"] = p_cap

        def cap_card(title, note=""):
            box = tk.Frame(p_cap, bg=ui.PANEL, highlightthickness=1, highlightbackground=ui.BORDER)
            box.pack(fill=tk.X, pady=(0, 10))
            head = tk.Frame(box, bg=ui.PANEL)
            head.pack(fill=tk.X, padx=14, pady=(10, 4))
            tk.Label(head, text=title, font=(F, ui.F_SMALL, "bold"), fg=ui.MUTED, bg=ui.PANEL).pack(side=tk.LEFT)
            if note:
                tk.Label(head, text=note, font=(F, -11), fg=ui.MUTED, bg=ui.PANEL).pack(side=tk.LEFT, padx=12)
            body = tk.Frame(box, bg=ui.PANEL)
            body.pack(fill=tk.X, padx=14, pady=(0, 12))
            return body

        folders = cap_card("CAPTURE FOLDERS", "choose a folder, then CAPTURE saves the camera image into it")
        self.capture_class_vars = {}
        self.capture_count_vars = {}
        self.active_capture_var = tk.StringVar(value=self.shared_state.active_capture_class_id)

        def update_all_counts():
            for i in range(1, 21):
                folder_name = self.shared_state.capture_classes.get(str(i), f"Class_{i}")
                target_dir = os.path.join(DATASET_DIR, folder_name)
                count = len([f for f in os.listdir(target_dir) if f.endswith('.jpg')]) if os.path.exists(target_dir) else 0
                if str(i) in self.capture_count_vars:
                    self.capture_count_vars[str(i)].set(str(count))
            self.after(1000, update_all_counts)

        self.after(1000, update_all_counts)
        cap_grid = tk.Frame(folders, bg=ui.PANEL)
        cap_grid.pack(fill=tk.X)
        cap_grid.columnconfigure(0, weight=1, uniform="cap"); cap_grid.columnconfigure(1, weight=1, uniform="cap")

        def update_capture_class_name(cid, var):
            try:
                val = var.get().strip()
                if not val: val = f"Class_{cid}"
                with self.shared_state.lock:
                    self.shared_state.capture_classes[str(cid)] = val
            except (ValueError, tk.TclError): pass

        def on_capture_select():
            with self.shared_state.lock:
                self.shared_state.active_capture_class_id = self.active_capture_var.get()
                self.shared_state.save_to_config(CFG_PATH)

        for i in range(1, 21):
            col = 0 if i <= 10 else 1
            row = i if i <= 10 else i - 10
            f1 = tk.Frame(cap_grid, bg=ui.PANEL)
            f1.grid(row=row, column=col, sticky="ew", padx=(0, 8) if col == 0 else (8, 0), pady=2)
            tk.Radiobutton(f1, text=f"{i:02d}", variable=self.active_capture_var, value=str(i), bg=ui.PANEL, fg=ui.TEXT, selectcolor=ui.FIELD,
                           activebackground=ui.PANEL, activeforeground=ui.ACCENT, highlightthickness=0, bd=0,
                           font=(F, ui.F_SMALL, "bold"), command=on_capture_select).pack(side=tk.LEFT)
            nm = tk.StringVar(value=self.shared_state.capture_classes.get(str(i), f"Class_{i}"))
            tk.Entry(f1, textvariable=nm, width=13, bg=ui.FIELD, fg=ui.TEXT, insertbackground=ui.TEXT, bd=0, font=(F, -13),
                     highlightthickness=1, highlightbackground=ui.BORDER, highlightcolor=ui.ACCENT).pack(side=tk.LEFT, padx=(4, 0), ipady=4)
            self.capture_class_vars[str(i)] = nm
            nm.trace_add("write", lambda *_, cid=i, v=nm: update_capture_class_name(cid, v))
            cnt_var = tk.StringVar(value="0")
            self.capture_count_vars[str(i)] = cnt_var

            def open_dir(cid=i):
                import subprocess
                folder_name = self.shared_state.capture_classes.get(str(cid), f"Class_{cid}")
                target_dir = os.path.join(DATASET_DIR, folder_name)
                os.makedirs(target_dir, exist_ok=True)
                subprocess.Popen(['xdg-open', target_dir])

            ui.FlatButton(f1, "OPEN", open_dir, kind="muted", height=26, width=56, font=(F, -11, "bold")).pack(side=tk.RIGHT)
            tk.Label(f1, textvariable=cnt_var, bg=ui.PANEL, fg=ui.MUTED, font=(F, ui.F_SMALL, "bold"), width=5, anchor="e").pack(side=tk.RIGHT, padx=(0, 8))

        buttons = tk.Frame(folders, bg=ui.PANEL)
        buttons.pack(fill=tk.X, pady=(12, 0))
        ui.FlatButton(buttons, "SAVE FOLDER NAMES", lambda: self.shared_state.save_to_config(CFG_PATH), kind="muted", height=40, width=200).pack(side=tk.LEFT)
        ui.FlatButton(buttons, "CAPTURE IMAGE TO FOLDER", self.shared_state.trigger_capture, kind="primary", height=40,
                      font=(F, ui.F_BODY, "bold")).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))

        console_frm = cap_card("AUTO CAPTURE", "saves cashew images by itself until the limit is reached; label them on the Teaching CNN page")
        limit_frm = tk.Frame(console_frm, bg=ui.PANEL)
        limit_frm.pack(fill=tk.X, pady=(0, 10))
        tk.Label(limit_frm, text="Limit", bg=ui.PANEL, fg=ui.TEXT, font=(F, ui.F_BODY)).pack(side=tk.LEFT)
        tk.Entry(limit_frm, textvariable=self.golden_count_var, width=8, bg=ui.FIELD, fg=ui.TEXT, insertbackground=ui.TEXT, bd=0, justify="center",
                 font=(F, -14), highlightthickness=1, highlightbackground=ui.BORDER, highlightcolor=ui.ACCENT).pack(side=tk.LEFT, padx=10, ipady=5)
        tk.Label(limit_frm, text="images", bg=ui.PANEL, fg=ui.MUTED, font=(F, ui.F_SMALL)).pack(side=tk.LEFT)
        mode_names = {"cropped": "CROP", "full": "FULL FRAME", "both": "BOTH"}
        self.capture_mode_selector = ui.Segmented(limit_frm, list(mode_names.values()), mode_names.get(self.capture_mode_var.get(), "CROP"),
                                                  lambda value: self.capture_mode_var.set(next(k for k, v in mode_names.items() if v == value)))
        self.capture_mode_selector.pack(side=tk.RIGHT)
        self.capture_mode_var.trace_add("write", lambda *_: self.capture_mode_selector.set(mode_names.get(self.capture_mode_var.get(), "CROP")))
        tk.Label(limit_frm, text="Mode", bg=ui.PANEL, fg=ui.TEXT, font=(F, ui.F_BODY)).pack(side=tk.RIGHT, padx=10)
        self.golden_prod_btn = tk.Button(console_frm, text="START AUTO CAPTURE", bg=self.colors["info"], fg="#06222A", activeforeground="#06222A",
                                         font=(F, ui.F_BODY, "bold"), bd=0, relief="flat", highlightthickness=0, cursor="hand2", pady=10)
        self.golden_prod_btn.config(command=toggle_golden)
        self.golden_prod_btn.pack(fill=tk.X)

        # Add updater to the main UI loop checking golden_prod_btn
        def check_golden_btn():
            is_active = self.shared_state.is_golden_capture_active()
            if hasattr(self, 'golden_prod_btn'):
                if not is_active:
                    self.golden_prod_btn.config(text="START AUTO CAPTURE", bg=self.colors["info"])
                else:
                    self.golden_prod_btn.config(text="STOP AUTO CAPTURE", bg=self.colors["danger"])
                    
            if hasattr(self, 'golden_pneumatic_btn'):
                if not is_active:
                    self.golden_pneumatic_btn.config(text="Auto Capture", bg=self.colors["info"])
                else:
                    self.golden_pneumatic_btn.config(text="Stop Capture", bg=self.colors["danger"])
                    
            self.after(500, check_golden_btn)
        self.after(500, check_golden_btn)

        # PAGE: AI Model
        p_model = tk.Frame(self.page_container, bg=self.colors["surface"], padx=15, pady=15)
        self.pages["AI Model Old"] = p_model          # replaced by model_ui.ModelPage below
        import model_ui
        p_models = tk.Frame(self.page_container, bg=ui.BG)
        self.pages["AI Model Select"] = p_models
        self.model_page = model_ui.ModelPage(
            self, p_models, str(MODEL_ROOT), self._active_model_name, self._activate_model, lambda: self.shared_state.phase == "RUNNING",
            self._class_settings, self._apply_class_settings, lambda: str(app_paths.TESTSET_DIR))
        ttk.Label(p_model, text="YOLO MODEL UPLOAD", style="Title.TLabel").pack(anchor="w", pady=(0, 20))
        
        upload_frm = tk.Frame(p_model, bg=self.colors["surface"])
        upload_frm.pack(fill="x", pady=(0, 15))
        
        model_frm = tk.Frame(p_model, bg=self.colors["surface"])
        model_frm.pack(fill="x", pady=10)
        
        ttk.Label(model_frm, text="Select Model:", style="Normal.TLabel").pack(side="left", padx=(0,10))
        
        self.model_combo_var = tk.StringVar()
        model_combo = ttk.Combobox(model_frm, textvariable=self.model_combo_var, state="readonly", width=25)
        model_combo.pack(side="left", padx=(0,10))
        
        insights_frm = tk.LabelFrame(p_model, text="MODEL INSIGHTS", bg=self.colors["surface"], fg=self.colors["info"], padx=10, pady=10)
        insights_frm.pack(fill="x", pady=15)
        
        self.insight_lbl = tk.Label(insights_frm, text="No model selected.", bg=self.colors["surface"], fg="white", justify="left", font=("Segoe UI", 12))
        self.insight_lbl.pack(side="left", padx=10, anchor="n")
        
        self.insight_img_lbl = tk.Label(insights_frm, bg=self.colors["surface"])
        self.insight_img_lbl.pack(side="right", padx=10)

        def load_insights(model_filename):
            base = os.path.splitext(model_filename)[0]
            csv_path = os.path.join(MODELS_DIR, f"{base}_results.csv")
            png_path = os.path.join(MODELS_DIR, f"{base}_confusion.png")
            
            import time
            from datetime import datetime
            
            info = f"Model: {model_filename}\n\n"
            
            if os.path.exists(csv_path):
                # Get modified time for training date
                mod_time = os.path.getmtime(csv_path)
                dt = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
                info += f"Trained On: {dt}\n\n"
                
                try:
                    with open(csv_path, 'r') as f:
                        lines = f.readlines()
                        if len(lines) >= 2:
                            headers = [h.strip() for h in lines[0].split(',')]
                            last_row = [v.strip() for v in lines[-1].split(',')]
                            data_dict = dict(zip(headers, last_row))
                            
                            epoch = data_dict.get('epoch', 'N/A')
                            map50 = float(data_dict.get('metrics/mAP50(B)', 0.0)) * 100
                            map75 = float(data_dict.get('metrics/mAP50-95(B)', 0.0)) * 100
                            prec = float(data_dict.get('metrics/precision(B)', 0.0)) * 100
                            rec = float(data_dict.get('metrics/recall(B)', 0.0)) * 100
                            
                            info += f"Training Epochs: {epoch}\n"
                            info += f"mAP (50): {map50:.2f}%\n"
                            info += f"mAP (50-95): {map75:.2f}%\n"
                            info += f"Precision: {prec:.2f}%\n"
                            info += f"Recall: {rec:.2f}%\n"
                except Exception as e:
                    info += f"Could not parse metrics: {e}\n"
            else:
                info += "No training metrics found.\n"
                
            self.insight_lbl.config(text=info)
            
            if os.path.exists(png_path):
                from PIL import Image, ImageTk
                try:
                    img = Image.open(png_path)
                    img.thumbnail((350, 350))
                    photo = ImageTk.PhotoImage(img)
                    self.insight_img_lbl.config(image=photo)
                    self.insight_img_lbl.image = photo 
                except Exception:
                    log.exception("Could not show %s", png_path)
                    self.insight_img_lbl.config(image="")
            else:
                self.insight_img_lbl.config(image="")

        def refresh_models():
            try:
                models = sorted(f for f in os.listdir(MODELS_DIR) if f.endswith(".pt"))
                model_combo['values'] = models
                active_name = os.path.basename(str(self.shared_state.yolo_model_path))
                if active_name in models:
                    self.model_combo_var.set(active_name)
                    load_insights(active_name)
                elif models:
                    self.model_combo_var.set(models[-1])
                    load_insights(models[-1])
            except Exception:
                log.exception("Could not list the models in %s", MODELS_DIR)
            
        def on_model_select(event):
            selected = self.model_combo_var.get()
            if selected:
                load_insights(selected)

        def apply_model():
            selected = self.model_combo_var.get()
            if not selected: return
            full_path = os.path.join(MODELS_DIR, selected)
            if not self.processing_thread.reload_model(full_path):
                messagebox.showerror("Model Error", f"Could not load {selected}.\nThe previous model stays active. See logs/grader.log.")
                return
            with self.shared_state.lock:
                self.shared_state.yolo_model_path = app_paths.relative(full_path)
            self.shared_state.save_to_config(CFG_PATH)
            self._refresh_class_map()
            messagebox.showinfo("Model Integrated", f"Successfully integrated model:\n{selected}")
            
        model_combo.bind("<<ComboboxSelected>>", on_model_select)
        
        btn_apply = ttk.Button(model_frm, text="Apply Model", command=apply_model, style="Success.TButton")
        btn_apply.pack(side="left", padx=10)
        
        def _model_report_files(model_filename):
            base = os.path.splitext(model_filename)[0]
            return [os.path.join(MODELS_DIR, f"{base}{suffix}") for suffix in ("_results.csv", "_confusion.png")]

        def delete_selected_model():
            selected = self.model_combo_var.get()
            if not selected:
                messagebox.showwarning("Warning", "No model selected.")
                return
            if not selected.endswith(".pt"):
                return
            if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete {selected}?"):
                try:
                    for path in [os.path.join(MODELS_DIR, selected)] + _model_report_files(selected):
                        if os.path.exists(path): os.remove(path)
                    refresh_models()
                    self.insight_lbl.config(text="Model deleted.")
                    self.insight_img_lbl.config(image="")
                except Exception as e:
                    log.exception("Failed to delete model %s", selected)
                    messagebox.showerror("Error", f"Failed to delete model: {e}")

        def clear_all_models():
            # Only YOLO .pt models and their result files. The CNN checkpoints (.pth) are never deleted here.
            yolo_models = sorted(f for f in os.listdir(MODELS_DIR) if f.endswith(".pt"))
            if not yolo_models:
                messagebox.showinfo("Clear All", "No YOLO (.pt) models to delete.")
                return
            if messagebox.askyesno("Confirm Clear All", f"Delete ALL {len(yolo_models)} YOLO (.pt) models?\n\n" + "\n".join(yolo_models)
                                   + "\n\nCNN models (.pth) are kept."):
                try:
                    for f in yolo_models:
                        for path in [os.path.join(MODELS_DIR, f)] + _model_report_files(f):
                            if os.path.isfile(path): os.remove(path)
                    refresh_models()
                    self.insight_lbl.config(text="All YOLO models cleared. CNN models (.pth) were kept.")
                    self.insight_img_lbl.config(image="")
                except Exception as e:
                    log.exception("Failed to clear models")
                    messagebox.showerror("Error", f"Failed to clear models: {e}")

        btn_delete = ttk.Button(upload_frm, text="Delete Selected", command=delete_selected_model, style="Danger.TButton")
        btn_delete.pack(side="left", padx=10)
        
        btn_clear = ttk.Button(upload_frm, text="Clear All", command=clear_all_models, style="Danger.TButton")
        btn_clear.pack(side="left", padx=10)
        
        def browse_model():
            path = filedialog.askopenfilename(title="Select YOLO model file", filetypes=[("YOLO model", "*.pt"), ("All Files", "*.*")])
            if path:
                try:
                    if not path.lower().endswith(".pt"):
                        messagebox.showerror("Upload Error", "Only YOLO .pt models can be uploaded here.")
                        return
                    base = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    dest_path = os.path.join(MODELS_DIR, f"{base}.pt")
                    shutil.copy2(path, dest_path)
                    
                    run_dir = os.path.dirname(os.path.dirname(path))
                    res_csv = os.path.join(run_dir, "results.csv")
                    conf_png = os.path.join(run_dir, "confusion_matrix.png")
                    if os.path.exists(res_csv):
                        shutil.copy2(res_csv, os.path.join(MODELS_DIR, f"{base}_results.csv"))
                    if os.path.exists(conf_png):
                        shutil.copy2(conf_png, os.path.join(MODELS_DIR, f"{base}_confusion.png"))
                    
                    if not self.processing_thread.reload_model(dest_path):
                        messagebox.showerror("Upload Error", "The file was copied but could not be loaded as a YOLO model.\nThe previous model stays active.")
                        refresh_models()
                        return
                    with self.shared_state.lock:
                        self.shared_state.yolo_model_path = app_paths.relative(dest_path)
                    self.shared_state.save_to_config(CFG_PATH)
                    self._refresh_class_map()
                    refresh_models()
                except Exception as e:
                    log.exception("Model upload failed")
                    messagebox.showerror("Upload Error", f"Failed to upload model: {e}")
                    
        ttk.Button(upload_frm, text="UPLOAD MODEL FILE", style="Primary.TButton", command=browse_model).pack(side="left")
        refresh_models()

        self._build_vision_page()

        p_sort = tk.Frame(self.page_container, bg=self.colors["surface"])
        self.pages["Sorting Logic"] = p_sort
        sort_canvas = tk.Canvas(p_sort, bg=self.colors["surface"], highlightthickness=0)
        sort_scroll = ttk.Scrollbar(p_sort, orient="vertical", command=sort_canvas.yview)
        sort_canvas.configure(bg=ui.BG)
        sort_outer = tk.Frame(sort_canvas, bg=ui.BG)
        sort_outer.bind("<Configure>", lambda e: sort_canvas.configure(scrollregion=sort_canvas.bbox("all")))
        sort_window = sort_canvas.create_window((0, 0), window=sort_outer, anchor="nw")
        sort_canvas.bind("<Configure>", lambda e: sort_canvas.itemconfig(sort_window, width=e.width))
        self._build_grading_section(sort_outer, bar_parent=p_sort)
        # Older blocks (combination rules, per-class confidence sliders of the old logic) stay below, unchanged.
        sort_inner = tk.Frame(sort_outer, bg=self.colors["surface"], padx=15, pady=15, width=520)
        sort_inner.pack(anchor="w")
        sort_canvas.configure(yscrollcommand=sort_scroll.set)
        sort_canvas.pack(side="left", fill="both", expand=True)
        sort_scroll.pack(side="right", fill="y")
        
        ttk.Label(sort_inner, text="COMBINATION RULES (classes seen together -> final)", style="Title.TLabel").pack(anchor="w", pady=(0,10))
        classes = ["Any", "Good", "UPW", "PKW", "White", "Defect_Skin"] + self.all_class_names
        classes = list(set(classes))
        
        prio_frm = tk.Frame(sort_inner, bg=self.colors["surface"])
        prio_frm.pack(fill=tk.X, pady=5)
        self.prio_vars = []
        for i in range(4):
            f_col = tk.Frame(prio_frm, bg=self.colors["surface"])
            f_col.pack(side=tk.LEFT, padx=2)
            ttk.Label(f_col, text=f"F{i+1}", font=("Roboto", 8), style="Surface.TLabel").pack()
            var = tk.StringVar(value="Any")
            ttk.Combobox(f_col, textvariable=var, values=classes, width=6).pack()
            self.prio_vars.append(var)
            if i < 3: ttk.Label(prio_frm, text="+", style="Surface.TLabel").pack(side=tk.LEFT, padx=2, pady=(15,0))
            
        ttk.Label(prio_frm, text="=", style="Surface.TLabel").pack(side=tk.LEFT, padx=5, pady=(15,0))
        f_fin = tk.Frame(prio_frm, bg=self.colors["surface"])
        f_fin.pack(side=tk.LEFT)
        ttk.Label(f_fin, text="FINAL", font=("Roboto", 8), style="Surface.TLabel").pack()
        self.final_grade_var = tk.StringVar(value="Good")
        ttk.Combobox(f_fin, textvariable=self.final_grade_var, values=classes, width=8).pack()
        
        ttk.Button(sort_inner, text="ADD RULE", style="Primary.TButton", command=self._add_priority_logic).pack(anchor="e", pady=10)
        
        cols_prio = ("f1", "f2", "f3", "f4", "final")
        self.logic_tree = ttk.Treeview(sort_inner, columns=cols_prio, show='headings', height=5)
        self.logic_tree.heading("f1", text="F1"); self.logic_tree.column("f1", width=50, anchor='center')
        self.logic_tree.heading("f2", text="F2"); self.logic_tree.column("f2", width=50, anchor='center')
        self.logic_tree.heading("f3", text="F3"); self.logic_tree.column("f3", width=50, anchor='center')
        self.logic_tree.heading("f4", text="F4"); self.logic_tree.column("f4", width=50, anchor='center')
        self.logic_tree.heading("final", text="FINAL"); self.logic_tree.column("final", width=80, anchor='center')
        self.logic_tree.pack(fill=tk.X, pady=5)
        
        for rule in self.shared_state.priority_rules:
            vals = rule["conditions"] + [rule["final"]]
            self.logic_tree.insert("", "end", values=vals)
            
        prio_acts = tk.Frame(sort_inner, bg=self.colors["surface"])
        prio_acts.pack(fill=tk.X, pady=5)
        ttk.Button(prio_acts, text="Undo", command=self._undo_last_priority_logic).pack(side=tk.LEFT, padx=2)
        ttk.Button(prio_acts, text="Delete", command=self._delete_selected_priority_logic).pack(side=tk.LEFT, padx=2)
        ttk.Button(prio_acts, text="Clear", command=self._clear_all_priority_rules).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(sort_inner, text="CONFIDENCE THRESHOLDS (old logic: not used by the grading rules above)", style="Title.TLabel").pack(anchor="w", pady=(20,10))
        self.conf_vars = {}
        for cname in sorted(self.all_class_names):
            cname_low = cname.lower()
            val = self.shared_state.class_confs.get(cname_low, 50)
            row = tk.Frame(sort_inner, bg=self.colors["surface"])
            row.pack(fill=tk.X, pady=4)
            ttk.Label(row, text=cname.upper(), width=12, style="Surface.TLabel").pack(side=tk.LEFT)
            var = tk.IntVar(value=val)
            self.conf_vars[cname_low] = var
            ttk.Scale(row, from_=1, to=100, orient=tk.HORIZONTAL, variable=var, length=180).pack(side=tk.LEFT, padx=10)
            tk.Entry(row, textvariable=var, width=4, bg=self.colors["surface_light"], fg=self.colors["text"], bd=0, justify="center").pack(side=tk.LEFT)
            var.trace_add("write", lambda *a, c=cname_low, v=var: self._update_class_conf(c, v))

        ttk.Label(sort_inner, text="AUTO-TEACH SIZE CALIBRATION", style="Title.TLabel").pack(anchor="w", pady=(20,10))
        at_row = tk.Frame(sort_inner, bg=self.colors["surface"])
        at_row.pack(fill=tk.X, pady=5)
        
        ttk.Label(at_row, text="Target Size:").pack(side=tk.LEFT, padx=(0,5))
        self.at_size_var = tk.StringVar(value="W180")
        ttk.Combobox(at_row, textvariable=self.at_size_var, values=["W180", "W210", "W240", "W320", "W450", "BUDDS"], width=8).pack(side=tk.LEFT, padx=(0,15))
        
        ttk.Button(at_row, text="TEACH FROM CAMERA", style="Primary.TButton", command=self._start_auto_teach).pack(side=tk.LEFT)

        ttk.Label(sort_inner, text="SIZE SORTING (MM)", style="Title.TLabel").pack(anchor="w", pady=(20,10))
        cfg = self.shared_state.size_config
        px_frm = tk.Frame(sort_inner, bg=self.colors["surface"])
        px_frm.pack(fill=tk.X, pady=5)
        ttk.Label(px_frm, text="1 MM = Pixels:", style="Surface.TLabel").pack(side=tk.LEFT)
        px_mm_var = tk.StringVar(value=str(cfg.get("px_per_mm", 10.0)))
        tk.Entry(px_frm, textvariable=px_mm_var, width=6, bg=self.colors["surface_light"], fg=self.colors["text"], bd=0).pack(side=tk.LEFT, padx=10)
        self.sz_vars = {"px_per_mm": px_mm_var}
        
        sz_grid = tk.Frame(sort_inner, bg=self.colors["surface"])
        sz_grid.pack(fill=tk.X)
        for i, h in enumerate(["Size", "Len <", "Wid <", "T1", "T2"]):
            ttk.Label(sz_grid, text=h, font=("Roboto", 9, "bold"), style="Surface.TLabel").grid(row=0, column=i, padx=5, pady=5, sticky="w")
            
        for i, sz in enumerate(["W180", "W210", "W240", "W320", "W450", "BUDDS"]):
            ttk.Label(sz_grid, text=sz, style="Surface.TLabel").grid(row=i+1, column=0, padx=5, pady=2, sticky="w")
            
            l_var = tk.StringVar(value=str(cfg.get(f"{sz}_L", 15)))
            tk.Entry(sz_grid, textvariable=l_var, width=5, bg=self.colors["surface_light"], fg=self.colors["text"], bd=0).grid(row=i+1, column=1, padx=5, pady=2)
            self.sz_vars[f"{sz}_L"] = l_var
            
            w_var = tk.StringVar(value=str(cfg.get(f"{sz}_W", 10)))
            tk.Entry(sz_grid, textvariable=w_var, width=5, bg=self.colors["surface_light"], fg=self.colors["text"], bd=0).grid(row=i+1, column=2, padx=5, pady=2)
            self.sz_vars[f"{sz}_W"] = w_var
            
            t1_var = tk.StringVar(value=str(cfg.get(f"T1_Act_{sz}", 0)))
            ttk.Combobox(sz_grid, textvariable=t1_var, values=[0] + list(range(1,21)), width=3).grid(row=i+1, column=3, padx=5, pady=2)
            self.sz_vars[f"T1_Act_{sz}"] = t1_var
            
            t2_var = tk.StringVar(value=str(cfg.get(f"T2_Act_{sz}", 0)))
            ttk.Combobox(sz_grid, textvariable=t2_var, values=[0] + list(range(21,41)), width=3).grid(row=i+1, column=4, padx=5, pady=2)
            self.sz_vars[f"T2_Act_{sz}"] = t2_var

        ttk.Label(sort_inner, text="DEFECT ALLOWANCE (MM)", style="Title.TLabel").pack(anchor="w", pady=(20,10))
        self.allow_vars = {}
        for cname in sorted(self.all_class_names):
            if "good" in cname.lower() or bool(re.search(r'\b(s|w)w\d*', cname.lower())): continue
            cname_low = cname.lower()
            val = self.shared_state.defect_allowances.get(cname_low, 0.0)
            row = tk.Frame(sort_inner, bg=self.colors["surface"])
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=cname.upper(), width=15, style="Surface.TLabel").pack(side=tk.LEFT)
            var = tk.DoubleVar(value=val)
            self.allow_vars[cname_low] = var
            tk.Entry(row, textvariable=var, width=5, bg=self.colors["surface_light"], fg=self.colors["text"], bd=0, justify="center").pack(side=tk.LEFT, padx=10)
            ttk.Label(row, text="Max Allowable Length (mm)", style="Surface.TLabel", foreground=self.colors["text_muted"]).pack(side=tk.LEFT)

        ttk.Label(sort_inner, text="DEFECT ACTUATOR MAPPING", style="Title.TLabel").pack(anchor="w", pady=(20,10))
        self.map_lbls = {}
        for trk, acts in [(1, range(1,21)), (2, range(21,41))]:
            ttk.Label(sort_inner, text=f"Track {trk}", style="Header.TLabel").pack(anchor="w", pady=5)
            for act in acts:
                row = tk.Frame(sort_inner, bg=self.colors["surface"])
                row.pack(fill=tk.X, pady=2)
                ttk.Button(row, text=f"A{act}", width=4, command=lambda a=act: self._popup_mapper(a)).pack(side=tk.LEFT, padx=(0,10))
                lbl = ttk.Label(row, text="(none)", style="Surface.TLabel", foreground=self.colors["text_muted"])
                lbl.pack(side=tk.LEFT)
                self.map_lbls[act] = lbl
        self._update_all_map_labels()
        
        ttk.Button(sort_inner, text="SAVE SORTING CONFIG", style="Primary.TButton", command=self._save_program).pack(fill=tk.X, pady=20)


        self._build_pneumatic_page()

        # PAGE: Network (5-track layout)
        F = ui.FONT
        p_net = tk.Frame(self.page_container, bg=ui.BG)
        self.pages["Network"] = p_net

        def net_card(title, note=""):
            box = tk.Frame(p_net, bg=ui.PANEL, highlightthickness=1, highlightbackground=ui.BORDER)
            box.pack(fill=tk.X, pady=(0, 12))
            head = tk.Frame(box, bg=ui.PANEL)
            head.pack(fill=tk.X, padx=16, pady=(12, 6))
            tk.Label(head, text=title, font=(F, ui.F_SMALL, "bold"), fg=ui.MUTED, bg=ui.PANEL).pack(side=tk.LEFT)
            if note:
                tk.Label(head, text=note, font=(F, -11), fg=ui.MUTED, bg=ui.PANEL).pack(side=tk.LEFT, padx=12)
            body = tk.Frame(box, bg=ui.PANEL)
            body.pack(fill=tk.X, padx=16, pady=(0, 14))
            return body

        def net_field(parent, row, column, label, var, width=16):
            tk.Label(parent, text=label, font=(F, ui.F_BODY), fg=ui.TEXT, bg=ui.PANEL, anchor="w").grid(row=row, column=column, sticky="w", pady=5, padx=(0, 12))
            tk.Entry(parent, textvariable=var, width=width, bg=ui.FIELD, fg=ui.TEXT, insertbackground=ui.TEXT, bd=0, justify="center", font=(F, -14),
                     highlightthickness=1, highlightbackground=ui.BORDER, highlightcolor=ui.ACCENT).grid(row=row, column=column + 1, sticky="w", pady=5, padx=(0, 40), ipady=5)

        plc_card = net_card("PLC", "Inovance EC320, Modbus TCP. Ejector 1 = M100 ... ejector 40 = M139; M140 conveyor and M141 feeder are never ejectors")
        fields = tk.Frame(plc_card, bg=ui.PANEL)
        fields.pack(fill=tk.X)
        self.net_plc_ip_var = tk.StringVar(value=getattr(self.controller, 'plc_ip', '192.168.1.88'))
        self.net_plc_port_var = tk.StringVar(value=str(getattr(self.controller, 'plc_port', 502)))
        self.net_plc_unit_var = tk.StringVar(value=str(getattr(self.controller, 'plc_unit_id', 1)))
        self.net_plc_regtype_var = tk.StringVar(value=getattr(self.controller, 'register_type', 'Coil (FC01/05/15)'))
        base_register = getattr(self.controller, 'relay1_register', 100)
        self.net_plc_basereg_var = tk.StringVar(value=str(base_register if isinstance(base_register, int) else 100))
        net_field(fields, 0, 0, "PLC IP address", self.net_plc_ip_var)
        net_field(fields, 0, 2, "Port", self.net_plc_port_var, 8)
        net_field(fields, 1, 0, "Unit ID", self.net_plc_unit_var, 8)
        net_field(fields, 1, 2, "Ejector 1 address (M100)", self.net_plc_basereg_var, 8)
        tk.Label(fields, text="Register type", font=(F, ui.F_BODY), fg=ui.TEXT, bg=ui.PANEL, anchor="w").grid(row=2, column=0, sticky="w", pady=5, padx=(0, 12))
        ttk.Combobox(fields, textvariable=self.net_plc_regtype_var, values=["Coil (FC01/05/15)", "Holding Reg (FC03/06/16)"], width=24,
                     state="readonly").grid(row=2, column=1, columnspan=2, sticky="w", pady=5)
        status_row = tk.Frame(plc_card, bg=ui.PANEL)
        status_row.pack(fill=tk.X, pady=(8, 0))
        tk.Label(status_row, text="Status", font=(F, ui.F_BODY), fg=ui.TEXT, bg=ui.PANEL).pack(side=tk.LEFT)
        self.net_plc_status = tk.Label(status_row, text="● CHECKING...", font=(F, ui.F_BODY, "bold"), fg=ui.WARN, bg=ui.PANEL)
        self.net_plc_status.pack(side=tk.LEFT, padx=14)

        def _apply_plc_settings():
            if self.shared_state.phase == "RUNNING":
                messagebox.showwarning("Stop first", "Stop the machine before changing the PLC connection.")
                return
            try:
                ip = self.net_plc_ip_var.get().strip()
                port = int(self.net_plc_port_var.get().strip())
                uid = int(self.net_plc_unit_var.get().strip())
                rtype = self.net_plc_regtype_var.get().strip()
                base_reg = int(self.net_plc_basereg_var.get().strip())
                if hasattr(self.controller, 'update_config'):
                    self.controller.update_config(ip=ip, port=port, unit_id=uid, start_reg=base_reg, reg_type=rtype)
                    messagebox.showinfo("PLC Config", "Inovance PLC settings saved! Reconnecting...")
            except Exception as e:
                messagebox.showerror("PLC Config Error", f"Failed to save PLC settings: {e}")

        net_buttons = tk.Frame(plc_card, bg=ui.PANEL)
        net_buttons.pack(fill=tk.X, pady=(14, 0))
        ui.FlatButton(net_buttons, "SAVE & CONNECT PLC", _apply_plc_settings, kind="primary", height=40, width=240, font=(F, ui.F_BODY, "bold")).pack(side=tk.LEFT)
        ui.FlatButton(net_buttons, "PLC DIAGNOSTICS / EJECTOR MAPPING ...", self._open_plc_diagnostics, height=40, width=330).pack(side=tk.LEFT, padx=12)

        cam_card = net_card("CAMERA", "USB3 Vision through Aravis; settings are on the Vision ROI page")
        self.lbl_net_camera = tk.Label(cam_card, text="", font=(F, ui.F_BODY), fg=ui.TEXT, bg=ui.PANEL, anchor="w", justify="left")
        self.lbl_net_camera.pack(fill=tk.X)

        self._build_tcm_studio_page()
        self._build_annotate_page()

    def _build_tcm_studio_page(self):
        """TEACHING CNN page as on the 5-track machine: capture, label, train, production inbox. See teach_ui.py."""
        import teach_ui
        p_tcm = tk.Frame(self.page_container, bg=ui.BG)
        self.pages["Teaching CNN"] = p_tcm
        self.teach_page = teach_ui.TeachPage(self, p_tcm, GOLDEN_DIR, MODEL_ROOT)

    # ── Training (training.py does the work; training_ui.py shows it) ──
    active_program_name = app_paths.PROGRAM_NAME
    plc_reconnects = 0                  # PLC came back after being offline (since program start)
    log_dir = app_paths.LOGS_DIR

    def _active_dataset_dir(self):
        """The clean dataset of the active program: one folder per final class name."""
        return str(app_paths.CLEAN_DATASET_DIR)

    def _active_cnn_path(self):
        """The CNN checkpoint the grader is using now."""
        return active_cnn_path()

    # ── Model activation (AI Model Select page, model_ui.py) ──
    def _active_model_name(self, kind):
        if kind == "CNN":
            return Path(self._active_cnn_path()).name
        return os.path.basename(str(self.shared_state.yolo_model_path or "")) or Path(str(getattr(self.processing_thread, "model_path", "") or "")).name

    def _activate_model(self, kind, name, path):
        """Make a model file the active one (called only after the class changes were confirmed, machine STOPPED)."""
        if kind == "YOLO":
            with self.shared_state.lock:
                self.shared_state.yolo_model_path = app_paths.relative(path)
            self.shared_state.save_to_config(CFG_PATH)
            self.processing_thread.reload_model(path)
        else:
            if not self.processing_thread.reload_cnn_model(path):
                messagebox.showerror("Activate model", f"{name} could not be loaded. The model in use was not changed.")
                return
            with open(app_paths.ACTIVE_MODELS_PATH, "w", encoding="utf-8") as f:
                json.dump({"cnn_model": name}, f)
        log.info("[MODELS] %s model activated: %s", kind, name)

    def _class_settings(self):
        """(FIRE classes, PASS classes) as stored now. PASS = counts as "no defect"; FIRE = a defect (its ejector, or reject)."""
        with self.shared_state.lock:
            g = self.shared_state.grading
            good = [str(c).upper() for c in g.get("good_classes", [])]
            fire = [str(c).upper() for c in list(g.get("priority", [])) + list(g.get("reject_classes", [])) + [decision.REJECTION, decision.DOUBLE_CLASS]]
        return [c for c in fire if c not in good], good

    def _apply_class_settings(self, choice):
        """Store the confirmed PASS / FIRE setting of every class of a newly activated model."""
        with self.shared_state.lock:
            g = copy.deepcopy(self.shared_state.grading)
        good = {normalize_name(c): c for c in g.get("good_classes", [])}
        known_fire = {normalize_name(c) for c in list(g.get("priority", [])) + list(g.get("reject_classes", [])) + [decision.REJECTION, decision.DOUBLE_CLASS]}
        for name, value in choice.items():
            key = normalize_name(name)
            if value == "PASS":
                good.setdefault(key, name)
            else:
                good.pop(key, None)
                if key not in known_fire:           # a new defect class: lowest priority until it is moved on Sorting Logic
                    g["priority"] = list(g.get("priority", [])) + [name]
                    known_fire.add(key)
        g["good_classes"] = list(good.values())
        self.shared_state.set_grading(g)
        log.info("[MODELS] class settings after activation: PASS %s, priority %s", g["good_classes"], g["priority"])

    def _set_production_capture(self, on):
        with self.shared_state.lock:
            self.shared_state.capture_cfg["enabled"] = bool(on)
        self.shared_state.save_to_config(CFG_PATH)
        log.info("Production capture %s", "ON" if on else "OFF")

    def _ask_password(self, title, key, who):
        """True when the password of `who` (supervisor / admin) was entered correctly. The first time, it is set here;
        only a salted hash is stored in machine_config.json. A password is never written to a log."""
        import supervisor
        from tkinter import simpledialog
        path = app_paths.MACHINE_CONFIG_PATH
        if not supervisor.is_set(path, key):
            if not messagebox.askyesno(title, f"No {who} password has been set on this machine yet.\n\nSet it now?"):
                return False
            first = simpledialog.askstring(title, f"New {who} password (at least 4 characters):", show="*", parent=self)
            if first is None:
                return False
            second = simpledialog.askstring(title, "The same password again:", show="*", parent=self)
            if first != second:
                messagebox.showerror(title, "The two entries are different. No password was set.")
                return False
            try:
                supervisor.set_password(path, first, key)
            except ValueError as e:
                messagebox.showerror(title, str(e))
                return False
            log.info("%s password set", who.capitalize())
            return True
        password = simpledialog.askstring(title, f"{who.capitalize()} password:", show="*", parent=self)
        if password is None:
            return False
        if not supervisor.check(path, password, key):
            messagebox.showerror(title, "Wrong password. Nothing was changed.")
            return False
        return True

    def _ask_supervisor_password(self, title):
        import supervisor
        return self._ask_password(title, supervisor.KEY, "supervisor")

    def _ask_admin_password(self, title):
        import supervisor
        return self._ask_password(title, supervisor.ADMIN_KEY, "admin")

    # ── AUTO LEARNING switch (auto_learning.py has the rules; the learning itself is a later phase) ──
    def _auto_learning_status(self):
        """(enabled, samples, target). The sample count is read from disk at most every 10 s."""
        import auto_learning
        now = time.time()
        if now - getattr(self, "_samples_at", 0.0) > 10.0:
            self._samples = auto_learning.sample_count(app_paths.CLEAN_DATASET_DIR, app_paths.TESTSET_DIR)
            self._samples_at = now
        cfg = auto_learning.settings(app_paths.MACHINE_CONFIG_PATH)
        return cfg["enabled"], self._samples, cfg["target"]

    def _auto_learning_toggle(self, want):
        import auto_learning
        enabled, samples, _ = self._auto_learning_status()
        page = getattr(self, "teach_page", None)
        if not self._ask_admin_password("AUTO LEARNING"):
            if page is not None:
                page.auto_switch.set(enabled)
            return
        self._samples_at = 0.0
        _, samples, _ = self._auto_learning_status()
        new, message = auto_learning.set_enabled(app_paths.MACHINE_CONFIG_PATH, bool(want), samples, app_paths.LOGS_DIR)
        if page is not None:
            page.auto_switch.set(new)
            page.lbl_auto_note.config(text=message, fg=ui.WARN if new != bool(want) else ui.MUTED)
        messagebox.showinfo("AUTO LEARNING", message)

    def _auto_learning_target(self):
        import auto_learning
        from tkinter import simpledialog
        if not self._ask_admin_password("AUTO LEARNING target"):
            return
        enabled, samples, target = self._auto_learning_status()
        value = simpledialog.askinteger("AUTO LEARNING target", "Samples needed before AUTO LEARNING can be turned ON:", initialvalue=target,
                                        minvalue=1, maxvalue=100_000_000, parent=self)
        if value is None:
            return
        try:
            auto_learning.set_target(app_paths.MACHINE_CONFIG_PATH, value, samples, app_paths.LOGS_DIR)
        except ValueError as e:
            messagebox.showerror("AUTO LEARNING target", str(e))

    def _begin_training(self, kind, epochs, work):
        """Run work(progress, stop) in a thread and show the live window. work returns the result dict."""
        import training_ui
        self.train_status = {"kind": kind, "running": True, "epochs": int(epochs), "history": [], "seconds_left": 0.0,
                             "result": None, "error": "", "stop": False, "started": time.time()}
        status = self.train_status
        self.is_training = True

        def progress(row):
            status["seconds_left"] = row.pop("seconds_left", 0.0)
            status["history"].append(row)

        def run():
            try:
                status["result"] = work(progress, lambda: status["stop"])
                if kind == "CNN":
                    dataset_ops.training_finished(self._active_dataset_dir())      # "new images since the last training" starts again
                log.info("[TRAIN] %s training finished: %s (not active until it is activated)", kind, status["result"].get("model"))
            except Exception as exc:
                log.exception("[TRAIN] %s training failed", kind)
                status["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                status["running"] = False
                self.is_training = False
        threading.Thread(target=run, daemon=True, name=f"{kind}Training").start()
        training_ui.open_training_window(self)

    def _training_blocked(self):
        if getattr(self, "is_training", False):
            messagebox.showwarning("Training", "A training is already running. Wait for it to finish or stop it in its window.")
            return True
        if self.shared_state.phase == "RUNNING":
            messagebox.showwarning("Training", "Stop the machine first: training uses the same graphics card as the grader "
                                               "and would slow the inspection down.")
            return True
        return False

    def _start_training_router(self):
        self._start_cnn_train()

    def _start_cnn_train(self):
        import training
        import training_ui
        if self._training_blocked():
            return
        dataset_dir = self._active_dataset_dir()
        images = training.list_class_images(dataset_dir)
        if not images:
            messagebox.showerror("Train CNN", f"No class folders found in\n{dataset_dir}")
            return
        try:
            epochs = int(self.tcm_epochs.get())
            if not 1 <= epochs <= 500:
                raise ValueError
        except (ValueError, tk.TclError):
            messagebox.showerror("Train CNN", "Epochs must be a whole number from 1 to 500.")
            return
        prefix = "cnn_" + "".join(ch if ch.isalnum() else "_" for ch in str(self.active_program_name))

        def start(classes):
            # training.py reads every image through preprocess.py, the same crop and padding as live grading.
            self._begin_training("CNN", epochs, lambda progress, stop: self._cnn_trainer(
                dataset_dir, classes, epochs, str(MODEL_ROOT), name_prefix=prefix, progress=progress, stop=stop,
                pass_classes=self._class_settings()[1]))
        training_ui.choose_classes(self, images, epochs, start)

    def _cnn_trainer(self, *args, **kwargs):
        import training
        return training.train_cnn(*args, **kwargs)

    CONFUSION_CONFIDENCE = 0.60      # CHECK CONFUSION also lists images the model labels right but below this confidence

    def _teach_changed(self):
        page = getattr(self, "teach_page", None)
        if page is not None:
            page._class_signature = None
            page._signature = None

    def _review_wrong_images(self):
        """After a training: the images the new model still gets wrong, to keep, move, or move to CONFUSION in one click."""
        import training
        import training_ui
        page = self.teach_page
        result = page._last_result()
        wrong = [m for m in (result or {}).get("wrong_images", []) if os.path.exists(m["path"])]
        if not wrong:
            messagebox.showinfo("Still wrong", "No image of the last training is listed as wrong (or they were all moved already).")
            return None
        dataset_dir = self._active_dataset_dir()
        folders = {name.upper().strip(): os.path.join(dataset_dir, name) for name in training.list_class_images(dataset_dir)}
        intro = (f"Model: {result['model']}.   After training it still gets {len(wrong)} of its own training / validation images wrong.\n"
                 "Either the label is wrong, or the image is unclear. MOVE ALL LISTED TO CONFUSION takes them out of the next training in one click; "
                 "review them later on the CONFUSION tab.")
        return training_ui.open_mismatch_window(self, {"mismatches": wrong, "skipped": {}}, result["model"], folders, on_changed=self._teach_changed,
                                                changes=page.changes, dataset_dir=dataset_dir, title="Images the new model still gets wrong", intro=intro)

    def _verify_mismatches(self):
        """Run the active CNN on the labelled images and list those where label and prediction disagree."""
        import training
        import training_ui
        if getattr(self, "_verify_running", False):
            return
        if self.shared_state.phase == "RUNNING":
            messagebox.showwarning("Check confusion", "Stop the machine first: the check uses the same graphics card as the grader.")
            return
        model_path = self._active_cnn_path()
        if not Path(model_path).exists():
            messagebox.showerror("Check confusion", f"The active CNN model was not found:\n{model_path}")
            return
        dataset_dir = self._active_dataset_dir()
        self._verify_running = True
        self.config(cursor="watch")

        def done(result=None, error=""):
            self._verify_running = False
            self.config(cursor="")
            if error:
                messagebox.showerror("Check confusion", f"The check failed:\n{error}")
                return
            folders = {name.upper().strip(): os.path.join(dataset_dir, name) for name in training.list_class_images(dataset_dir)}
            if not result["checked"]:
                messagebox.showwarning("Check confusion", f"Nothing could be checked: the model ({', '.join(result['model_classes'])}) "
                                                        f"knows none of the class folders in\n{dataset_dir}")
                return
            training_ui.open_mismatch_window(self, result, Path(model_path).name, folders, on_changed=self._teach_changed,
                                             changes=self.teach_page.changes, dataset_dir=dataset_dir, title="Check confusion")

        def run():
            try:
                result = training.verify_labels(str(model_path), dataset_dir, low_confidence=self.CONFUSION_CONFIDENCE)
                self.after(0, lambda: done(result))
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                self.after(0, lambda: done(error=message))
        threading.Thread(target=run, daemon=True, name="VerifyLabels").start()

    def _launch_tcm_studio(self):
        import subprocess
        script = app_paths.TCM_STUDIO_SCRIPT
        if not script.exists():
            messagebox.showerror("Not available", f"TCM Studio Pro was not found:\n{script}")
            return
        try:
            subprocess.Popen([sys.executable, str(script)], cwd=str(script.parent))
        except Exception as e:
            log.exception("Could not start TCM Studio Pro")
            messagebox.showerror("Launch Error", f"Failed to start TCM Studio Pro: {e}")

    def _start_yolo_train(self, arch=None):
        if self._training_blocked():
            return
        if arch is None:        # buttons that do not pass an architecture use the selector of the Teaching CNN page
            arch = self.tcm_arch_var.get() if hasattr(self, 'tcm_arch_var') else ""
            if "YOLO" not in arch:
                arch = "YOLO11-Classifier"
        self.is_training = True
        self.ann_train_btn.config(state=tk.DISABLED)
        
        def train_thread():
            try:
                import shlex
                
                # Since we are doing YOLO Classification, the dataset is already in DATASET_DIR (cashew_dataset_images/ClassX/...)
                # YOLO classify requires a specific folder structure (train, val).
                # We will copy the dataset to a temp folder and split into train/val.
                
                temp_dataset_dir = str(app_paths.RUNS_DIR / "temp_yolo_cls_dataset")
                if os.path.exists(temp_dataset_dir):
                    shutil.rmtree(temp_dataset_dir)
                
                train_dir = os.path.join(temp_dataset_dir, "train")
                val_dir = os.path.join(temp_dataset_dir, "val")
                os.makedirs(train_dir)
                os.makedirs(val_dir)
                
                class_folders = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
                
                for cname in class_folders:
                    src_c = os.path.join(DATASET_DIR, cname)
                    imgs = glob.glob(os.path.join(src_c, "*.jpg"))
                    if not imgs: continue
                    
                    dst_train = os.path.join(train_dir, cname)
                    dst_val = os.path.join(val_dir, cname)
                    os.makedirs(dst_train)
                    os.makedirs(dst_val)
                    
                    # 80/20 split
                    split_idx = int(len(imgs) * 0.8)
                    for img in imgs[:split_idx]: shutil.copy(img, dst_train)
                    for img in imgs[split_idx:]: shutil.copy(img, dst_val)


                # 4. Run Subprocess Training
                train_script = str(app_paths.RUNS_DIR / "temp_train.py")
                res_file = str(app_paths.RUNS_DIR / "temp_results.json")
                if os.path.exists(res_file): os.remove(res_file)
                
                # We determine if they picked YOLO11 or YOLOv8 based on the dropdown
                # Assuming 'yolo11n-cls.pt' for YOLO11, and 'yolov8n-cls.pt' for YOLOv8
                base_model = 'yolo11n-cls.pt' if '11' in arch else 'yolov8n-cls.pt'
                
                with open(train_script, "w") as f:
                    f.write(f"""
import sys, json, os
from ultralytics import YOLO

try:
    model = YOLO({base_model!r})
    results = model.train(
        data={temp_dataset_dir!r},
        epochs={int(self.ann_epochs.get())},
        imgsz=224,
        project={str(app_paths.RUNS_DIR)!r},
        verbose=True, workers=0
    )
    # Classification metrics
    if hasattr(results, 'top1'):
        top1 = getattr(results, 'top1', 0.0)
        top5 = getattr(results, 'top5', 0.0)
        with open({res_file!r}, 'w') as rf:
            json.dump({{'top1': top1, 'top5': top5}}, rf)
            
except Exception as e:
    print(f"SUBPROCESS ERROR: {{e}}")
    sys.exit(1)
""")
                import subprocess
                cmd = ["gnome-terminal", "--wait", "--", "bash", "-c",
                       f"cd {shlex.quote(str(app_paths.BASE_DIR))} && {shlex.quote(sys.executable)} {shlex.quote(train_script)}; read -p 'Press Enter to exit...'"]
                process = subprocess.Popen(cmd)
                process.wait()
                
                if process.returncode == 0 and os.path.exists(res_file):
                    with open(res_file, 'r') as rf:
                        r = json.load(rf)
                    summary = f"\n\n--- FINAL RESULTS ---\nTop-1 Accuracy: {r.get('top1', 0)*100:.2f}%\nTop-5 Accuracy: {r.get('top5', 0)*100:.2f}%\n---------------------\n"
                    print(summary)
                else:
                    log.warning("Training terminal closed with code %s and no result file.", process.returncode)
                
            except Exception:
                log.exception("YOLO classifier training failed")
            finally:
                self.is_training = False
                self.after(0, lambda: self.ann_train_btn.config(state=tk.NORMAL))
                
        threading.Thread(target=train_thread, daemon=True).start()


    def _show_mismatches(self, mismatches):
        self.tcm_log.insert(tk.END, f"Found {len(mismatches)} mismatched images.\n")
        if not mismatches: return
        
        popup = tk.Toplevel(self)
        popup.title("Mismatched Images")
        popup.geometry("600x400")
        
        ttk.Label(popup, text=f"{len(mismatches)} images have potential label mismatches.", font=("Roboto", 12, "bold")).pack(pady=10)
        
        tree = ttk.Treeview(popup, columns=("Image Path",), show="headings")
        tree.heading("Image Path", text="Image Path")
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        for path in mismatches:
            tree.insert("", "end", values=(path,))
            
        def on_double_click(event):
            selected = tree.selection()
            if not selected: return
            item = tree.item(selected[0])
            path = item["values"][0]
            self._switch_page("Teaching YOLO")
            self.show_mismatch_preds = True
            
            
            folder = os.path.dirname(path)
            if not self.ann_image_list or folder != os.path.dirname(self.ann_image_list[0]):
                import glob
                self.ann_image_list = sorted(glob.glob(os.path.join(folder, "*.jpg")))
            
            if path in self.ann_image_list:
                self.ann_image_idx = self.ann_image_list.index(path)
                self._ann_load_image()
            
            popup.destroy()
            
        tree.bind("<Double-1>", on_double_click)
        ttk.Label(popup, text="Double-click an image to fix labels.").pack(pady=5)

    def _build_annotate_page(self):
        p_ann = tk.Frame(self.page_container, bg=self.colors["surface"], padx=15, pady=15)
        self.pages["Annotate"] = p_ann
        
        ttk.Label(p_ann, text="MANUAL ANNOTATION (MAGIC WAND)", style="Title.TLabel").pack(anchor="w", pady=(0, 10))
        
        t_row = tk.Frame(p_ann, bg=self.colors["bg"], padx=10, pady=5)
        t_row.pack(fill=tk.X, pady=(0, 10))
        
        self.ann_image_list = []
        self.ann_image_idx = -1
        self.ann_current_img = None
        self.ann_display_img = None
        self.ann_current_labels = []
        
        ttk.Button(t_row, text="Open Folder", command=self._ann_open_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(t_row, text="◀ Prev", command=self._ann_prev).pack(side=tk.LEFT, padx=5)
        ttk.Button(t_row, text="Next ▶", command=self._ann_next).pack(side=tk.LEFT, padx=5)
        
        self.ann_lbl_file = ttk.Label(t_row, text="No image loaded.")
        self.ann_lbl_file.pack(side=tk.LEFT, padx=15)
        
        fast_row = tk.Frame(p_ann, bg=self.colors["surface"], pady=10)
        fast_row.pack(fill=tk.X)
        ttk.Label(fast_row, text="Select Class to Mark:", style="Header.TLabel").pack(anchor="w", pady=(0, 5))
        
        self.ann_fast_class_var = tk.StringVar()
        if self.all_class_names: self.ann_fast_class_var.set(self.all_class_names[0])
        
        btn_grid = tk.Frame(fast_row, bg=self.colors["surface"])
        btn_grid.pack(fill=tk.X)
        
        cols = 3
        for i, cname in enumerate(self.all_class_names):
            r, c = divmod(i, cols)
            ttk.Radiobutton(btn_grid, text=cname, variable=self.ann_fast_class_var, value=cname, style="Toolbutton").grid(row=r, column=c, padx=2, pady=2, sticky="ew")
            btn_grid.columnconfigure(c, weight=1)
        
        ttk.Label(p_ann, text="Current Annotations:", style="Header.TLabel").pack(anchor="w", pady=(15, 5))
        
        edit_row = tk.Frame(p_ann, bg=self.colors["surface"], pady=5)
        edit_row.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Dataset Verification & Training
        train_frm = tk.Frame(p_ann, bg=self.colors["bg"], padx=10, pady=5)
        train_frm.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
        
        ttk.Label(train_frm, text="YOLOv11 Model Training & Verification", style="Header.TLabel").pack(anchor="w", pady=(0,5))
        
        row2 = tk.Frame(train_frm, bg=self.colors["bg"])
        row2.pack(fill=tk.X, pady=2)
        
        ttk.Label(row2, text="Epochs:").pack(side=tk.LEFT, padx=(0,5))
        self.ann_epochs = tk.IntVar(value=100)
        tk.Entry(row2, textvariable=self.ann_epochs, width=5).pack(side=tk.LEFT, padx=(0,15))
        
        ttk.Label(row2, text="Base Model:").pack(side=tk.LEFT, padx=(0,5))
        self.tcm_base_model = tk.StringVar(value="yolo11n.pt")
        tk.Entry(row2, textvariable=self.tcm_base_model, width=15).pack(side=tk.LEFT, padx=(0,15))
        
        row3 = tk.Frame(train_frm, bg=self.colors["bg"])
        row3.pack(fill=tk.X, pady=5)
        
        ttk.Button(row3, text="Cross Check Labels", style="Primary.TButton", command=self._launch_tcm_studio).pack(side=tk.LEFT, padx=(0,10))
        
        self.ann_train_btn = ttk.Button(row3, text="START TRAINING", style="Success.TButton", command=self._start_yolo_train)
        self.ann_train_btn.pack(side=tk.LEFT)
        
        self.tcm_train_progress = ttk.Progressbar(train_frm, orient=tk.HORIZONTAL, mode='determinate')
        self.tcm_train_progress.pack(fill=tk.X, pady=(5,2))
        
        self.tcm_log = tk.Text(train_frm, bg="black", fg="white", height=5)
        self.tcm_log.pack(fill=tk.X, pady=(2,0))        
        ttk.Button(edit_row, text="Rename Selected", command=self._ann_rename).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(edit_row, text="Delete Selected", command=self._ann_delete, style="Danger.TButton").pack(side=tk.LEFT)

        tree_frm = tk.Frame(p_ann, bg=self.colors["surface"])
        tree_frm.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.ann_tree = ttk.Treeview(tree_frm, columns=("ID", "Class Name"), show="headings", height=5)
        self.ann_tree.heading("ID", text="ID")
        self.ann_tree.heading("Class Name", text="Class Name")
        self.ann_tree.column("ID", width=50, anchor="center")
        self.ann_tree.column("Class Name", width=200, anchor="w")
        self.ann_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        sb = ttk.Scrollbar(tree_frm, orient=tk.VERTICAL, command=self.ann_tree.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.ann_tree.configure(yscrollcommand=sb.set)
        
        self.ann_canvas = tk.Canvas(self.center_frm, bg="black", cursor="crosshair")
        self.ann_canvas.bind("<Button-1>", self._ann_on_click)
        
        def _on_ann_resize(event):
            if hasattr(self, '_ann_resize_id'):
                self.after_cancel(self._ann_resize_id)
            self._ann_resize_id = self.after(50, self._ann_draw_image)
        self.ann_canvas.bind("<Configure>", _on_ann_resize)

    def _ann_open_folder(self):
        from tkinter import filedialog
        folder = filedialog.askdirectory(initialdir=DATASET_DIR)
        if folder:
            import glob
            self.ann_image_list = sorted(glob.glob(os.path.join(folder, "*.jpg")))
            self.ann_image_idx = 0 if self.ann_image_list else -1
            if self.ann_image_list:
                self.video_label.pack_forget()
                if hasattr(self, 'ann_canvas'):
                    self.ann_canvas.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
            self._ann_load_image()

    def _ann_prev(self):
        if self.ann_image_list and self.ann_image_idx > 0:
            self.ann_image_idx -= 1
            self._ann_load_image()

    def _ann_next(self):
        if self.ann_image_list and self.ann_image_idx < len(self.ann_image_list) - 1:
            self.ann_image_idx += 1
            self._ann_load_image()

    def _start_auto_teach(self):
        with self.shared_state.lock:
            frame = self.shared_state.frame
        if frame is None:
            messagebox.showerror("Error", "No camera feed available.")
            return
            
        target_size = self.at_size_var.get()
        if not target_size: return
        
        popup = tk.Toplevel(self)
        popup.title(f"Auto-Teach Size: {target_size}")
        popup.geometry("800x600")
        
        lbl = ttk.Label(popup, text=f"Click on the {target_size} Cashew to calculate dimensions.", font=("Roboto", 12))
        lbl.pack(pady=10)
        
        canvas = tk.Canvas(popup, bg="black", cursor="crosshair")
        canvas.pack(fill=tk.BOTH, expand=True)
        
        cw, ch = 800, 500
        h, w = frame.shape[:2]
        scale = min(cw/w, ch/h)
        nw, nh = int(w*scale), int(h*scale)
        
        resized = cv2.resize(frame, (nw, nh))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        from PIL import Image, ImageTk
        pil_img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(pil_img)
        canvas.create_image((cw-nw)//2, (ch-nh)//2, anchor=tk.NW, image=photo)
        canvas.image = photo 
        
        def on_click(event):
            ox = int((event.x - (cw-nw)//2) / scale)
            oy = int((event.y - (ch-nh)//2) / scale)
            
            if 0 <= ox < w and 0 <= oy < h:
                if tcm_magic_wand is None:
                    messagebox.showerror("Not available", f"The magic wand tool was not found in\n{app_paths.EXTERNAL_TOOLS_DIR}", parent=popup)
                    return
                try:
                    pts = magic_polygon(frame, ox, oy)
                    if len(pts) > 0:
                        pts_np = np.array(pts, dtype=np.int32)
                        area = cv2.contourArea(pts_np)
                        x, y, bw, bh = cv2.boundingRect(pts_np)
                        
                        px_per_mm = float(self.sz_vars["px_per_mm"].get())
                        if px_per_mm == 0: px_per_mm = 10.0
                        
                        mm_L = max(bw, bh) / px_per_mm
                        mm_W = min(bw, bh) / px_per_mm
                        
                        if f"{target_size}_L" in self.sz_vars:
                            self.sz_vars[f"{target_size}_L"].set(f"{mm_L:.1f}")
                        if f"{target_size}_W" in self.sz_vars:
                            self.sz_vars[f"{target_size}_W"].set(f"{mm_W:.1f}")
                            
                        self._save_program()
                        
                        msg = f"Taught {target_size}:\nLength: {mm_L:.1f} mm\nWidth: {mm_W:.1f} mm\nArea: {area} px"
                        print(f"[AUTO-TEACH] {msg.replace(chr(10), ' | ')}")
                        messagebox.showinfo("Success", msg, parent=popup)
                        popup.destroy()
                    else:
                        print("[AUTO-TEACH] Warning: Could not find cashew boundary.")
                        messagebox.showwarning("Warning", "Could not find cashew boundary.", parent=popup)
                except Exception as e:
                    log.exception("[AUTO-TEACH] Magic wand failed")
                    messagebox.showerror("Error", f"Magic Wand failed: {e}", parent=popup)
                    
        canvas.bind("<Button-1>", on_click)

    def _ann_load_image(self):
        if self.ann_image_idx < 0 or self.ann_image_idx >= len(self.ann_image_list): return
        path = self.ann_image_list[self.ann_image_idx]
        self.ann_lbl_file.config(text=os.path.basename(path) + f" ({self.ann_image_idx+1}/{len(self.ann_image_list)})")
        
        img = cv2.imread(path)
        if img is None: return
        self.ann_current_img = img
        
        self.ann_current_labels = []
        txt_path = path.replace(".jpg", ".txt")
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        parts = line.split()
                        cls_id = int(parts[0])
                        cname = self.class_map.get(cls_id, str(cls_id)) if hasattr(self, 'class_map') else str(cls_id)
                        poly_str = " ".join(parts[1:])
                        self.ann_current_labels.append({"cid": cls_id, "coords": poly_str, "cname": cname})
            except Exception:
                log.exception("Could not read the label file %s", txt_path)
            
        self.ann_pred_labels = []
        if getattr(self, 'show_mismatch_preds', False) and os.path.exists(self.tcm_base_model.get()):
            try:
                from ultralytics import YOLO
                model = YOLO(self.tcm_base_model.get())
                results = model(self.ann_current_img, verbose=False)
                if len(results) > 0:
                    for box in results[0].boxes:
                        cid = int(box.cls.item())
                        cname = self.class_map.get(cid, str(cid)) if hasattr(self, 'class_map') else str(cid)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        self.ann_pred_labels.append({"cname": f"AI: {cname}", "coords": [int(x1), int(y1), int(x2), int(y2)]})
            except Exception:
                log.exception("Could not run the model on %s", path)
            
        self._ann_update_tree()
        self._ann_draw_image()

    def _ann_update_tree(self):
        for item in self.ann_tree.get_children():
            self.ann_tree.delete(item)
        for i, lbl in enumerate(self.ann_current_labels):
            self.ann_tree.insert("", "end", iid=str(i), values=(i+1, lbl["cname"]))
            
    def _ann_save_txt(self):
        if self.ann_image_idx < 0: return
        path = self.ann_image_list[self.ann_image_idx]
        txt_path = path.replace(".jpg", ".txt")
        with open(txt_path, 'w') as f:
            for lbl in self.ann_current_labels:
                f.write(f"{lbl['cid']} {lbl['coords']}\n")

    def _get_ann_color(self, cid: int):
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 165, 255), (255, 255, 0), (0, 255, 255),
            (255, 0, 255), (200, 200, 200), (128, 0, 0), (128, 128, 0), (0, 128, 0),
            (128, 0, 128), (0, 128, 128), (0, 0, 128), (255, 128, 0), (255, 0, 128),
            (100, 149, 237), (220, 20, 60), (50, 205, 50), (255, 105, 180), (255, 215, 0)
        ]
        return colors[cid % len(colors)]

    def _ann_draw_image(self):
        if self.ann_current_img is None: return
        img = self.ann_current_img.copy()
        
        h, w = img.shape[:2]
        
        for lbl in self.ann_current_labels:
            parts = lbl["coords"].split()
            cname = lbl["cname"]
            cid = lbl["cid"]
            color = self._get_ann_color(cid)
            
            if len(parts) == 4:
                xc, yc, bw, bh = map(float, parts)
                x1 = int((xc - bw/2) * w)
                y1 = int((yc - bh/2) * h)
                x2 = int((xc + bw/2) * w)
                y2 = int((yc + bh/2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, cname, (x1, max(10, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            elif len(parts) >= 6 and len(parts) % 2 == 0:
                coords = list(map(float, parts))
                pts = []
                for i in range(0, len(coords), 2):
                    pts.append([int(coords[i]*w), int(coords[i+1]*h)])
                pts = np.array(pts, np.int32).reshape((-1, 1, 2))
                
                overlay = img.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
                cv2.polylines(img, [pts], True, color, 2)
                
                cv2.putText(img, cname, (pts[0][0][0], max(10, pts[0][0][1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if getattr(self, 'show_mismatch_preds', False) and hasattr(self, 'ann_pred_labels'):
            for pred in self.ann_pred_labels:
                x1, y1, x2, y2 = pred["coords"]
                cname = pred["cname"]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for AI
                cv2.putText(img, cname, (x1, max(10, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cw = self.ann_canvas.winfo_width()
        ch = self.ann_canvas.winfo_height()
        if cw < 10 or ch < 10: 
            cw, ch = 800, 600
            
        scale = min(cw/w, ch/h)
        nw, nh = int(w*scale), int(h*scale)
        self.ann_scale = scale
        self.ann_nw = nw
        self.ann_nh = nh
        self.ann_pad_x = (cw - nw) // 2
        self.ann_pad_y = (ch - nh) // 2
        
        resized = cv2.resize(img, (nw, nh))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        from PIL import Image, ImageTk
        pil_img = Image.fromarray(rgb)
        self.ann_display_img = ImageTk.PhotoImage(pil_img)
        self.ann_canvas.delete("all")
        self.ann_canvas.create_image(self.ann_pad_x, self.ann_pad_y, anchor=tk.NW, image=self.ann_display_img)

    def _ann_on_click(self, event):
        if self.ann_current_img is None or not tcm_magic_wand: return
        
        x = event.x - getattr(self, 'ann_pad_x', 0)
        y = event.y - getattr(self, 'ann_pad_y', 0)
        scale = getattr(self, 'ann_scale', 1.0)
        
        if x < 0 or y < 0 or x >= getattr(self, 'ann_nw', 0) or y >= getattr(self, 'ann_nh', 0): return
        
        orig_x = int(x / scale)
        orig_y = int(y / scale)
        
        poly = magic_polygon(self.ann_current_img, orig_x, orig_y)
        if poly is not None and len(poly) > 2:
            cname = self.ann_fast_class_var.get()
            cid = 0
            for k,v in self.class_map.items():
                if v == cname: cid = k; break
                
            h, w = self.ann_current_img.shape[:2]
            normalized_points = []
            for pt in poly:
                nx = pt[0][0] / float(w)
                ny = pt[0][1] / float(h)
                normalized_points.append(f"{nx:.6f} {ny:.6f}")
                
            poly_str = " ".join(normalized_points)
            self.ann_current_labels.append({"cid": cid, "coords": poly_str, "cname": cname})
            
            self._ann_save_txt()
            self._ann_update_tree()
            self._ann_draw_image()

    def _ann_rename(self):
        selected = self.ann_tree.selection()
        if not selected: return
        idx = int(selected[0])
        
        cname = self.ann_fast_class_var.get()
        cid = 0
        for k,v in self.class_map.items():
            if v == cname: cid = k; break
            
        self.ann_current_labels[idx]["cname"] = cname
        self.ann_current_labels[idx]["cid"] = cid
        self._ann_save_txt()
        self._ann_update_tree()
        self._ann_draw_image()

    def _ann_delete(self):
        selected = self.ann_tree.selection()
        if not selected: return
        idx = int(selected[0])
        del self.ann_current_labels[idx]
        self._ann_save_txt()
        self._ann_update_tree()
        self._ann_draw_image()

    def _build_test_console_page(self):
        p_test = tk.Frame(self.page_container, bg=self.colors["surface"], padx=15, pady=15)
        self.pages["Test Console"] = p_test
        
        ttk.Label(p_test, text="TEST CONSOLE (INFERENCE)", style="Title.TLabel").pack(anchor="w", pady=(0, 10))
        
        t_row = tk.Frame(p_test, bg=self.colors["bg"], padx=10, pady=5)
        t_row.pack(fill=tk.X, pady=(0, 10))
        
        self.test_model_path = tk.StringVar(value=self.shared_state.yolo_model_path or app_paths.relative(self.model_path or MODEL_ROOT / "best.pt"))
        ttk.Label(t_row, text="Model:").pack(side=tk.LEFT, padx=(0,5))
        tk.Entry(t_row, textvariable=self.test_model_path, width=40, bg=self.colors["surface"], fg="white").pack(side=tk.LEFT, padx=5)
        
        ttk.Button(t_row, text="Load Media", command=self._test_load_media).pack(side=tk.LEFT, padx=(15, 5))
        ttk.Button(t_row, text="Load Folder", command=self._test_load_folder).pack(side=tk.LEFT, padx=5)
        
        # Row 2 for Navigation
        t_row2 = tk.Frame(p_test, bg=self.colors["bg"], padx=10, pady=5)
        t_row2.pack(fill=tk.X, pady=(0, 10))
        
        nav_frm = tk.Frame(t_row2, bg=self.colors["bg"])
        nav_frm.pack(side=tk.LEFT, padx=5)
        
        self.test_prev_btn = ttk.Button(nav_frm, text="< Prev", command=self._test_prev_img, width=6)
        self.test_prev_btn.pack(side=tk.LEFT, padx=2)
        
        self.test_counter_lbl = tk.Label(nav_frm, text="0 / 0", bg=self.colors["bg"], fg="white", width=10)
        self.test_counter_lbl.pack(side=tk.LEFT, padx=5)
        
        self.test_next_btn = ttk.Button(nav_frm, text="Next >", command=self._test_next_img, width=6)
        self.test_next_btn.pack(side=tk.LEFT, padx=2)
        
        self.test_prev_btn.config(state=tk.DISABLED)
        self.test_next_btn.config(state=tk.DISABLED)
        
        self.test_play_btn = ttk.Button(t_row2, text="Play Video", command=self._test_toggle_play)
        self.test_play_btn.pack(side=tk.LEFT, padx=15)
        self.test_play_btn.config(state=tk.DISABLED)
        
        self.test_status_lbl = ttk.Label(t_row2, text="Ready.")
        self.test_status_lbl.pack(side=tk.RIGHT, padx=15)
        
        self.test_canvas = tk.Canvas(self.center_frm, bg="black")
        
        self.test_is_playing = False
        self.test_cap = None
        self.test_model = None
        self.test_current_model_path = ""
        
        self.test_folder_images = []
        self.test_idx = 0

    def _test_ensure_model(self) -> bool:
        """Load the YOLO model named in the Test Console (only when it changed)."""
        model_path = str(app_paths.resolve(self.test_model_path.get().strip()))
        if not model_path.endswith(".pt") or not os.path.exists(model_path):
            self.test_status_lbl.config(text=f"YOLO .pt model not found: {model_path}")
            return False
        if self.test_model is None or self.test_current_model_path != model_path:
            try:
                self.test_model = YOLO(model_path)
                self.test_class_map = {int(k): str(v) for k, v in self.test_model.names.items()}
                self.test_current_model_path = model_path
            except Exception as e:
                log.exception("Test Console: could not load %s", model_path)
                self.test_status_lbl.config(text=f"Model error: {e}")
                return False
        return True

    def _test_load_folder(self):
        folder = filedialog.askdirectory()
        if not folder: return
        if not self._test_ensure_model():
            return
                
        # Gather images
        images = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG'):
            images.extend(glob.glob(os.path.join(folder, ext)))
            
        self.test_folder_images = sorted(images)
        self.test_idx = 0
        
        if not self.test_folder_images:
            self.test_status_lbl.config(text="No images found in folder.")
            self.test_prev_btn.config(state=tk.DISABLED)
            self.test_next_btn.config(state=tk.DISABLED)
            self.test_counter_lbl.config(text="0 / 0")
            return
            
        self.test_prev_btn.config(state=tk.NORMAL)
        self.test_next_btn.config(state=tk.NORMAL)
        self.test_play_btn.config(state=tk.DISABLED)
        
        self._test_show_current_img()

    def _test_prev_img(self):
        if not self.test_folder_images: return
        self.test_idx -= 1
        if self.test_idx < 0: self.test_idx = len(self.test_folder_images) - 1
        self._test_show_current_img()
        
    def _test_next_img(self):
        if not self.test_folder_images: return
        self.test_idx += 1
        if self.test_idx >= len(self.test_folder_images): self.test_idx = 0
        self._test_show_current_img()
        
    def _test_show_current_img(self):
        if not self.test_folder_images: return
        path = self.test_folder_images[self.test_idx]
        self.test_counter_lbl.config(text=f"{self.test_idx + 1} / {len(self.test_folder_images)}")
        img = cv2.imread(path)
        if img is not None:
            self._test_run_inference(img)

    def _test_load_media(self):
        path = filedialog.askopenfilename(filetypes=[("Media Files", "*.jpg *.jpeg *.png *.mp4 *.avi")])
        if not path: return
        if not self._test_ensure_model():
            return
                
        self.test_is_playing = False
        if self.test_cap is not None:
            self.test_cap.release()
            self.test_cap = None
            
        ext = os.path.splitext(path)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png']:
            self.test_play_btn.config(state=tk.DISABLED)
            img = cv2.imread(path)
            if img is not None:
                self._test_run_inference(img)
        elif ext in ['.mp4', '.avi']:
            self.test_cap = cv2.VideoCapture(path)
            self.test_play_btn.config(state=tk.NORMAL, text="Play")
            ret, frame = self.test_cap.read()
            if ret:
                self._test_run_inference(frame)
                
    def _test_run_inference(self, img):
        if self.test_model is None: return
        results = self.test_model(img, verbose=False)
        if len(results) > 0:
            res_img = results[0].plot()
            count = len(results[0].boxes)
            self.test_status_lbl.config(text=f"Detections: {count}")
        else:
            res_img = img.copy()
            self.test_status_lbl.config(text="Detections: 0")
        cw = self.test_canvas.winfo_width()
        ch = self.test_canvas.winfo_height()
        if cw < 10 or ch < 10: cw, ch = 800, 600
        
        h, w = res_img.shape[:2]
        scale = min(cw/w, ch/h)
        nw, nh = int(w*scale), int(h*scale)
        
        resized = cv2.resize(res_img, (nw, nh))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        from PIL import Image, ImageTk
        pil_img = Image.fromarray(rgb)
        self.test_display_img = ImageTk.PhotoImage(pil_img)
        self.test_canvas.delete("all")
        self.test_canvas.create_image((cw-nw)//2, (ch-nh)//2, anchor=tk.NW, image=self.test_display_img)
        
    def _test_toggle_play(self):
        if self.test_cap is None: return
        self.test_is_playing = not self.test_is_playing
        if self.test_is_playing:
            self.test_play_btn.config(text="Pause")
            self._test_play_loop()
        else:
            self.test_play_btn.config(text="Play")
            
    def _test_play_loop(self):
        if not self.test_is_playing or self.test_cap is None: return
        ret, frame = self.test_cap.read()
        if ret:
            import time
            start = time.time()
            self._test_run_inference(frame)
            elapsed = time.time() - start
            delay = max(1, int(30 - elapsed*1000))
            self.after(delay, self._test_play_loop)
        else:
            self.test_is_playing = False
            self.test_play_btn.config(text="Play")
            self.test_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Settings-only pages use the whole width; the camera picture is shown where it is needed
    # (Home, Vision ROI, Dataset Capture; Teaching YOLO and Test Console have their own picture).
    PAGES_WITHOUT_CAMERA = ("Pneumatic", "Teaching CNN", "Colour Size", "Sorting Logic", "AI Model Select", "Network", "Info")

    def _switch_page(self, page_name):
        if hasattr(self, 'active_page'): self.active_page.set(page_name)
        for name, btn in self.nav_buttons.items():
            if hasattr(btn, 'set_active'):
                btn.set_active(name == page_name)
        for p in self.pages.values():
            p.pack_forget()
        if page_name in self.pages:
            self.pages[page_name].pack(fill=tk.BOTH, expand=True)
            
        # Full Screen Logic for specific pages
        if page_name in self.PAGES_WITHOUT_CAMERA:
            if hasattr(self, 'center_frm'):
                self.center_frm.pack_forget()
            if hasattr(self, 'right_frm'):
                self.right_frm.pack_forget()
                self.right_frm.config(width=900)
                self.right_frm.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=16, pady=16)
        else:
            if hasattr(self, 'center_frm'):
                self.center_frm.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=16, pady=16)
            if hasattr(self, 'right_frm'):
                self.right_frm.pack_forget()
                self.right_frm.config(width=ui.RIGHT_W)
                self.right_frm.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 16), pady=16)

        if page_name == "Teaching YOLO":
            if hasattr(self, 'test_canvas'): self.test_canvas.pack_forget()
            if getattr(self, 'ann_image_list', []):
                self.video_label.pack_forget()
                if hasattr(self, 'ann_canvas'):
                    self.ann_canvas.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
            else:
                if hasattr(self, 'ann_canvas'): self.ann_canvas.pack_forget()
                self.video_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        elif page_name == "Test Console":
            self.video_label.pack_forget()
            if hasattr(self, 'ann_canvas'): self.ann_canvas.pack_forget()
            if hasattr(self, 'test_canvas'):
                self.test_canvas.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        else:
            if hasattr(self, 'ann_canvas'): self.ann_canvas.pack_forget()
            if hasattr(self, 'test_canvas'): self.test_canvas.pack_forget()
            self.video_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

    def _set_phase(self, phase):
        with self.shared_state.lock:
            self.shared_state.phase = phase

    def _build_vision_page(self):
        """Vision ROI page in the 5-track layout: ROIs and decision line, calibration (ROI length -> px per mm),
        image sensor and colour gain, camera / test source, white calibration, recorder. One SAVE button."""
        F = ui.FONT
        page = tk.Frame(self.page_container, bg=ui.BG)
        self.pages["Vision ROI"] = page
        save_bar = tk.Frame(page, bg=ui.BG)
        save_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0))
        canvas = tk.Canvas(page, bg=ui.BG, highlightthickness=0)
        scroll = ttk.Scrollbar(page, orient="vertical", command=canvas.yview)
        p_vis = tk.Frame(canvas, bg=ui.BG)
        p_vis.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        window_id = canvas.create_window((0, 0), window=p_vis, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(window_id, width=e.width))
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        def card(title, note=""):
            box = tk.Frame(p_vis, bg=ui.PANEL, highlightthickness=1, highlightbackground=ui.BORDER)
            box.pack(fill=tk.X, pady=(0, 10), padx=(0, 6))
            head = tk.Frame(box, bg=ui.PANEL)
            head.pack(fill=tk.X, padx=14, pady=(10, 4))
            tk.Label(head, text=title, font=(F, ui.F_SMALL, "bold"), fg=ui.MUTED, bg=ui.PANEL).pack(side=tk.LEFT)
            if note:
                tk.Label(head, text=note, font=(F, -11), fg=ui.MUTED, bg=ui.PANEL).pack(side=tk.LEFT, padx=12)
            body = tk.Frame(box, bg=ui.PANEL)
            body.pack(fill=tk.X, padx=14, pady=(0, 12))
            return body

        def line(parent, pady=3):
            row = tk.Frame(parent, bg=ui.PANEL)
            row.pack(fill=tk.X, pady=pady)
            return row

        def text(parent, value, bold=False, colour=ui.TEXT, size=ui.F_SMALL, **kw):
            return tk.Label(parent, text=value, font=(F, size, "bold") if bold else (F, size), fg=colour, bg=ui.PANEL, **kw)

        def entry(parent, var, width=8):
            return tk.Entry(parent, textvariable=var, width=width, bg=ui.FIELD, fg=ui.TEXT, insertbackground=ui.TEXT, bd=0, justify="center",
                            disabledbackground=ui.BG, font=(F, -14), highlightthickness=1, highlightbackground=ui.BORDER, highlightcolor=ui.ACCENT)

        def note(parent, value):
            label = text(parent, value, colour=ui.MUTED, size=-11, justify="left", anchor="w", wraplength=570)
            label.pack(fill=tk.X, pady=(4, 0))
            return label

        # ── ROIs and decision line ──
        rois = card("ROIS AND DECISION LINE", "press DRAW, then drag on the camera image")
        self.lbl_roi_coords = {}
        for number, start, reset in ((1, "ROI1_START", self.shared_state.reset_roi1), (2, "ROI2_START", self.shared_state.reset_roi2)):
            row = line(rois)
            text(row, f"ROI {number}", bold=True, size=ui.F_BODY, width=6, anchor="w").pack(side=tk.LEFT)
            self.lbl_roi_coords[number] = text(row, "", colour=ui.MUTED)
            self.lbl_roi_coords[number].pack(side=tk.LEFT, padx=(4, 0))
            ui.FlatButton(row, "RESET", reset, kind="muted", height=30, width=84).pack(side=tk.RIGHT)
            ui.FlatButton(row, "DRAW", lambda s=start: self._set_phase(s), height=30, width=84).pack(side=tk.RIGHT, padx=8)
        tk.Frame(rois, bg=ui.BORDER, height=1).pack(fill=tk.X, pady=8)
        row = line(rois)
        text(row, "Decision line", bold=True, size=ui.F_BODY).pack(side=tk.LEFT)
        self.lbl_line_pos = text(row, "", bold=True, colour=ui.ACCENT)
        self.lbl_line_pos.pack(side=tk.LEFT, padx=14)
        ui.FlatButton(row, "RESET", self.shared_state.reset_line, kind="muted", height=30, width=84).pack(side=tk.RIGHT)
        ui.FlatButton(row, "SET LINE", lambda: self._set_phase("LINE_SET_MODE"), height=30, width=84).pack(side=tk.RIGHT, padx=8)

        line_names = {fg.VERTICAL: "Vertical  |", fg.HORIZONTAL: "Horizontal  —"}
        dir_frm = line(rois, pady=6)
        text(dir_frm, "Line").pack(side=tk.LEFT)
        self.line_orient_var = tk.StringVar(value=line_names[fg.line_orientation(self.shared_state.flow_direction)])
        orient_combo = ttk.Combobox(dir_frm, textvariable=self.line_orient_var, values=list(line_names.values()), state="readonly", width=13)
        orient_combo.pack(side=tk.LEFT, padx=(6, 16))
        text(dir_frm, "Cashews move").pack(side=tk.LEFT)
        self.flow_var = tk.StringVar(value=fg.FLOW_LABELS[self.shared_state.flow_direction])
        flow_combo = ttk.Combobox(dir_frm, textvariable=self.flow_var, state="readonly", width=14)
        flow_combo.pack(side=tk.LEFT, padx=6)

        def flows_of_selected_line():
            orientation = next(k for k, v in line_names.items() if v == self.line_orient_var.get())
            return fg.FLOWS_FOR_LINE[orientation]

        def show_current_direction():
            current = self.shared_state.flow_direction
            self.line_orient_var.set(line_names[fg.line_orientation(current)])
            flow_combo['values'] = [fg.FLOW_LABELS[f] for f in fg.FLOWS_FOR_LINE[fg.line_orientation(current)]]
            self.flow_var.set(fg.FLOW_LABELS[current])
            self._refresh_line_buttons()

        def change_flow(flow):
            if flow == self.shared_state.flow_direction:
                return
            if self.shared_state.phase == "RUNNING":
                messagebox.showwarning("Stop first", "Stop the machine before changing the decision line or the flow direction.")
            else:
                self.shared_state.set_flow_direction(flow)
            show_current_direction()

        def on_orientation(_event=None):
            flows = flows_of_selected_line()
            if self.shared_state.flow_direction not in flows:
                change_flow(flows[0])

        def on_flow(_event=None):
            change_flow(next(f for f, label in fg.FLOW_LABELS.items() if label == self.flow_var.get()))

        orient_combo.bind("<<ComboboxSelected>>", on_orientation)
        flow_combo.bind("<<ComboboxSelected>>", on_flow)

        man_frm = line(rois)
        self.manual_line_var = tk.StringVar()
        self.btn_line_minus = ui.FlatButton(man_frm, "◀ Shift Left", lambda: self._shift_line(-5), height=30, width=120)
        self.btn_line_minus.pack(side=tk.LEFT)
        entry(man_frm, self.manual_line_var, 7).pack(side=tk.LEFT, padx=8, ipady=5)
        self.btn_line_plus = ui.FlatButton(man_frm, "Shift Right ▶", lambda: self._shift_line(5), height=30, width=120)
        self.btn_line_plus.pack(side=tk.LEFT)
        text(man_frm, "px", colour=ui.MUTED).pack(side=tk.LEFT, padx=8)
        self.link_lines_var = tk.BooleanVar(value=self.shared_state.link_lines)
        link = line(rois, pady=(6, 0))
        self.link_switch = ui.ToggleSwitch(link, ui.PANEL, self.shared_state.link_lines,
                                           lambda on: (self.link_lines_var.set(on), self.shared_state.set_link_lines(on)))
        self.link_switch.pack(side=tk.LEFT)
        text(link, "One line for both ROIs  (off: a line per ROI, click or drag inside that ROI)", colour=ui.MUTED, size=-11).pack(side=tk.LEFT, padx=10)
        self.manual_line_var.trace_add("write", self._on_manual_line_var_write)
        self.shared_state.line_pixel_callback = self._on_line_change_callback
        show_current_direction()
        self.shared_state._notify_line()

        # ── Calibration: ROI length in mm -> px per mm ──
        cal = card("CALIBRATION", "used for mm sizes, the double rule, belt speed and valve timing")
        row = line(cal)
        text(row, "ROI 1 length", size=ui.F_BODY).pack(side=tk.LEFT)
        length = self.shared_state.size_config.get("roi_length_mm")
        self.roi_length_var = tk.StringVar(value="" if not length else f"{float(length):g}")
        entry(row, self.roi_length_var, 7).pack(side=tk.LEFT, padx=10, ipady=5)
        text(row, "mm", colour=ui.MUTED).pack(side=tk.LEFT)
        self.lbl_px_per_mm = text(row, "", bold=True, colour=ui.ACCENT, size=ui.F_HEAD)
        self.lbl_px_per_mm.pack(side=tk.RIGHT)
        self.lbl_roi_px = text(row, "", colour=ui.MUTED)
        self.lbl_roi_px.pack(side=tk.RIGHT, padx=14)
        self.lbl_calibration = note(cal, "")
        self.roi_length_var.trace_add("write", lambda *_: self._refresh_calibration())

        # ── Image sensor ──
        sensor = card("IMAGE SENSOR", "camera exposure and gain, software colour gain")
        cs = self.shared_state.camera_settings
        self.cam_vars = {
            "source": tk.StringVar(value=cs.source),
            "exposure_us": tk.StringVar(value="" if cs.exposure_us is None else f"{cs.exposure_us:g}"),
            "gain": tk.StringVar(value="" if cs.gain is None else f"{cs.gain:g}"),
            "fps_limit_enabled": tk.BooleanVar(value=cs.fps_limit_enabled),
            "fps_limit": tk.StringVar(value=f"{cs.fps_limit:g}"),
            "test_path": tk.StringVar(value=cs.test_path), "test_loop": tk.BooleanVar(value=cs.test_loop),
            "test_fps": tk.StringVar(value=f"{cs.test_fps:g}"),
        }
        grid = tk.Frame(sensor, bg=ui.PANEL)
        grid.pack(fill=tk.X)
        text(grid, "Gain", size=ui.F_BODY).grid(row=0, column=0, sticky="w", pady=4)
        entry(grid, self.cam_vars["gain"], 9).grid(row=0, column=1, padx=12, pady=4, ipady=5)
        self.lbl_gain_range = text(grid, "", colour=ui.MUTED)
        self.lbl_gain_range.grid(row=0, column=2, sticky="w")
        text(grid, "Exposure", size=ui.F_BODY).grid(row=1, column=0, sticky="w", pady=4)
        entry(grid, self.cam_vars["exposure_us"], 9).grid(row=1, column=1, padx=12, pady=4, ipady=5)
        self.lbl_exposure_range = text(grid, "", colour=ui.MUTED)
        self.lbl_exposure_range.grid(row=1, column=2, sticky="w")
        text(grid, "µs", colour=ui.MUTED).grid(row=1, column=3, sticky="w", padx=6)
        fps_row = line(sensor)
        fps_entry = entry(fps_row, self.cam_vars["fps_limit"], 6)

        def on_fps_limit(on=None):
            if on is not None:
                self.cam_vars["fps_limit_enabled"].set(on)
            fps_entry.config(state=tk.NORMAL if self.cam_vars["fps_limit_enabled"].get() else tk.DISABLED)

        ui.ToggleSwitch(fps_row, ui.PANEL, cs.fps_limit_enabled, on_fps_limit).pack(side=tk.LEFT)
        text(fps_row, "Limit FPS to").pack(side=tk.LEFT, padx=10)
        fps_entry.pack(side=tk.LEFT, ipady=4)
        text(fps_row, "off = free running", colour=ui.MUTED, size=-11).pack(side=tk.LEFT, padx=10)
        on_fps_limit()
        self.vars_cam = {
            "gain": tk.StringVar(value=str(self.shared_state.cam_gain)),
            "r": tk.StringVar(value=str(self.shared_state.cam_r)),
            "g": tk.StringVar(value=str(self.shared_state.cam_g)),
            "b": tk.StringVar(value=str(self.shared_state.cam_b)),
            "a_gain": tk.StringVar(value=str(self.shared_state.analog_gain)),
            "d_gain": tk.StringVar(value=str(self.shared_state.digital_gain)),
        }
        gain_row = line(sensor, pady=(8, 0))
        text(gain_row, "Colour gain", size=ui.F_BODY).pack(side=tk.LEFT)
        for label, key in (("ALL", "gain"), ("R", "r"), ("G", "g"), ("B", "b")):
            text(gain_row, label, bold=True, colour=ui.MUTED).pack(side=tk.LEFT, padx=(14, 5))
            entry(gain_row, self.vars_cam[key], 5).pack(side=tk.LEFT, ipady=5)
            self.vars_cam[key].trace_add("write", self._on_cam_setting_change)
        note(sensor, "An empty gain or exposure field leaves the camera's own value unchanged. Colour gain is applied in software to every frame "
                     "before the AI and before any image is saved.")

        # ── Camera / test source ──
        source = card("CAMERA SOURCE")
        row = line(source)
        names = {"aravis": "CAMERA (ARAVIS)", "file": "TEST SOURCE"}
        self.source_selector = ui.Segmented(row, list(names.values()), names.get(cs.source, "CAMERA (ARAVIS)"),
                                            lambda value: (self.cam_vars["source"].set(next(k for k, v in names.items() if v == value)),
                                                           self.source_selector.set(value)))
        self.source_selector.pack(side=tk.LEFT)
        self.lbl_cam_model = text(source, "", bold=True, colour=ui.ACCENT, justify="left", anchor="w", wraplength=570)
        self.lbl_cam_model.pack(fill=tk.X, pady=(6, 2))
        test_frm = line(source)
        text(test_frm, "Test source").pack(side=tk.LEFT)
        path_entry = entry(test_frm, self.cam_vars["test_path"], 24)
        path_entry.config(justify="left")
        path_entry.pack(side=tk.LEFT, padx=8, fill=tk.X, expand=True, ipady=5)

        def browse_test_file():
            path = filedialog.askopenfilename(title="Test video or image", initialdir=str(app_paths.BASE_DIR),
                                              filetypes=[("Video / image", "*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")])
            if path: self.cam_vars["test_path"].set(path)

        def browse_test_folder():
            start = app_paths.RECORDINGS_DIR if app_paths.RECORDINGS_DIR.exists() else app_paths.BASE_DIR
            path = filedialog.askdirectory(title="Folder of test images (for example a recording)", initialdir=str(start))
            if path: self.cam_vars["test_path"].set(path)

        ui.FlatButton(test_frm, "FOLDER", browse_test_folder, kind="muted", height=30, width=80).pack(side=tk.RIGHT)
        ui.FlatButton(test_frm, "FILE", browse_test_file, kind="muted", height=30, width=64).pack(side=tk.RIGHT, padx=6)
        test_opt = line(source)
        ui.ToggleSwitch(test_opt, ui.PANEL, cs.test_loop, lambda on: self.cam_vars["test_loop"].set(on)).pack(side=tk.LEFT)
        text(test_opt, "Repeat").pack(side=tk.LEFT, padx=10)
        text(test_opt, "FPS when the files have no timestamps", colour=ui.MUTED).pack(side=tk.LEFT, padx=(14, 6))
        entry(test_opt, self.cam_vars["test_fps"], 5).pack(side=tk.LEFT, ipady=4)
        self.lbl_cam_status = text(source, "", colour=ui.MUTED, justify="left", anchor="w", wraplength=570)
        self.lbl_cam_status.pack(fill=tk.X, pady=(6, 0))

        # ── White calibration (flat-field correction) ──
        white = card("WHITE CALIBRATION (FFC)")
        self.ffc_enabled_var = tk.BooleanVar(value=self.flat_field.enabled)

        def on_ffc_toggle(on):
            if on and not self.flat_field.available:
                messagebox.showinfo("White calibration", "No calibration yet. Put a plain white sheet under the camera and press TEACH WHITE.")
                on = False
            self.ffc_enabled_var.set(on)
            self.flat_field.enabled = on
            self.ffc_switch.set(on)

        def teach_white():
            if self.shared_state.phase == "RUNNING":
                messagebox.showwarning("Stop first", "Stop the machine before the white calibration.")
                return
            if self.camera_thread.state != "ONLINE":
                messagebox.showwarning("White calibration", "The camera must be ONLINE (not the test source).")
                return
            if not messagebox.askyesno("White calibration", "Is a plain white sheet covering the whole camera view now?\n\n"
                                       "The next frame becomes the white reference and replaces the saved one."):
                return
            self.flat_field.enabled = True      # so that the request is picked up with the next frame
            self.flat_field.request_calibration()
            self.ffc_enabled_var.set(True)
            self.ffc_switch.set(True)

        row = line(white)
        self.ffc_switch = ui.ToggleSwitch(row, ui.PANEL, self.flat_field.enabled, on_ffc_toggle)
        self.ffc_switch.pack(side=tk.LEFT)
        text(row, "Use white calibration").pack(side=tk.LEFT, padx=10)
        ui.FlatButton(row, "TEACH WHITE", teach_white, height=30, width=130).pack(side=tk.RIGHT)
        self.lbl_ffc_status = note(white, "")

        # ── Recorder: real camera frames for replay tests ──
        rec = card("RECORD TEST FRAMES")
        rec_frm = line(rec)
        self.rec_seconds_var, self.rec_fps_var = tk.StringVar(value="10"), tk.StringVar(value="25")
        text(rec_frm, "Seconds").pack(side=tk.LEFT)
        entry(rec_frm, self.rec_seconds_var, 5).pack(side=tk.LEFT, padx=(6, 14), ipady=4)
        text(rec_frm, "FPS").pack(side=tk.LEFT)
        entry(rec_frm, self.rec_fps_var, 5).pack(side=tk.LEFT, padx=(6, 14), ipady=4)
        ui.FlatButton(rec_frm, "STOP", lambda: self.camera_thread.recorder.stop(), kind="muted", height=30, width=80).pack(side=tk.RIGHT)
        ui.FlatButton(rec_frm, "RECORD", self._start_recording, height=30, width=100).pack(side=tk.RIGHT, padx=6)
        self.lbl_rec_status = note(rec, "Saves camera frames (before colour gain) into recordings/. Replay them as a test source.")

        # ── bottom: one SAVE for the page ──
        self.lbl_vision_saved = tk.Label(save_bar, text="ROIs and the line are in pixels of the full camera image.", font=(F, -11), fg=ui.MUTED, bg=ui.BG, anchor="w")
        self.lbl_vision_saved.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))
        ui.FlatButton(save_bar, "TEST IMAGE CAPTURE", self.shared_state.trigger_capture, kind="muted", height=44, width=190).pack(side=tk.LEFT)
        ui.FlatButton(save_bar, "SAVE VISION SETTINGS", self._save_vision_settings, kind="primary", height=44,
                      font=(F, ui.F_BODY, "bold")).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 6))
        self._refresh_calibration()

    def _roi_length_px(self):
        """Length of ROI 1 along the direction the cashews move, in pixels of the full image (0 when not drawn)."""
        roi = self.shared_state.roi1
        if not roi:
            return 0
        return abs(roi[2] - roi[0]) if self.shared_state.line_is_vertical else abs(roi[3] - roi[1])

    def _calibration_from_ui(self):
        """(ROI length mm, px per mm) from the field, or (None, None) when it is empty or not a number above 0."""
        try:
            mm = float(self.roi_length_var.get().strip().replace(",", "."))
        except ValueError:
            return None, None
        px = self._roi_length_px()
        return (mm, px / mm) if mm > 0 and px > 0 else (None, None)

    def _refresh_calibration(self):
        if not hasattr(self, "lbl_px_per_mm"):
            return
        ss = self.shared_state
        for number, roi in ((1, ss.roi1), (2, ss.roi2)):
            shown = f"x {roi[0]} - {roi[2]}    y {roi[1]} - {roi[3]}" if roi else "not drawn"
            if self.lbl_roi_coords[number].cget("text") != shown:
                self.lbl_roi_coords[number].config(text=shown)
        px = self._roi_length_px()
        mm, px_mm = self._calibration_from_ui()
        saved = float(ss.size_config.get("px_per_mm", 0) or 0)
        calibrated = bool(ss.size_config.get("roi_length_mm"))
        self.lbl_roi_px.config(text=f"{px} px in the image" if px else "ROI 1 is not drawn")
        self.lbl_px_per_mm.config(text=f"{px_mm:.2f} px per mm" if px_mm else "- px per mm", fg=ui.ACCENT if px_mm else ui.WARN)
        if not calibrated:
            state, colour = (f"NOT CALIBRATED: the program is using {saved:g} px per mm, which is only a placeholder. Sizes in mm, the double rule "
                             "and speeds are wrong until this is set."), ui.WARN
        elif px_mm and abs(px_mm - saved) > 0.005:
            state, colour = f"In use now: {saved:.2f} px per mm. The new value is used after SAVE VISION SETTINGS.", ui.WARN
        else:
            state, colour = f"In use: {saved:.2f} px per mm.", ui.MUTED
        self.lbl_calibration.config(text="Measure the real length of ROI 1 on the roller, along the direction the cashews move, with a ruler and enter it. "
                                         + state, fg=colour)

    def _save_vision_settings(self):
        """SAVE VISION SETTINGS: calibration, colour gain and the camera settings; ROIs and line are saved with them."""
        if self.shared_state.phase == "RUNNING":
            messagebox.showwarning("Stop first", "Stop the machine before saving the vision settings.")
            return
        mm, px_mm = self._calibration_from_ui()
        typed = self.roi_length_var.get().strip()
        if typed and px_mm is None:
            messagebox.showerror("Calibration", "ROI length must be a number above 0 in mm, and ROI 1 must be drawn.")
            return
        with self.shared_state.lock:
            if px_mm:
                self.shared_state.size_config["roi_length_mm"] = mm
                self.shared_state.size_config["px_per_mm"] = round(px_mm, 4)
        if px_mm and hasattr(self, "sz_vars") and "px_per_mm" in self.sz_vars:
            self.sz_vars["px_per_mm"].set(f"{px_mm:.4f}")
        self._on_cam_setting_change()               # colour gain, and writes the config file
        self.shared_state.save_to_config(CFG_PATH)
        self._apply_camera_settings()
        self._refresh_calibration()
        self.lbl_vision_saved.config(text=f"Saved {datetime.now():%H:%M:%S}" + (f"   {px_mm:.2f} px per mm" if px_mm else "   (calibration not set)"), fg=ui.GOOD)

    def _camera_settings_from_ui(self) -> CameraSettings:
        v = self.cam_vars

        def optional_number(key):
            raw = v[key].get().strip().replace(",", ".")
            return float(raw) if raw else None

        settings = CameraSettings(
            source=v["source"].get(), device_id=self.shared_state.camera_settings.device_id,
            exposure_us=optional_number("exposure_us"), gain=optional_number("gain"),
            fps_limit_enabled=bool(v["fps_limit_enabled"].get()), fps_limit=float(v["fps_limit"].get()),
            test_path=v["test_path"].get().strip(), test_loop=bool(v["test_loop"].get()), test_fps=float(v["test_fps"].get()))
        info = self.camera_thread.info if self.camera_thread.state == "ONLINE" else {}
        for value, bounds, name in ((settings.exposure_us, info.get("exposure_bounds"), "Exposure"), (settings.gain, info.get("gain_bounds"), "Gain")):
            if value is not None and bounds and not (bounds[0] <= value <= bounds[1]):
                raise ValueError(f"{name} must be between {bounds[0]:g} and {bounds[1]:g}.")
        if settings.exposure_us is not None and settings.exposure_us <= 0:
            raise ValueError("Exposure must be more than 0.")
        if not (0.1 <= settings.fps_limit <= 1000) or not (1 <= settings.test_fps <= 240):
            raise ValueError("FPS limit must be 0.1 to 1000, test FPS 1 to 240.")
        return settings

    def _apply_camera_settings(self):
        try:
            settings = self._camera_settings_from_ui()
        except ValueError as e:
            messagebox.showerror("Camera settings", f"Please check the numbers.\n{e}")
            return False
        current = self.shared_state.camera_settings
        if current.needs_reopen(settings) and self.shared_state.phase == "RUNNING":
            messagebox.showwarning("Stop first", "Stop the machine before changing the camera source.")
            return False
        with self.shared_state.lock:
            self.shared_state.camera_settings = settings
        self.shared_state.save_to_config(CFG_PATH)
        self.camera_thread.apply_settings(settings)
        log.info("Camera settings applied: %s", settings)
        return True

    def _start_recording(self):
        if not self.camera_thread.is_connected:
            messagebox.showwarning("Record", "There is no picture to record.")
            return
        try:
            seconds, fps = float(self.rec_seconds_var.get()), float(self.rec_fps_var.get())
            if not (1 <= seconds <= 120) or not (1 <= fps <= 100):
                raise ValueError
        except ValueError:
            messagebox.showerror("Record", "Seconds must be 1 to 120, FPS 1 to 100.")
            return
        folder = app_paths.RECORDINGS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
        ok, message = self.camera_thread.recorder.start(str(folder), seconds, fps)
        if not ok:
            messagebox.showwarning("Record", message)

    def _refresh_camera_status(self):
        cam = self.camera_thread
        self.cam_indicator.config(text=f"● Camera {cam.state}",
                                  fg=self.colors["success"] if cam.is_connected else (self.colors["warning"] if cam.state in ("CONNECTING", "RECONNECTING") else self.colors["danger"]))
        if not hasattr(self, 'lbl_cam_status'):
            return
        self._refresh_calibration()
        info = cam.info or {}
        if cam.is_connected and cam.state == "ONLINE":
            model = f"{info.get('vendor', '')} {info.get('model', '')}  |  {info.get('pixel_format', '?')}  |  {info.get('width', '?')} x {info.get('height', '?')}"
            limit = f"limit {info.get('fps_limit', 0):g} fps" if info.get("fps_limit_enabled") else "free running"
            text = (f"ONLINE: {cam.fps:.1f} fps measured ({limit}). Exposure {info.get('exposure_us', 0):.0f} µs, gain {info.get('gain', 0):g}."
                    f" Auto: exposure {info.get('exposure_auto', '?')}, gain {info.get('gain_auto', '?')}, white balance {info.get('white_balance_auto', '?')}.")
            if cam.reconnect_count:
                text += f"\nReconnected {cam.reconnect_count} time(s) since start."
            eb, gb = info.get("exposure_bounds"), info.get("gain_bounds")
            self.lbl_exposure_range.config(text=f"{eb[0]:g} - {eb[1]:g}" if eb else "")
            self.lbl_gain_range.config(text=f"{gb[0]:g} - {gb[1]:g}" if gb else "")
        elif cam.is_connected:
            model = f"Test source: {info.get('model', '')}  |  {info.get('frames', '?')} frame(s)" + ("  |  recorded timing" if info.get("recorded_timing") else "")
            text = f"TEST: {cam.fps:.1f} fps"
        else:
            model, text = "", f"{cam.state}: {cam.last_error}"
        if self.lbl_cam_model.cget("text") != model:
            self.lbl_cam_model.config(text=model)
        if self.lbl_cam_status.cget("text") != text:
            self.lbl_cam_status.config(text=text, fg=self.colors["text_muted"] if cam.is_connected else self.colors["warning"])
        ffc = self.flat_field
        ffc_text = ("White calibration is ON." if ffc.enabled else "White calibration is saved but switched off.") if ffc.available else "No white calibration saved."
        if self.lbl_ffc_status.cget("text") != ffc_text:
            self.lbl_ffc_status.config(text=ffc_text)
            self.ffc_enabled_var.set(ffc.enabled)
        rec = cam.recorder
        if rec.active:
            self.lbl_rec_status.config(text=f"Recording... {rec.saved} frames saved, {rec.dropped} dropped", fg=self.colors["warning"])
        elif rec.message and self.lbl_rec_status.cget("text") != rec.message:
            self.lbl_rec_status.config(text=rec.message, fg=self.colors["text_muted"])

    def _refresh_line_buttons(self):
        if not hasattr(self, 'btn_line_minus'):
            return
        vertical = self.shared_state.line_is_vertical
        self.btn_line_minus.config(text="◀ Shift Left" if vertical else "▲ Shift Up")
        self.btn_line_plus.config(text="Shift Right ▶" if vertical else "Shift Down ▼")

    def _shift_line(self, amount):
        try:
            val = int(self.manual_line_var.get()) + amount
            self.manual_line_var.set(str(val))
        except (ValueError, tk.TclError): pass

    def _on_manual_line_var_write(self, *args):
        try:
            val = int(self.manual_line_var.get())
        except (ValueError, tk.TclError):
            return
        if val != self.shared_state.line_pos:
            self.shared_state.set_line_pos(val, notify=False)
            axis = "X" if self.shared_state.line_is_vertical else "Y"
            self.lbl_line_pos.config(text=f"Line {axis}: {val} px")

    def _on_line_change_callback(self, txt, val):
        self.lbl_line_pos.config(text=txt)
        shown = "" if val is None else str(val)
        if shown != self.manual_line_var.get():
            self.manual_line_var.set(shown)
            
    def _on_cam_setting_change(self, *args):
        try:
            with self.shared_state.lock:
                self.shared_state.cam_gain = float(self.vars_cam["gain"].get())
                self.shared_state.cam_r = float(self.vars_cam["r"].get())
                self.shared_state.cam_g = float(self.vars_cam["g"].get())
                self.shared_state.cam_b = float(self.vars_cam["b"].get())
                self.shared_state.analog_gain = float(self.vars_cam["a_gain"].get())
                self.shared_state.digital_gain = float(self.vars_cam["d_gain"].get())
            self.shared_state.save_to_config(CFG_PATH)
        except (ValueError, tk.TclError):
            pass        # a field is empty or half typed
        except Exception:
            log.exception("Could not apply the colour gain")
    # ── manual relay tests (operator actions; all through relay_output, relays 1-40 only) ──
    def _manual_test_allowed(self) -> bool:
        if self.shared_state.phase == "RUNNING":
            messagebox.showwarning("Stop first", "Relay tests are only possible when the machine is STOPPED.")
            return False
        return True

    def _test_pulse_ms(self) -> int:
        try:
            pulse = int(self.test_pulse_var.get())
        except (ValueError, tk.TclError, AttributeError):
            pulse = int(self.shared_state.common_pulse_ms)
        return min(1000, max(10, pulse))

    def _result_text(self, relay, result):
        return f"{time.strftime('%H:%M:%S')}  {self.relay_output.describe(relay)}: {RESULT_TEXT.get(result, result)}"

    def _confirm_fire_many(self, title) -> bool:
        """Question before FIRE ALL / SEQUENCE ALL (a single pulse button fires at once, without a question)."""
        return messagebox.askyesno(title, "Make sure nobody is at the nozzles.\n\nThe REAL valves of ejectors 1-40 will fire.\n\nGo on?")

    def _update_row_pulse(self, relay):
        try:
            value = int(self.pulse_vars[relay].get())
        except (ValueError, tk.TclError, KeyError):
            return
        if 5 <= value <= 2000:
            self.shared_state.set_pulse(relay, value)

    def _all_relays_off(self):
        """ALL RELAYS OFF NOW: always allowed; cancels every waiting pulse and writes OFF to ejectors 1-40."""
        self._sequence_running = False
        done, error = self.relay_output.all_ejectors_off()
        self._manual_shown = None
        self._show_manual_result(f"{time.strftime('%H:%M:%S')}  ALL RELAYS OFF: {done} of 40 written OFF." + (f"  PLC error: {error}" if error else ""), not error)

    def _show_manual_result(self, text, sent):
        if hasattr(self, 'lbl_test_relay_result'):
            self.lbl_test_relay_result.config(text=text, fg=self.colors["success"] if sent else self.colors["warning"])

    def _ejector_check_blocked(self) -> str:
        """"" when the ejector check may run; otherwise the reason. Asked before the start and before every relay."""
        if self.shared_state.phase == "RUNNING":
            return "The machine is RUNNING."
        if self.shared_state.is_conveyor_on:
            return "The conveyor is ON."
        return ""

    def _open_ejector_check(self):
        ejector_check_ui.open_ejector_check(self, self.ejector_check, self.relay_output,
                                            is_stopped=lambda: self.shared_state.phase != "RUNNING",
                                            conveyor_off=lambda: not self.shared_state.is_conveyor_on)

    def _open_plc_diagnostics(self):
        """PLC DIAGNOSTICS window of this program (plc_diag_ui.py): everything in it goes through relay_output."""
        import plc_diag_ui
        plc_diag_ui.open_plc_diagnostics(self)

    def _open_plc_vendor_tool(self):
        """The PLC vendor tool is a separate program that writes straight to the PLC with its own connection; it can
        also switch conveyor and feeder, and what is done there is not in the manual fire log. Opened deliberately."""
        if not hasattr(self.controller, 'open_diagnostics_gui'):
            return
        if not self._manual_test_allowed():
            return
        if not messagebox.askyesno("PLC vendor tool", "Make sure nobody is at the nozzles.\n\nThis tool writes directly to the PLC: its buttons fire REAL "
                                   "valves at once and can also switch CONVEYOR (M140) and FEEDER (M141). "
                                   "What is done there is NOT in the manual fire log.\n\nOpen it?"):
            return
        log.warning("PLC vendor tool opened by the operator (direct PLC access)")
        self.relay_output.log_manual("PLC VENDOR TOOL opened", "", "", "DIRECT")
        self.controller.open_diagnostics_gui()

    def _flash_relay_led(self, relay, pulse):
        if hasattr(self, 'pn_leds') and relay in self.pn_leds:
            self.pn_leds[relay].config(fg=ui.GOOD)
            self.after(max(int(pulse), 200), lambda r=relay: self.pn_leds[r].config(fg=ui.BORDER))

    def _row_pulse_ms(self, relay) -> int:
        return int(self.shared_state.get_timing(relay)[1])

    def _test_fire_relay(self, relay, action="PULSE", pulse=None):
        """Fire one ejector once, by hand, at once (real PLC). Only when STOPPED. Returns the result code."""
        if not self._manual_test_allowed():
            return None
        pulse = self._row_pulse_ms(relay) if pulse is None else pulse
        self._manual_shown, self._manual_at = relay, time.time()
        result = self.relay_output.manual_pulse(relay, pulse, action=action)
        self._flash_relay_led(relay, pulse)
        if result != "SCHEDULED":           # refused or PLC offline: known at once; otherwise the page shows OK / PLC error when the pulse is over
            self._show_manual_result(self._result_text(relay, result), False)
        return result

    def _test_fire_selected_relay(self):
        try:
            relay = int(self.test_relay_var.get())
        except (ValueError, tk.TclError):
            return
        self._test_fire_relay(relay, action="TEST FIRE", pulse=self._test_pulse_ms())

    def _pulse_and_flash(self, act_idx):
        if hasattr(self, 'test_relay_var'):
            self.test_relay_var.set(str(act_idx))
        self._test_fire_relay(act_idx, action="ROW PULSE")

    def _fire_all(self):
        if not self._manual_test_allowed():
            return
        if not self._confirm_fire_many("FIRE ALL"):
            return
        if not self._manual_test_allowed():      # the machine may have been started while the question was open
            return
        self._manual_shown = None
        results = {relay: self.relay_output.manual_pulse(relay, self._row_pulse_ms(relay), "FIRE ALL") for relay in range(FIRST_EJECTOR, LAST_EJECTOR + 1)}
        for relay in results:
            self._flash_relay_led(relay, self._row_pulse_ms(relay))
        refused = sorted({RESULT_TEXT.get(r, r) for r in results.values() if r != "SCHEDULED"})
        self._show_manual_result(f"{time.strftime('%H:%M:%S')}  FIRE ALL (ejectors 1-40, M100-M139): " + ("; ".join(refused) if refused else "fired; see LAST FIRE of each row"),
                                 not refused)

    def _sequence_all(self):
        """Ejectors 1-40 one after another. Runs on the UI timer (no extra thread); stops if the machine is started."""
        if not self._manual_test_allowed() or getattr(self, '_sequence_running', False):
            return
        if not self._confirm_fire_many("SEQUENCE ALL"):
            return
        if not self._manual_test_allowed():
            return
        self._sequence_running = True

        def step(relay):
            if relay > LAST_EJECTOR or not self._sequence_running or self.stop_event.is_set() or self.shared_state.phase == "RUNNING":
                self._sequence_running = False
                return
            pulse = self._row_pulse_ms(relay)
            self.test_relay_var.set(str(relay))
            self._test_fire_relay(relay, action="SEQUENCE ALL", pulse=pulse)
            self.after(max(150, pulse + 80), lambda: step(relay + 1))

        step(FIRST_EJECTOR)

    def _update_common_pulse(self, *args):
        """The "set all" field: the same pulse for every ejector (each row can then be changed on its own)."""
        try:
            pulse = int(self.common_pulse_var.get())
        except (ValueError, tk.TclError):
            return
        if not 5 <= pulse <= 2000:
            return
        self.shared_state.set_common_pulse(pulse)
        for var in getattr(self, "pulse_vars", {}).values():
            if var.get() != str(pulse):
                var.set(str(pulse))

    def _update_conveyor_speed(self, *args):
        try:
            val_str = str(self.conveyor_speed_var.get()).strip()
            if val_str:
                spd = float(val_str)
                if spd > 0:
                    self.shared_state.conveyor_speed_mm_s = spd
        except Exception:
            pass

    def _on_distance_changed(self, act_id):
        if getattr(self, '_recalc_active', False):
            return
        try:
            val_str = str(self.dist_vars[act_id].get()).strip()
            if not val_str:
                return
            dist_mm = float(val_str)
            if dist_mm < 0:
                return
            
            self.shared_state.distances_mm[act_id - 1] = dist_mm
            speed = float(getattr(self.shared_state, 'conveyor_speed_mm_s', 300.0))
            if speed <= 0:
                speed = 300.0

            # Auto calculate delay
            # ROI 1: A01 is from decision line; A02..A20 are from previous nozzle hole.
            # ROI 2: A21 is from decision line; A22..A40 are from previous nozzle hole.
            if act_id in (1, 21):
                new_delay = int(round((dist_mm / speed) * 1000.0))
            else:
                prev_delay = self.shared_state.delays_ms[act_id - 2]
                delta_ms = int(round((dist_mm / speed) * 1000.0))
                new_delay = prev_delay + delta_ms

            self._recalc_active = True
            try:
                self.act_vars[act_id].set(str(new_delay))
                self.shared_state.set_delay(act_id, new_delay)
            finally:
                self._recalc_active = False
        except Exception:
            pass

    def _recalculate_all_from_distances(self):
        self._recalc_active = True
        try:
            speed = float(getattr(self.shared_state, 'conveyor_speed_mm_s', 300.0))
            if speed <= 0:
                speed = 300.0
            pulse = float(self.shared_state.common_pulse_ms)
            # delay = distance from the Decision line / speed - pulse / 2 (fire_timing.py), distances added up per ROI
            spacing = {i + 1: self.shared_state.distances_mm[i] for i in range(NUM_ACTUATORS)}
            for act_id, d in fire_timing.delays_from_distances(spacing, speed, pulse).items():
                self.act_vars[act_id].set(str(d))
                self.shared_state.set_delay(act_id, d)
            messagebox.showinfo("Recalculate", f"All delays recalculated: distance / {speed:.1f} mm/s - pulse {pulse:.0f} ms / 2.")
        except Exception as e:
            messagebox.showerror("Recalculate Error", str(e))
        finally:
            self._recalc_active = False

    def _update_act_timing(self, act_id):
        if getattr(self, '_recalc_active', False):
            return
        try:
            val_str = str(self.act_vars[act_id].get()).strip()
            if val_str.isdigit():
                self.shared_state.set_delay(act_id, int(val_str))
        except Exception:
            pass

    def _save_pneumatics(self):
        problems = self._mapping_problems()
        if problems:
            messagebox.showerror("SAVE PNEUMATICS", "NOT saved. Repair the class -> ejector mapping first:\n\n" + "\n".join(problems))
            return
        if not self._confirm_mapping_summary():
            return
        try:
            with self.shared_state.lock:
                for act_id in range(1, NUM_ACTUATORS + 1):
                    if act_id in self.act_vars:
                        try:
                            val_str = str(self.act_vars[act_id].get()).strip()
                            if val_str.isdigit():
                                self.shared_state.delays_ms[act_id - 1] = int(val_str)
                        except Exception:
                            pass
                    if hasattr(self, 'dist_vars') and act_id in self.dist_vars:
                        try:
                            d_str = str(self.dist_vars[act_id].get()).strip()
                            if d_str:
                                self.shared_state.distances_mm[act_id - 1] = float(d_str)
                        except Exception:
                            pass
                if hasattr(self, 'common_pulse_var'):
                    try:
                        val_str = str(self.common_pulse_var.get()).strip()
                        if val_str.isdigit():
                            self.shared_state.common_pulse_ms = int(val_str)
                    except Exception:
                        pass
                if hasattr(self, 'conveyor_speed_var'):
                    try:
                        spd_str = str(self.conveyor_speed_var.get()).strip()
                        if spd_str:
                            self.shared_state.conveyor_speed_mm_s = float(spd_str)
                    except Exception:
                        pass
            try:
                weight = float(self.weight_var.get())
                if 0.1 <= weight <= 20.0:
                    self.shared_state.grading["avg_cashew_weight_g"] = weight
            except (ValueError, AttributeError, tk.TclError):
                pass
            self.shared_state.save_to_config(CFG_PATH)
            self._mapping_dirty = False
            messagebox.showinfo("Saved", "Pneumatic delays, distances and the class -> ejector mapping are saved.")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save pneumatics: {e}")

    def _save_program(self):
        try:
            with self.shared_state.lock:
                for k, v in self.sz_vars.items():
                    if k == "px_per_mm" or k.endswith("_L") or k.endswith("_W"):
                        self.shared_state.size_config[k] = float(v.get())
                    else:
                        self.shared_state.size_config[k] = int(v.get())
            self.shared_state.save_to_config(CFG_PATH)
            messagebox.showinfo("Saved", "Configuration Saved.")
        except Exception as e:
            log.exception("Could not save the size configuration")
            messagebox.showerror("Save Error", f"Configuration NOT saved: {e}")

    def _update_class_conf(self, cname, var):
        try:
            val = int(var.get())
            self.shared_state.set_class_conf(cname, val)
        except (ValueError, tk.TclError): pass

    def _add_priority_logic(self):
        vals = [v.get() for v in self.prio_vars] + [self.final_grade_var.get()]
        self.logic_tree.insert("", "end", values=vals)
        self._save_priority_rules()

    def _undo_last_priority_logic(self):
        children = self.logic_tree.get_children()
        if children:
            self.logic_tree.delete(children[-1])
            self._save_priority_rules()

    def _delete_selected_priority_logic(self):
        selected_items = self.logic_tree.selection()
        if not selected_items: return
        for item in selected_items:
            self.logic_tree.delete(item)
        self._save_priority_rules()

    def _clear_all_priority_rules(self):
        for item in self.logic_tree.get_children(): 
            self.logic_tree.delete(item)
        self._save_priority_rules()

    def _save_priority_rules(self):
        rules = []
        for child in self.logic_tree.get_children():
            row = self.logic_tree.item(child)["values"]
            rules.append({"conditions": list(row[:4]), "final": row[4]})
        with self.shared_state.lock:
            self.shared_state.priority_rules = rules
        self.shared_state.save_to_config(CFG_PATH)

    # ------------------------------------------------------------------ class / grade -> ejector (relay_mapping.py has the rules)
    def _mapping_names(self):
        """(classes, grades) that need an ejector: the defect classes of the priority list, then the size grades."""
        fixed = {normalize_name(n) for n in (decision.REJECTION, decision.DOUBLE, decision.UNDEFINED)}
        fixed |= {normalize_name(c) for c in self.shared_state.grading.get("good_classes", [])}       # a PASS class needs no ejector
        classes = [c for c in self.shared_state.grading.get("priority", []) if normalize_name(c) not in fixed]
        return classes, relay_mapping.GRADES_WW + relay_mapping.GRADES_SW

    def _update_all_map_labels(self):
        with self.shared_state.lock:
            mapping = relay_mapping.clean(self.shared_state.mapping_by_act)
        for i in range(1, NUM_ACTUATORS + 1):
            text = relay_mapping.label(mapping, i)
            if relay_mapping.is_reject(i):
                fg_color = self.colors["warning"]
            else:
                fg_color = self.colors["success"] if mapping[i] else self.colors["text_muted"]
            if i in self.map_lbls:
                self.map_lbls[i].config(text=text, foreground=fg_color)
            if hasattr(self, 'pn_map_lbls') and i in self.pn_map_lbls:
                self.pn_map_lbls[i].config(text=text, foreground=fg_color)
            if hasattr(self, 'pn_leds') and i in self.pn_leds:
                self.pn_leds[i].config(fg=self.colors["surface_light"]) # LED is always OFF until fired
        if hasattr(self, 'lbl_map_unassigned'):
            classes, grades = self._mapping_names()
            lines = []
            for roi in (1, 2):
                missing = relay_mapping.unassigned(mapping, classes, grades, roi, self.shared_state.no_fire)
                if missing:
                    lines.append(f"ROI {roi} -> reject ejector A{relay_mapping.REJECT_RELAY[roi]:02d}:  " + ", ".join(missing))
            end = f"   End of belt, no fire: {', '.join(self.shared_state.no_fire) or 'none'}."
            if not self.shared_state.size_config.get("size_ready", False):
                end += "\nSIZE NOT READY: until px per mm and the size grades are taught, every GOOD cashew goes to the reject ejector (its grade is only logged)."
            self.lbl_map_unassigned.config(text=("NOT ASSIGNED (these go to the reject ejector)\n" + "\n".join(lines) + "\n" + end.strip()) if lines
                                           else "Every class and grade has an ejector." + end,
                                           fg=self.colors["warning"] if lines else self.colors["success"])
            problems = self._mapping_problems(mapping)
            self.lbl_map_problem.config(text=("SAVE IS BLOCKED\n" + "\n".join(problems)) if problems else "")

    def _dialog(self, title):
        """A modal dialog window in the theme. In the self-screenshot mode it is placed outside the visible screen."""
        win = tk.Toplevel(self)
        if os.environ.get("GSN_UI_OFFSCREEN"):
            win.overrideredirect(True)
            win.geometry(f"+{self.winfo_screenwidth() + 4200}+0")
        win.title(title)
        win.configure(bg=ui.BG)
        win.transient(self)
        win.grab_set()
        return win

    def _mapping_problems(self, mapping=None):
        """Texts of what must be repaired before the mapping can be saved (duplicates, a class on a reject ejector)."""
        if mapping is None:
            with self.shared_state.lock:
                mapping = relay_mapping.clean(self.shared_state.mapping_by_act)
        out = []
        for roi, found in relay_mapping.duplicates(mapping).items():
            for name, relays in found.items():
                out.append(f"ROI {roi}: {name} is on ejectors " + " and ".join(f"A{r:02d}" for r in relays) + " (only one is allowed)")
        for relay, names in relay_mapping.wrongly_on_reject(mapping).items():
            out.append(f"A{relay:02d} is the reject ejector but has {', '.join(names)}")
        return out

    def _popup_mapper(self, act_id):
        """Multi-select picker of one ejector: several classes / grades may share it; a class that is on another
        ejector of the ROI is moved here only after the question "move it here?"."""
        if relay_mapping.is_reject(act_id):
            messagebox.showinfo(f"A{act_id:02d}", f"A{act_id:02d} is the reject ejector of ROI {relay_mapping.roi_of(act_id)} (Double, Undefined, "
                                "REJECTION and every class without an ejector).\n\nNo class can be assigned to it.")
            return
        roi, mirror_id = relay_mapping.roi_of(act_id), relay_mapping.mirror(act_id)
        with self.shared_state.lock:
            mapping = relay_mapping.clean(self.shared_state.mapping_by_act)
        win = self._dialog(f"Ejector A{act_id:02d}")
        F = ui.FONT
        tk.Label(win, text=f"EJECTOR A{act_id:02d}   (ROI {roi})", font=(F, ui.F_HEAD, "bold"), fg=ui.TEXT, bg=ui.BG).pack(anchor="w", padx=20, pady=(16, 2))
        tk.Label(win, text="Tick every class and grade this ejector takes. One ejector can take several.\n"
                           "A class can be on only one ejector per ROI: ticking one that is elsewhere asks to move it here.",
                 font=(F, ui.F_SMALL), fg=ui.MUTED, bg=ui.BG, justify="left").pack(anchor="w", padx=20, pady=(0, 10))

        classes, grades = self._mapping_names()
        known = {normalize_name(n) for n in classes + grades + relay_mapping.GROUPS}
        extra = [n for r in relay_mapping.ALL_RELAYS for n in mapping[r] if normalize_name(n) not in known]     # names already in the mapping
        seen, options = set(), []
        for title, names in (("DEFECT CLASSES", classes), ("GOOD GROUPS (catch every grade without its own ejector)", relay_mapping.GROUPS),
                             ("WW GRADES", relay_mapping.GRADES_WW), ("SW GRADES", relay_mapping.GRADES_SW), ("OTHER NAMES IN THE MAPPING", extra)):
            group = [n for n in names if normalize_name(n) not in seen and not seen.add(normalize_name(n))]
            if group:
                options.append((title, group))

        current = {normalize_name(n) for n in mapping[act_id]}
        vars_map, labels = {}, {}
        moved = {}                      # class -> the ejector it will be taken from

        def paint(name):
            other = relay_mapping.owner(mapping, roi, name)
            ticked = vars_map[name].get()
            if ticked and name in moved:
                text, colour = f"{name}   (moves here from A{moved[name]:02d})", ui.WARN
            elif ticked:
                text, colour = name, ui.GOOD
            elif other is not None and other != act_id:
                text, colour = f"{name}   (on A{other:02d})", ui.MUTED
            else:
                text, colour = name, ui.TEXT
            labels[name].config(text=text, fg=colour, activeforeground=colour)

        def on_tick(name):
            if vars_map[name].get():
                other = relay_mapping.owner(mapping, roi, name)
                if other is not None and other != act_id:
                    if messagebox.askyesno("Move it here?", f"{name} is on ejector A{other:02d} now.\n\nMove it here to A{act_id:02d}?\n"
                                           f"(A{other:02d} keeps its other classes; with none left it becomes free.)", parent=win):
                        moved[name] = other
                    else:
                        vars_map[name].set(False)
            else:
                moved.pop(name, None)
            paint(name)

        body = tk.Frame(win, bg=ui.BG)
        body.pack(fill=tk.BOTH, padx=20)
        for title, names in options:
            tk.Label(body, text=title, font=(F, ui.F_SMALL, "bold"), fg=ui.ACCENT, bg=ui.BG, anchor="w").pack(fill=tk.X, pady=(8, 2))
            grid = tk.Frame(body, bg=ui.BG)
            grid.pack(fill=tk.X)
            for idx, name in enumerate(names):
                var = tk.BooleanVar(value=normalize_name(name) in current)
                vars_map[name] = var
                chk = tk.Checkbutton(grid, text=name, variable=var, command=lambda n=name: on_tick(n), bg=ui.BG, selectcolor=ui.FIELD,
                                     activebackground=ui.BG, font=(F, ui.F_BODY), anchor="w", width=30, bd=0, highlightthickness=0)
                chk.grid(row=idx // 3, column=idx % 3, sticky="w", padx=(0, 10), pady=1)
                labels[name] = chk
                paint(name)

        mirror_var = tk.BooleanVar(value=True)
        tk.Checkbutton(win, text=f"The same for ROI {relay_mapping.roi_of(mirror_id)}: ejector A{mirror_id:02d}", variable=mirror_var, bg=ui.BG, fg=ui.TEXT,
                       selectcolor=ui.FIELD, activebackground=ui.BG, activeforeground=ui.TEXT, font=(F, ui.F_BODY)).pack(anchor="w", padx=20, pady=(12, 0))
        note = tk.Label(win, text="", font=(F, ui.F_SMALL), fg=ui.BAD, bg=ui.BG, justify="left", wraplength=820)
        note.pack(anchor="w", padx=20)

        def on_ok():
            chosen = [name for name, var in vars_map.items() if var.get()]
            with self.shared_state.lock:
                now = relay_mapping.clean(self.shared_state.mapping_by_act)
            if mirror_var.get():        # classes that are elsewhere in the OTHER ROI move as well; say so first
                elsewhere = relay_mapping.conflicts(now, mirror_id, chosen)
                if elsewhere and not messagebox.askyesno("Move it here?", f"In ROI {relay_mapping.roi_of(mirror_id)}: " +
                                                         ", ".join(f"{n} is on A{r:02d}" for n, r in elsewhere.items()) +
                                                         f".\n\nMove to A{mirror_id:02d} as well?", parent=win):
                    return
            try:
                new = relay_mapping.assign(now, act_id, chosen, move=True, mirror_roi=mirror_var.get())
            except ValueError as e:
                note.config(text=str(e))
                return
            self.shared_state.set_mapping(new)
            with self.shared_state.lock:
                self.shared_state.no_fire = [n for n in self.shared_state.no_fire if normalize_name(n) not in {normalize_name(c) for c in chosen}]
            self._mapping_dirty = True
            self._update_all_map_labels()
            win.destroy()

        buttons = tk.Frame(win, bg=ui.BG)
        buttons.pack(fill=tk.X, padx=20, pady=16)
        ui.FlatButton(buttons, "OK", command=on_ok, kind="primary", height=40, width=160, font=(F, ui.F_BODY, "bold")).pack(side=tk.RIGHT)
        ui.FlatButton(buttons, "CANCEL", command=win.destroy, kind="muted", height=40, width=120).pack(side=tk.RIGHT, padx=(0, 8))
        ui.FlatButton(buttons, "FREE THIS EJECTOR", command=lambda: [v.set(False) for v in vars_map.values()] + [moved.clear()] + [paint(n) for n in vars_map],
                      kind="danger", height=40, width=200).pack(side=tk.LEFT)
        tk.Label(win, text="Not saved to the settings file until SAVE PNEUMATICS (it shows the whole table first).", font=(F, ui.F_SMALL), fg=ui.MUTED,
                 bg=ui.BG).pack(anchor="w", padx=20, pady=(0, 14))
        self._mapper_window = (win, vars_map, on_tick, on_ok, mirror_var)      # for the self-test

    def _confirm_mapping_summary(self) -> bool:
        """The final class -> ejector table. True when the operator confirms it."""
        with self.shared_state.lock:
            mapping = relay_mapping.clean(self.shared_state.mapping_by_act)
        classes, grades = self._mapping_names()
        rows = relay_mapping.summary(mapping, classes, grades, self.shared_state.no_fire)
        win = self._dialog("Class -> ejector")
        F = ui.FONT
        tk.Label(win, text="CLASS  ->  EJECTOR", font=(F, ui.F_HEAD, "bold"), fg=ui.TEXT, bg=ui.BG).pack(anchor="w", padx=20, pady=(16, 2))
        tk.Label(win, text="Check the table. REJECT = no ejector assigned: it goes to the reject ejector. REJECTION, DOUBLE and Undefined always go to A01 / A21.",
                 font=(F, ui.F_SMALL), fg=ui.MUTED, bg=ui.BG, wraplength=640, justify="left").pack(anchor="w", padx=20, pady=(0, 8))
        style = ttk.Style(self)
        style.configure("Summary.Treeview", rowheight=22, font=(F, -13), background=ui.PANEL, fieldbackground=ui.PANEL, foreground=ui.TEXT, borderwidth=0)
        style.layout("Summary.Treeview", [("Treeview.treearea", {"sticky": "nswe"})])
        tree = ttk.Treeview(win, columns=("name", "roi1", "roi2"), show="headings", height=min(26, len(rows) + 2), style="Summary.Treeview")
        for col, text, width in (("name", "CLASS / GRADE", 220), ("roi1", "ROI 1 EJECTOR", 210), ("roi2", "ROI 2 EJECTOR", 210)):
            tree.heading(col, text=text)
            tree.column(col, width=width, anchor="w")
        tree.tag_configure("reject", foreground=ui.WARN)
        tree.tag_configure("ok", foreground=ui.GOOD)
        for name, roi1, roi2 in rows:
            tree.insert("", "end", values=(name, roi1, roi2), tags=("reject" if "REJECT" in roi1 + roi2 else "ok",))
        free = "   ".join(f"ROI {roi}: " + (", ".join(f"A{r:02d}" for r in relay_mapping.free_relays(mapping, roi)) or "none") for roi in (1, 2))
        tree.pack(padx=20)
        tk.Label(win, text=f"Free ejectors (stay OFF):   {free}", font=(F, ui.F_SMALL), fg=ui.MUTED, bg=ui.BG, wraplength=640, justify="left").pack(anchor="w", padx=20, pady=(8, 0))
        answer = {"ok": False}

        def confirm():
            answer["ok"] = True
            win.destroy()
        buttons = tk.Frame(win, bg=ui.BG)
        buttons.pack(fill=tk.X, padx=20, pady=16)
        ui.FlatButton(buttons, "CONFIRM AND SAVE", command=confirm, kind="primary", height=40, width=220, font=(F, ui.F_BODY, "bold")).pack(side=tk.RIGHT)
        ui.FlatButton(buttons, "CANCEL", command=win.destroy, kind="muted", height=40, width=120).pack(side=tk.RIGHT, padx=(0, 8))
        self._summary_window = (win, rows, confirm)
        self.wait_window(win)
        return answer["ok"]

    def _toggle_conv(self):
        self.shared_state.is_conveyor_on = not self.shared_state.is_conveyor_on
        state = self.shared_state.is_conveyor_on
        color = self.colors["success"] if state else self.colors["surface_light"]
        fg_col = "white" if state else self.colors["text_muted"]
        self.conv_btn.config(text=f"CONVEYOR: {'ON' if state else 'OFF'}", bg=color, fg=fg_col)
        self.conv_btn.set_kind("primary" if state else "muted")
        if hasattr(self.controller, 'set_conveyor'):
            self.controller.set_conveyor(state)
        elif hasattr(self.controller, 'set_relay'):
            self.controller.set_relay(41, state)

    def _toggle_feed(self):
        self.shared_state.is_feeder_on = not self.shared_state.is_feeder_on
        state = self.shared_state.is_feeder_on
        color = self.colors["success"] if state else self.colors["surface_light"]
        fg_col = "white" if state else self.colors["text_muted"]
        self.feed_btn.config(text=f"FEEDER: {'ON' if state else 'OFF'}", bg=color, fg=fg_col)
        self.feed_btn.set_kind("primary" if state else "muted")
        if hasattr(self.controller, 'set_feeder'):
            self.controller.set_feeder(state)
        elif hasattr(self.controller, 'set_relay'):
            self.controller.set_relay(42, state)

    def _on_start_stop(self):
        ph = self.shared_state.get_processing_state()["phase"]
        if ph == "RUNNING":
            self.shared_state.phase = "READY"
            self.relay_output.cancel_pending()      # STOPPED: nothing fires from grading, also not what was still waiting
        elif ph == "READY" and self.ejector_check.busy():
            messagebox.showwarning("Ejector check", "The ejector check is running. Stop it first.")
        elif ph == "READY":
            st = self.shared_state.get_processing_state()
            if st["roi1"] and st["roi2"] and st["line_pos"] is not None and st["line_pos2"] is not None:
                self.shared_state.phase = "RUNNING"
            else:
                messagebox.showwarning("Incomplete", "Please draw ROI1, ROI2 and set the Decision Line (of both ROIs when they are not linked) in Vision Setup.")

    def _on_mouse(self, event):
        if not self.current_video_geometry: return 
        geo = self.current_video_geometry
        lbl_w, lbl_h = self.video_label.winfo_width(), self.video_label.winfo_height()
        tk_off_x = (lbl_w - VIDEO_PANE_W) // 2 if lbl_w > VIDEO_PANE_W else 0
        tk_off_y = (lbl_h - VIDEO_PANE_H) // 2 if lbl_h > VIDEO_PANE_H else 0
        eff_x, eff_y = event.x - tk_off_x, event.y - tk_off_y
        x = int(np.clip((eff_x - geo["x_off"]) / geo["scale"], 0, geo["orig_w"] - 1))
        y = int(np.clip((eff_y - geo["y_off"]) / geo["scale"], 0, geo["orig_h"] - 1))
        
        if event.type == tk.EventType.ButtonPress: self.shared_state.handle_mouse_down((x,y))
        elif event.type == tk.EventType.Motion: self.shared_state.handle_mouse_move((x,y))
        elif event.type == tk.EventType.ButtonRelease: self.shared_state.handle_mouse_up((x,y))

    def _update_loop(self):
        try:
            (frame, geo) = self.display_queue.get_nowait()
            self.current_video_geometry = geo
            img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.video_label.configure(image=img_tk)
            self.video_label.image = img_tk 
        except queue.Empty: pass
        
        a_conn = self.controller.is_connected()
        self._refresh_camera_status()
        self.ard_indicator.config(fg=self.colors["success"] if a_conn else self.colors["danger"])
        if hasattr(self, 'lbl_net_camera'):
            cam, info = self.camera_thread, (self.camera_thread.info or {})
            shown = (f"{cam.state}     {info.get('vendor', '')} {info.get('model', '')}     {info.get('width', '?')} x {info.get('height', '?')}"
                     f"     {info.get('pixel_format', '')}     {cam.fps:.0f} fps") if cam.is_connected else f"{cam.state}     {cam.last_error}"
            if self.lbl_net_camera.cget("text") != shown:
                self.lbl_net_camera.config(text=shown, fg=ui.TEXT if cam.is_connected else ui.WARN)
        if hasattr(self, 'net_plc_status'):
            plc_ip = getattr(self.controller, 'plc_ip', 'PLC')
            plc_port = getattr(self.controller, 'plc_port', 502)
            self.net_plc_status.config(
                text=f"● CONNECTED ({plc_ip}:{plc_port})" if a_conn else "● DISCONNECTED",
                foreground=self.colors["success"] if a_conn else self.colors["danger"]
            )

        
        # Sync Start/Stop UI with actual phase
        ph = self.shared_state.get_processing_state()["phase"]
        if ph == "RUNNING":
            if self.start_btn.cget("text") != "■ STOP SYSTEM":
                self.start_btn.config(text="■ STOP SYSTEM", style="Danger.TButton")
                self.lbl_state_badge.config(text="● RUNNING", fg=self.colors["success"])
        else:
            if self.start_btn.cget("text") != "START RUNNING":
                self.start_btn.config(text="START RUNNING", style="Success.TButton")
                self.lbl_state_badge.config(text="● READY", fg=self.colors["warning"])
        
        self.fps_lbl.config(text=f"CAM {self.camera_thread.fps:.0f} fps   •   AI {self.processing_thread.ai_fps:.0f} fps")

        now_dt = datetime.now()
        self.lbl_clock.config(text=now_dt.strftime("%H:%M:%S"))
        self.lbl_date.config(text=now_dt.strftime("%a %d-%m-%Y"))
        
        current_fires = self.shared_state.get_recent_fires()
        if current_fires != self._last_fires_state:
            self.fire_tree.delete(*self.fire_tree.get_children())
            for row in current_fires:
                self.fire_tree.insert("", "end", values=row)
            self._last_fires_state = list(current_fires)

        self._grading_tick = getattr(self, "_grading_tick", 0) + 1
        if self._grading_tick % 5 == 0:
            self._refresh_status()
        if self._grading_tick % 15 == 2:
            self._refresh_pneumatic()
        if self._grading_tick % 15 == 7:
            self._refresh_colour_size()
        if self._grading_tick % 50 == 1:
            self._refresh_teaching()
            self._fit_pills()
        if self._grading_tick % 10 == 0 and hasattr(self, "lbl_view_live"):
            self._refresh_grading_live()

        if hasattr(self, 'lbl_prod_count'):
            det_cnt = self.shared_state.get_detected_count()
            if getattr(self, '_last_det_cnt', -1) != det_cnt:
                self.lbl_prod_count.config(text=f"[ Detected: {det_cnt} ]")
                self._last_det_cnt = det_cnt

        self.after(30, self._update_loop)

    def _build_teaching_console_page(self):
        """Teaching YOLO page in the 5-track layout: open saved images, mark the cashews with a class, train."""
        F = ui.FONT
        p_teach = tk.Frame(self.page_container, bg=ui.BG)
        self.pages["Teaching YOLO"] = p_teach

        def card(title, note="", expand=False):
            box = tk.Frame(p_teach, bg=ui.PANEL, highlightthickness=1, highlightbackground=ui.BORDER)
            box.pack(fill=tk.BOTH if expand else tk.X, expand=expand, pady=(0, 10))
            head = tk.Frame(box, bg=ui.PANEL)
            head.pack(fill=tk.X, padx=14, pady=(10, 4))
            tk.Label(head, text=title, font=(F, ui.F_SMALL, "bold"), fg=ui.MUTED, bg=ui.PANEL).pack(side=tk.LEFT)
            if note:
                tk.Label(head, text=note, font=(F, -11), fg=ui.MUTED, bg=ui.PANEL).pack(side=tk.LEFT, padx=12)
            body = tk.Frame(box, bg=ui.PANEL)
            body.pack(fill=tk.BOTH if expand else tk.X, expand=expand, padx=14, pady=(0, 12))
            return body

        images = card("1   IMAGES", "open a folder of saved camera images")
        ui.FlatButton(images, "OPEN FOLDER", self._ann_open_folder, height=34, width=140).pack(side=tk.LEFT)
        ui.FlatButton(images, "< PREV", self._ann_prev, kind="muted", height=34, width=84).pack(side=tk.LEFT, padx=(10, 0))
        ui.FlatButton(images, "NEXT >", self._ann_next, kind="muted", height=34, width=84).pack(side=tk.LEFT, padx=(8, 0))
        self.ann_lbl_file = tk.Label(images, text="No image loaded", font=(F, ui.F_SMALL), bg=ui.PANEL, fg=ui.MUTED)
        self.ann_lbl_file.pack(side=tk.LEFT, padx=14)

        classes = card("2   CLASS TO MARK", "tick a class, then click a cashew in the image")
        btn_row = tk.Frame(classes, bg=ui.PANEL)
        btn_row.pack(fill=tk.X, pady=(0, 8))
        ui.FlatButton(btn_row, "ADD", self._add_label_class, height=30, width=70).pack(side=tk.LEFT)
        ui.FlatButton(btn_row, "RENAME", self._rename_label_class, kind="muted", height=30, width=90).pack(side=tk.LEFT, padx=8)
        ui.FlatButton(btn_row, "DELETE", self._delete_label_class, kind="danger", height=30, width=84).pack(side=tk.LEFT)
        self.ann_fast_class_var = tk.StringVar()
        self._class_check_vars = {}  # {class_name: tk.BooleanVar}
        self.class_btn_grid = tk.Frame(classes, bg=ui.PANEL)
        self.class_btn_grid.pack(fill=tk.X)
        self._refresh_class_grid()

        train = tk.Frame(p_teach, bg=ui.PANEL, highlightthickness=1, highlightbackground=ui.BORDER)
        train.pack(side=tk.BOTTOM, fill=tk.X)
        head = tk.Frame(train, bg=ui.PANEL)
        head.pack(fill=tk.X, padx=14, pady=(10, 4))
        tk.Label(head, text="4   TRAIN", font=(F, ui.F_SMALL, "bold"), fg=ui.MUTED, bg=ui.PANEL).pack(side=tk.LEFT)
        tk.Label(head, text="machine STOPPED; a new model is not active until you activate it", font=(F, -11), fg=ui.MUTED, bg=ui.PANEL).pack(side=tk.LEFT, padx=12)
        row = tk.Frame(train, bg=ui.PANEL)
        row.pack(fill=tk.X, padx=14, pady=(0, 12))
        ui.FlatButton(row, "START TRAINING", self._start_yolo_train, kind="primary", height=40, width=220, font=(F, ui.F_BODY, "bold")).pack(side=tk.LEFT)
        ui.FlatButton(row, "ACTIVATE A MODEL ...", lambda: self._switch_page("AI Model Select"), height=40, width=210).pack(side=tk.RIGHT)

        marks = card("3   MARKS IN THIS IMAGE", expand=True)
        edit_row = tk.Frame(marks, bg=ui.PANEL)
        edit_row.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0))
        ui.FlatButton(edit_row, "RENAME SELECTED", self._ann_rename, kind="muted", height=30, width=170).pack(side=tk.LEFT)
        ui.FlatButton(edit_row, "DELETE SELECTED", self._ann_delete, kind="danger", height=30, width=170).pack(side=tk.LEFT, padx=8)
        self.ann_tree = ttk.Treeview(marks, columns=("ID", "Class Name"), show="headings", height=4, style="Home.Treeview")
        self.ann_tree.heading("ID", text="ID", anchor="w"); self.ann_tree.column("ID", width=60, anchor="w")
        self.ann_tree.heading("Class Name", text="CLASS", anchor="w"); self.ann_tree.column("Class Name", width=300, anchor="w")
        sb = ttk.Scrollbar(marks, orient=tk.VERTICAL, command=self.ann_tree.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.ann_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.ann_tree.configure(yscrollcommand=sb.set)

    # ------------------------------------------------------------------ grading rules (step c1)
    def _build_grading_section(self, parent, bar_parent=None):
        """Sorting Logic page, full width (no camera): grading rules and priority, view limits with live values
        and a small live picture per ROI, belt colour and tracker. One SAVE button for the page."""
        F, g = ui.FONT, self.shared_state.grading
        self.grading_vars = {}
        self._sorting_dirty = False

        def changed(*_):
            if not getattr(self, "_sorting_loading", False):
                self._sorting_dirty = True
                self.lbl_grading_status.config(text="●  UNSAVED CHANGES", fg=ui.WARN)

        def panel(column, title, note=""):
            box = tk.Frame(column, bg=ui.PANEL, highlightthickness=1, highlightbackground=ui.BORDER)
            box.pack(fill=tk.X, pady=(0, 2 * ui.GRID))
            head = tk.Frame(box, bg=ui.PANEL)
            head.pack(fill=tk.X, padx=20, pady=(16, 6))
            tk.Label(head, text=title, font=(F, ui.F_SMALL, "bold"), fg=ui.MUTED, bg=ui.PANEL).pack(side=tk.LEFT)
            if note:
                tk.Label(box, text=note, font=(F, ui.F_SMALL), fg=ui.MUTED, bg=ui.PANEL, justify="left", anchor="w",
                         wraplength=480).pack(fill=tk.X, padx=20, pady=(0, 8))
            body = tk.Frame(box, bg=ui.PANEL)
            body.pack(fill=tk.X, padx=20, pady=(0, 16))
            return body

        def number(body, section, key, label, unit=""):
            row = tk.Frame(body, bg=ui.PANEL)
            row.pack(fill=tk.X, pady=5)
            tk.Label(row, text=label, font=(F, ui.F_BODY), fg=ui.TEXT, bg=ui.PANEL, anchor="w").pack(side=tk.LEFT)
            tk.Label(row, text=unit, font=(F, ui.F_SMALL), fg=ui.MUTED, bg=ui.PANEL, width=7, anchor="w").pack(side=tk.RIGHT)
            var = tk.StringVar(value=f"{g[section].get(key, ''):g}")
            tk.Entry(row, textvariable=var, width=7, bg=ui.FIELD, fg=ui.TEXT, insertbackground=ui.TEXT, bd=0, justify="center",
                     font=(F, ui.F_BODY, "bold"), highlightthickness=1, highlightbackground=ui.BORDER, highlightcolor=ui.ACCENT
                     ).pack(side=tk.RIGHT, padx=(8, 8), ipady=4)
            var.trace_add("write", changed)
            self.grading_vars[(section, key)] = var

        def switch(body, label, var, note=""):
            row = tk.Frame(body, bg=ui.PANEL)
            row.pack(fill=tk.X, pady=6)
            tk.Checkbutton(row, variable=var, command=changed, bg=ui.PANEL, activebackground=ui.PANEL, selectcolor=ui.FIELD,
                           fg=ui.ACCENT, bd=0, highlightthickness=0).pack(side=tk.LEFT)
            text = tk.Frame(row, bg=ui.PANEL)
            text.pack(side=tk.LEFT, fill=tk.X, expand=True)
            tk.Label(text, text=label, font=(F, ui.F_BODY), fg=ui.TEXT, bg=ui.PANEL, anchor="w", justify="left", wraplength=470).pack(fill=tk.X)
            if note:
                tk.Label(text, text=note, font=(F, ui.F_SMALL), fg=ui.MUTED, bg=ui.PANEL, anchor="w", justify="left", wraplength=470).pack(fill=tk.X)

        columns = tk.Frame(parent, bg=ui.BG)
        columns.pack(fill=tk.BOTH, expand=True)
        cols = []
        for i in range(3):
            columns.columnconfigure(i, weight=1, uniform="sorting")
            col = tk.Frame(columns, bg=ui.BG)
            col.grid(row=0, column=i, sticky="new", padx=(0 if i == 0 else 2 * ui.GRID, 0))
            cols.append(col)

        # ── column 1: grading rules, priority ──
        body = panel(cols[0], "GRADING RULES", "One decision per cashew at its Decision line. A class is \"seen\" with enough frames at or above its confidence.")
        number(body, "voting", "defect_conf", "Defect: confidence at least", "0 - 1")
        number(body, "voting", "defect_min_frames", "Defect: in at least", "frames")
        number(body, "voting", "good_conf", "GOOD: confidence at least", "0 - 1")
        number(body, "voting", "good_min_frames", "GOOD: in at least", "frames")
        number(body, "voting", "neck_min_frames", "GOOD: NECK view in at least (M)", "frames")
        tk.Label(body, text="M = 0: the not-rolled rule is on hold. A cashew whose NECK was never seen is still written to the log as \"would be not rolled\".",
                 font=(F, ui.F_SMALL), fg=ui.MUTED, bg=ui.PANEL, justify="left", anchor="w", wraplength=500).pack(fill=tk.X, pady=(0, 6))
        number(body, "voting", "neck_weight", "TESTA vote from a NECK frame counts as", "frames")
        number(body, "voting", "rejection_conf", "REJECTION (YOLO): confidence at least", "0 - 1")
        number(body, "voting", "rejection_min_frames", "REJECTION (YOLO): in at least", "frames")
        self.single_defect_var = tk.BooleanVar(value=bool(g["voting"].get("single_defect_blocks_good", False)))
        switch(body, "One defect frame blocks GOOD", self.single_defect_var,
               "ON: a single defect frame (below the frame limit) makes the cashew Undefined. OFF: it is ignored.")

        body = panel(cols[0], "PRIORITY (highest first)", "Before this list: tracking faults, DOUBLE and REJECTION go to the reject relay; skin together with another defect is UPW.")
        prio = tk.Frame(body, bg=ui.PANEL)
        prio.pack(fill=tk.X)
        self.priority_list = tk.Listbox(prio, height=10, width=24, bg=ui.FIELD, fg=ui.TEXT, selectbackground=ui.ACCENT, selectforeground="#06222A",
                                        bd=0, highlightthickness=1, highlightbackground=ui.BORDER, font=(F, ui.F_BODY), exportselection=False, activestyle="none")
        for name in g["priority"]:
            self.priority_list.insert(tk.END, name)
        self.priority_list.pack(side=tk.LEFT)

        def move(step):
            sel = self.priority_list.curselection()
            if not sel or not (0 <= sel[0] + step < self.priority_list.size()):
                return
            i = sel[0]
            name = self.priority_list.get(i)
            self.priority_list.delete(i)
            self.priority_list.insert(i + step, name)
            self.priority_list.selection_set(i + step)
            changed()
        side = tk.Frame(prio, bg=ui.PANEL)
        side.pack(side=tk.LEFT, padx=16, anchor="n")
        ui.FlatButton(side, "▲  UP", command=lambda: move(-1), height=36, width=120).pack(pady=(0, 8))
        ui.FlatButton(side, "▼  DOWN", command=lambda: move(1), height=36, width=120).pack()

        # ── column 2: view limits, live values ──
        body = panel(cols[1], "VIEW LIMITS (empty = not used)",
                     "A frame is NECK, BACK or TOP when all filled limits of that view hold (checked in this order), otherwise OTHER. "
                     "The letters are the values drawn under each cashew.")
        grid = tk.Frame(body, bg=ui.PANEL)
        grid.pack(fill=tk.X)
        for c, view in enumerate(view_features.VIEWS):
            tk.Label(grid, text=view, font=(F, ui.F_SMALL, "bold"), fg=ui.ACCENT, bg=ui.PANEL).grid(row=0, column=1 + 2 * c, columnspan=2, pady=(0, 2))
            for k, word in enumerate(("min", "max")):
                tk.Label(grid, text=word, font=(F, -11), fg=ui.MUTED, bg=ui.PANEL).grid(row=1, column=1 + 2 * c + k)
        self.view_vars = {}
        for r, (name, (letter, description)) in enumerate(view_features.FEATURES.items(), start=2):
            tk.Label(grid, text=f"{letter}  {name}", font=(F, ui.F_BODY), fg=ui.TEXT, bg=ui.PANEL, anchor="w").grid(row=r, column=0, sticky="w", padx=(0, 10), pady=4)
            for c, view in enumerate(view_features.VIEWS):
                limits = (g["views"]["rules"].get(view) or {}).get(name) or [None, None]
                for k in (0, 1):
                    var = tk.StringVar(value="" if limits[k] is None else f"{limits[k]:g}")
                    tk.Entry(grid, textvariable=var, width=5, bg=ui.FIELD, fg=ui.TEXT, insertbackground=ui.TEXT, bd=0, justify="center",
                             font=(F, ui.F_BODY), highlightthickness=1, highlightbackground=ui.BORDER, highlightcolor=ui.ACCENT
                             ).grid(row=r, column=1 + 2 * c + k, padx=(8 if k == 0 else 2, 2), pady=4, ipady=3)
                    var.trace_add("write", changed)
                    self.view_vars[(view, name, k)] = var
        legend = "   ".join(f"{letter} = {description}" for name, (letter, description) in view_features.FEATURES.items())
        tk.Label(body, text=legend, font=(F, -11), fg=ui.MUTED, bg=ui.PANEL, justify="left", anchor="w", wraplength=480).pack(fill=tk.X, pady=(8, 0))
        self.show_values_var = tk.BooleanVar(value=bool(g["views"].get("show_values", True)))
        switch(body, "Draw the values under each cashew in the camera picture", self.show_values_var)

        body = panel(cols[1], "LIVE VALUES", "The latest measured cashew of each ROI while the machine is RUNNING (for tuning the view limits).")
        self.lbl_view_thumb, self.lbl_view_live_roi, self._thumb_images = {}, {}, {}
        for rid in (1, 2):
            row = tk.Frame(body, bg=ui.PANEL)
            row.pack(fill=tk.X, pady=4)
            thumb = tk.Label(row, bg=ui.FIELD, width=14, height=6, highlightthickness=1, highlightbackground=ui.BORDER)
            thumb.pack(side=tk.LEFT)
            text = tk.Label(row, text=f"ROI {rid}:  -", font=(F, ui.F_BODY), fg=ui.WARN, bg=ui.PANEL, justify="left", anchor="nw", wraplength=380)
            text.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=12)
            self.lbl_view_thumb[rid], self.lbl_view_live_roi[rid] = thumb, text
        self.lbl_view_live = self.lbl_view_live_roi[1]      # marks "this page has live values"
        self.lbl_result_counts = tk.Label(body, text="", font=(F, ui.F_SMALL), fg=ui.TEXT, bg=ui.PANEL, justify="left", anchor="w", wraplength=520)
        self.lbl_result_counts.pack(fill=tk.X, pady=(8, 6))
        ui.FlatButton(body, "RESET COUNTERS", command=self.shared_state.reset_result_counts, kind="muted", height=32).pack(anchor="w")

        # ── column 3: belt colour, tracker, other switches ──
        body = panel(cols[2], "BELT COLOUR (cashew mask)", "The cashew is everything that is not belt blue and not a dark roller gap.")
        number(body, "segmentation", "blue_h_min", "Belt hue from", "0 - 179")
        number(body, "segmentation", "blue_h_max", "Belt hue to", "0 - 179")
        number(body, "segmentation", "blue_s_min", "Belt saturation at least", "0 - 255")
        number(body, "segmentation", "dark_v_max", "Dark (gap) brightness up to", "0 - 255")
        number(body, "segmentation", "min_area_px", "Smallest cashew area", "px")
        number(body, "segmentation", "edge_margin_px", "Cut-off: box this close to the ROI edge", "px")

        def teach():
            if messagebox.askyesno("Teach belt colour", "The belt must be EMPTY and lit as in production.\n\nTeach the belt colour from the picture now?"):
                with self.shared_state.lock:
                    self.shared_state.teach_belt_requested = True
                    self.shared_state.teach_belt_message = "Teaching..."
        ui.FlatButton(body, "TEACH BELT COLOUR (empty belt)", command=teach, height=36).pack(anchor="w", pady=(8, 4))
        self.lbl_teach_belt = tk.Label(body, text="", font=(F, ui.F_SMALL), fg=ui.MUTED, bg=ui.PANEL, justify="left", anchor="w", wraplength=520)
        self.lbl_teach_belt.pack(fill=tk.X)

        body = panel(cols[2], "TRACKER", "First values; they are tuned on a recording with real cashews.")
        number(body, "tracker", "max_step_px", "Largest step per frame (speed unknown)", "px")
        number(body, "tracker", "gate_extra_px", "Allowed error around the prediction", "px")
        number(body, "tracker", "speed_factor", "... plus this x the predicted step", "x")
        number(body, "tracker", "back_tol_px", "Backward tolerance", "px")
        number(body, "tracker", "max_lateral_px", "Sideways limit", "px")
        number(body, "tracker", "hold_ms", "Keep a missed cashew's ID for", "ms")

        body = panel(cols[2], "OTHER")
        self.old_filter_var = tk.BooleanVar(value=bool(self.shared_state.opencv_filter.get("enabled", False)))
        switch(body, "Old dark-spot filter for CNN SW / WHITE_DW", self.old_filter_var, "Uses the values of the Colour Size page. Default OFF.")
        self.frame_log_var = tk.BooleanVar(value=bool(g["frame_log"].get("enabled", True)))
        switch(body, "Per-frame log", self.frame_log_var, "logs/frames_*.jsonl, kept 7 days.")

        # ── save bar: outside the scrolling part, so it is always on screen ──
        bar = tk.Frame(bar_parent or parent, bg=ui.PANEL, highlightthickness=1, highlightbackground=ui.BORDER)
        bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(ui.GRID, 0))
        ui.FlatButton(bar, "SAVE SORTING LOGIC", command=self._apply_grading_settings, kind="primary", height=44, width=300,
                      font=(F, ui.F_BODY, "bold")).pack(side=tk.RIGHT, padx=16, pady=12)
        self.lbl_grading_status = tk.Label(bar, text="All changes saved", font=(F, ui.F_BODY, "bold"), fg=ui.MUTED, bg=ui.PANEL, anchor="e",
                                           justify="right", wraplength=900)
        self.lbl_grading_status.pack(side=tk.RIGHT, padx=16)
        tk.Label(bar, text="Saved with the settings file pneumatic_config.json (block \"grading\").", font=(F, ui.F_SMALL), fg=ui.MUTED,
                 bg=ui.PANEL).pack(side=tk.LEFT, padx=16)

    def _apply_grading_settings(self):
        g = copy.deepcopy(self.shared_state.grading)
        ints = {"defect_min_frames", "good_min_frames", "neck_min_frames", "neck_weight", "rejection_min_frames",
                "blue_h_min", "blue_h_max", "blue_s_min", "dark_v_max", "min_area_px", "edge_margin_px"}
        try:
            for (section, key), var in self.grading_vars.items():
                value = float(var.get())
                if value < 0 or (key.endswith("_conf") and value > 1):
                    raise ValueError(f"{key}: {var.get()}")
                if key in ints:
                    if value != int(value):
                        raise ValueError(f"{key} must be a whole number")
                    value = int(value)
                g[section][key] = value
            if g["voting"]["neck_weight"] < 1 or g["voting"]["defect_min_frames"] < 1 or g["voting"]["good_min_frames"] < 1 or g["voting"]["rejection_min_frames"] < 1:
                raise ValueError("frame counts and the NECK weight must be at least 1")
            if g["segmentation"]["blue_h_min"] >= g["segmentation"]["blue_h_max"] or g["segmentation"]["blue_h_max"] > 179:
                raise ValueError("belt hue: from must be below to, at most 179")
            rules = {view: {} for view in view_features.VIEWS}
            for view in view_features.VIEWS:
                for name in view_features.FEATURES:
                    lo, hi = (self.view_vars[(view, name, k)].get().strip() for k in (0, 1))
                    if lo or hi:
                        limits = [float(lo) if lo else None, float(hi) if hi else None]
                        if limits[0] is not None and limits[1] is not None and limits[0] > limits[1]:
                            raise ValueError(f"{view} {name}: min is above max")
                        rules[view][name] = limits
        except ValueError as e:
            messagebox.showerror("Sorting Logic", f"NOT saved. Wrong value: {e}")
            self.lbl_grading_status.config(text=f"NOT SAVED: wrong value ({e})", fg=ui.BAD)
            return
        g["voting"]["single_defect_blocks_good"] = bool(self.single_defect_var.get())
        g["priority"] = list(self.priority_list.get(0, tk.END))
        g["views"]["rules"] = rules
        g["views"]["show_values"] = bool(self.show_values_var.get())
        g["frame_log"]["enabled"] = bool(self.frame_log_var.get())
        with self.shared_state.lock:
            self.shared_state.opencv_filter["enabled"] = bool(self.old_filter_var.get())
        self.shared_state.set_grading(g)
        self.processing_thread.cashew_log.frames_enabled = g["frame_log"]["enabled"]
        used = [view for view in view_features.VIEWS if rules[view]]
        view_mode = g.get("engines", {}).get("view", engines.SHADOW)
        note = ""
        if "NECK" not in used and g["voting"]["neck_min_frames"] > 0 and view_mode == engines.ACTIVE:
            note = "   NOTE: NECK has no limits and the View engine is ACTIVE, so no cashew can become GOOD."
        self._sorting_dirty = False
        self.lbl_grading_status.config(text=f"✔  Saved {datetime.now().strftime('%H:%M:%S')}   (views with limits: {', '.join(used) or 'none'}){note}",
                                       fg=ui.WARN if note else ui.GOOD)
        log.info("Grading settings applied: %s", g)

    def _refresh_grading_live(self):
        ss = self.shared_state
        with ss.lock:
            live, speed, teach_msg = dict(ss.view_live), dict(ss.lane_speed), ss.teach_belt_message
            counts = dict(ss.result_counts)
            seg = dict(ss.grading["segmentation"])
        px_per_mm = float(ss.size_config.get("px_per_mm", 10.0)) or 10.0
        with ss.lock:
            thumbs = dict(ss.view_thumb)
        for r in (1, 2):
            self.lbl_view_live_roi[r].config(text=f"ROI {r}:  {live[r] or '-'}\nlane speed {speed[r] * 1000:.0f} px/s = "
                                                  f"{speed[r] * 1000 / px_per_mm:.0f} mm/s at {px_per_mm:g} px/mm")
            crop = thumbs.get(r)
            if crop is not None and crop is not self._thumb_images.get((r, "src")):
                image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                self._thumb_images[r], self._thumb_images[(r, "src")] = image, crop
                self.lbl_view_thumb[r].config(image=image, width=image.width(), height=image.height())
        total = sum(n for k, n in counts.items() if not k.startswith(("UNDEFINED: ", decision.IGNORED_PREFIX)))
        self.lbl_result_counts.config(text=f"Decided: {total}   " + "   ".join(f"{k} {n}" for k, n in sorted(counts.items())))
        if teach_msg != self.lbl_teach_belt.cget("text"):
            self.lbl_teach_belt.config(text=teach_msg)
            if teach_msg.startswith("Belt colour taught"):
                for key in ("blue_h_min", "blue_h_max", "blue_s_min"):
                    self.grading_vars[("segmentation", key)].set(f"{seg[key]:g}")

    def _show_decision_detail(self, _event=None):
        """Double click on a history row: the frame-by-frame votes of that cashew."""
        sel = self.fire_tree.selection()
        if not sel:
            return
        track_id = str(self.fire_tree.item(sel[0])["values"][1])
        with self.shared_state.lock:
            detail = self.shared_state.recent_decisions.get(track_id)
        if not detail:
            return
        win = tk.Toplevel(self)
        win.title(f"Cashew {track_id}")
        win.geometry("1150x420")
        box = tk.Text(win, bg=self.colors["surface"], fg=self.colors["text"], font=("Roboto Mono", 9), wrap="none")
        box.insert("1.0", detail)
        box.config(state="disabled")
        box.pack(fill=tk.BOTH, expand=True)

    def _build_colour_size_page(self):
        """Colour Size page, step (c2). Left: SW / WW colour (metric, threshold, TEACH WW / TEACH SW with both
        histograms and a suggested threshold) and the last measured cashews. Right: SIZE GRADES READY, the grade
        tables W... / BUDDS / SW... with TEACH per grade, shape limits. One SAVE button."""
        import size_teach
        F = ui.FONT
        ss = self.shared_state
        page = tk.Frame(self.page_container, bg=ui.BG)
        self.pages["Colour Size"] = page
        self.cs_vars = {}
        self._size_teach = None
        self._colour_teach = {"active": None, "since": 0.0, colour_size.WW: {}, colour_size.SW: {}}      # colour -> {id: lab}
        self._colour_suggestion = None

        bar = tk.Frame(page, bg=ui.PANEL, highlightthickness=1, highlightbackground=ui.BORDER)
        bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
        self.lbl_cs_saved = tk.Label(bar, text=f"Saved with the program:  {self.active_program_name}", font=(F, ui.F_SMALL), fg=ui.MUTED, bg=ui.PANEL)
        self.lbl_cs_saved.pack(side=tk.LEFT, padx=16, pady=14)
        ui.FlatButton(bar, "SAVE COLOUR & SIZE", self._save_colour_size, kind="primary", height=40, width=260, font=(F, ui.F_BODY, "bold")).pack(side=tk.RIGHT, padx=14)
        self.lbl_cs_unsaved = tk.Label(bar, text="", font=(F, ui.F_SMALL, "bold"), fg=ui.WARN, bg=ui.PANEL)
        self.lbl_cs_unsaved.pack(side=tk.RIGHT, padx=10)

        def card(parent, title, note_text, **pack):
            box = tk.Frame(parent, bg=ui.PANEL, highlightthickness=1, highlightbackground=ui.BORDER)
            box.pack(**pack)
            head = tk.Frame(box, bg=ui.PANEL)
            head.pack(fill=tk.X, padx=16, pady=(12, 6))
            tk.Label(head, text=title, font=(F, ui.F_SMALL, "bold"), fg=ui.TEXT, bg=ui.PANEL).pack(side=tk.LEFT)
            tk.Label(head, text=note_text, font=(F, ui.F_SMALL), fg=ui.MUTED, bg=ui.PANEL).pack(side=tk.LEFT, padx=16)
            return box

        def text(parent, value, bold=False, colour=ui.TEXT, size=ui.F_SMALL, **kw):
            return tk.Label(parent, text=value, font=(F, size, "bold") if bold else (F, size), fg=colour, bg=parent.cget("bg"), **kw)

        def entry(parent, var, width=6):
            return tk.Entry(parent, textvariable=var, width=width, bg=ui.FIELD, fg=ui.TEXT, insertbackground=ui.TEXT, bd=0, justify="center",
                            font=(F, -14), highlightthickness=1, highlightbackground=ui.BORDER, highlightcolor=ui.ACCENT)

        def row_of(parent, pady=4):
            row = tk.Frame(parent, bg=ui.PANEL)
            row.pack(fill=tk.X, padx=16, pady=pady)
            return row

        def changed(*_):
            if hasattr(self, "lbl_cs_unsaved"):
                self.lbl_cs_unsaved.config(text="●  UNSAVED CHANGES")
        self._cs_changed = changed

        def rule(parent):
            tk.Frame(parent, bg=ui.BORDER, height=1).pack(fill=tk.X, padx=16, pady=(8, 4))

        body = tk.Frame(page, bg=ui.BG)
        body.pack(fill=tk.BOTH, expand=True)

        # ───────────────────────────── left: colour ─────────────────────────────
        left = card(body, "COLOUR  SW / WW", "median inside the cashew mask over all frames of the cashew", side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))
        colour_cfg = {**colour_size.COLOUR_DEFAULTS, **ss.size_config.get("colour", {})}
        self._metric_names = {"b": "b*  YELLOW", "a": "a*  RED", "L": "L*  LIGHT"}
        row = row_of(left)
        text(row, "Value", size=ui.F_BODY, width=11, anchor="w").pack(side=tk.LEFT)
        self.cs_metric = ui.Segmented(row, list(self._metric_names.values()), self._metric_names[colour_cfg["metric"]], self._cs_set_metric)
        self.cs_metric.pack(side=tk.LEFT)
        text(row, "SW is", size=ui.F_BODY).pack(side=tk.LEFT, padx=(24, 8))
        self.cs_side = ui.Segmented(row, ("ABOVE", "BELOW"), "ABOVE" if colour_cfg["sw_above"] else "BELOW", lambda v: (self.cs_side.set(v), changed(), self._refresh_colour_size(force=True)))
        self.cs_side.pack(side=tk.LEFT)
        text(row, "the threshold", colour=ui.MUTED).pack(side=tk.LEFT, padx=8)
        row = row_of(left)
        text(row, "Threshold", size=ui.F_BODY, width=11, anchor="w").pack(side=tk.LEFT)
        self.cs_threshold_var = tk.StringVar(value="" if colour_cfg["threshold"] is None else f"{float(colour_cfg['threshold']):g}")
        self.cs_threshold_state = text(row, "", bold=True, colour=ui.WARN)
        self.cs_threshold_state.pack(side=tk.RIGHT)
        entry(row, self.cs_threshold_var, 7).pack(side=tk.RIGHT, padx=8, ipady=5)
        ui.Slider(row, self.cs_threshold_var, -30, 100, 0.5, ui.PANEL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.cs_threshold_var.trace_add("write", lambda *_: (changed(), self._refresh_colour_size(force=True)))
        row = row_of(left, pady=(6, 2))
        text(row, "CURRENT CASHEW", bold=True, colour=ui.MUTED).pack(side=tk.LEFT)
        self.lbl_cs_live = text(row, "-", bold=True, size=ui.F_HEAD)
        self.lbl_cs_live.pack(side=tk.LEFT, padx=16)
        self.lbl_cs_live_all = text(row, "", colour=ui.MUTED)
        self.lbl_cs_live_all.pack(side=tk.RIGHT)

        rule(left)
        row = row_of(left)
        text(row, "TEACH COLOUR", bold=True, colour=ui.MUTED).pack(side=tk.LEFT)
        text(row, "run only WW cashews, then only SW cashews (machine RUNNING)", colour=ui.MUTED, size=-11).pack(side=tk.LEFT, padx=12)
        row = row_of(left)
        self.btn_teach_ww = ui.FlatButton(row, "TEACH WW", lambda: self._colour_teach_toggle(colour_size.WW), height=34, width=150)
        self.btn_teach_ww.pack(side=tk.LEFT)
        self.btn_teach_sw = ui.FlatButton(row, "TEACH SW", lambda: self._colour_teach_toggle(colour_size.SW), height=34, width=150)
        self.btn_teach_sw.pack(side=tk.LEFT, padx=8)
        ui.FlatButton(row, "CLEAR", self._colour_teach_clear, kind="muted", height=34, width=80).pack(side=tk.LEFT)
        self.btn_cs_colour_apply = ui.FlatButton(row, "APPLY SUGGESTION", self._colour_teach_apply, kind="primary", height=34, width=200)
        self.btn_cs_colour_apply.pack(side=tk.RIGHT)
        self.btn_cs_colour_apply.set_enabled(False)
        self.cs_colour_hist = ui.DualHistogram(left, height=150)
        self.cs_colour_hist.pack(fill=tk.X, padx=16, pady=(6, 0))
        self.lbl_cs_suggest = text(left, "", colour=ui.MUTED, size=-12, justify="left", anchor="w", wraplength=780)
        self.lbl_cs_suggest.pack(fill=tk.X, padx=16, pady=(6, 0))

        rule(left)
        head = row_of(left, pady=0)
        text(head, "LAST CASHEWS", bold=True, colour=ui.MUTED).pack(side=tk.LEFT)
        text(head, "measured while RUNNING", colour=ui.MUTED, size=-11).pack(side=tk.LEFT, padx=12)
        self.cs_last_tree = ttk.Treeview(left, columns=("id", "result", "colour", "value", "l", "w", "a", "grade", "relay"), show="headings", style="Home.Treeview", height=7)
        for col, title, width in (("id", "ID", 96), ("result", "CLASS / RESULT", 230), ("colour", "SW/WW", 60), ("value", "COLOUR", 70), ("l", "L mm", 60),
                                  ("w", "W mm", 60), ("a", "AREA", 60), ("grade", "GRADE", 110), ("relay", "EJECTOR", 70)):
            self.cs_last_tree.heading(col, text=title, anchor="w")
            self.cs_last_tree.column(col, width=width, anchor="w", stretch=(col == "result"))
        self.cs_last_tree.pack(fill=tk.BOTH, expand=True, padx=16, pady=(6, 14))

        # ───────────────────────────── right: size ─────────────────────────────
        right = card(body, "SIZE GRADES", "length, width, area in mm from the contour, median over the frames", side=tk.LEFT, fill=tk.BOTH, expand=True)
        ready = row_of(right, pady=0)
        self.cs_ready_switch = ui.ToggleSwitch(ready, ui.PANEL, bool(ss.size_config.get("size_ready", False)), self._set_size_ready)
        self.cs_ready_switch.pack(side=tk.LEFT)
        self.lbl_cs_ready = text(ready, "", bold=True)
        self.lbl_cs_ready.pack(side=tk.LEFT, padx=12)
        self.lbl_cs_calibration = text(ready, "", colour=ui.MUTED)
        self.lbl_cs_calibration.pack(side=tk.RIGHT)

        # The table is built from plain rows (pack), every cell with a fixed width: no grid spans or weights.
        widths = (8, 7, 7, 7, 7, 7, 7)
        table = tk.Frame(right, bg=ui.PANEL)
        table.pack(fill=tk.X, padx=16, pady=(8, 0))
        header = tk.Frame(table, bg=ui.PANEL)
        header.pack(fill=tk.X)
        for title, width in zip(("GRADE", "L min", "L max", "W min", "W max", "AREA min", "AREA max"), widths):
            text(header, title, bold=True, colour=ui.MUTED, size=-11, width=width + 1, anchor="w").pack(side=tk.LEFT)
        text(header, "TAUGHT", bold=True, colour=ui.MUTED, size=-11, anchor="w").pack(side=tk.LEFT, padx=6)
        self.cs_grade_taught, self.cs_teach_buttons = {}, {}
        grades = ss.size_config.get("grades", {})
        for grade in colour_size.WW_GRADES + (colour_size.BUDDS,) + colour_size.SW_GRADES:
            if grade == colour_size.SW_GRADES[0]:
                gap = tk.Frame(table, bg=ui.PANEL)
                gap.pack(fill=tk.X, pady=(6, 2))
                text(gap, "SW grades", bold=True, colour=ui.MUTED, size=-11).pack(side=tk.LEFT)
                ui.FlatButton(gap, "COPY W LIMITS TO SW", self._copy_w_to_sw, kind="muted", height=24, width=190, font=(F, -11, "bold")).pack(side=tk.LEFT, padx=12)
            line = tk.Frame(table, bg=ui.PANEL)
            line.pack(fill=tk.X, pady=1)
            text(line, grade, bold=True, size=ui.F_BODY, width=7, anchor="w").pack(side=tk.LEFT)
            for key in colour_size.LIMIT_KEYS:
                value = grades.get(grade, {}).get(key)
                var = tk.StringVar(value="" if value is None else f"{value:g}")
                self.cs_vars[("grade", grade, key)] = var
                entry(line, var, 6).pack(side=tk.LEFT, padx=(0, 6), ipady=3)
                var.trace_add("write", changed)
            button = ui.FlatButton(line, "TEACH", lambda g=grade: self._size_teach_toggle(g), height=26, width=72)
            button.pack(side=tk.RIGHT)
            self.cs_teach_buttons[grade] = button
            self.cs_grade_taught[grade] = text(line, "", colour=ui.MUTED, anchor="w")
            self.cs_grade_taught[grade].pack(side=tk.LEFT, padx=6)
        text(right, "BUDDS: one row for both colours (no fire: end of belt). A cashew outside every grade of its colour goes to the reject ejector.",
             colour=ui.MUTED, size=-11, anchor="w").pack(fill=tk.X, padx=16, pady=(4, 0))

        teach = row_of(right, pady=(8, 0))
        self.lbl_cs_teach = text(teach, "TEACH FROM CAMERA", bold=True, colour=ui.MUTED)
        self.lbl_cs_teach.pack(side=tk.LEFT)
        self.cs_k_var = tk.StringVar(value=f"{float(ss.size_config.get('teach_k', size_teach.K_DEFAULT)):g}")
        entry(teach, self.cs_k_var, 5).pack(side=tk.RIGHT, ipady=3)
        text(teach, "mean -/+ k x std,  k", colour=ui.MUTED).pack(side=tk.RIGHT, padx=8)
        histograms = tk.Frame(right, bg=ui.PANEL)
        histograms.pack(fill=tk.X, padx=16, pady=(6, 0))
        self.cs_histograms = {}
        for index, (key, title, unit) in enumerate(size_teach.KEYS):
            holder = tk.Frame(histograms, bg=ui.PANEL)
            holder.place(relx=index / 3.0, rely=0, relwidth=1 / 3.0, relheight=1.0)
            histogram = ui.Histogram(holder, title, unit, height=104)
            histogram.pack(fill=tk.BOTH, expand=True, padx=(0, 6) if index < 2 else 0)
            self.cs_histograms[key] = histogram
        histograms.config(height=106)
        apply_row = row_of(right, pady=(8, 0))
        self.btn_cs_apply = ui.FlatButton(apply_row, "APPLY", self._size_teach_apply, kind="primary", height=30, width=170)
        self.btn_cs_apply.pack(side=tk.LEFT)
        self.btn_cs_apply.set_enabled(False)
        text(apply_row, f"Run only cashews of that grade. At least {size_teach.MIN_SAMPLES} cashews. Yellow lines = the min / max that will be set.",
             colour=ui.MUTED, size=-11).pack(side=tk.LEFT, padx=12)

        rule(right)
        text(right, "SHAPE LIMITS  (GOOD cashews only; SHADOW / ACTIVE on the Pneumatic page)      DOUBLE RULE", bold=True, colour=ui.MUTED, anchor="w").pack(fill=tk.X, padx=16)
        shape, double = ss.grading["shape"], ss.grading["double"]
        limits = row_of(right, pady=(6, 12))
        for key, label, value, unit in ((("shape", "min_solidity"), "Min solidity", shape["min_solidity"], ""),
                                        (("shape", "min_length_width"), "Min L / W", shape["min_length_width"], ""),
                                        (("shape", "min_length_mm"), "Min length", shape["min_length_mm"], "mm"),
                                        (("shape", "min_area_mm2"), "Min area", shape["min_area_mm2"], "mm2"),
                                        (("double", "max_single_length_mm"), "Longest single", double["max_single_length_mm"], "mm")):
            var = tk.StringVar(value=f"{value:g}")
            self.cs_vars[key] = var
            text(limits, label, colour=ui.TEXT).pack(side=tk.LEFT)
            entry(limits, var, 5).pack(side=tk.LEFT, padx=(6, 2), ipady=3)
            text(limits, unit, colour=ui.MUTED, size=-11).pack(side=tk.LEFT, padx=(0, 12))
            var.trace_add("write", changed)
        self.lbl_cs_unsaved.config(text="")
        self._refresh_colour_size(force=True)

    # ── colour: metric, threshold, teach ──
    def _cs_metric(self):
        return next(k for k, v in self._metric_names.items() if v == self.cs_metric.value)

    def _cs_set_metric(self, shown):
        self.cs_metric.set(shown)
        self._cs_changed()
        self._refresh_colour_size(force=True)

    def _cs_colour_cfg(self):
        """Colour settings as shown on the page (threshold None while the field is empty or not a number)."""
        try:
            threshold = float(self.cs_threshold_var.get().strip().replace(",", "."))
        except ValueError:
            threshold = None
        return {"metric": self._cs_metric(), "threshold": threshold, "sw_above": self.cs_side.value == "ABOVE"}

    def _colour_teach_toggle(self, colour):
        """TEACH WW / TEACH SW: collect the colour of every cashew measured from now on; a second press stops."""
        teach = self._colour_teach
        if teach["active"] == colour:
            teach["active"] = None
        else:
            teach["active"], teach["since"] = colour, time.time()
        self._refresh_colour_size(force=True)

    def _colour_teach_clear(self):
        self._colour_teach.update({"active": None, colour_size.WW: {}, colour_size.SW: {}})
        self._colour_suggestion = None
        self._refresh_colour_size(force=True)

    def _colour_teach_collect(self):
        teach = self._colour_teach
        if not teach["active"]:
            return
        with self.shared_state.lock:
            new = [s for s in self.shared_state.size_samples if s["t"] >= teach["since"] and s.get("lab") and not s["double"]]
        for sample in new:
            teach[teach["active"]][sample["id"]] = sample["lab"]

    def _colour_teach_apply(self):
        """APPLY SUGGESTION: the best metric, its threshold and side go into the fields and are saved at once."""
        suggestion = self._colour_suggestion
        if not suggestion:
            return
        if self.shared_state.phase == "RUNNING":
            messagebox.showwarning("Stop first", "Stop the machine before applying the colour threshold.")
            return
        best = suggestion["metrics"][suggestion["best"]]
        self.cs_metric.set(self._metric_names[suggestion["best"]])
        self.cs_side.set("ABOVE" if best["sw_above"] else "BELOW")
        self.cs_threshold_var.set(f"{best['threshold']:g}")
        self._colour_teach["active"] = None
        with self.shared_state.lock:
            self.shared_state.size_config["colour"] = {**self._cs_colour_cfg(), "taught": f"{datetime.now():%d.%m.%Y %H:%M}",
                                                       "ww": suggestion["ww"], "sw": suggestion["sw"], "wrong_pct": best["wrong_pct"]}
        self.shared_state.save_to_config(CFG_PATH)
        log.info("Colour threshold taught: %s", self.shared_state.size_config["colour"])
        self._refresh_colour_size(force=True)

    def _copy_w_to_sw(self):
        """COPY W LIMITS TO SW: the fields of SW180 ... SW450 get the values of W180 ... W450 (saved with SAVE)."""
        copied = []
        for size in colour_size.SIZES:
            values = [self.cs_vars[("grade", f"W{size}", key)].get().strip() for key in colour_size.LIMIT_KEYS]
            if values[0] and values[1]:
                for key, value in zip(colour_size.LIMIT_KEYS, values):
                    self.cs_vars[("grade", f"SW{size}", key)].set(value)
                self._size_taught_pending = getattr(self, "_size_taught_pending", {})
                self._size_taught_pending[f"SW{size}"] = {"taught": f"copied from W{size}"}
                copied.append(f"SW{size}")
        if not copied:
            messagebox.showinfo("Copy W limits", "No W grade has limits yet. Teach or type the W grades first.")
        self._refresh_colour_size(force=True)

    def _size_ready_blockers(self):
        """What is still missing before SIZE GRADES READY may be switched on."""
        size_config = self.shared_state.size_config
        missing = []
        if not size_config.get("roi_length_mm"):
            missing.append("px per mm is not calibrated (Vision ROI page: ROI 1 length in mm, SAVE VISION SETTINGS)")
        if size_config.get("colour", {}).get("threshold") is None:
            missing.append("the SW / WW threshold is not set (TEACH WW, TEACH SW, APPLY SUGGESTION, or type it and SAVE)")
        grades = size_config.get("grades", {})
        for colour, names in ((colour_size.WW, colour_size.WW_GRADES), (colour_size.SW, colour_size.SW_GRADES)):
            if not any(grades.get(n, {}).get("l_min") is not None for n in names):
                missing.append(f"no {colour} grade has limits (TEACH a grade and SAVE" + ("; or COPY W LIMITS TO SW)" if colour == colour_size.SW else ")"))
        return missing

    def _set_size_ready(self, on):
        """SIZE GRADES READY: off = SHADOW (every GOOD cashew goes to the reject ejector, the grade is only logged)."""
        ss = self.shared_state
        if on:
            missing = self._size_ready_blockers()
            if missing:
                messagebox.showwarning("Size grades", "SIZE GRADES READY cannot be switched on yet:\n\n- " + "\n- ".join(missing))
                self.cs_ready_switch.set(False)
                return
            if self.lbl_cs_unsaved.cget("text"):
                messagebox.showwarning("Size grades", "There are unsaved changes on this page. Press SAVE COLOUR & SIZE first.")
                self.cs_ready_switch.set(False)
                return
            taught = [n for n in colour_size.ALL_GRADES if ss.size_config.get("grades", {}).get(n, {}).get("l_min") is not None]
            if not messagebox.askyesno("Size grades", "Switch SIZE GRADES READY on?\n\nGOOD cashews will then be fired by the ejector of their colour "
                                       "and size grade instead of the reject ejector.\n\nGrades with limits: " + ", ".join(taught) +
                                       "\nA GOOD cashew outside every grade of its colour goes to the reject ejector.\n\n"
                                       "Do this only after the \"would be\" grades in RECENT RESULTS were checked with real cashews.", icon="warning"):
                self.cs_ready_switch.set(False)
                return
        with ss.lock:
            ss.size_config["size_ready"] = bool(on)
        g = copy.deepcopy(ss.grading)
        for name in ("sw_ww", "size"):
            g["engines"][name] = engines.ACTIVE if on else engines.SHADOW
        ss.set_grading(g)                           # also writes the settings file
        log.info("SIZE GRADES READY -> %s", bool(on))
        self._refresh_pneumatic()
        self._refresh_colour_size(force=True)

    def _size_teach_toggle(self, grade):
        """TEACH on a grade row: start collecting the cashews measured from now on; a second press stops it."""
        if self._size_teach and self._size_teach["grade"] == grade:
            self._size_teach = None
        else:
            if not self.shared_state.size_config.get("roi_length_mm"):
                messagebox.showwarning("Teach size", "Calibrate px per mm first: Vision ROI page, ROI length in mm, SAVE VISION SETTINGS.\n"
                                                     "Without it the mm values taught here would be wrong.")
                return
            self._size_teach = {"grade": grade, "since": time.time()}
        self._refresh_colour_size(force=True)

    def _size_teach_samples(self):
        import size_teach
        if not self._size_teach:
            return []
        with self.shared_state.lock:
            samples = [s for s in self.shared_state.size_samples if s["t"] >= self._size_teach["since"]]
        return size_teach.usable(samples)

    def _size_teach_k(self):
        import size_teach
        try:
            return min(6.0, max(0.5, float(self.cs_k_var.get().replace(",", "."))))
        except ValueError:
            return size_teach.K_DEFAULT

    def _size_teach_apply(self):
        """APPLY: write mean -/+ k x std of the taught cashews into the grade's limit fields (saved with SAVE)."""
        import size_teach
        if not self._size_teach:
            return
        grade = self._size_teach["grade"]
        result = size_teach.limits(self._size_teach_samples(), self._size_teach_k())
        if not result:
            return
        for key in colour_size.LIMIT_KEYS:
            self.cs_vars[("grade", grade, key)].set(f"{result[key]:g}")
        self._size_taught_pending = getattr(self, "_size_taught_pending", {})
        self._size_taught_pending[grade] = {"n": result["n"], "k": result["k"], "taught": f"{datetime.now():%d.%m %H:%M}"}
        self._size_teach = None
        self._refresh_colour_size(force=True)

    def _save_colour_size(self):
        """SAVE COLOUR & SIZE: colour metric / threshold / side, grade limits, shape limits and the double length."""
        ss = self.shared_state
        if ss.phase == "RUNNING":
            messagebox.showwarning("Stop first", "Stop the machine before saving colour and size settings.")
            return False
        try:
            number = lambda var: float(var.get().strip().replace(",", "."))
            shape = {key[1]: number(var) for key, var in self.cs_vars.items() if key[0] == "shape"}
            double_length = number(self.cs_vars[("double", "max_single_length_mm")])
            grades = {}
            for grade in colour_size.ALL_GRADES:
                row = {key: self.cs_vars[("grade", grade, key)].get().strip() for key in colour_size.LIMIT_KEYS}
                if any(row.values()):
                    grades[grade] = {key: float(value.replace(",", ".")) for key, value in row.items() if value}
        except ValueError:
            messagebox.showerror("Colour & Size", "Please check the numbers: one field is not a number.")
            return False
        for grade, row in grades.items():
            if "l_min" not in row or "l_max" not in row:
                messagebox.showerror("Colour & Size", f"{grade}: L min and L max are needed (width and area limits are optional).")
                return False
            for low, high in (("l_min", "l_max"), ("w_min", "w_max"), ("a_min", "a_max")):
                if low in row and high in row and row[low] >= row[high]:
                    messagebox.showerror("Colour & Size", f"{grade}: {low.replace('_', ' ')} must be below {high.replace('_', ' ')}.")
                    return False
        if double_length <= 0:
            messagebox.showerror("Colour & Size", "Sizes must be above 0.")
            return False
        colour_cfg = self._cs_colour_cfg()
        if self.cs_threshold_var.get().strip() and colour_cfg["threshold"] is None:
            messagebox.showerror("Colour & Size", "The SW / WW threshold is not a number.")
            return False
        if ss.size_config.get("size_ready") and (colour_cfg["threshold"] is None or not grades):
            messagebox.showerror("Colour & Size", "SIZE GRADES READY is on: the threshold and the grade limits cannot be emptied. Switch it off first.")
            return False
        pending = getattr(self, "_size_taught_pending", {})
        with ss.lock:
            old_colour = ss.size_config.get("colour", {})
            keep = {k: v for k, v in old_colour.items() if k in ("taught", "ww", "sw", "wrong_pct")} \
                if all(old_colour.get(k) == colour_cfg[k] for k in colour_cfg) else {}
            ss.size_config["colour"] = {**colour_cfg, **keep}
            stored = ss.size_config.get("grades", {})
            ss.size_config["grades"] = {grade: {**{k: v for k, v in stored.get(grade, {}).items() if k in ("n", "k", "taught") and grade not in pending},
                                                **row, **pending.get(grade, {})} for grade, row in grades.items()}
            ss.size_config["teach_k"] = self._size_teach_k()
        self._size_taught_pending = {}
        g = copy.deepcopy(ss.grading)
        g["shape"].update(shape)
        g["double"]["max_single_length_mm"] = double_length
        ss.set_grading(g)                           # also writes the settings file
        self.lbl_cs_unsaved.config(text="")
        self.lbl_cs_saved.config(text=f"Saved {datetime.now():%H:%M:%S} with the program:  {self.active_program_name}", fg=ui.GOOD)
        self._refresh_colour_size(force=True)
        return True

    def _refresh_colour_size(self, force=False):
        import size_teach
        if not hasattr(self, "cs_last_tree") or not (force or self.pages["Colour Size"].winfo_ismapped()):
            return
        ss = self.shared_state
        with ss.lock:
            samples = list(ss.size_samples)[-7:]
            size_config = copy.deepcopy(ss.size_config)
        cfg = self._cs_colour_cfg()
        metric, star = cfg["metric"], {"b": "b*", "a": "a*", "L": "L*"}
        # colour: state of the threshold, the current cashew, the teach histograms and the suggestion
        saved = {**colour_size.COLOUR_DEFAULTS, **size_config.get("colour", {})}
        if cfg["threshold"] is None:
            state, colour = "NOT SET", ui.WARN
        elif any(saved.get(k) != cfg[k] for k in cfg):
            state, colour = "not saved yet", ui.WARN
        else:
            state, colour = (f"taught {saved['taught']}" if saved.get("taught") else "saved"), ui.GOOD
        self.cs_threshold_state.config(text=state, fg=colour)
        last = next((s for s in reversed(samples) if s.get("lab")), None)
        if last:
            name, value = colour_size.colour_of(last["lab"], cfg)
            self.lbl_cs_live.config(text=f"{star[metric]} {value:.1f}" + (f"   ->  {name}" if name else "   (no threshold)"),
                                    fg=ui.TEXT if name is None else (ui.WARN if name == colour_size.SW else ui.ACCENT))
            self.lbl_cs_live_all.config(text=f"{last['id']}     L* {last['lab']['L']:.1f}   a* {last['lab']['a']:.1f}   b* {last['lab']['b']:.1f}")
        self._colour_teach_collect()
        teach = self._colour_teach
        ww, sw = list(teach[colour_size.WW].values()), list(teach[colour_size.SW].values())
        for button, name, group in ((self.btn_teach_ww, colour_size.WW, ww), (self.btn_teach_sw, colour_size.SW, sw)):
            button.set_text(f"STOP {name}  ({len(group)})" if teach["active"] == name else f"TEACH {name}  ({len(group)})")
            button.set_kind("danger" if teach["active"] == name else "outline")
        self.cs_colour_hist.show([("WW", ui.ACCENT, [s[metric] for s in ww]), ("SW", ui.WARN, [s[metric] for s in sw])], cfg["threshold"],
                                 f"{star[metric]} OF THE TAUGHT CASHEWS      white line = threshold")
        self._colour_suggestion = colour_size.suggest(ww, sw)
        self.btn_cs_colour_apply.set_enabled(self._colour_suggestion is not None)
        if self._colour_suggestion:
            rows = self._colour_suggestion["metrics"]
            best = self._colour_suggestion["best"]
            lines = [f"{star[m]}: threshold {rows[m]['threshold']:g}, SW {'above' if rows[m]['sw_above'] else 'below'}, "
                     f"{rows[m]['wrong_pct']:g} % of the taught cashews on the wrong side" + ("   <- BEST" if m == best else "") for m in ("b", "a", "L")]
            self.lbl_cs_suggest.config(text="Suggestion from the taught cashews:\n" + "\n".join(lines), fg=ui.TEXT)
        else:
            need = colour_size.MIN_COLOUR_SAMPLES
            self.lbl_cs_suggest.config(text=f"A suggestion needs at least {need} WW and {need} SW cashews (now {len(ww)} and {len(sw)}). "
                                            "The three values L*, a*, b* are compared and the one that separates your cashews best is proposed.", fg=ui.MUTED)
        # size
        ready, calibrated = bool(size_config.get("size_ready")), bool(size_config.get("roi_length_mm"))
        self.cs_ready_switch.set(ready)
        self.lbl_cs_ready.config(text="SIZE GRADES READY: GOOD goes to its grade ejector" if ready
                                 else "SHADOW: grade is logged (\"would be\"), GOOD goes to the reject ejector", fg=ui.GOOD if ready else ui.WARN)
        self.lbl_cs_calibration.config(text=f"{float(size_config.get('px_per_mm', 0)):.2f} px per mm" if calibrated else "px per mm NOT CALIBRATED (Vision ROI)",
                                       fg=ui.MUTED if calibrated else ui.BAD)
        rows = [(s["id"], s["label"].replace("UNDEF SIZE_NOT_READY ", "GOOD "), s.get("colour") or "-",
                 "-" if s.get("colour_value") is None else f"{s['colour_value']:.1f}", f"{s['l']:.1f}", f"{s['w']:.1f}", f"{s['a']:.0f}",
                 s.get("grade") or "-", f"A{s['relay']:02d}" if s.get("relay") else "-") for s in reversed(samples)]
        if rows != getattr(self, "_cs_last_rows", None):
            self._cs_last_rows = rows
            self.cs_last_tree.delete(*self.cs_last_tree.get_children())
            for row in rows:
                self.cs_last_tree.insert("", "end", values=row)
        pending = getattr(self, "_size_taught_pending", {})
        teaching = self._size_teach["grade"] if self._size_teach else None
        taught = self._size_teach_samples()
        for grade in colour_size.ALL_GRADES:
            info = pending.get(grade) or size_config.get("grades", {}).get(grade, {})
            if grade == teaching:
                label, colour = f"{len(taught)} / {size_teach.TARGET_SAMPLES} teaching now", ui.ACCENT
            elif grade in pending:
                label, colour = (f"{info['n']} cashews" if info.get("n") else info.get("taught", "")) + ", not saved yet", ui.WARN
            elif info.get("taught"):
                label, colour = (f"{info['n']} cashews, " if info.get("n") else "") + str(info["taught"]), ui.MUTED
            else:
                label, colour = ("typed" if info else "not taught"), ui.MUTED
            self.cs_grade_taught[grade].config(text=label, fg=colour)
            self.cs_teach_buttons[grade].set_text("STOP" if grade == teaching else "TEACH")
        k = self._size_teach_k()
        for key, histogram in self.cs_histograms.items():
            histogram.show([s[key] for s in taught], k)
        enough = teaching is not None and len(taught) >= size_teach.MIN_SAMPLES
        self.btn_cs_apply.set_enabled(enough)
        self.btn_cs_apply.set_text(f"APPLY TO {teaching}" if teaching else "APPLY")
        self.lbl_cs_teach.config(text=(f"TEACH {teaching}:  run only {teaching} cashews now (machine RUNNING)   "
                                       f"{len(taught)} of {size_teach.TARGET_SAMPLES} measured") if teaching else "TEACH FROM CAMERA",
                                 fg=ui.ACCENT if teaching else ui.MUTED)

    def _refresh_class_grid(self):
        try:
            for widget in self.class_btn_grid.winfo_children():
                widget.destroy()
            valid_classes = [c for c in self.all_class_names if not (c.startswith("sw_") or c.startswith("ww_"))]
            if valid_classes and not self.ann_fast_class_var.get(): 
                self.ann_fast_class_var.set(valid_classes[0])
            elif self.ann_fast_class_var.get() not in valid_classes and valid_classes:
                self.ann_fast_class_var.set(valid_classes[0])

            # Rebuild checkbox variables
            self._class_check_vars = {}
            cols = 3
            for i, cname in enumerate(sorted(valid_classes)):
                r, c = divmod(i, cols)
                var = tk.BooleanVar(value=False)
                self._class_check_vars[cname] = var
                cb = tk.Checkbutton(
                    self.class_btn_grid, text=cname, variable=var,
                    onvalue=True, offvalue=False,
                    command=lambda cn=cname: self._on_class_check_toggle(cn),
                    bg=ui.FIELD, fg=ui.TEXT, selectcolor=ui.BG, activebackground=ui.FIELD, activeforeground=ui.ACCENT,
                    font=(ui.FONT, ui.F_SMALL, "bold"), anchor="w", width=14, relief=tk.FLAT, bd=0, padx=8, pady=6,
                    highlightthickness=1, highlightbackground=ui.BORDER
                )
                self.class_btn_grid.columnconfigure(c, weight=1, uniform="cls")
                cb.grid(row=r, column=c, padx=(0, 6), pady=3, sticky="ew")
            self.class_btn_grid.update_idletasks()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh grid: {e}")

    def _on_class_check_toggle(self, cname):
        """When a checkbox is toggled ON, set it as the active annotation class."""
        var = self._class_check_vars.get(cname)
        if var and var.get():
            self.ann_fast_class_var.set(cname)

    def _refresh_class_map(self):
        """Take the GUI copy of the class list from the registry again (after a model or class change)."""
        self.class_map = self.registry.snapshot()
        with self.fire_lock:
            for cid in self.class_map:
                self.fire_counts.setdefault(cid, 0)
        for name in self.class_map.values():
            if name not in self.all_class_names:
                self.all_class_names.append(name)
            self.shared_state.class_confs.setdefault(name.lower(), 50)
            self.shared_state.defect_allowances.setdefault(name.lower(), 0.0)
        if hasattr(self, 'class_btn_grid'):
            self._refresh_class_grid()

    def _save_class_mapping(self):
        """cashew_class_mapping.json: the YOLO classes (with the model's own IDs) plus the classes the user manages."""
        self.class_map = self.registry.snapshot()
        managed = {str(cid): name for cid, name in sorted(self.class_map.items())
                   if self.registry.is_yolo_class(name) or name in self.all_class_names}
        tmp_path = f"{CLASS_MAPPING_PATH}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(managed, f, indent=4)
        os.replace(tmp_path, CLASS_MAPPING_PATH)

    def _add_label_class(self):
        import tkinter.simpledialog as sd
        new_cname = sd.askstring("Add Class", "Enter new class name:", parent=self)
        if not new_cname:
            return
        new_cname = new_cname.strip()
        if not new_cname:
            return
        if new_cname in self.all_class_names or self.registry.id_for(new_cname) is not None:
            messagebox.showwarning("Warning", f"Class '{new_cname}' already exists!")
            return
        try:
            with self.shared_state.lock:
                keys = [int(k) for k in self.shared_state.capture_classes.keys() if str(k).isdigit()]
                self.shared_state.capture_classes[str(max(keys + [-1]) + 1)] = new_cname
            self.shared_state.save_to_config(CFG_PATH)
            self.all_class_names.append(new_cname)
            self.registry.id_for(new_cname, register=True)      # gets an ID after the YOLO IDs
            self._save_class_mapping()
            self._refresh_class_grid()
            messagebox.showinfo("Success", f"Added class '{new_cname}'")
        except Exception as e:
            log.exception("Could not add class %s", new_cname)
            messagebox.showerror("Error", f"Could not add class: {e}")

    def _rename_label_class(self):
        cname = self.ann_fast_class_var.get()
        if not cname: 
            messagebox.showwarning("Warning", "No class selected!")
            return
        if self.registry.is_yolo_class(cname):
            messagebox.showwarning("Not possible", f"'{cname}' is a class of the YOLO model.\nIts name comes from the model and cannot be changed here.")
            return
        import tkinter.simpledialog as sd
        new_cname = sd.askstring("Rename Class", f"Rename {cname} to:", parent=self)
        if not new_cname:
            return
        new_cname = new_cname.strip()
        if not new_cname:
            return
        if new_cname in self.all_class_names or self.registry.id_for(new_cname) is not None:
            messagebox.showwarning("Warning", f"Class '{new_cname}' already exists!")
            return
        try:
            with self.shared_state.lock:
                for k, v in self.shared_state.capture_classes.items():
                    if v == cname:
                        self.shared_state.capture_classes[k] = new_cname
                if new_cname not in self.shared_state.capture_classes.values():
                    keys = [int(k) for k in self.shared_state.capture_classes.keys() if str(k).isdigit()]
                    self.shared_state.capture_classes[str(max(keys + [-1]) + 1)] = new_cname
            self.shared_state.save_to_config(CFG_PATH)
            self.all_class_names = [new_cname if c == cname else c for c in self.all_class_names]
            if not self.registry.rename(cname, new_cname):
                self.registry.id_for(new_cname, register=True)
            self._save_class_mapping()
            self.ann_fast_class_var.set(new_cname)
            self._refresh_class_grid()
            messagebox.showinfo("Success", f"Renamed to '{new_cname}'")
        except Exception as e:
            log.exception("Could not rename class %s", cname)
            messagebox.showerror("Error", f"Could not rename class: {e}")

    def _delete_label_class(self):
        # Batch-delete all checked classes
        to_delete = [cn for cn, var in self._class_check_vars.items() if var.get()]
        if not to_delete:
            messagebox.showwarning("Warning", "No classes selected! Check the boxes first.")
            return
        yolo_owned = [cn for cn in to_delete if self.registry.is_yolo_class(cn)]
        if yolo_owned:
            messagebox.showwarning("Not possible", "These are classes of the YOLO model and cannot be deleted:\n\n" + ", ".join(yolo_owned))
            to_delete = [cn for cn in to_delete if cn not in yolo_owned]
            if not to_delete:
                return
        names_str = ", ".join(to_delete)
        if not messagebox.askyesno("Confirm", f"Delete {len(to_delete)} class(es)?\n\n{names_str}\n\nImages in the dataset folders are not deleted."):
            return
        try:
            with self.shared_state.lock:
                for cname in to_delete:
                    for k in [k for k, v in self.shared_state.capture_classes.items() if v == cname]:
                        del self.shared_state.capture_classes[k]
            for cname in to_delete:
                self.all_class_names = [c for c in self.all_class_names if c != cname]
                self.registry.remove(cname)
            self.shared_state.save_to_config(CFG_PATH)
            self._save_class_mapping()
            self.ann_fast_class_var.set("")
            self._refresh_class_grid()
            messagebox.showinfo("Success", f"Deleted {len(to_delete)} class(es): {names_str}")
        except Exception as e:
            log.exception("Could not delete classes %s", names_str)
            messagebox.showerror("Error", f"Could not delete classes: {e}")

    HARD_EXIT = True            # False in the self-test, which must go on after closing

    def _on_close(self, then=None):
        """Stop everything, save, end the program. `then` runs after everything is stopped (SHUTDOWN: power off)."""
        print("[SYSTEM] Initiating shutdown...")
        self.stop_event.set()
        self.controller.stop()
        self.shared_state.save_to_config(CFG_PATH)
        
        # Give daemon threads a chance to gracefully release hardware (camera)
        if hasattr(self, 'camera_thread') and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1.0)
        if hasattr(self, 'processing_thread') and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        if hasattr(self, 'processing_thread'):
            self.processing_thread.cashew_log.stop()
        if hasattr(self, 'ejector_check') and self.ejector_check.busy():
            self.ejector_check.stop()
        if hasattr(self, 'processing_thread'):
            self.processing_thread.capture.stop()
            self.processing_thread.undefined_capture.stop()
        if hasattr(self, 'relay_output'):
            self.relay_output.cancel_pending()
            self.relay_output.stop()
            
        print("[SYSTEM] Shutdown complete.")
        if then is not None:
            try:
                then()
            except Exception:
                log.exception("The step after closing failed")
        if self.HARD_EXIT:
            # Everything is stopped and saved above. Tk can stall while it destroys the widgets of pages that were
            # never shown (seen in the self-test, 2026-10-10), which would leave a frozen window after EXIT. So the
            # window is hidden and the process ends here instead of taking the window apart widget by widget.
            try:
                self.withdraw()
                self.update_idletasks()
            except tk.TclError:
                pass
            import logging
            logging.shutdown()
            sys.stdout.flush()
            os._exit(0)
        self.destroy()


def main():
    app = None
    try:
        app = App()
        app.mainloop()
    except KeyboardInterrupt:
        print("\n[SYSTEM] KeyboardInterrupt received. Shutting down...")
        if app:
            app._on_close()
    except Exception as e:
        if app:
            try: app._on_close()
            except (ValueError, tk.TclError): pass
        log.exception("Fatal error: %s", e)

if __name__ == "__main__":
    main()
