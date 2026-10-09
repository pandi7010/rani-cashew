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
        self.view_live = {1: "", 2: ""}                     # latest measured values per ROI (view tuning)
        self.lane_speed = {1: 0.0, 2: 0.0}                  # px/ms, from the tracker
        self.teach_belt_requested = False
        self.teach_belt_message = ""
        self.decision_times = deque(maxlen=5000)            # for pcs/min on the Home page
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
        with self.lock: return (self.delays_ms[act_id-1], self.common_pulse_ms)

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
    def add_decision(self, track_id: str, label: str, frames_txt: str, relay: int, count_keys, detail: str):
        with self.lock:
            self.recent_fires.appendleft((time.strftime("%H:%M:%S"), track_id, label, frames_txt, f"R{relay}" if relay else "-"))
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
        with self.lock: self.common_pulse_ms = pulse_ms
        self.save_to_config(CFG_PATH)

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
            self.recent_fires.appendleft((ts, f"Trk {track}", label, f"{conf:.2f}", f"A{act_id}"))

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
        self.mapping_t1, self.mapping_t2 = {}, {}
        for act_id, class_list in self.mapping_by_act.items():
            try:
                aid = int(act_id)
            except (ValueError, TypeError):
                continue
            for c in class_list:
                norm_c = normalize_name(c)
                if 1 <= aid <= 20: self.mapping_t1[norm_c] = aid
                elif 21 <= aid <= 40: self.mapping_t2[norm_c] = aid

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
                "mapping_t1": self.mapping_t1.copy(), "mapping_t2": self.mapping_t2.copy(),
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
        if _CNN_AVAILABLE and CNN_MODEL_PATH.exists():
            try:
                self.cnn_device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
                self.cnn_model, self.cnn_classes = preprocess.load_cnn_checkpoint(CNN_MODEL_PATH, self.cnn_device)
                self.cnn_transform = preprocess.build_transform(train=False)
                # CNN classes are looked up by name; names YOLO does not have get an ID after the YOLO IDs.
                self.registry.add_names(self.cnn_classes)
                log.info("[CNN] Loaded %s (%d classes: %s)", CNN_MODEL_PATH.name, len(self.cnn_classes), self.cnn_classes)
            except Exception:
                self.cnn_model = None
                log.exception("[CNN] Failed to load %s", CNN_MODEL_PATH)
        else:
            log.warning("[CNN] %s not found or PyTorch missing - YOLO classes are used.", CNN_MODEL_PATH)

        log.info("Class names in use: %s", self.registry.snapshot())

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
                    results = self.model.predict(combined_img, conf=state.get("target_conf", 0.30), verbose=False, device=self.device)[0]
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
            tracker = self.trackers[rid]
            tracker.params = {**tracker.params, **grading.get("tracker", {})}
            tracks = tracker.update(detections, t_ms, flow, roi)
            for det in detections:
                if not det.track.decided:
                    det.track.frames.append(det.data)
                det.track.info["last"] = det.data
            live = next((d.data for d in detections if d.data.get("feat")), None)
            for lost in tracker.dropped:        # disappeared before its line: LOST, nothing is fired
                if not lost.decided:
                    self._log_lost(lost, rid)
            with self.shared_state.lock:
                self.shared_state.lane_speed[rid] = tracker.lane_speed
                self.shared_state.belt_stopped[rid] = tracker.stopped
                if live is not None:
                    self.shared_state.view_live[rid] = f"{live['view']}  {view_features.short_text(live['feat'])}"

            line = lines[rid]
            for trk in tracks:
                if not trk.counted and trk.hits >= 2:   # seen in two frames: a cashew, not noise
                    trk.counted = True
                    self.shared_state.increment_detected_count()
                progress = fg.progress(flow, line, trk.cx, trk.cy) if line is not None else None
                if progress is not None and progress < 0 and not trk.missed:
                    self._golden_capture(clean_frame, trk, state)
                if progress is not None and progress >= 0 and not trk.decided:
                    self._decide_track(trk, rid, line, state, t_ms)
                self._draw_track(frame, trk, grading)

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
                            view_engine=engine_mode["view"])
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

        relay, grade = None, ""
        if d.reject:
            relay = REJECT_RELAY[rid]
            label = d.result if d.result != decision.UNDEFINED else f"UNDEF {d.reason}"
        elif d.result == decision.GOOD:
            # Until step (c2): the old size table and mapping decide the relay of a GOOD cashew.
            mm_length = (length_px or 0.0) / px_per_mm
            mm_width = (width_px or 0.0) / px_per_mm
            _, s_lbl = self._get_size_data(mm_length, mm_width, state["size_config"], rid)
            prefix = "sw" if sw_votes * 2 > d.good_frames else "ww"
            names = ([f"sw{s_lbl[1:]}" if s_lbl.startswith("W") else f"sw_{s_lbl}"] if prefix == "sw" else []) + \
                    [s_lbl, f"{prefix}_{s_lbl}", prefix, "good_cashew", "good"]
            relay = next((map_dict[normalize_name(n)] for n in names if normalize_name(n) in map_dict), None)
            grade = f"{prefix.upper()}_{s_lbl}"
            label = f"GOOD {grade}"
        else:
            relay = map_dict.get(normalize_name(d.result))
            label = d.result
            if relay is None:       # a class without a relay cannot be sorted: reject it
                d.reason, d.reject = f"{decision.NOT_MAPPED}:{d.result}", True
                relay, label = REJECT_RELAY[rid], f"UNDEF no relay ({d.result})"
                d.result = decision.UNDEFINED

        fire_result = ""
        if relay is not None and 1 <= relay <= NUM_ACTUATORS:
            d_ms, p_ms = self.shared_state.get_timing(relay)
            fire_result = self.controller.fire(relay, max(0.0, d_ms - late_ms), p_ms, source=f"{trk.id} {label}")
            cid = self.registry.id_for(d.result)
            if cid is not None:
                with self.fire_lock:
                    self.fire_counts[int(cid)] = self.fire_counts.get(int(cid), 0) + 1

        speed = track_speed(trk.path, flow)
        t_cross = crossing_time(trk.path, flow, line)
        trk.info.update(label=label, relay=relay, result=d.result)
        why = d.reason.split(":")[0].split(" ")[0]
        why = "LOST_OVER_LIMIT" if why.startswith("LOST_") else why
        count_keys = [d.result] + ([f"UNDEFINED: {why}"] if d.result == decision.UNDEFINED else [])
        frame_rows = [{"t": round(f["t"] - trk.born_ms, 1), "cnn": f["cnn"], "cnn_conf": round(f["cnn_conf"], 3),
                       "yolo": f["yolo"], "yolo_conf": round(f["yolo_conf"], 3), "view": f["view"], "edge_cut": f["edge_cut"],
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
        self.shared_state.add_decision(trk.id, label, f"{d.frames_used}f", relay or 0, count_keys, "\n".join(detail))
        self.cashew_log.write({
            "id": trk.id, "roi": rid, "frames": d.frames_total, "frames_used": d.frames_used, "result": d.result, "reason": d.reason,
            "reject": int(d.reject), "good_frames": d.good_frames, "neck_frames": d.neck_frames, "seen": d.seen, "votes": d.votes,
            "views": d.views, "cnn_sw_votes": sw_votes, "length_px": length_px, "width_px": width_px, "area_px": top("area_px"),
            "length_mm": (length_px / px_per_mm) if length_px else "", "width_mm": (width_px / px_per_mm) if width_px else "",
            "L": med("L"), "a": med("a"), "b": med("b"), "dark_pct_max": top("dark_pct"), "red_pct_max": top("red_pct"),
            "spots_max": top("spots"), "speed_px_ms": speed if speed is not None else "", "t_cross_ms": t_cross if t_cross is not None else "",
            "grade": grade, "relay": relay or "", "fire": fire_result, "late": "", "shadow": "; ".join(shadow),
        }, frames=frame_rows)

    def _log_lost(self, trk, rid):
        """A cashew that disappeared before its line (5-track: decision LOST). Counted and logged, never fired."""
        if trk.hits < 2:
            reason = decision.SEEN_ONLY_ONCE
        else:
            reason = decision.LEFT_ROI_UNDECIDED if trk.info.get("left_roi") else decision.LOST_BEFORE_LINE
        self.shared_state.count_result(decision.LOST)
        self.cashew_log.write({"id": trk.id, "roi": rid, "frames": len(trk.frames), "result": decision.LOST, "reason": reason,
                               "fire": "NOT_FIRED"})

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
        self.title("Newton Automation | Industrial HMI")
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
        self.relay_output = RelayOutput(self.controller)
        self.relay_output.manual_log_dir = app_paths.LOGS_DIR     # every manual fire is logged
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
        tk.Label(brand, text="NEWTON", font=(F, ui.F_BODY, "bold"), fg=ui.TEXT, bg=ui.PANEL).pack(anchor="w", padx=16, pady=(12, 0))
        tk.Label(brand, text="AUTOMATION", font=(F, ui.F_SMALL), fg=ui.MUTED, bg=ui.PANEL).pack(anchor="w", padx=16)

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
        tk.Label(prog, text="NEW / COPY: NOT AVAILABLE", font=(F, -10), fg=ui.IDLE, bg=ui.PANEL).pack(side=tk.LEFT, padx=(8, 10))

        self.pills = {}
        self._pill_bar = top
        for key, name, short in (("camera", "CAMERA", "CAM"), ("plc", "PLC", "PLC"), ("safety", "SAFETY", "SAFETY"), ("master", "MASTER EJECTOR", "EJECTOR"),
                                 ("mode", "MODE", "MODE"), ("conveyor", "CONVEYOR", "CONV"), ("state", "STATE", "STATE")):
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
        nav_items = ["Home", "Pneumatic", "Sorting Logic", "Colour Size", "Vision ROI", "Teaching CNN", "Dataset Capture", "Teaching YOLO", "AI Model Select", "Test Console", "Network"]
        self.nav_buttons = {}
        for item in nav_items:
            btn = ui.NavItem(nav_frm, item, command=lambda i=item: self._switch_page(i))
            btn.pack(fill=tk.X)
            self.nav_buttons[item] = btn
        foot = tk.Frame(nav_frm, bg=ui.PANEL)
        foot.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 12))
        tk.Frame(foot, bg=ui.BORDER, height=1).pack(fill=tk.X, padx=10, pady=(0, 10))
        tk.Label(foot, text="GSN V21", font=(F, ui.F_HEAD, "bold"), fg=ui.TEXT, bg=ui.PANEL).pack(anchor="w", padx=14)
        tk.Label(foot, text="CASHEW GRADER", font=(F, -11, "bold"), fg=ui.ACCENT, bg=ui.PANEL).pack(anchor="w", padx=14)
        tk.Label(foot, text=f"Software {_software_version()}", font=(F, -10), fg=ui.MUTED, bg=ui.PANEL).pack(anchor="w", padx=14, pady=(6, 0))

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
        self._switch_page("Home")

    def _fit_pills(self):
        """Keep the status pills inside the top bar: tighter padding, then short names, then a smaller font."""
        bar = self._pill_bar
        bar.update_idletasks()
        for level in (0, 1, 2, 3):
            for pill in self.pills.values():
                pill.set_level(level)
            bar.update_idletasks()
            if sum(child.winfo_reqwidth() + 16 for child in bar.winfo_children()) <= bar.winfo_width() or level == 3:
                break

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
            self._on_close()
            import subprocess
            subprocess.Popen(["systemctl", "poweroff"])

    def _refresh_status(self):
        """Top bar pills, Home tiles and lane LEDs (called a few times per second)."""
        ss, cam = self.shared_state, self.camera_thread
        cam_colour = ui.GOOD if cam.is_connected else (ui.WARN if cam.state in ("CONNECTING", "RECONNECTING") else ui.BAD)
        self.pills["camera"].set(str(cam.state), cam_colour)
        plc_ok = self.controller.is_connected()
        self.pills["plc"].set("ONLINE" if plc_ok else "OFFLINE", ui.GOOD if plc_ok else ui.BAD)
        master, dry = self.relay_output.master_enabled, self.relay_output.dry_run
        self.pills["master"].set("ON" if master else "OFF", ui.GOOD if master else ui.MUTED)
        self.pills["mode"].set("DRY-RUN" if dry else "LIVE", ui.WARN if dry else ui.GOOD)
        self.pills["conveyor"].set("ON" if ss.is_conveyor_on else "OFF", ui.GOOD if ss.is_conveyor_on else ui.MUTED)
        phase = ss.phase
        running = phase == "RUNNING"
        self.pills["state"].set("RUNNING" if running else ("STOPPED" if phase == "READY" else "SETUP"),
                                ui.GOOD if running else (ui.MUTED if phase == "READY" else ui.WARN))
        self.lbl_machine_state.config(text="RUNNING" if running else "STOPPED", fg=ui.GOOD if running else ui.BAD)

        counts = ss.get_result_counts()
        total = sum(n for key, n in counts.items() if not key.startswith("UNDEFINED: "))
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
        with ss.lock:
            engine_counts = dict(ss.engine_counts)
        if engine_counts:
            parts.append("SHADOW would reject: " + ", ".join(f"{name} {n}" for name, n in sorted(engine_counts.items())))
        parts += [f"{key} {n}" for key, n in sorted(counts.items()) if key not in shown + (decision.DOUBLE, decision.LOST) and not key.startswith("UNDEFINED: ")]
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
                                alarms.min_safe_fps(lane_px_s, max_step), free_gb, list(ss.missed_rejects), dry)
        self.pills["safety"].set("FAULT" if active else "OK", ui.BAD if active else ui.GOOD)
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
        tk.Label(ops, text="MASTER EJECTOR is OFF and DRY-RUN is ON at every start.", font=(F, -11), fg=ui.MUTED, bg=ui.PANEL,
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

        cols = ("time", "track", "label", "conf", "act")
        self.fire_tree = ttk.Treeview(tree_frame, columns=cols, show='headings', height=8)
        self.fire_tree.heading("time", text="TIME"); self.fire_tree.column("time", width=80, anchor='center')
        self.fire_tree.heading("track", text="ID"); self.fire_tree.column("track", width=110, anchor='center')
        self.fire_tree.heading("label", text="RESULT / CLASS"); self.fire_tree.column("label", width=280, anchor='w')
        self.fire_tree.heading("conf", text="FRAMES"); self.fire_tree.column("conf", width=70, anchor='center')
        self.fire_tree.heading("act", text="RELAY"); self.fire_tree.column("act", width=70, anchor='center')
        self.fire_tree.bind("<Double-1>", self._show_decision_detail)     # frame-by-frame votes of that cashew

        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.fire_tree.yview)
        self.fire_tree.configure(yscrollcommand=tree_scroll.set)

        self.fire_tree.pack(side="left", fill=tk.BOTH, expand=True)
        tree_scroll.pack(side="right", fill="y")
        

        # PAGE: Dataset Capture
        p_cap = tk.Frame(self.page_container, bg=self.colors["surface"])
        self.pages["Dataset Capture"] = p_cap
        cap_canvas = tk.Canvas(p_cap, bg=self.colors["surface"], highlightthickness=0)
        cap_scroll = ttk.Scrollbar(p_cap, orient="vertical", command=cap_canvas.yview)
        cap_inner = tk.Frame(cap_canvas, bg=self.colors["surface"], padx=15, pady=15)
        cap_inner.bind("<Configure>", lambda e: cap_canvas.configure(scrollregion=cap_canvas.bbox("all")))
        cap_canvas.create_window((0, 0), window=cap_inner, anchor="nw", width=520)
        cap_canvas.configure(yscrollcommand=cap_scroll.set)
        cap_canvas.pack(side="left", fill="both", expand=True)
        cap_scroll.pack(side="right", fill="y")
        
        ttk.Label(cap_inner, text="DATASET CAPTURE FOLDERS", style="Title.TLabel").pack(anchor="w", pady=(0,15))
        
        # Capture button at top of capture page
        ttk.Button(cap_inner, text="CAPTURE IMAGE TO FOLDER", style="Success.TButton", command=self.shared_state.trigger_capture).pack(fill=tk.X, pady=(0, 20))
        
        self.capture_class_vars = {}
        self.capture_count_vars = {}
        self.active_capture_var = tk.StringVar(value=self.shared_state.active_capture_class_id)
        
        def update_all_counts():
            for i in range(1, 21):
                folder_name = self.shared_state.capture_classes.get(str(i), f"Class_{i}")
                target_dir = os.path.join(DATASET_DIR, folder_name)
                count = len([f for f in os.listdir(target_dir) if f.endswith('.jpg')]) if os.path.exists(target_dir) else 0
                if str(i) in self.capture_count_vars:
                    self.capture_count_vars[str(i)].set(f"[{count}]")
            self.after(1000, update_all_counts)
                
        self.after(1000, update_all_counts)
        
        cap_grid = tk.Frame(cap_inner, bg=self.colors["surface"])
        cap_grid.pack(fill=tk.X)
        cap_grid.columnconfigure(0, weight=1); cap_grid.columnconfigure(1, weight=1)
        
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
            
            f1 = tk.Frame(cap_grid, bg=self.colors["surface"])
            f1.grid(row=row, column=col, sticky="ew", padx=(0,10) if col==0 else (10,0), pady=4)
            
            rb = tk.Radiobutton(f1, text=f"{i:02d}", variable=self.active_capture_var, value=str(i), 
                                bg=self.colors["surface"], fg=self.colors["text"], selectcolor=self.colors["bg"],
                                activebackground=self.colors["surface"], activeforeground=self.colors["text"],
                                font=("Roboto", 10, "bold"), command=on_capture_select)
            rb.pack(side=tk.LEFT)
            
            nm = tk.StringVar(value=self.shared_state.capture_classes.get(str(i), f"Class_{i}"))
            ent = tk.Entry(f1, textvariable=nm, width=10, bg=self.colors["surface_light"], fg=self.colors["text"], bd=0)
            ent.pack(side=tk.LEFT, padx=(5,0))
            self.capture_class_vars[str(i)] = nm
            nm.trace_add("write", lambda *_, cid=i, v=nm: update_capture_class_name(cid, v))
            
            cnt_var = tk.StringVar(value="[0]")
            self.capture_count_vars[str(i)] = cnt_var
            tk.Label(f1, textvariable=cnt_var, bg=self.colors["surface"], fg=self.colors["text"], font=("Roboto", 8)).pack(side=tk.LEFT, padx=(2,2))
            
            def open_dir(cid=i):
                import subprocess
                folder_name = self.shared_state.capture_classes.get(str(cid), f"Class_{cid}")
                target_dir = os.path.join(DATASET_DIR, folder_name)
                os.makedirs(target_dir, exist_ok=True)
                subprocess.Popen(['xdg-open', target_dir])
                
            tk.Button(f1, text="VIEW", bg=self.colors["bg"], fg=self.colors["text"], bd=0, font=("Roboto", 7), padx=2, pady=0, command=open_dir).pack(side=tk.LEFT)
            
        ttk.Button(cap_inner, text="SAVE CLASS NAMES", style="Primary.TButton", command=lambda: self.shared_state.save_to_config(CFG_PATH)).pack(fill=tk.X, pady=(20, 40))

        # ── AUTO CAPTURE CONSOLE ──
        ttk.Label(cap_inner, text="AUTO CAPTURE CONSOLE", style="Title.TLabel").pack(anchor="w", pady=(0, 10))
        
        console_frm = tk.Frame(cap_inner, bg=self.colors["surface_light"], padx=15, pady=15)
        console_frm.pack(fill=tk.X, pady=(0, 20))
        
        # Limit
        limit_frm = tk.Frame(console_frm, bg=self.colors["surface_light"])
        limit_frm.pack(fill=tk.X, pady=(0, 15))
        tk.Label(limit_frm, text="Capture Image Limit:", bg=self.colors["surface_light"], fg=self.colors["text"], font=("Roboto", 10, "bold")).pack(side=tk.LEFT)
        ttk.Combobox(limit_frm, textvariable=self.golden_count_var, values=["100", "500", "1000", "2000", "5000"], width=10, font=("Roboto", 10)).pack(side=tk.LEFT, padx=10)
        
        # Mode
        mode_frm = tk.Frame(console_frm, bg=self.colors["surface_light"])
        mode_frm.pack(fill=tk.X, pady=(0, 20))
        tk.Label(mode_frm, text="Capture Mode:", bg=self.colors["surface_light"], fg=self.colors["text"], font=("Roboto", 10, "bold")).pack(side=tk.LEFT, padx=(0, 10))
        
        style_rb = {"bg": self.colors["surface_light"], "fg": self.colors["text"], "selectcolor": self.colors["bg"], "activebackground": self.colors["surface_light"], "activeforeground": self.colors["text"], "font": ("Roboto", 10)}
        tk.Radiobutton(mode_frm, text="Cropped images", variable=self.capture_mode_var, value="cropped", **style_rb).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(mode_frm, text="Full frame", variable=self.capture_mode_var, value="full", **style_rb).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(mode_frm, text="Both", variable=self.capture_mode_var, value="both", **style_rb).pack(side=tk.LEFT, padx=5)
        
        # Toggle Button
        self.golden_prod_btn = tk.Button(console_frm, text="START AUTO CAPTURE", 
                                         bg=self.colors["info"], fg="white", font=("Roboto", 12, "bold"),
                                         bd=0, cursor="hand2", pady=10)
                                         
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
        self.pages["AI Model Select"] = p_model
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
        sort_inner = tk.Frame(sort_canvas, bg=self.colors["surface"], padx=15, pady=15)
        sort_inner.bind("<Configure>", lambda e: sort_canvas.configure(scrollregion=sort_canvas.bbox("all")))
        sort_canvas.create_window((0, 0), window=sort_inner, anchor="nw", width=430)
        sort_canvas.configure(yscrollcommand=sort_scroll.set)
        sort_canvas.pack(side="left", fill="both", expand=True)
        sort_scroll.pack(side="right", fill="y")
        
        self._build_grading_section(sort_inner)

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
        
        ttk.Label(sort_inner, text="CONFIDENCE THRESHOLDS", style="Title.TLabel").pack(anchor="w", pady=(20,10))
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


        p_pneu = tk.Frame(self.page_container, bg=self.colors["surface"])
        self.pages["Pneumatic"] = p_pneu
        # Right controls panel
        pn_controls = tk.Frame(p_pneu, bg=self.colors["surface"], width=320, padx=15, pady=15)
        pn_controls.pack(side="right", fill="y", padx=(10,0))
        pn_controls.pack_propagate(False)
        
        ttk.Button(pn_controls, text="SAVE PNEUMATICS", style="Primary.TButton", command=self._save_pneumatics).pack(fill=tk.X, pady=(0, 30))
        ttk.Label(pn_controls, text="GLOBAL CONTROLS", style="Header.TLabel").pack(anchor="w", pady=(0, 10))
        
        cp_frm = tk.Frame(pn_controls, bg=self.colors["surface"])
        cp_frm.pack(fill=tk.X, pady=(0,15))
        ttk.Label(cp_frm, text="Common Pulse Time:", style="Surface.TLabel").pack(side=tk.LEFT)
        self.common_pulse_var = tk.StringVar(value=str(self.shared_state.common_pulse_ms))
        tk.Entry(cp_frm, textvariable=self.common_pulse_var, width=8, bg=self.colors["surface_light"], fg=self.colors["text"], bd=0, font=("Roboto", 12)).pack(side=tk.RIGHT)
        self.common_pulse_var.trace_add("write", self._update_common_pulse)
        
        cs_frm = tk.Frame(pn_controls, bg=self.colors["surface"])
        cs_frm.pack(fill=tk.X, pady=(0,15))
        ttk.Label(cs_frm, text="Conveyor Speed (mm/s):", style="Surface.TLabel").pack(side=tk.LEFT)
        self.conveyor_speed_var = tk.StringVar(value=str(int(getattr(self.shared_state, 'conveyor_speed_mm_s', 300))))
        tk.Entry(cs_frm, textvariable=self.conveyor_speed_var, width=8, bg=self.colors["surface_light"], fg=self.colors["text"], bd=0, font=("Roboto", 12)).pack(side=tk.RIGHT)
        self.conveyor_speed_var.trace_add("write", self._update_conveyor_speed)
        
        ttk.Button(pn_controls, text="RECALCULATE FROM DISTANCES", style="Secondary.TButton", command=self._recalculate_all_from_distances).pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(pn_controls, text="OPEN PLC DIAGNOSTICS", style="Primary.TButton", command=self._open_plc_diagnostics).pack(fill=tk.X, pady=(0, 10))

        ttk.Separator(pn_controls, orient="horizontal").pack(fill=tk.X, pady=10)

        surf = self.colors["surface"]
        # DRY-RUN: on at every start. Nothing reaches the PLC; what would fire is only counted and logged.
        self.dry_run_var = tk.BooleanVar(value=True)
        # MASTER EJECTOR: off at every start. Blocks firing that comes from grading.
        self.master_ejector_var = tk.BooleanVar(value=False)

        def refresh_safety_labels():
            dry, master = self.relay_output.dry_run, self.relay_output.master_enabled
            self.dry_run_btn.config(text="DRY-RUN: ON (nothing is sent to the PLC)" if dry else "DRY-RUN: OFF (REAL valves)",
                                    fg=self.colors["success"] if dry else self.colors["danger"])
            self.master_ej_btn.config(text="MASTER EJECTOR: ON" if master else "MASTER EJECTOR: OFF")
            if not master:
                text, colour = "● Grading does not fire", self.colors["text_muted"]
            elif dry:
                text, colour = "● Grading: dry-run only", self.colors["warning"]
            else:
                text, colour = "● Grading fires REAL valves", self.colors["danger"]
            self.master_ej_led.config(text=text, fg=colour)

        def toggle_dry_run():
            want = self.dry_run_var.get()
            if not want and not messagebox.askyesno("DRY-RUN OFF", "With DRY-RUN off, test buttons and grading (if MASTER EJECTOR is on) energise REAL valves.\n\nSwitch DRY-RUN off?"):
                self.dry_run_var.set(True)
                want = True
            self.relay_output.dry_run = want
            log.info("DRY-RUN %s", "ON" if want else "OFF")
            refresh_safety_labels()

        def toggle_ejector():
            self.relay_output.master_enabled = self.master_ejector_var.get()
            log.info("MASTER EJECTOR %s", "ON" if self.relay_output.master_enabled else "OFF")
            refresh_safety_labels()

        self.dry_run_btn = tk.Checkbutton(pn_controls, variable=self.dry_run_var, command=toggle_dry_run, bg=surf, selectcolor=self.colors["surface_light"],
                                          activebackground=surf, font=("Roboto", 9, "bold"), indicatoron=False, wraplength=240, pady=4)
        self.dry_run_btn.pack(fill=tk.X, pady=(0, 6))
        self.master_ej_btn = tk.Checkbutton(pn_controls, variable=self.master_ejector_var, command=toggle_ejector, bg=surf, fg="white", selectcolor=self.colors["surface_light"],
                                            activebackground=surf, font=("Roboto", 11, "bold"), indicatoron=False, pady=4)
        self.master_ej_btn.pack(fill=tk.X, pady=(0, 6))
        self.master_ej_led = tk.Label(pn_controls, text="", font=("Roboto", 10, "bold"), bg=surf)
        self.master_ej_led.pack(anchor="center")
        refresh_safety_labels()

        ttk.Separator(pn_controls, orient="horizontal").pack(fill=tk.X, pady=10)

        # ── Single relay test: check that each M coil drives the right valve ──
        ttk.Label(pn_controls, text="SINGLE RELAY TEST", style="Header.TLabel").pack(anchor="w", pady=(0, 6))
        test_row = tk.Frame(pn_controls, bg=surf)
        test_row.pack(fill=tk.X)
        ttk.Label(test_row, text="Relay", style="Surface.TLabel").pack(side=tk.LEFT)
        self.test_relay_var = tk.StringVar(value="1")
        ttk.Combobox(test_row, textvariable=self.test_relay_var, values=[str(r) for r in range(FIRST_EJECTOR, LAST_EJECTOR + 1)],
                     width=4, state="readonly").pack(side=tk.LEFT, padx=(5, 10))
        ttk.Label(test_row, text="Pulse ms", style="Surface.TLabel").pack(side=tk.LEFT)
        self.test_pulse_var = tk.StringVar(value="100")
        tk.Entry(test_row, textvariable=self.test_pulse_var, width=5, bg=self.colors["surface_light"], fg=self.colors["text"], bd=0, font=("Roboto", 11)).pack(side=tk.LEFT, padx=5)
        self.lbl_test_relay_addr = tk.Label(pn_controls, text="", bg=surf, fg=self.colors["info"], font=("Roboto", 10, "bold"), anchor="w")
        self.lbl_test_relay_addr.pack(fill=tk.X, pady=(6, 2))
        nav_row = tk.Frame(pn_controls, bg=surf)
        nav_row.pack(fill=tk.X, pady=2)
        nav_row.columnconfigure(0, weight=1); nav_row.columnconfigure(1, weight=3); nav_row.columnconfigure(2, weight=1)

        def show_test_relay(*_):
            try:
                relay = int(self.test_relay_var.get())
            except ValueError:
                return
            roi = 1 if relay <= 20 else 2
            role = "REJECT relay" if relay in (1, 21) else "class / grade relay"
            self.lbl_test_relay_addr.config(text=f"{self.relay_output.describe(relay)}\nROI {roi}, {role}", justify="left")

        def step_test_relay(step):
            relay = min(LAST_EJECTOR, max(FIRST_EJECTOR, int(self.test_relay_var.get()) + step))
            self.test_relay_var.set(str(relay))

        ttk.Button(nav_row, text="◀", width=3, command=lambda: step_test_relay(-1)).grid(row=0, column=0, sticky="ew", padx=(0, 3))
        ttk.Button(nav_row, text="TEST FIRE THIS RELAY", style="Danger.TButton", command=self._test_fire_selected_relay).grid(row=0, column=1, sticky="ew")
        ttk.Button(nav_row, text="▶", width=3, command=lambda: step_test_relay(1)).grid(row=0, column=2, sticky="ew", padx=(3, 0))
        self.lbl_test_relay_result = tk.Label(pn_controls, text="Only when the machine is STOPPED.", bg=surf, fg=self.colors["text_muted"],
                                              font=("Roboto", 9), anchor="w", justify="left", wraplength=250)
        self.lbl_test_relay_result.pack(fill=tk.X, pady=(4, 8))
        self.test_relay_var.trace_add("write", show_test_relay)
        show_test_relay()

        all_row = tk.Frame(pn_controls, bg=surf)
        all_row.pack(fill=tk.X)
        all_row.columnconfigure(0, weight=1); all_row.columnconfigure(1, weight=1)
        ttk.Button(all_row, text="SEQUENCE 1-40", command=self._sequence_all).grid(row=0, column=0, sticky="ew", padx=(0, 3))
        ttk.Button(all_row, text="FIRE 1-40", command=self._fire_all).grid(row=0, column=1, sticky="ew", padx=(3, 0))

        pn_canvas = tk.Canvas(p_pneu, bg=self.colors["surface"], highlightthickness=0)
        pn_scroll = ttk.Scrollbar(p_pneu, orient="vertical", command=pn_canvas.yview)
        pn_inner = tk.Frame(pn_canvas, bg=self.colors["surface"], padx=15, pady=15)
        pn_inner.bind("<Configure>", lambda e: pn_canvas.configure(scrollregion=pn_canvas.bbox("all")))
        pn_inner_id = pn_canvas.create_window((0, 0), window=pn_inner, anchor="nw")
        pn_canvas.bind("<Configure>", lambda e: pn_canvas.itemconfig(pn_inner_id, width=e.width))
        pn_canvas.configure(yscrollcommand=pn_scroll.set)
        pn_canvas.pack(side="left", fill="both", expand=True)
        pn_scroll.pack(side="right", fill="y")
        
        ttk.Label(pn_inner, text="PNEUMATIC TIMING (MS) & DISTANCE (MM)", style="Title.TLabel").pack(anchor="w", pady=(0,10))
        
        act_grid = tk.Frame(pn_inner, bg=self.colors["surface"])
        act_grid.pack(fill=tk.X)
        act_grid.columnconfigure(0, weight=1); act_grid.columnconfigure(1, weight=1)
        
        # Header Row 0
        h_f1 = tk.Frame(act_grid, bg=self.colors["surface"])
        h_f1.grid(row=0, column=0, sticky="ew", padx=(0,10), pady=(0,8))
        ttk.Label(h_f1, text="ROI LEFT (A01-A20)", font=("Roboto", 12, "bold"), foreground=self.colors["info"], style="Surface.TLabel").pack(side=tk.LEFT)
        ttk.Label(h_f1, text="Dist(mm)  Delay(ms)", font=("Roboto", 9, "bold"), foreground=self.colors["text_muted"], style="Surface.TLabel").pack(side=tk.RIGHT, padx=(0,40))

        h_f2 = tk.Frame(act_grid, bg=self.colors["surface"])
        h_f2.grid(row=0, column=1, sticky="ew", padx=(10,0), pady=(0,8))
        ttk.Label(h_f2, text="ROI RIGHT (A21-A40)", font=("Roboto", 12, "bold"), foreground=self.colors["info"], style="Surface.TLabel").pack(side=tk.LEFT)
        ttk.Label(h_f2, text="Dist(mm)  Delay(ms)", font=("Roboto", 9, "bold"), foreground=self.colors["text_muted"], style="Surface.TLabel").pack(side=tk.RIGHT, padx=(0,40))

        self.act_vars = {}
        self.dist_vars = {}
        self._recalc_active = False

        for i in range(1, 21):
            f1 = tk.Frame(act_grid, bg=self.colors["surface"])
            f1.grid(row=i, column=0, sticky="ew", padx=(0,10), pady=4)
            d1 = tk.StringVar(value=str(self.shared_state.delays_ms[i-1]))
            cur_dist1 = self.shared_state.distances_mm[i-1]
            dist_str1 = f"{cur_dist1:.1f}" if cur_dist1 > 0 else ""
            dist1 = tk.StringVar(value=dist_str1)
            self.dist_vars[i] = dist1
            self.act_vars[i] = d1

            ttk.Label(f1, text=f"A{i:02d}", font=("Roboto", 10, "bold"), foreground=self.colors["text_muted"], style="Surface.TLabel").pack(side=tk.LEFT, padx=(0,3))
            
            # Distance input box (mm)
            tk.Entry(f1, textvariable=dist1, width=5, bg=self.colors["surface_light"], fg=self.colors.get("accent", "#38BDF8"), bd=0, font=("Roboto", 10)).pack(side=tk.LEFT, padx=(0,1))
            ttk.Label(f1, text="mm", font=("Roboto", 8), foreground=self.colors["text_muted"], style="Surface.TLabel").pack(side=tk.LEFT, padx=(0,4))

            # Delay input box (ms)
            tk.Entry(f1, textvariable=d1, width=5, bg=self.colors["surface_light"], fg=self.colors["text"], bd=0, font=("Roboto", 10)).pack(side=tk.LEFT, padx=(0,1))
            ttk.Label(f1, text="ms", font=("Roboto", 8), foreground=self.colors["text_muted"], style="Surface.TLabel").pack(side=tk.LEFT, padx=(0,4))

            if hasattr(self, 'img_test') and self.img_test:
                tk.Button(f1, image=self.img_test, bg=self.colors["surface_light"], bd=0, command=lambda x=i: self._pulse_and_flash(x)).pack(side=tk.LEFT)
            else:
                tk.Button(f1, text="T", bg=self.colors["surface_light"], fg=self.colors["text"], bd=0, command=lambda x=i: self._pulse_and_flash(x)).pack(side=tk.LEFT)
                
            if hasattr(self, 'img_gear') and self.img_gear:
                tk.Button(f1, image=self.img_gear, bg=self.colors["surface_light"], bd=0, command=lambda x=i: self._popup_mapper(x)).pack(side=tk.LEFT, padx=(2,0))
            else:
                tk.Button(f1, text="M", bg=self.colors["info"], fg="white", bd=0, command=lambda x=i: self._popup_mapper(x)).pack(side=tk.LEFT, padx=(2,0))
            
            if not hasattr(self, 'pn_map_lbls'): self.pn_map_lbls = {}
            if not hasattr(self, 'pn_leds'): self.pn_leds = {}
            
            self.pn_leds[i] = tk.Label(f1, text="●", fg=self.colors["surface_light"], bg=self.colors["surface"], font=("Roboto", 10))
            self.pn_leds[i].pack(side=tk.LEFT, padx=(2,0))
            
            self.pn_map_lbls[i] = ttk.Label(f1, text="(none)", width=24, anchor="w", style="Surface.TLabel")
            self.pn_map_lbls[i].pack(side=tk.LEFT, padx=(3,0))

            dist1.trace_add("write", lambda *_, ii=i: self._on_distance_changed(ii))
            d1.trace_add("write", lambda *_, ii=i: self._update_act_timing(ii))
            
            i2 = i + 20
            f2 = tk.Frame(act_grid, bg=self.colors["surface"])
            f2.grid(row=i, column=1, sticky="ew", padx=(10,0), pady=4)
            d2 = tk.StringVar(value=str(self.shared_state.delays_ms[i2-1]))
            cur_dist2 = self.shared_state.distances_mm[i2-1]
            dist_str2 = f"{cur_dist2:.1f}" if cur_dist2 > 0 else ""
            dist2 = tk.StringVar(value=dist_str2)
            self.dist_vars[i2] = dist2
            self.act_vars[i2] = d2

            ttk.Label(f2, text=f"A{i2:02d}", font=("Roboto", 10, "bold"), foreground=self.colors["text_muted"], style="Surface.TLabel").pack(side=tk.LEFT, padx=(0,3))
            
            # Distance input box (mm)
            tk.Entry(f2, textvariable=dist2, width=5, bg=self.colors["surface_light"], fg=self.colors.get("accent", "#38BDF8"), bd=0, font=("Roboto", 10)).pack(side=tk.LEFT, padx=(0,1))
            ttk.Label(f2, text="mm", font=("Roboto", 8), foreground=self.colors["text_muted"], style="Surface.TLabel").pack(side=tk.LEFT, padx=(0,4))

            # Delay input box (ms)
            tk.Entry(f2, textvariable=d2, width=5, bg=self.colors["surface_light"], fg=self.colors["text"], bd=0, font=("Roboto", 10)).pack(side=tk.LEFT, padx=(0,1))
            ttk.Label(f2, text="ms", font=("Roboto", 8), foreground=self.colors["text_muted"], style="Surface.TLabel").pack(side=tk.LEFT, padx=(0,4))

            if hasattr(self, 'img_test') and self.img_test:
                tk.Button(f2, image=self.img_test, bg=self.colors["surface_light"], bd=0, command=lambda x=i2: self._pulse_and_flash(x)).pack(side=tk.LEFT)
            else:
                tk.Button(f2, text="T", bg=self.colors["surface_light"], fg=self.colors["text"], bd=0, command=lambda x=i2: self._pulse_and_flash(x)).pack(side=tk.LEFT)
                
            if hasattr(self, 'img_gear') and self.img_gear:
                tk.Button(f2, image=self.img_gear, bg=self.colors["surface_light"], bd=0, command=lambda x=i2: self._popup_mapper(x)).pack(side=tk.LEFT, padx=(2,0))
            else:
                tk.Button(f2, text="M", bg=self.colors["info"], fg="white", bd=0, command=lambda x=i2: self._popup_mapper(x)).pack(side=tk.LEFT, padx=(2,0))
                
            self.pn_leds[i2] = tk.Label(f2, text="●", fg=self.colors["surface_light"], bg=self.colors["surface"], font=("Roboto", 10))
            self.pn_leds[i2].pack(side=tk.LEFT, padx=(2,0))
            
            self.pn_map_lbls[i2] = ttk.Label(f2, text="(none)", width=24, anchor="w", style="Surface.TLabel")
            self.pn_map_lbls[i2].pack(side=tk.LEFT, padx=(3,0))

            dist2.trace_add("write", lambda *_, ii=i2: self._on_distance_changed(ii))
            d2.trace_add("write", lambda *_, ii=i2: self._update_act_timing(ii))
            
        self._update_all_map_labels()

        p_net = tk.Frame(self.page_container, bg=self.colors["surface"], padx=15, pady=15)
        self.pages["Network"] = p_net
        ttk.Label(p_net, text="NETWORK CONFIGURATION", style="Title.TLabel").pack(anchor="w", pady=(0,15))
        
        # --- Inovance PLC (Modbus TCP) Configuration ---
        ttk.Label(p_net, text="Inovance EC320 PLC (Modbus TCP)", style="Header.TLabel").pack(anchor="w", pady=(0, 10))

        plc_card = tk.Frame(p_net, bg=self.colors["surface_light"], padx=12, pady=12)
        plc_card.pack(fill=tk.X, pady=(0, 15))

        # Row 1: IP & Port
        pr1 = tk.Frame(plc_card, bg=self.colors["surface_light"])
        pr1.pack(fill=tk.X, pady=4)
        ttk.Label(pr1, text="PLC IP:", font=("Roboto", 10, "bold"), width=16, style="Surface.TLabel").pack(side=tk.LEFT)
        self.net_plc_ip_var = tk.StringVar(value=getattr(self.controller, 'plc_ip', '192.168.1.88'))
        tk.Entry(pr1, textvariable=self.net_plc_ip_var, width=16, bg=self.colors["surface"], fg=self.colors["text"], bd=0, font=("Roboto", 11)).pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(pr1, text="Port:", font=("Roboto", 10, "bold"), width=6, style="Surface.TLabel").pack(side=tk.LEFT)
        self.net_plc_port_var = tk.StringVar(value=str(getattr(self.controller, 'plc_port', 502)))
        tk.Entry(pr1, textvariable=self.net_plc_port_var, width=8, bg=self.colors["surface"], fg=self.colors["text"], bd=0, font=("Roboto", 11)).pack(side=tk.LEFT)

        # Row 2: Unit ID & Register Type
        pr2 = tk.Frame(plc_card, bg=self.colors["surface_light"])
        pr2.pack(fill=tk.X, pady=4)
        ttk.Label(pr2, text="Unit ID:", font=("Roboto", 10, "bold"), width=16, style="Surface.TLabel").pack(side=tk.LEFT)
        self.net_plc_unit_var = tk.StringVar(value=str(getattr(self.controller, 'plc_unit_id', 1)))
        tk.Entry(pr2, textvariable=self.net_plc_unit_var, width=8, bg=self.colors["surface"], fg=self.colors["text"], bd=0, font=("Roboto", 11)).pack(side=tk.LEFT, padx=(0, 28))

        ttk.Label(pr2, text="Register Type:", font=("Roboto", 10, "bold"), width=14, style="Surface.TLabel").pack(side=tk.LEFT)
        self.net_plc_regtype_var = tk.StringVar(value=getattr(self.controller, 'register_type', 'Coil (FC01/05/15)'))
        ttk.Combobox(pr2, textvariable=self.net_plc_regtype_var, values=["Coil (FC01/05/15)", "Holding Reg (FC03/06/16)"], width=22, state="readonly").pack(side=tk.LEFT)

        # Row 3: Base Register (M100) & Status
        pr3 = tk.Frame(plc_card, bg=self.colors["surface_light"])
        pr3.pack(fill=tk.X, pady=4)
        ttk.Label(pr3, text="Relay 1 Addr (M100):", font=("Roboto", 10, "bold"), width=16, style="Surface.TLabel").pack(side=tk.LEFT)
        self.net_plc_basereg_var = tk.StringVar(value=str(getattr(self.controller, 'relay1_register', 100)))
        tk.Entry(pr3, textvariable=self.net_plc_basereg_var, width=8, bg=self.colors["surface"], fg=self.colors["text"], bd=0, font=("Roboto", 11)).pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(pr3, text="PLC Status:", font=("Roboto", 10, "bold"), width=12, style="Surface.TLabel").pack(side=tk.LEFT)
        self.net_plc_status = ttk.Label(pr3, text="● CHECKING...", style="Surface.TLabel", foreground=self.colors["warning"])
        self.net_plc_status.pack(side=tk.LEFT)

        # Row 4: Buttons
        pr4 = tk.Frame(plc_card, bg=self.colors["surface_light"])
        pr4.pack(fill=tk.X, pady=(10, 0))

        def _apply_plc_settings():
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

        ttk.Button(pr4, text="Save & Connect PLC", style="Primary.TButton", command=_apply_plc_settings).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(pr4, text="Open PLC Diagnostics Tool", style="Warning.TButton", command=self._open_plc_diagnostics).pack(side=tk.LEFT)

        self._build_tcm_studio_page()
        self._build_annotate_page()

    def _build_tcm_studio_page(self):
        p_tcm = tk.Frame(self.page_container, bg=self.colors["surface"], padx=15, pady=15)
        self.pages["Teaching CNN"] = p_tcm
        
        ttk.Label(p_tcm, text="TCM STUDIO - AUTO DATASET & TRAIN", style="Title.TLabel").pack(anchor="w", pady=(0, 20))
        

        
        # Section 1: Manage Golden Images
        rev_frm = tk.Frame(p_tcm, bg=self.colors["bg"], padx=10, pady=10)
        rev_frm.pack(fill=tk.X, pady=(0, 20))
        ttk.Label(rev_frm, text="1. Manage Golden Images", style="Header.TLabel").pack(anchor="w", pady=(0,10))
        
        row2 = tk.Frame(rev_frm, bg=self.colors["bg"])
        row2.pack(fill=tk.X, pady=5)
        
        def open_dataset_folders():
            import subprocess
            import os
            os.makedirs(GOLDEN_DIR, exist_ok=True)
            os.makedirs(DATASET_DIR, exist_ok=True)
            subprocess.Popen(["xdg-open", GOLDEN_DIR])
            subprocess.Popen(["xdg-open", DATASET_DIR])
            
        def launch_auto_labeler():
            import subprocess
            try:
                subprocess.Popen([sys.executable, str(app_paths.AUTO_LABELER_SCRIPT)], cwd=str(app_paths.BASE_DIR))
            except Exception as e:
                log.exception("Could not start the AI Auto-Labeler")
                messagebox.showerror("Launch Error", f"Failed to start AI Auto-Labeler: {e}")
                
        ttk.Button(row2, text="Open Dataset Folders", style="Primary.TButton", command=open_dataset_folders).pack(side=tk.LEFT, padx=(0,10))
        ttk.Button(row2, text="Cross Check Labels (TCM)", style="Warning.TButton", command=self._launch_tcm_studio).pack(side=tk.LEFT, padx=(0,10))
        ttk.Button(row2, text="AI Auto-Label (Fast Review)", style="Success.TButton", command=launch_auto_labeler).pack(side=tk.LEFT, padx=(0,10))
        ttk.Label(row2, text="Run AI Auto-Label to sort Golden crops quickly.", style="Surface.TLabel").pack(side=tk.LEFT, padx=(10,0))
        
        # Section 2: Training Setup
        trn_frm = tk.Frame(p_tcm, bg=self.colors["bg"], padx=10, pady=10)
        trn_frm.pack(fill=tk.X, pady=(0, 20))
        ttk.Label(trn_frm, text="2. Train Model", style="Header.TLabel").pack(anchor="w", pady=(0,10))
        
        t_row = tk.Frame(trn_frm, bg=self.colors["bg"])
        t_row.pack(fill=tk.X, pady=5)
        
        ttk.Label(t_row, text="Model Architecture:").pack(side=tk.LEFT, padx=(0,5))
        self.tcm_arch_var = tk.StringVar(value="CNN (MobileNet)")
        arch_combo = ttk.Combobox(t_row, textvariable=self.tcm_arch_var, values=["CNN (MobileNet)", "YOLOv8-Classifier", "YOLO11-Classifier"], state="readonly", width=18)
        arch_combo.pack(side=tk.LEFT, padx=(0,15))
        
        ttk.Label(t_row, text="Epochs:").pack(side=tk.LEFT, padx=(0,5))
        self.tcm_epochs = tk.IntVar(value=100)
        tk.Entry(t_row, textvariable=self.tcm_epochs, width=5).pack(side=tk.LEFT, padx=(0,15))
        
        self.tcm_train_btn = ttk.Button(t_row, text="START TRAINING", style="Primary.TButton", command=self._start_training_router)
        self.tcm_train_btn.pack(side=tk.LEFT)
        
        self.is_training = False
        
        # Section 3: Verify Mismatches
        ver_frm = tk.Frame(p_tcm, bg=self.colors["bg"], padx=10, pady=10)
        ver_frm.pack(fill=tk.X, pady=(0, 20))
        ttk.Label(ver_frm, text="3. Verify Confusion Images", style="Header.TLabel").pack(anchor="w", pady=(0,10))
        
        v_row = tk.Frame(ver_frm, bg=self.colors["bg"])
        v_row.pack(fill=tk.X, pady=5)
        
        self.tcm_verify_btn = ttk.Button(v_row, text="Verify Mismatches", style="Warning.TButton", command=self._verify_mismatches)
        self.tcm_verify_btn.pack(side=tk.LEFT, padx=(0,10))
        ttk.Label(v_row, text="Double-click confusing images to re-assign class.").pack(side=tk.LEFT)
        
        self.tcm_log = tk.Text(ver_frm, bg="black", fg="white", height=5)
        self.tcm_log.pack(fill=tk.X, pady=(10, 0))


    def _start_training_router(self):
        arch = self.tcm_arch_var.get()
        if "CNN" in arch:
            self._start_cnn_train()
        else:
            self._start_yolo_train(arch)

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

    def _start_cnn_train(self):
        if getattr(self, 'is_training', False):
            return
        train_script = app_paths.BASE_DIR / "train_cnn.py"
        if not train_script.exists():
            messagebox.showerror("Not available", "CNN training is not available yet:\ntrain_cnn.py is not in the project folder.")
            return
        dataset_dir = DATASET_DIR
        disabled_dir = str(app_paths.DATASET_DISABLED_DIR)
        
        # In case previous run crashed, restore disabled folders first
        if os.path.exists(disabled_dir):
            for d in os.listdir(disabled_dir):
                if not os.path.exists(os.path.join(dataset_dir, d)):
                    shutil.move(os.path.join(disabled_dir, d), os.path.join(dataset_dir, d))
                
        if not os.path.exists(dataset_dir):
            messagebox.showerror("Error", "Dataset folder not found!")
            return
            
        classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        if not classes:
            messagebox.showerror("Error", "No class folders found in dataset!")
            return
            
        # Create Popup
        popup = tk.Toplevel(self)
        popup.title("Select Classes for Training")
        popup.geometry("500x680")
        popup.transient(self)
        popup.grab_set()
        
        ttk.Label(popup, text="Select classes to include in this training run:", font=("Roboto", 12, "bold")).pack(pady=10)
        
        frame = tk.Frame(popup)
        frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Read counts and create checkboxes
        self.class_vars = {}
        for cls in sorted(classes):
            cls_path = os.path.join(dataset_dir, cls)
            count = len([f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            var = tk.BooleanVar(value=(count > 0)) # Default check if has images
            self.class_vars[cls] = var
            chk = ttk.Checkbutton(scrollable_frame, text=f"{cls} ({count} images)", variable=var)
            chk.pack(anchor="w", pady=2)
            
        def on_start():
            selected = [cls for cls, var in self.class_vars.items() if var.get()]
            unselected = [cls for cls, var in self.class_vars.items() if not var.get()]
            
            if len(selected) < 2:
                messagebox.showerror("Error", "You must select at least 2 classes to train a model!")
                return
                
            popup.destroy()
            self.is_training = True
            self.tcm_train_btn.config(state=tk.DISABLED)
            
            def cnn_thread():
                try:
                    # Move unselected classes to disabled_dir
                    os.makedirs(disabled_dir, exist_ok=True)
                    for cls in unselected:
                        shutil.move(os.path.join(dataset_dir, cls), os.path.join(disabled_dir, cls))
                        
                    import subprocess
                    import shlex
                    cmd = ["gnome-terminal", "--wait", "--", "bash", "-c",
                           f"cd {shlex.quote(str(app_paths.BASE_DIR))} && {shlex.quote(sys.executable)} {shlex.quote(str(train_script))}; read -p 'Press Enter to exit...'"]
                    process = subprocess.Popen(cmd)
                    process.wait()
                    
                except Exception:
                    log.exception("CNN training could not be started")
                finally:
                    # Restore unselected classes
                    if os.path.exists(disabled_dir):
                        for cls in unselected:
                            src = os.path.join(disabled_dir, cls)
                            if os.path.exists(src):
                                shutil.move(src, os.path.join(dataset_dir, cls))
                                
                    self.is_training = False
                    self.after(0, lambda: self.tcm_train_btn.config(state=tk.NORMAL))
                    
            threading.Thread(target=cnn_thread, daemon=True).start()
            
        ttk.Button(popup, text="CONFIRM & TRAIN", style="Success.TButton", command=on_start).pack(pady=20, ipadx=20)

    def _start_yolo_train(self, arch=None):
        if getattr(self, 'is_training', False):
            return
        if arch is None:        # buttons that do not pass an architecture use the selector of the Teaching CNN page
            arch = self.tcm_arch_var.get() if hasattr(self, 'tcm_arch_var') else ""
            if "YOLO" not in arch:
                arch = "YOLO11-Classifier"
        self.is_training = True
        self.tcm_train_btn.config(state=tk.DISABLED)
        
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
        epochs={int(self.tcm_epochs.get())},
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
                self.after(0, lambda: self.tcm_train_btn.config(state=tk.NORMAL))
                
        threading.Thread(target=train_thread, daemon=True).start()


    def _verify_mismatches(self):
        self.tcm_verify_btn.config(state=tk.DISABLED)
        self.tcm_log.insert(tk.END, "\nStarting Dataset Verification...\n")
        
        def verify_thread():
            try:
                from ultralytics import YOLO
                import glob
                
                model_path = self.tcm_base_model.get()
                if not os.path.exists(model_path):
                    self.after(0, lambda: self.tcm_log.insert(tk.END, f"Error: {model_path} not found.\n"))
                    return
                    
                model = YOLO(model_path)
                images = glob.glob(os.path.join(DATASET_DIR, "*.jpg"))
                
                mismatches = []
                self.after(0, lambda: self.tcm_log.insert(tk.END, f"Checking {len(images)} images...\n"))
                
                for img_path in images:
                    txt_path = img_path.replace(".jpg", ".txt")
                    if not os.path.exists(txt_path): continue
                    
                    manual_labels = []
                    with open(txt_path, 'r') as f:
                        for line in f:
                            if not line.strip(): continue
                            manual_labels.append(int(line.split()[0]))
                            
                    if not manual_labels: continue
                    
                    results = model(img_path, verbose=False)
                    if len(results) > 0:
                        pred_classes = results[0].boxes.cls.cpu().numpy().astype(int).tolist()
                        
                        manual_set = set(manual_labels)
                        pred_set = set(pred_classes)
                        
                        if manual_set != pred_set:
                            mismatches.append(img_path)
                            
                self.after(0, lambda: self._show_mismatches(mismatches))
                
            except Exception as e:
                self.after(0, lambda: self.tcm_log.insert(tk.END, f"Error: {e}\n"))
            finally:
                self.after(0, lambda: self.tcm_verify_btn.config(state=tk.NORMAL))
                
        threading.Thread(target=verify_thread, daemon=True).start()

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
        self.tcm_epochs = tk.IntVar(value=100)
        tk.Entry(row2, textvariable=self.tcm_epochs, width=5).pack(side=tk.LEFT, padx=(0,15))
        
        ttk.Label(row2, text="Base Model:").pack(side=tk.LEFT, padx=(0,5))
        self.tcm_base_model = tk.StringVar(value="yolo11n.pt")
        tk.Entry(row2, textvariable=self.tcm_base_model, width=15).pack(side=tk.LEFT, padx=(0,15))
        
        row3 = tk.Frame(train_frm, bg=self.colors["bg"])
        row3.pack(fill=tk.X, pady=5)
        
        ttk.Button(row3, text="Cross Check Labels", style="Primary.TButton", command=self._launch_tcm_studio).pack(side=tk.LEFT, padx=(0,10))
        
        self.tcm_train_btn = ttk.Button(row3, text="START TRAINING", style="Success.TButton", command=self._start_yolo_train)
        self.tcm_train_btn.pack(side=tk.LEFT)
        
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
        if page_name in ["Pneumatic", "Teaching CNN", "Colour Size"]:
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
        """Vision ROI page: ROIs, decision line (either orientation), USB camera / test source, colour gain."""
        surf, light, text, muted = self.colors["surface"], self.colors["surface_light"], self.colors["text"], self.colors["text_muted"]
        page = tk.Frame(self.page_container, bg=surf)
        self.pages["Vision ROI"] = page
        canvas = tk.Canvas(page, bg=surf, highlightthickness=0)
        scroll = ttk.Scrollbar(page, orient="vertical", command=canvas.yview)
        p_vis = tk.Frame(canvas, bg=surf, padx=15, pady=15)
        p_vis.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        window_id = canvas.create_window((0, 0), window=p_vis, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(window_id, width=e.width))
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        def entry(parent, var, width=8):
            return tk.Entry(parent, textvariable=var, width=width, bg=light, fg=text, insertbackground=text,
                            disabledbackground=self.colors["bg"], bd=0, font=("Roboto", 11))

        def check(parent, label, var, command=None):
            return tk.Checkbutton(parent, text=label, variable=var, command=command, bg=surf, fg=text, selectcolor=self.colors["bg"],
                                  activebackground=surf, activeforeground=text, font=("Roboto", 10))

        # ── ROIs ──
        ttk.Label(p_vis, text="VISION & ROI SETUP", style="Title.TLabel").pack(anchor="w", pady=(0,10))
        roi_grid = tk.Frame(p_vis, bg=surf)
        roi_grid.pack(fill=tk.X, pady=5)
        roi_grid.columnconfigure(0, weight=1); roi_grid.columnconfigure(1, weight=1)
        ttk.Button(roi_grid, text="Draw ROI 1", command=lambda: self._set_phase("ROI1_START")).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ttk.Button(roi_grid, text="Reset ROI 1", command=self.shared_state.reset_roi1).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(roi_grid, text="Draw ROI 2", command=lambda: self._set_phase("ROI2_START")).grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        ttk.Button(roi_grid, text="Reset ROI 2", command=self.shared_state.reset_roi2).grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        # ── Decision line ──
        ttk.Label(p_vis, text="DECISION LINE", style="Header.TLabel").pack(anchor="w", pady=(15,5))
        line_names = {fg.VERTICAL: "Vertical  |", fg.HORIZONTAL: "Horizontal  —"}
        dir_frm = tk.Frame(p_vis, bg=surf)
        dir_frm.pack(fill=tk.X, pady=5)
        ttk.Label(dir_frm, text="Line:", style="Surface.TLabel").pack(side=tk.LEFT)
        self.line_orient_var = tk.StringVar(value=line_names[fg.line_orientation(self.shared_state.flow_direction)])
        orient_combo = ttk.Combobox(dir_frm, textvariable=self.line_orient_var, values=list(line_names.values()), state="readonly", width=13)
        orient_combo.pack(side=tk.LEFT, padx=(5, 15))
        ttk.Label(dir_frm, text="Cashews move:", style="Surface.TLabel").pack(side=tk.LEFT)
        self.flow_var = tk.StringVar(value=fg.FLOW_LABELS[self.shared_state.flow_direction])
        flow_combo = ttk.Combobox(dir_frm, textvariable=self.flow_var, state="readonly", width=14)
        flow_combo.pack(side=tk.LEFT, padx=5)

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

        line_grid = tk.Frame(p_vis, bg=surf)
        line_grid.pack(fill=tk.X, pady=5)
        line_grid.columnconfigure(0, weight=1); line_grid.columnconfigure(1, weight=1)
        ttk.Button(line_grid, text="Set Decision Line", style="Primary.TButton", command=lambda: self._set_phase("LINE_SET_MODE")).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ttk.Button(line_grid, text="Reset Line", command=self.shared_state.reset_line).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.link_lines_var = tk.BooleanVar(value=self.shared_state.link_lines)
        tk.Checkbutton(p_vis, text="One line for both ROIs (untick to set a line per ROI: click or drag inside that ROI)",
                       variable=self.link_lines_var, command=lambda: self.shared_state.set_link_lines(self.link_lines_var.get()),
                       bg=surf, fg=text, selectcolor=light, activebackground=surf, activeforeground=text, font=("Roboto", 9),
                       anchor="w", wraplength=430, justify="left").pack(fill=tk.X)

        man_frm = tk.Frame(p_vis, bg=surf)
        man_frm.pack(fill=tk.X, pady=5)
        self.manual_line_var = tk.StringVar()
        self.btn_line_minus = ttk.Button(man_frm, text="", command=lambda: self._shift_line(-5))
        self.btn_line_minus.pack(side=tk.LEFT, padx=5)
        entry(man_frm, self.manual_line_var, 6).pack(side=tk.LEFT, padx=10, fill=tk.Y)
        self.btn_line_plus = ttk.Button(man_frm, text="", command=lambda: self._shift_line(5))
        self.btn_line_plus.pack(side=tk.LEFT, padx=5)
        self.lbl_line_pos = ttk.Label(p_vis, text="", font=("Roboto", 10, "bold"), foreground=self.colors["warning"], style="Surface.TLabel")
        self.lbl_line_pos.pack(pady=5)
        self.manual_line_var.trace_add("write", self._on_manual_line_var_write)
        self.shared_state.line_pixel_callback = self._on_line_change_callback
        show_current_direction()
        self.shared_state._notify_line()

        # ── Camera ──
        ttk.Label(p_vis, text="CAMERA", style="Header.TLabel").pack(anchor="w", pady=(20,5))
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
        src_frm = tk.Frame(p_vis, bg=surf)
        src_frm.pack(fill=tk.X, pady=2)
        for label, value in (("Camera (Aravis)", "aravis"), ("Test source (video / images)", "file")):
            tk.Radiobutton(src_frm, text=label, variable=self.cam_vars["source"], value=value, bg=surf, fg=text, selectcolor=self.colors["bg"],
                           activebackground=surf, activeforeground=text, font=("Roboto", 10, "bold")).pack(side=tk.LEFT, padx=(0, 15))
        self.lbl_cam_model = tk.Label(p_vis, text="", bg=surf, fg=self.colors["info"], font=("Roboto", 10, "bold"), justify="left", anchor="w", wraplength=430)
        self.lbl_cam_model.pack(fill=tk.X, pady=(2, 4))

        cam_set = tk.Frame(p_vis, bg=surf)
        cam_set.pack(fill=tk.X, pady=2)
        ttk.Label(cam_set, text="Exposure (µs)", style="Surface.TLabel").grid(row=0, column=0, sticky="w", pady=3)
        entry(cam_set, self.cam_vars["exposure_us"], 9).grid(row=0, column=1, padx=10, pady=3, sticky="w")
        self.lbl_exposure_range = ttk.Label(cam_set, text="", style="Surface.TLabel", foreground=muted)
        self.lbl_exposure_range.grid(row=0, column=2, sticky="w")
        ttk.Label(cam_set, text="Camera gain", style="Surface.TLabel").grid(row=1, column=0, sticky="w", pady=3)
        entry(cam_set, self.cam_vars["gain"], 9).grid(row=1, column=1, padx=10, pady=3, sticky="w")
        self.lbl_gain_range = ttk.Label(cam_set, text="", style="Surface.TLabel", foreground=muted)
        self.lbl_gain_range.grid(row=1, column=2, sticky="w")
        fps_entry = entry(cam_set, self.cam_vars["fps_limit"], 9)

        def on_fps_limit():
            fps_entry.config(state=tk.NORMAL if self.cam_vars["fps_limit_enabled"].get() else tk.DISABLED)

        check(cam_set, "Limit FPS to", self.cam_vars["fps_limit_enabled"], on_fps_limit).grid(row=2, column=0, sticky="w", pady=3)
        fps_entry.grid(row=2, column=1, padx=10, pady=3, sticky="w")
        ttk.Label(cam_set, text="off = free running", style="Surface.TLabel", foreground=muted).grid(row=2, column=2, sticky="w")
        on_fps_limit()
        tk.Label(p_vis, text="An empty exposure or gain field leaves the camera's own value unchanged.",
                 bg=surf, fg=muted, font=("Roboto", 9), justify="left", anchor="w", wraplength=430).pack(fill=tk.X)

        test_frm = tk.Frame(p_vis, bg=surf)
        test_frm.pack(fill=tk.X, pady=(8, 2))
        ttk.Label(test_frm, text="Test source:", style="Surface.TLabel").pack(side=tk.LEFT)
        entry(test_frm, self.cam_vars["test_path"], 24).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        def browse_test_file():
            path = filedialog.askopenfilename(title="Test video or image", initialdir=str(app_paths.BASE_DIR),
                                              filetypes=[("Video / image", "*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")])
            if path: self.cam_vars["test_path"].set(path)

        def browse_test_folder():
            start = app_paths.RECORDINGS_DIR if app_paths.RECORDINGS_DIR.exists() else app_paths.BASE_DIR
            path = filedialog.askdirectory(title="Folder of test images (for example a recording)", initialdir=str(start))
            if path: self.cam_vars["test_path"].set(path)

        ttk.Button(test_frm, text="File", width=5, command=browse_test_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(test_frm, text="Folder", width=7, command=browse_test_folder).pack(side=tk.LEFT, padx=2)
        test_opt = tk.Frame(p_vis, bg=surf)
        test_opt.pack(fill=tk.X)
        check(test_opt, "Repeat", self.cam_vars["test_loop"]).pack(side=tk.LEFT)
        ttk.Label(test_opt, text="   FPS when the files have no timestamps:", style="Surface.TLabel").pack(side=tk.LEFT)
        entry(test_opt, self.cam_vars["test_fps"], 5).pack(side=tk.LEFT, padx=5)

        ttk.Button(p_vis, text="APPLY CAMERA SETTINGS", style="Primary.TButton", command=self._apply_camera_settings).pack(fill=tk.X, pady=(8, 2))
        self.lbl_cam_status = tk.Label(p_vis, text="", bg=surf, fg=muted, font=("Roboto", 10), justify="left", anchor="w", wraplength=430)
        self.lbl_cam_status.pack(fill=tk.X, pady=(6, 0))

        # ── White calibration (flat-field correction) ──
        ttk.Label(p_vis, text="WHITE CALIBRATION (FFC)", style="Header.TLabel").pack(anchor="w", pady=(20,5))
        self.ffc_enabled_var = tk.BooleanVar(value=self.flat_field.enabled)

        def on_ffc_toggle():
            if self.ffc_enabled_var.get() and not self.flat_field.available:
                messagebox.showinfo("White calibration", "No calibration yet. Put a plain white sheet under the camera and press TEACH WHITE.")
                self.ffc_enabled_var.set(False)
            self.flat_field.enabled = self.ffc_enabled_var.get()

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

        check(p_vis, "Use white calibration", self.ffc_enabled_var, on_ffc_toggle).pack(anchor="w")
        ttk.Button(p_vis, text="TEACH WHITE (white sheet under the camera)", command=teach_white).pack(fill=tk.X, pady=(4, 2))
        self.lbl_ffc_status = tk.Label(p_vis, text="", bg=surf, fg=muted, font=("Roboto", 9), justify="left", anchor="w", wraplength=430)
        self.lbl_ffc_status.pack(fill=tk.X)

        # ── Recorder: real camera frames for replay tests ──
        ttk.Label(p_vis, text="RECORD TEST FRAMES", style="Header.TLabel").pack(anchor="w", pady=(20,5))
        rec_frm = tk.Frame(p_vis, bg=surf)
        rec_frm.pack(fill=tk.X)
        self.rec_seconds_var, self.rec_fps_var = tk.StringVar(value="10"), tk.StringVar(value="25")
        ttk.Label(rec_frm, text="Seconds", style="Surface.TLabel").pack(side=tk.LEFT)
        entry(rec_frm, self.rec_seconds_var, 4).pack(side=tk.LEFT, padx=(5, 12))
        ttk.Label(rec_frm, text="FPS", style="Surface.TLabel").pack(side=tk.LEFT)
        entry(rec_frm, self.rec_fps_var, 4).pack(side=tk.LEFT, padx=(5, 12))
        ttk.Button(rec_frm, text="RECORD", command=self._start_recording).pack(side=tk.LEFT, padx=2)
        ttk.Button(rec_frm, text="STOP", command=lambda: self.camera_thread.recorder.stop()).pack(side=tk.LEFT, padx=2)
        self.lbl_rec_status = tk.Label(p_vis, text="Saves camera frames (before colour gain) into recordings/. Replay them as a test source.",
                                       bg=surf, fg=muted, font=("Roboto", 9), justify="left", anchor="w", wraplength=430)
        self.lbl_rec_status.pack(fill=tk.X, pady=(4, 0))
        tk.Label(p_vis, text="ROIs and the decision line are in pixels of the full camera image.",
                 bg=surf, fg=muted, font=("Roboto", 9), justify="left", anchor="w", wraplength=430).pack(fill=tk.X, pady=(8, 0))

        # ── Colour gain (software, applied to every frame before the AI and before any image is saved) ──
        ttk.Label(p_vis, text="COLOUR GAIN (SOFTWARE)", style="Header.TLabel").pack(anchor="w", pady=(20,5))
        self.vars_cam = {
            "gain": tk.StringVar(value=str(self.shared_state.cam_gain)),
            "r": tk.StringVar(value=str(self.shared_state.cam_r)),
            "g": tk.StringVar(value=str(self.shared_state.cam_g)),
            "b": tk.StringVar(value=str(self.shared_state.cam_b)),
            "a_gain": tk.StringVar(value=str(self.shared_state.analog_gain)),
            "d_gain": tk.StringVar(value=str(self.shared_state.digital_gain)),
        }
        cam_grid = tk.Frame(p_vis, bg=surf)
        cam_grid.pack(fill=tk.X)
        for idx, (txt, key) in enumerate([("Global Gain", "gain"), ("R Gain", "r"), ("G Gain", "g"), ("B Gain", "b")]):
            ttk.Label(cam_grid, text=txt, style="Surface.TLabel").grid(row=idx, column=0, sticky="w", pady=3)
            entry(cam_grid, self.vars_cam[key], 10).grid(row=idx, column=1, padx=20, pady=3)
            self.vars_cam[key].trace_add("write", self._on_cam_setting_change)
            
        def save_cam_settings():
            self._on_cam_setting_change()
            messagebox.showinfo("Saved", "Colour gain saved.")
            
        ttk.Button(p_vis, text="SAVE COLOUR GAIN", style="Primary.TButton", command=save_cam_settings).pack(pady=(10, 0), fill=tk.X)
        ttk.Button(p_vis, text="Test Image Capture", command=self.shared_state.trigger_capture).pack(pady=15, fill=tk.X)

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
        """Question before FIRE 1-40 / SEQUENCE 1-40. Asked every time; it also says whether DRY-RUN is on."""
        if self.relay_output.dry_run:
            mode = "DRY-RUN is ON: nothing will be sent to the PLC."
        else:
            mode = "DRY-RUN is OFF: the REAL valves of relays 1-40 will fire."
        return messagebox.askyesno(title, f"Make sure nobody is at the nozzles.\n\n{mode}\n\nGo on?")

    def _show_manual_result(self, text, sent):
        if hasattr(self, 'lbl_test_relay_result'):
            self.lbl_test_relay_result.config(text=text, fg=self.colors["success"] if sent else self.colors["warning"])

    def _open_plc_diagnostics(self):
        """The PLC diagnostics tool is a separate program that writes straight to the PLC: it does not know
        DRY-RUN or MASTER EJECTOR and can also switch conveyor and feeder. So it is only opened deliberately."""
        if not hasattr(self.controller, 'open_diagnostics_gui'):
            return
        if not self._manual_test_allowed():
            return
        if self.relay_output.dry_run:
            messagebox.showwarning("PLC diagnostics", "DRY-RUN is ON.\n\nThe PLC diagnostics tool writes directly to the PLC and would ignore DRY-RUN, "
                                   "so it is blocked.\n\nSwitch DRY-RUN off first if you really want to drive the PLC by hand.")
            self.relay_output.log_manual("PLC DIAGNOSTICS TOOL blocked (DRY-RUN)", "", "", "DRY_RUN")
            return
        if not messagebox.askyesno("PLC diagnostics", "Make sure nobody is at the nozzles.\n\nThis tool writes directly to the PLC: its buttons fire REAL "
                                   "valves at once, ignore DRY-RUN and MASTER EJECTOR, and can also switch CONVEYOR (M140) and FEEDER (M141). "
                                   "What is done there is NOT in the manual fire log.\n\nOpen it?"):
            return
        log.warning("PLC diagnostics tool opened by the operator (direct PLC access, outside DRY-RUN)")
        self.relay_output.log_manual("PLC DIAGNOSTICS TOOL opened", "", "", "DIRECT")
        self.controller.open_diagnostics_gui()

    def _flash_relay_led(self, relay, pulse):
        if hasattr(self, 'pn_leds') and relay in self.pn_leds:
            self.pn_leds[relay].config(fg=self.colors["success"])
            self.after(max(pulse, 200), lambda r=relay: self.pn_leds[r].config(fg=self.colors["surface_light"]))

    def _test_fire_relay(self, relay, action="SINGLE RELAY TEST"):
        """Fire one relay once, by hand. Returns the result code."""
        if not self._manual_test_allowed():
            return None
        pulse = self._test_pulse_ms()
        result = self.relay_output.test_fire(relay, pulse, action=action)
        self._flash_relay_led(relay, pulse)
        self._show_manual_result(self._result_text(relay, result), result == "SENT")
        return result

    def _test_fire_selected_relay(self):
        try:
            relay = int(self.test_relay_var.get())
        except (ValueError, tk.TclError):
            return
        self._test_fire_relay(relay)

    def _pulse_and_flash(self, act_idx):
        if hasattr(self, 'test_relay_var'):
            self.test_relay_var.set(str(act_idx))
        self._test_fire_relay(act_idx, action="ROW TEST")

    def _fire_all(self):
        if not self._manual_test_allowed():
            return
        if not self._confirm_fire_many("FIRE 1-40"):
            return
        if not self._manual_test_allowed():      # the machine may have been started while the question was open
            return
        pulse = self._test_pulse_ms()
        results = self.relay_output.test_all(pulse)     # 40 x test_fire(): the same path as the single relay test
        for relay in results:
            self._flash_relay_led(relay, pulse)
        kinds = sorted(set(results.values()))
        summary = "; ".join(f"{sum(1 for r in results.values() if r == kind)} x {RESULT_TEXT.get(kind, kind)}" for kind in kinds)
        self._show_manual_result(f"{time.strftime('%H:%M:%S')}  FIRE 1-40 (relays 1-40, M100-M139): {summary}", kinds == ["SENT"])
            
    def _sequence_all(self):
        """Relays 1-40 one after another. Runs on the UI timer (no extra thread); stops if the machine is started."""
        if not self._manual_test_allowed() or getattr(self, '_sequence_running', False):
            return
        if not self._confirm_fire_many("SEQUENCE 1-40"):
            return
        if not self._manual_test_allowed():
            return
        pulse = self._test_pulse_ms()
        interval = max(120, pulse + 40)
        self._sequence_running = True

        def step(relay):
            if relay > LAST_EJECTOR or self.stop_event.is_set() or self.shared_state.phase == "RUNNING":
                self._sequence_running = False
                return
            result = self.relay_output.test_fire(relay, pulse, action="SEQUENCE 1-40")
            self.test_relay_var.set(str(relay))
            self._flash_relay_led(relay, pulse)
            self._show_manual_result("SEQUENCE 1-40  " + self._result_text(relay, result), result == "SENT")
            self.after(interval, lambda: step(relay + 1))

        step(FIRST_EJECTOR)

    def _update_common_pulse(self, *args):
        try:
            pulse = int(self.common_pulse_var.get())
            self.shared_state.set_common_pulse(pulse)
        except (ValueError, tk.TclError): pass

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
            self.shared_state.save_to_config(CFG_PATH)
            messagebox.showinfo("Saved", "Pneumatic Delays, Distances, and Configuration Saved Successfully!")
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

    def _update_all_map_labels(self):
        for i in range(1, NUM_ACTUATORS + 1): 
            cls_lst = self.shared_state.get_mapping_for_actuator(i)
            text = ", ".join(cls_lst) if cls_lst else "(Unmapped)"
            fg_color = self.colors["success"] if cls_lst else self.colors["text_muted"]
            if i in self.map_lbls:
                self.map_lbls[i].config(text=text, foreground=fg_color)
            if hasattr(self, 'pn_map_lbls') and i in self.pn_map_lbls:
                self.pn_map_lbls[i].config(text=text, foreground=fg_color)
            if hasattr(self, 'pn_leds') and i in self.pn_leds:
                self.pn_leds[i].config(fg=self.colors["surface_light"]) # LED is always OFF until fired

    def _popup_mapper(self, act_id):
        win = tk.Toplevel(self)
        win.title(f"A{act_id} Assignment")
        win.configure(bg=self.colors["bg"])
        win.transient(self); win.grab_set() 
        
        track_num = 1 if act_id <= 20 else 2
        
        used_map = {}
        start_a, end_a = (1, 20) if track_num == 1 else (21, 40)
        for a in range(start_a, end_a + 1):
            if a != act_id:
                for c in self.shared_state.get_mapping_for_actuator(a):
                    used_map[c] = f"A{a:02d}"
        
        base_classes = [c.upper() for c in self.all_class_names if not c.startswith("ww_") and not c.startswith("sw_")]
        size_classes = [
            "SW", "WW",
            "SW180", "SW210", "SW240", "SW320", "SW450",
            "W180", "W210", "W240", "W320", "W450",
            "BUDDS"
        ]
        
        # Deduplicate while preserving order
        options = []
        for opt in base_classes + size_classes:
            if opt not in options and opt != "CAPTURE_ONLY_MODE":
                options.append(opt)
                
        vars_map = {}
        opts_frame = tk.Frame(win, bg=self.colors["bg"])
        opts_frame.pack(padx=10, pady=10)
        
        for idx, name in enumerate(options):
            is_current = name in self.shared_state.get_mapping_for_actuator(act_id)
            is_in_other = name in used_map
            
            var = tk.BooleanVar(value=is_current)
            vars_map[name] = var
            
            if is_current:
                fg_col = "#FFFF00" # Bright Yellow
                txt = name
                fnt = ("Roboto", 10, "bold")
            elif is_in_other:
                fg_col = self.colors["warning"] # Orange warning text showing where it's currently mapped
                txt = f"{name} ({used_map[name]})"
                fnt = ("Roboto", 10)
            else:
                fg_col = self.colors["success"]
                txt = name
                fnt = ("Roboto", 10)
                
            chk = tk.Checkbutton(opts_frame, text=txt, variable=var, bg=self.colors["bg"], 
                                 fg=fg_col, activeforeground=fg_col, disabledforeground=fg_col, 
                                 selectcolor=self.colors["surface"], font=fnt)
                
            row = idx // 2
            col = idx % 2
            chk.grid(row=row, column=col, sticky="w", padx=10, pady=2)
            
        def on_save():
            sel = {name for name, var in vars_map.items() if var.get()}
            start_a, end_a = (1, 20) if track_num == 1 else (21, 40)
            other_act_id = act_id + 20 if act_id <= 20 else act_id - 20
            start_a2, end_a2 = (21, 40) if track_num == 1 else (1, 20)

            # Auto-transfer selected classes from old actuators to current actuator
            with self.shared_state.lock:
                for a in range(start_a, end_a + 1):
                    if a != act_id and a in self.shared_state.mapping_by_act:
                        self.shared_state.mapping_by_act[a] = [c for c in self.shared_state.mapping_by_act[a] if c not in sel]
                        
                for a in range(start_a2, end_a2 + 1):
                    if a != other_act_id and a in self.shared_state.mapping_by_act:
                        self.shared_state.mapping_by_act[a] = [c for c in self.shared_state.mapping_by_act[a] if c not in sel]

            self.shared_state.set_mapping_for_actuator(act_id, list(sel))
            self.shared_state.set_mapping_for_actuator(other_act_id, list(sel))
            
            self._update_all_map_labels()
            self.shared_state.save_to_config(CFG_PATH)
            win.destroy()

        # Select All / Deselect All Action Row
        btn_row = tk.Frame(win, bg=self.colors["bg"])
        btn_row.pack(fill=tk.X, padx=20, pady=(5, 10))
        btn_row.columnconfigure(0, weight=1)
        btn_row.columnconfigure(1, weight=1)

        def select_all():
            for v in vars_map.values():
                v.set(True)

        def deselect_all():
            for v in vars_map.values():
                v.set(False)

        tk.Button(
            btn_row, text="SELECT ALL", font=("Roboto", 9, "bold"),
            bg=self.colors["surface_light"], fg=self.colors["info"],
            activebackground=self.colors["info"], activeforeground="white",
            bd=0, pady=6, cursor="hand2", command=select_all
        ).grid(row=0, column=0, sticky="ew", padx=(0, 5))

        tk.Button(
            btn_row, text="DESELECT ALL", font=("Roboto", 9, "bold"),
            bg=self.colors["surface_light"], fg=self.colors["text_muted"],
            activebackground=self.colors["danger"], activeforeground="white",
            bd=0, pady=6, cursor="hand2", command=deselect_all
        ).grid(row=0, column=1, sticky="ew", padx=(5, 0))
            
        ttk.Button(win, text="Confirm", style="Primary.TButton", command=on_save).pack(pady=(0, 15), padx=20, fill=tk.X)

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
        if getattr(self, 'is_training', False):
            self.after(500, self._update_loop)
            return
            
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
        if self._grading_tick % 10 == 0 and hasattr(self, "lbl_view_live"):
            self._refresh_grading_live()

        if hasattr(self, 'lbl_prod_count'):
            det_cnt = self.shared_state.get_detected_count()
            if getattr(self, '_last_det_cnt', -1) != det_cnt:
                self.lbl_prod_count.config(text=f"[ Detected: {det_cnt} ]")
                self._last_det_cnt = det_cnt

        self.after(30, self._update_loop)

    def _build_teaching_console_page(self):
        p_teach = tk.Frame(self.page_container, bg=self.colors["bg"])
        self.pages["Teaching YOLO"] = p_teach
        
        canvas = tk.Canvas(p_teach, bg=self.colors["bg"], highlightthickness=0)
        scroll = ttk.Scrollbar(p_teach, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=self.colors["bg"], padx=15, pady=15)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        
        def _on_canvas_resize(e):
            canvas.itemconfig(window_id, width=e.width)
        canvas.bind("<Configure>", _on_canvas_resize)
        
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        
        # --- CNN (DEFECTS) ---
        ttk.Label(inner, text="CNN (DEFECTS) - MANUAL ANNOTATION", style="Title.TLabel").pack(anchor="w", pady=(0, 10))
        cnn_frm = tk.Frame(inner, bg=self.colors["surface"], padx=10, pady=10)
        cnn_frm.pack(fill=tk.X, pady=(0, 20))
        
        ctrl_row = tk.Frame(cnn_frm, bg=self.colors["surface"])
        ctrl_row.pack(fill=tk.X, pady=5)
        ttk.Button(ctrl_row, text="Open Folder", command=self._ann_open_folder).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl_row, text="◄ Prev", command=self._ann_prev).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl_row, text="Next ►", command=self._ann_next).pack(side=tk.LEFT, padx=2)
        self.ann_lbl_file = tk.Label(ctrl_row, text="No image loaded", bg=self.colors["surface"], fg=self.colors["text"])
        self.ann_lbl_file.pack(side=tk.LEFT, padx=15)
        
        fast_row = tk.Frame(cnn_frm, bg=self.colors["surface"], pady=10)
        fast_row.pack(fill=tk.X)
        
        hdr_row = tk.Frame(fast_row, bg=self.colors["surface"])
        hdr_row.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(hdr_row, text="Select Class to Mark:", style="Header.TLabel").pack(side=tk.LEFT)
        
        btn_row = tk.Frame(fast_row, bg=self.colors["surface"])
        btn_row.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(btn_row, text="Add", style="Primary.TButton", command=self._add_label_class).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Rename", command=self._rename_label_class).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Delete", style="Danger.TButton", command=self._delete_label_class).pack(side=tk.LEFT, padx=2)
        
        self.ann_fast_class_var = tk.StringVar()
        self._class_check_vars = {}  # {class_name: tk.BooleanVar}
        self.class_btn_grid = tk.Frame(fast_row, bg=self.colors["bg"])
        self.class_btn_grid.pack(fill=tk.X)
        self._refresh_class_grid()
            
        tree_frm = tk.Frame(cnn_frm, bg=self.colors["surface"])
        tree_frm.pack(fill=tk.X, pady=10)
        ttk.Label(tree_frm, text="Current Annotations:", style="Header.TLabel").pack(anchor="w", pady=(0, 5))
        
        # We reuse the existing tree variables
        self.ann_tree = ttk.Treeview(tree_frm, columns=("ID", "Class Name"), show="headings", height=4)
        self.ann_tree.heading("ID", text="ID"); self.ann_tree.column("ID", width=50, anchor="center")
        self.ann_tree.heading("Class Name", text="Class Name"); self.ann_tree.column("Class Name", width=200, anchor="w")
        self.ann_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(tree_frm, orient=tk.VERTICAL, command=self.ann_tree.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.ann_tree.configure(yscrollcommand=sb.set)
        
        edit_row = tk.Frame(cnn_frm, bg=self.colors["surface"], pady=5)
        edit_row.pack(fill=tk.X)
        ttk.Button(edit_row, text="Rename Selected", command=self._ann_rename).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(edit_row, text="Delete Selected", command=self._ann_delete, style="Danger.TButton").pack(side=tk.LEFT)
        ttk.Button(edit_row, text="START TRAINING", style="Success.TButton", command=self._start_yolo_train).pack(side=tk.RIGHT, padx=10)
        

    # ------------------------------------------------------------------ grading rules (step c1)
    def _build_grading_section(self, parent):
        """Voting limits, priority order, view limits, belt colour, tracker and live values."""
        surf, light, text, muted = self.colors["surface"], self.colors["surface_light"], self.colors["text"], self.colors["text_muted"]
        g = self.shared_state.grading
        self.grading_vars = {}

        def entry_row(frame, section, key, label, row, col=0):
            tk.Label(frame, text=label, bg=surf, fg=text, font=("Roboto", 9), anchor="w").grid(row=row, column=col, sticky="w", padx=(0, 6), pady=2)
            var = tk.StringVar(value=f"{g[section].get(key, ''):g}")
            tk.Entry(frame, textvariable=var, width=7, bg=light, fg=text, insertbackground=text, bd=0, font=("Roboto", 10)).grid(row=row, column=col + 1, sticky="w", pady=2)
            self.grading_vars[(section, key)] = var

        def check(label, var):
            tk.Checkbutton(parent, text=label, variable=var, bg=surf, fg=text, selectcolor=light, activebackground=surf,
                           activeforeground=text, font=("Roboto", 9), anchor="w", wraplength=400, justify="left").pack(fill=tk.X)

        ttk.Label(parent, text="GRADING RULES", style="Title.TLabel").pack(anchor="w", pady=(0, 4))
        tk.Label(parent, text="One decision per cashew at its Decision line. A class is \"seen\" with enough frames at or above its confidence.",
                 bg=surf, fg=muted, font=("Roboto", 9), wraplength=410, justify="left").pack(anchor="w")

        vote = tk.Frame(parent, bg=surf); vote.pack(fill=tk.X, pady=6)
        for row, (key, label) in enumerate((
                ("defect_conf", "Defect: confidence at least (0-1)"), ("defect_min_frames", "Defect: in at least N frames"),
                ("good_conf", "GOOD: confidence at least (0-1)"), ("good_min_frames", "GOOD: in at least N frames"),
                ("neck_min_frames", "GOOD: NECK view in at least M frames (0 = off)"),
                ("neck_weight", "TESTA vote from a NECK frame counts as"),
                ("rejection_conf", "REJECTION (YOLO): confidence at least"), ("rejection_min_frames", "REJECTION (YOLO): in at least N frames"))):
            entry_row(vote, "voting", key, label, row)
        self.single_defect_var = tk.BooleanVar(value=bool(g["voting"].get("single_defect_blocks_good", True)))
        check("One defect frame (below N) blocks GOOD -> Undefined", self.single_defect_var)

        ttk.Label(parent, text="PRIORITY (highest first)", style="Header.TLabel").pack(anchor="w", pady=(12, 4))
        prio = tk.Frame(parent, bg=surf); prio.pack(fill=tk.X)
        self.priority_list = tk.Listbox(prio, height=10, width=22, bg=light, fg=text, selectbackground=self.colors["info"], bd=0,
                                        highlightthickness=0, font=("Roboto", 10), exportselection=False)
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
        side = tk.Frame(prio, bg=surf); side.pack(side=tk.LEFT, padx=8, anchor="n")
        ttk.Button(side, text="UP", command=lambda: move(-1)).pack(fill=tk.X, pady=2)
        ttk.Button(side, text="DOWN", command=lambda: move(1)).pack(fill=tk.X, pady=2)
        tk.Label(side, text="Before this list:\nDOUBLE, REJECTION\n-> reject relay.\nSkin + other defect\n-> UPW.", bg=surf, fg=muted,
                 font=("Roboto", 8), justify="left").pack(anchor="w", pady=(6, 0))

        ttk.Label(parent, text="VIEW LIMITS (empty = not used)", style="Header.TLabel").pack(anchor="w", pady=(12, 2))
        tk.Label(parent, text="A frame is NECK, BACK or TOP when all filled limits of that view hold (checked in this order), otherwise OTHER. "
                              "The letters are the values drawn under each cashew.",
                 bg=surf, fg=muted, font=("Roboto", 9), wraplength=410, justify="left").pack(anchor="w")
        grid = tk.Frame(parent, bg=surf); grid.pack(fill=tk.X, pady=4)
        tk.Label(grid, text="", bg=surf).grid(row=0, column=0)
        for c, view in enumerate(view_features.VIEWS):
            tk.Label(grid, text=f"{view} min / max", bg=surf, fg=self.colors["info"], font=("Roboto", 8, "bold")).grid(row=0, column=1 + 2 * c, columnspan=2)
        self.view_vars = {}
        for r, (name, (letter, description)) in enumerate(view_features.FEATURES.items(), start=1):
            tk.Label(grid, text=f"{letter} {name}", bg=surf, fg=text, font=("Roboto", 9), anchor="w").grid(row=r, column=0, sticky="w", padx=(0, 4))
            for c, view in enumerate(view_features.VIEWS):
                limits = (g["views"]["rules"].get(view) or {}).get(name) or [None, None]
                for k in (0, 1):
                    var = tk.StringVar(value="" if limits[k] is None else f"{limits[k]:g}")
                    tk.Entry(grid, textvariable=var, width=5, bg=light, fg=text, insertbackground=text, bd=0, font=("Roboto", 9)).grid(row=r, column=1 + 2 * c + k, padx=1, pady=1)
                    self.view_vars[(view, name, k)] = var
        self.show_values_var = tk.BooleanVar(value=bool(g["views"].get("show_values", True)))
        check("Draw the values under each cashew", self.show_values_var)

        ttk.Label(parent, text="LIVE VALUES", style="Header.TLabel").pack(anchor="w", pady=(12, 2))
        self.lbl_view_live = tk.Label(parent, text="", bg=surf, fg=self.colors["warning"], font=("Roboto Mono", 9), justify="left", anchor="w", wraplength=410)
        self.lbl_view_live.pack(fill=tk.X)
        self.lbl_result_counts = tk.Label(parent, text="", bg=surf, fg=text, font=("Roboto", 9), justify="left", anchor="w", wraplength=410)
        self.lbl_result_counts.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(parent, text="RESET COUNTERS", command=self.shared_state.reset_result_counts).pack(anchor="w", pady=4)

        ttk.Label(parent, text="BELT COLOUR (cashew mask)", style="Header.TLabel").pack(anchor="w", pady=(12, 2))
        seg = tk.Frame(parent, bg=surf); seg.pack(fill=tk.X)
        for row, (key, label) in enumerate((("blue_h_min", "Belt hue from (0-179)"), ("blue_h_max", "Belt hue to"),
                                            ("blue_s_min", "Belt saturation at least"), ("dark_v_max", "Dark (gap) brightness up to"),
                                            ("min_area_px", "Smallest cashew area (px)"), ("edge_margin_px", "Cut-off: box within px of ROI edge"))):
            entry_row(seg, "segmentation", key, label, row)

        def teach():
            if messagebox.askyesno("Teach belt colour", "The belt must be EMPTY and lit as in production.\n\nTeach the belt colour from the picture now?"):
                with self.shared_state.lock:
                    self.shared_state.teach_belt_requested = True
                    self.shared_state.teach_belt_message = "Teaching..."
        ttk.Button(parent, text="TEACH BELT COLOUR (empty belt)", command=teach).pack(anchor="w", pady=4)
        self.lbl_teach_belt = tk.Label(parent, text="", bg=surf, fg=muted, font=("Roboto", 9), justify="left", anchor="w", wraplength=410)
        self.lbl_teach_belt.pack(fill=tk.X)

        ttk.Label(parent, text="TRACKER", style="Header.TLabel").pack(anchor="w", pady=(12, 2))
        trk = tk.Frame(parent, bg=surf); trk.pack(fill=tk.X)
        for row, (key, label) in enumerate((("max_step_px", "Largest step per frame, speed unknown (px)"), ("gate_extra_px", "Allowed error around prediction (px)"),
                                            ("speed_factor", "... plus this x predicted step"), ("back_tol_px", "Backward tolerance (px)"),
                                            ("max_lateral_px", "Sideways limit (px)"), ("hold_ms", "Keep a missed cashew's ID for (ms)"))):
            entry_row(trk, "tracker", key, label, row)

        self.old_filter_var = tk.BooleanVar(value=bool(self.shared_state.opencv_filter.get("enabled", False)))
        check("Old dark-spot filter for CNN SW / WHITE_DW (Colour Size page values)", self.old_filter_var)
        self.frame_log_var = tk.BooleanVar(value=bool(g["frame_log"].get("enabled", True)))
        check("Per-frame log (logs/frames_*.jsonl, kept 7 days)", self.frame_log_var)

        ttk.Button(parent, text="APPLY GRADING SETTINGS", style="Primary.TButton", command=self._apply_grading_settings).pack(fill=tk.X, pady=(10, 2))
        self.lbl_grading_status = tk.Label(parent, text="", bg=surf, fg=muted, font=("Roboto", 9), anchor="w", wraplength=410, justify="left")
        self.lbl_grading_status.pack(fill=tk.X, pady=(0, 20))

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
            messagebox.showerror("Grading settings", f"Not applied. Wrong value: {e}")
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
        note = "" if "NECK" in used or g["voting"]["neck_min_frames"] == 0 else "  NOTE: NECK has no limits, so no cashew can become GOOD while M is above 0."
        self.lbl_grading_status.config(text=f"Applied {datetime.now().strftime('%H:%M:%S')}. Views with limits: {', '.join(used) or 'none'}.{note}",
                                       fg=self.colors["warning"] if note else self.colors["success"])
        log.info("Grading settings applied: %s", g)

    def _refresh_grading_live(self):
        ss = self.shared_state
        with ss.lock:
            live, speed, teach_msg = dict(ss.view_live), dict(ss.lane_speed), ss.teach_belt_message
            counts = dict(ss.result_counts)
            seg = dict(ss.grading["segmentation"])
        px_per_mm = float(ss.size_config.get("px_per_mm", 10.0)) or 10.0
        self.lbl_view_live.config(text="\n".join(
            f"ROI {r}: {live[r] or '-'}\n       lane speed {speed[r] * 1000:.0f} px/s = {speed[r] * 1000 / px_per_mm:.0f} mm/s at {px_per_mm:g} px/mm" for r in (1, 2)))
        total = sum(n for k, n in counts.items() if not k.startswith("UNDEFINED: "))
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
        """Build the dedicated Colour & Size settings page (OpenCV Filter + Size Logic)."""
        p_cs = tk.Frame(self.page_container, bg=self.colors["bg"])
        self.pages["Colour Size"] = p_cs
        
        canvas = tk.Canvas(p_cs, bg=self.colors["bg"], highlightthickness=0)
        scroll = ttk.Scrollbar(p_cs, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=self.colors["bg"], padx=15, pady=15)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        
        def _on_canvas_resize(e):
            canvas.itemconfig(window_id, width=e.width)
        canvas.bind("<Configure>", _on_canvas_resize)
        
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        
        # --- OPENCV FILTER ---
        ttk.Label(inner, text="OPENCV FILTER (COLOR & DEFECT TOLERANCE)", style="Title.TLabel").pack(anchor="w", pady=(0, 10))
        cv_frm = tk.Frame(inner, bg=self.colors["surface"], padx=10, pady=10)
        cv_frm.pack(fill=tk.X, pady=(0, 20))
        
        def make_slider(parent, label, var_name, from_, to, default):
            row = tk.Frame(parent, bg=self.colors["surface"], pady=5)
            row.pack(fill=tk.X)
            ttk.Label(row, text=label, width=25).pack(side=tk.LEFT)
            var = tk.IntVar(value=self.shared_state.opencv_filter.get(var_name, default))
            setattr(self, f"var_{var_name}", var)
            ttk.Scale(row, from_=from_, to=to, variable=var, orient=tk.HORIZONTAL, length=150).pack(side=tk.LEFT, padx=10)
            tk.Entry(row, textvariable=var, width=5).pack(side=tk.LEFT)
            def _on_change(*args, v=var, n=var_name):
                self.shared_state.opencv_filter[n] = v.get()
                self.shared_state.save_to_config(CFG_PATH)
            var.trace_add("write", _on_change)
            
        make_slider(cv_frm, "SW Color Threshold:", "sw_color_thresh", 0, 255, 45)
        make_slider(cv_frm, "SW Defect Tolerance (px):", "sw_defect_size_px", 0, 100, 20)
        make_slider(cv_frm, "WW Color Threshold:", "ww_color_thresh", 0, 255, 85)
        make_slider(cv_frm, "WW Defect Tolerance (px):", "ww_defect_size_px", 0, 100, 16)
        make_slider(cv_frm, "Max Defect Area (px):", "max_defect_px", 0, 5000, 1200)
        
        # --- SIZE LOGIC ---
        ttk.Label(inner, text="SIZE LOGIC (AUTO TEACH)", style="Title.TLabel").pack(anchor="w", pady=(10, 10))
        sz_frm = tk.Frame(inner, bg=self.colors["surface"], padx=10, pady=10)
        sz_frm.pack(fill=tk.X)
        
        self.at_size_var = tk.StringVar(value="W180")
        def _do_teach(sz):
            self.at_size_var.set(sz)
            self._start_auto_teach()
            
        sz_btn_grid = tk.Frame(sz_frm, bg=self.colors["surface"])
        sz_btn_grid.pack(fill=tk.X)
        
        for i, sz in enumerate(["W180", "W210", "W240", "W320", "W450", "BUDDS"]):
            row_frm = tk.Frame(sz_btn_grid, bg=self.colors["surface"])
            row_frm.pack(fill=tk.X, pady=2)
            
            ttk.Label(row_frm, text=f"[{sz}]", font=("Roboto", 12, "bold"), width=8).pack(side=tk.LEFT)
            
            # Re-use the existing StringVars from Pneumatics tab so both UIs update together!
            l_var = self.sz_vars.get(f"{sz}_L", tk.StringVar(value=str(self.shared_state.size_config.get(f"{sz}_L", 15))))
            w_var = self.sz_vars.get(f"{sz}_W", tk.StringVar(value=str(self.shared_state.size_config.get(f"{sz}_W", 10))))
            
            ttk.Label(row_frm, text="L(mm):").pack(side=tk.LEFT)
            tk.Entry(row_frm, textvariable=l_var, width=5, bg=self.colors["bg"], fg="white", justify="center").pack(side=tk.LEFT, padx=(0,10))
            ttk.Label(row_frm, text="W(mm):").pack(side=tk.LEFT)
            tk.Entry(row_frm, textvariable=w_var, width=5, bg=self.colors["bg"], fg="white", justify="center").pack(side=tk.LEFT, padx=(0,10))
            
            def update_sz_cfg(*args, s=sz, lv=l_var, wv=w_var):
                try:
                    self.shared_state.size_config[f"{s}_L"] = float(lv.get())
                    self.shared_state.size_config[f"{s}_W"] = float(wv.get())
                    self.shared_state.save_to_config(CFG_PATH)
                except (ValueError, tk.TclError): pass
            
            l_var.trace_add("write", update_sz_cfg)
            w_var.trace_add("write", update_sz_cfg)
            
            btn = ttk.Button(row_frm, text="TEACH FROM CAMERA", command=lambda s=sz: _do_teach(s))
            btn.pack(side=tk.LEFT, padx=10)

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
                    bg=self.colors["bg"], fg=self.colors["text"],
                    selectcolor=self.colors.get("surface", "#333"),
                    activebackground=self.colors["bg"],
                    activeforeground=self.colors["text"],
                    font=("Segoe UI", 10), anchor="w", width=14,
                    relief=tk.RIDGE, bd=1, padx=4, pady=2
                )
                cb.grid(row=r, column=c, padx=4, pady=3, sticky="ew")
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

    def _on_close(self):
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
            
        print("[SYSTEM] Shutdown complete.")
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
