from datetime import datetime
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Instant Fire Grader - Final Master Release
(Stable Zero-Mask Tracking, Multi-Frame Priority Logic, Individual Conf, MM Sizing)
"""
import os

import sys
import shutil
import time
import json
import csv
import glob
import threading
import re
import queue
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from collections import deque

import cv2
cv2.setNumThreads(0)
import numpy as np
from ultralytics import YOLO

sys.path.append("/home/phoenix/Documents/RaniVision_Latest")
try:
    import tcm_magic_wand
except ImportError:
    tcm_magic_wand = None

# MindVision capture (used for IP config only)
try:
    import mvsdk
except Exception as e:
    mvsdk = None
    print("[WARN] mvsdk not found! IP Config might not work.")

# Aravis capture (used for streaming)
import gi
gi.require_version('Aravis', '0.8')
from gi.repository import Aravis
import platform

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
_tv_models = None
_tv_transforms = None
_F = None
try:
    import torch as _torch
    import torchvision.models as _tv_models
    import torchvision.transforms as _tv_transforms
    import torch.nn.functional as _F
    _CNN_AVAILABLE = True
    print("[CNN] PyTorch + torchvision loaded OK")
except ImportError as e:
    print(f"[CNN] PyTorch/torchvision not available – CNN disabled: {e}")


# ---------------- CONFIG ----------------
CSV_PATH = "cashew_classes.csv" 
CFG_PATH = str(Path(__file__).resolve().parent / "pneumatic_config.json")
MODEL_ROOT = Path("saved_models")

# CNN Color Classifier checkpoint
CNN_MODEL_PATH = Path("saved_models/cashew_color_cnn_new.pth")
CNN_OVERRIDE_YOLO_BELOW = 0.98   # YOLO conf < 98% → CNN can override
CNN_CONF_THRESHOLD      = 0.75   # CNN must be >= 75% to override

DATASET_DIR = "dataset" 
os.makedirs(DATASET_DIR, exist_ok=True) 

GOLDEN_DIR = "cnn_final/golden_crops"
os.makedirs(GOLDEN_DIR, exist_ok=True)

FULL_FRAME_DIR = "cnn_final/golden_full_frame"
os.makedirs(FULL_FRAME_DIR, exist_ok=True)


MODELS_DIR = "saved_models"
os.makedirs(MODELS_DIR, exist_ok=True) 

NUM_ACTUATORS = 40
DEFAULT_ON_MS = 25

VIDEO_PANE_W = 1000 
VIDEO_PANE_H = 800 

BG_COLOR = (30, 30, 30)
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
def normalize_name(s: str) -> str:
    return re.sub(r'\s+',' ', re.sub(r'[^a-z0-9\s-]',' ', str(s).lower())).strip()

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
        p = Path(n)
        if p.exists(): return p
    return None

def get_centroid(box):
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0

def centroid_distance(box1, box2):
    cx1, cy1 = get_centroid(box1)
    cx2, cy2 = get_centroid(box2)
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

def bayer_to_bgr(data, w, h):
    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        if arr.size == h*w:
            img = arr.reshape((h,w))
            return cv2.cvtColor(img, cv2.COLOR_BAYER_GR2BGR)
        elif arr.size == h*w*3:
            return arr.reshape((h,w,3)).copy()
        else:
            return None
    except Exception as e:
        return None

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
        self.analog_gain, self.digital_gain, self.exposure = 1.0, 1.0, 10000
        self.cam_settings_changed = False
        self.opencv_filter = {
            "sw_color_thresh": 45,
            "sw_defect_size_px": 20,
            "ww_color_thresh": 85,
            "ww_defect_size_px": 16,
            "max_defect_px": 1200
        }
        
        self.keras_preprocess = "MobileNet / TM (-1 to 1)"
        
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

        self.gain_map_8u = None
        self.ffc_enabled = False
        self.calibrate_ffc_requested = False
        
        if os.path.exists("ffc_gain_map.png"):
            self.gain_map_8u = cv2.imread("ffc_gain_map.png", cv2.IMREAD_GRAYSCALE)
            if self.gain_map_8u is not None:
                self.ffc_enabled = True
                print("[INIT] FFC Gain Map Loaded.")

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
    def reset_line(self):
        with self.lock: 
            self.line_x_full = None
            if self.line_pixel_callback: self.line_pixel_callback("Line X: Not Set", None)

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

    def trigger_ffc_calibration(self):
        with self.lock:
            self.calibrate_ffc_requested = True

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
            cfg = json.load(open(cfg_path,'r'))
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
                self.keras_preprocess = cfg.get("keras_preprocess", "MobileNet / TM (-1 to 1)")
                if "roi1" in cfg and cfg["roi1"]: self.roi1 = tuple(cfg["roi1"])
                if "roi2" in cfg and cfg["roi2"]: self.roi2 = tuple(cfg["roi2"])
                self.line_x_full = cfg.get("line_x_full")
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
                    self.exposure = c.get("exposure", 10000)
                
                self._rebuild_processing_map()
        except Exception: pass

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
                    "keras_preprocess": self.keras_preprocess,
                    "roi1": self.roi1,
                    "roi2": self.roi2,
                    "line_x_full": self.line_x_full,
                    "capture_classes": self.capture_classes,
                    "active_capture_class_id": self.active_capture_class_id,
                    "camera": {
                        "gain": self.cam_gain, "r": self.cam_r, "g": self.cam_g, "b": self.cam_b,
                        "a_gain": self.analog_gain, "d_gain": self.digital_gain, "exposure": self.exposure
                    }
                }
            with open(cfg_path, 'w') as f: json.dump(cfg, f, indent=2)
            # Auto-sync saved JSON to parent directory so old JSON files are eliminated
            for sync_dst in [
                Path(__file__).resolve().parent.parent / "pneumatic_config.json",
                Path("/home/phoenix/Documents/CNN_final/CNN/pneumatic_config.json")
            ]:
                try:
                    if sync_dst.parent.exists():
                        with open(sync_dst, 'w') as f: json.dump(cfg, f, indent=2)
                except Exception:
                    pass
        except Exception: pass

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
                "line_x": self.line_x_full,
                "mouse_x": self.mouse_x, "mouse_y": self.mouse_y,
                "mapping_t1": self.mapping_t1.copy(), "mapping_t2": self.mapping_t2.copy(),
                "priority_rules": list(self.priority_rules),
                "size_config": self.size_config.copy(),
                "class_confs": self.class_confs.copy(),
                "defect_allowances": self.defect_allowances.copy(),
                "keras_preprocess": self.keras_preprocess,
                "is_conveyor_on": self.is_conveyor_on,
                "gain": self.cam_gain, "r": self.cam_r, "g": self.cam_g, "b": self.cam_b,
                "a_gain": self.analog_gain, "d_gain": self.digital_gain, "exposure": self.exposure
            }

    def handle_mouse_down(self, pt: Tuple[int, int]):
        with self.lock:
            if self.phase == "ROI1_START" or self.phase == "ROI2_START":
                self.temp_start_pt = pt; self.temp_end_pt = pt
                self.phase = "ROI1_DRAW" if self.phase == "ROI1_START" else "ROI2_DRAW"
            elif self.phase == "LINE_SET_MODE":
                self.line_x_full = pt[0]
                self.phase = "READY"
                if self.line_pixel_callback: self.line_pixel_callback(f"Line X: {self.line_x_full} px", self.line_x_full)
                self.save_to_config(CFG_PATH)
            elif self.line_x_full is not None and abs(pt[0] - self.line_x_full) < 15:
                self.dragging_line = True

    def handle_mouse_move(self, pt: Tuple[int, int]):
        with self.lock:
            self.mouse_x, self.mouse_y = pt
            if self.phase in ("ROI1_DRAW", "ROI2_DRAW"): 
                self.temp_end_pt = pt
            elif self.dragging_line:
                self.line_x_full = pt[0]
                if self.line_pixel_callback: self.line_pixel_callback(f"Line X: {self.line_x_full} px", self.line_x_full)

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

# ---------------- CameraThread ----------------
class CameraThread(threading.Thread):
    def __init__(self, output_queue: queue.Queue, stop_event: threading.Event, shared_state: SharedState):
        super().__init__(daemon=True, name="CameraThread")
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.shared_state = shared_state
        self.camera = None
        self.stream = None
        self.is_connected = False
        self.reconnect_flag = threading.Event()
        self.fps = 0.0

    def request_reconnect(self):
        self.reconnect_flag.set()

    def run(self):
        frame_count = 0
        start_time = time.time()
        self.current_cam_phase = "READY"
        self.hw_canvas = None
        self.offset_x = 0
        self.offset_y = 0
        self.max_w = 0
        self.max_h = 0
        
        try:
            Aravis.update_device_list()
            n = Aravis.get_n_devices()
            if n == 0:
                self.output_queue.put(("ERROR", "No cameras found"))
                return
            
            device_id = Aravis.get_device_id(0)
            self.camera = Aravis.Camera.new(device_id)
            self.stream = self.camera.create_stream(None, None)
            payload = self.camera.get_payload()
            for _ in range(10): 
                self.stream.push_buffer(Aravis.Buffer.new_allocate(payload))
            
            try:
                device = self.camera.get_device()
                device.set_string_feature("ExposureAuto", "Off")
            except Exception as e: print(f"[CAM] ExposureAuto Off failed: {e}")
            try:
                device.set_string_feature("ExposureMode", "Timed")
            except Exception: pass
            
            def safe_set_exposure(dev, val):
                try: dev.set_float_feature_value("ExposureTime", float(val))
                except Exception:
                    try: dev.set_integer_feature_value("ExposureTime", int(val))
                    except Exception:
                        try: dev.set_float_feature_value("ExposureTimeAbs", float(val))
                        except Exception as e: 
                            err_msg = f"Exposure set failed: {e}"
                            print(f"[CAM] {err_msg}")
                            import tkinter.messagebox
                            try: tkinter.messagebox.showerror("Camera Error", err_msg)
                            except: pass
                        
            safe_set_exposure(device, self.shared_state.exposure)
                
            try:
                self.max_w = device.get_integer_feature("WidthMax")
                self.max_h = device.get_integer_feature("HeightMax")
            except Exception:
                self.max_w = 2448
                self.max_h = 2048
            
            self.camera.start_acquisition()
            self.is_connected = True
            
            while not self.stop_event.is_set():
                target_phase = self.shared_state.phase
                if target_phase in ["RUNNING", "READY"] and target_phase != self.current_cam_phase:
                    self.current_cam_phase = target_phase

                if self.shared_state.cam_settings_changed:
                    try:
                        device = self.camera.get_device()
                        device.set_string_feature_value("ExposureAuto", "Off")
                    except Exception: pass
                    safe_set_exposure(device, self.shared_state.exposure)
                    self.shared_state.cam_settings_changed = False
                    
                buf = self.stream.pop_buffer()
                if buf is None:
                    time.sleep(0.001) 
                    continue
                
                try:
                    w = buf.get_image_width()
                    h = buf.get_image_height()
                    data = buf.get_data()
                    
                    if self.shared_state.calibrate_ffc_requested:
                        self.shared_state.calibrate_ffc_requested = False
                        bayer_arr = np.frombuffer(data, dtype=np.uint8).reshape((h, w))
                        # Aggressive downsample to remove noise and texture quickly
                        small = cv2.resize(bayer_arr, (w//16, h//16))
                        smoothed = cv2.GaussianBlur(small, (15, 15), 0)
                        smoothed = cv2.resize(smoothed, (w, h))
                        mean_val = np.mean(smoothed)
                        gain_map = mean_val / (smoothed.astype(np.float32) + 1e-5)
                        self.shared_state.gain_map_8u = np.clip(gain_map * 128, 0, 255).astype(np.uint8)
                        self.shared_state.ffc_enabled = True
                        cv2.imwrite("ffc_gain_map.png", self.shared_state.gain_map_8u)
                        print("[CAM] FFC Calibration Complete and Saved.")
                        
                    if self.shared_state.ffc_enabled and self.shared_state.gain_map_8u is not None:
                        bayer_arr = np.frombuffer(data, dtype=np.uint8).reshape((h, w))
                        # cv2.multiply with scale applies the float multiplier mapped to 8-bit, running in fast C++
                        bayer_arr = cv2.multiply(bayer_arr, self.shared_state.gain_map_8u, scale=1/128.0)
                        frame_full = cv2.cvtColor(bayer_arr, cv2.COLOR_BAYER_GR2BGR)
                    else:
                        # Directly convert Bayer to BGR without making a copy of the Bayer buffer.
                        # cv2.cvtColor will allocate the new BGR array natively.
                        frame_full = bayer_to_bgr(data, w, h)
                        
                    if frame_full is None:
                        continue
                        
                    frame_count += 1
                    ct = time.time()
                    if ct - start_time >= 1.0:
                        self.fps = frame_count / (ct - start_time)
                        frame_count = 0
                        start_time = ct
                    
                    try: self.output_queue.get_nowait()
                    except queue.Empty: pass 
                    self.output_queue.put((frame_full, self.max_w, self.max_h))
                except Exception as e: pass
                finally:
                    self.stream.push_buffer(buf) 
                    
        except Exception as e:
            self.is_connected = False
            try: self.output_queue.put_nowait(("ERROR", str(e)))
            except queue.Full: pass
        finally:
            self.is_connected = False
            try:
                if self.camera: self.camera.stop_acquisition()
            except Exception: pass
            self.stream = None
            self.camera = None

# ---------------- ProcessingThread ----------------
class ProcessingThread(threading.Thread):
    def __init__(self, model_path: Optional[Path], class_names: Dict[int, str],
                 camera_queue: queue.Queue, display_queue: queue.Queue, 
                 stop_event: threading.Event, shared_state: SharedState,
                 controller: Any, fire_counts: Dict[int, int],
                 fire_lock: threading.Lock):

        super().__init__(daemon=True, name="ProcessingThread")
        self.model_path = model_path
        self.class_names = class_names
        self.camera_queue = camera_queue
        self.display_queue = display_queue
        self.stop_event = stop_event
        self.shared_state = shared_state
        self.controller = controller
        self.fire_counts = fire_counts 
        self.fire_lock = fire_lock      
        self.is_keras_model = False
        self.model = None
        self.model_lock = threading.Lock()
        
        # Load torch to ensure GPU availability check
        try:
            import torch
            self.device = 0 if torch.cuda.is_available() else 'cpu'
            print(f"[DEBUG-INIT] PyTorch available. Using device: {self.device}")
        except ImportError:
            self.device = 'cpu'
            print("[DEBUG-INIT] PyTorch not imported. Defaulting to cpu.")
            
        print(f"[DEBUG-INIT] ProcessingThread initialized with model_path: {self.model_path}")
        if self.model_path:
            self.reload_model(self.model_path)

        # ── CNN Color Classifier ──────────────────────────────────────────────
        self.cnn_model = None
        self.cnn_classes = []
        self.cnn_device = None
        self.cnn_transform = None
        if _CNN_AVAILABLE and CNN_MODEL_PATH.exists():
            try:
                ckpt = _torch.load(str(CNN_MODEL_PATH), map_location="cpu")
                self.cnn_classes = ckpt["class_names"]
                _m = _tv_models.mobilenet_v2(weights=None)
                _m.classifier[1] = _torch.nn.Linear(_m.last_channel, len(self.cnn_classes))
                _m.load_state_dict(ckpt.get("model_state_dict", ckpt.get("model_state")))
                _m.eval()
                self.cnn_device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
                self.cnn_model = _m.to(self.cnn_device)
                self.cnn_transform = _tv_transforms.Compose([
                    _tv_transforms.Resize((224, 224)),
                    _tv_transforms.ToTensor(),
                    _tv_transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225]),
                ])
                print(f"[CNN] Loaded cashew_color_cnn.pth "
                      f"({len(self.cnn_classes)} classes)")
            except Exception as e:
                print(f"[CNN] Failed to load CNN model: {e}")
        else:
            print("[CNN] CNN model not found – skipping.")
            
        # Ensure cashew_class_mapping.json and cnn_classes are populated in self.class_names
        if os.path.exists("cashew_class_mapping.json"):
            try:
                import json
                with open("cashew_class_mapping.json", "r") as f:
                    json_map = {int(k): str(v) for k, v in json.load(f).items()}
                    for k, v in json_map.items():
                        self.class_names[k] = v
            except Exception as e:
                print(f"[WARN] Failed to merge class_mapping JSON: {e}")

        for cname in self.cnn_classes:
            if not any(v.lower() == cname.lower() for v in self.class_names.values()):
                new_id = max(self.class_names.keys(), default=-1) + 1
                self.class_names[new_id] = cname

        print(f"[DEBUG-INIT] Final self.class_names: {self.class_names}")
        
        self.tracked_objects = []
        self.next_obj_id = 1

    def _get_good_class_id(self) -> int:
        for k, v in self.class_names.items():
            if v.lower() in GOOD_CLASSES:
                return int(k)
        return 0

    def _cnn_predict(self, crop_bgr: np.ndarray):
        """Run CNN on a BGR crop. Returns (class_name, confidence) or (None, 0.0)."""
        if self.cnn_model is None or crop_bgr is None or crop_bgr.size == 0:
            return None, 0.0
        try:
            from PIL import Image as _PIL_Image
            # Pad crop to square to preserve cashew aspect ratio
            h, w = crop_bgr.shape[:2]
            max_dim = max(h, w)
            pad_h = (max_dim - h) // 2
            pad_w = (max_dim - w) // 2
            square_crop = cv2.copyMakeBorder(crop_bgr, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE)
            
            rgb = cv2.cvtColor(square_crop, cv2.COLOR_BGR2RGB)
            pil = _PIL_Image.fromarray(rgb)
            tensor = self.cnn_transform(pil).unsqueeze(0).to(self.cnn_device)
            with _torch.no_grad():
                probs = _F.softmax(self.cnn_model(tensor), dim=1)[0].cpu().numpy()
            idx = int(np.argmax(probs))
            return self.cnn_classes[idx], float(probs[idx])
        except Exception:
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
        except Exception as e:
            print(f"[CV-ERR] Defect area calculation failed: {e}")
            return 999999

    def _get_cashew_contour_rect(self, crop_bgr: np.ndarray):
        """Returns (cashew_area, (x, y, w, h)) inside the crop to localize cashew vs empty roller."""
        if crop_bgr is None or crop_bgr.size == 0:
            return 0, None
        try:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            # Threshold to separate bright cashew from dark blue roller channel
            _, mask = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(c)
                if area > 100:
                    x, y, w, h = cv2.boundingRect(c)
                    return int(area), (x, y, w, h)
            return 0, None
        except Exception:
            return 0, None
        
    def get_color_for_class(self, cid: int):
        return CLASS_COLORS[cid % len(CLASS_COLORS)]
        
    def reload_model(self, model_path):
        try:
            model_str = str(model_path)
            if model_str.endswith(".pt"):
                from ultralytics import YOLO
                new_model = YOLO(model_str)
                with self.model_lock:
                    with self.shared_state.lock:
                        self.model_path = model_path
                        self.model = new_model
                        self.is_keras_model = False
                        model_names = {int(k): str(v) for k, v in new_model.names.items()}
                        if not hasattr(self, 'class_names') or not self.class_names:
                            self.class_names = model_names
                        else:
                            for k, v in model_names.items():
                                if k not in self.class_names:
                                    self.class_names[k] = v
            else:
                print(f"[ERROR] Only .pt YOLO models are supported. {model_str} is invalid.")
        except Exception as e:
            print(f"[ERROR] Failed to load new model: {e}")

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
                
                (frame_full, w, h) = data
                if frame_full is None: continue
                
                with self.shared_state.lock:
                    self.shared_state.frame = frame_full.copy()
                
                state = self.shared_state.get_processing_state()
                
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
                            f.write(f"Exposure: {state.get('exposure', 5000):.2f}\n")
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
                
                display_frame = None
                
                if state["phase"] == "RUNNING" and self.model and (state["roi1"] or state["roi2"]):
                    # Don't draw rulers here, just process it directly to save massive amounts of CPU on the 5MP frame
                    frame_drawn = self._process_frame(frame_full, state)
                    
                    # Crop and stack only the ROIs for the GUI display!
                    crops_to_show = []
                    if state["roi1"]:
                        rx1, ry1, rx2, ry2 = state["roi1"]
                        crops_to_show.append(frame_drawn[ry1:ry2, rx1:rx2])
                    if state["roi2"]:
                        rx1, ry1, rx2, ry2 = state["roi2"]
                        crops_to_show.append(frame_drawn[ry1:ry2, rx1:rx2])
                        
                    if len(crops_to_show) > 0:
                        target_w = max(c.shape[1] for c in crops_to_show)
                        padded = []
                        for c in crops_to_show:
                            if c.shape[1] < target_w:
                                c = cv2.copyMakeBorder(c, 0, 0, 0, target_w - c.shape[1], cv2.BORDER_CONSTANT, value=(0,0,0))
                            padded.append(c)
                        display_img = cv2.vconcat(padded)
                    else:
                        display_img = frame_drawn
                        
                    display_frame, scale, x_off, y_off = resize_letterbox(display_img, VIDEO_PANE_W, VIDEO_PANE_H, bg_color=BG_COLOR)
                    geometry = {"scale": scale, "x_off": x_off, "y_off": y_off, "orig_w": w, "orig_h": h}
                else:
                    self.tracked_objects = []
                    self.next_obj_id = 1
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
            except Exception as e: pass

    # FIXED: Small arrow in the top corner (approx 25mm length x 5mm height visually)
    def _draw_arrows(self, frame, state):
        line_x = state.get("line_x")
        for idx, roi in enumerate([state["roi1"], state["roi2"]]):
            if roi:
                x1, y1, x2, y2 = roi
                cx_roi = (x1 + x2) // 2
                
                # Flow direction logic
                flow_right = True
                if line_x and cx_roi > line_x:
                    flow_right = False
                
                # Placed exactly 20 pixels below the top edge of ROI, neatly in the corner
                arrow_y = y1 + 20 
                arrow_length = 60 # Approx 25mm visually
                
                if flow_right:
                    start_pt = (x1 + 10, arrow_y)
                    end_pt = (x1 + 10 + arrow_length, arrow_y)
                    cv2.arrowedLine(frame, start_pt, end_pt, ARROW_COLOR, 2, tipLength=0.3)
                    cv2.putText(frame, f"T{idx+1} FLOW", (end_pt[0] + 10, arrow_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ARROW_COLOR, 2)
                else:
                    start_pt = (x2 - 10, arrow_y)
                    end_pt = (x2 - 10 - arrow_length, arrow_y)
                    cv2.arrowedLine(frame, start_pt, end_pt, ARROW_COLOR, 2, tipLength=0.3)
                    cv2.putText(frame, f"FLOW T{idx+1}", (end_pt[0] - 80, arrow_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ARROW_COLOR, 2)

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
        if state["line_x"]:
            cv2.line(frame, (state["line_x"], 0), (state["line_x"], frame.shape[0]), LINE_COLOR, 3)
        
        ph = state["phase"]
        msg = ""
        if ph == "ROI1_START": msg = "Draw ROI 1"
        elif ph == "ROI2_START": msg = "Draw ROI 2"
        elif ph == "LINE_SET_MODE": msg = "Click to set Decision Line"
        if msg: cv2.putText(frame, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        self._draw_arrows(frame, state)

    def _resolve_priority(self, history):
        if not history: return 0
        from collections import Counter
        counts = Counter(history)
        
        defects = []
        non_defects = []
        for cid, count in counts.items():
            cname = self.class_names.get(cid, "UNK").lower()
            if cname in ["good_cashew", "sw", "white_dw"] or cname.startswith("ww_") or cname.startswith("sw_"):
                non_defects.append((cid, count))
            else:
                defects.append((cid, count))
                
        if defects:
            defects.sort(key=lambda x: x[1], reverse=True)
            best_defect, def_count = defects[0]
            
            # REMOVED NOISE FILTER: 
            # If the cashew is rolling 360 degrees and shows a defect for even 1 frame, 
            # we MUST reject it. We do NOT ignore single-frame defect blips anymore.
            return best_defect
        else:
            if non_defects:
                non_defects.sort(key=lambda x: x[1], reverse=True)
                return non_defects[0][0]
            return history[-1]

    def _process_frame(self, frame, state):
        current_time_ms = time.time() * 1000.0
        
        # Clean copy for inference and image capturing (no overlay lines/dots!)
        clean_frame = frame.copy()
        
        # Use user-defined YOLO confidence threshold to filter out background noise (empty rollers)
        global_min_conf = state.get("target_conf", 0.30)
        
        line_x = state["line_x"]
        if line_x: cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), LINE_COLOR, 2)

        boxes, classes, confs = [], [], []
        
        with self.model_lock:
            rois = []
            if state.get("roi1"): rois.append(state["roi1"])
            if state.get("roi2"): rois.append(state["roi2"])
            if not rois:
                rois = [(0, 0, frame.shape[1], frame.shape[0])]

            if self.model is not None:
                crops = []
                offsets = []
                for rx1, ry1, rx2, ry2 in rois:
                    roi_crop = clean_frame[ry1:ry2, rx1:rx2]
                    if roi_crop.size > 0:
                        crops.append(roi_crop)
                        offsets.append((rx1, ry1))
                        
                if crops:
                    target_w = max(c.shape[1] for c in crops)
                    padded = []
                    for c in crops:
                        if c.shape[1] < target_w:
                            c = cv2.copyMakeBorder(c, 0, 0, 0, target_w - c.shape[1], cv2.BORDER_CONSTANT, value=(0,0,0))
                        padded.append(c)
                    
                    combined_img = cv2.vconcat(padded)
                    results = self.model.predict(combined_img, conf=global_min_conf, verbose=False, device=self.device)[0]
                    
                    for box in results.boxes:
                        xy = box.xyxy[0].cpu().numpy().tolist()
                        cy = (xy[1] + xy[3]) / 2.0
                        
                        y_accum = 0
                        for i, (c, (rx1, ry1)) in enumerate(zip(padded, offsets)):
                            if y_accum <= cy <= y_accum + c.shape[0]:
                                local_x1, local_y1 = xy[0], xy[1] - y_accum
                                local_x2, local_y2 = xy[2], xy[3] - y_accum
                                
                                c_conf = float(box.conf[0].cpu().numpy())
                                cls_id = int(box.cls[0].cpu().numpy())
                                
                                global_x1 = int(local_x1) + rx1
                                global_y1 = int(local_y1) + ry1
                                global_x2 = int(local_x2) + rx1
                                global_y2 = int(local_y2) + ry1
                                
                                boxes.append([global_x1, global_y1, global_x2, global_y2])
                                classes.append(cls_id)
                                confs.append(c_conf)
                                break
                            y_accum += c.shape[0]

        # Golden Capture logic moved to run after tracker assignment


        # ── CNN as Primary Classifier ──────────────
        if self.cnn_model is not None and boxes:
            for i in range(len(boxes)):
                bx1, by1, bx2, by2 = boxes[i]
                
                crop = clean_frame[max(0,by1):min(clean_frame.shape[0],by2),
                             max(0,bx1):min(clean_frame.shape[1],bx2)]
                if crop.size == 0:
                    continue
                    
                cashew_area, cashew_rect = self._get_cashew_contour_rect(crop)
                # Localize tight bounding box around the cashew inside the roller slot if contour is present
                if cashew_rect is not None and cashew_area >= 150:
                    rx, ry, rw, rh = cashew_rect
                    margin = 4
                    nbx1 = max(0, bx1 + rx - margin)
                    nby1 = max(0, by1 + ry - margin)
                    nbx2 = min(clean_frame.shape[1], bx1 + rx + rw + margin)
                    nby2 = min(clean_frame.shape[0], by1 + ry + rh + margin)
                    boxes[i] = [nbx1, nby1, nbx2, nby2]
                    crop = clean_frame[nby1:nby2, nbx1:nbx2]

                cnn_name, cnn_conf = self._cnn_predict(crop)
                if cnn_name is None:
                    continue
                    
                # ── CNN Primary Class Preserved ──────────────────────────────
                cname_lower = cnn_name.lower().replace("_", " ")
                cv_cfg = state.get("opencv_filter", {})

                # Only apply noise filter for SW / WHITE DW if defect is below noise threshold
                if cname_lower in ["sw", "white dw"]:
                    if cname_lower == "sw":
                        def_px = self._calculate_defect_area(crop, color_thresh=cv_cfg.get("sw_color_thresh", 45))
                        max_px = cv_cfg.get("sw_defect_size_px", 20)
                    else:
                        def_px = self._calculate_defect_area(crop, color_thresh=cv_cfg.get("ww_color_thresh", 85))
                        max_px = cv_cfg.get("ww_defect_size_px", 16)
                    if def_px < max_px:
                        cnn_name = "good_cashew"
                    
                # Map CNN class name back to class ID (auto-register if new)
                cnn_cid = None
                for k, v in self.class_names.items():
                    if v.lower() == cnn_name.lower() or normalize_name(v) == normalize_name(cnn_name):
                        cnn_cid = k
                        break
                        
                if cnn_cid is None:
                    new_id = max(self.class_names.keys(), default=-1) + 1
                    self.class_names[new_id] = cnn_name
                    cnn_cid = new_id

                if cnn_cid is not None:
                    req_conf = state.get("class_confs", {}).get(cnn_name.lower(), 50)
                    if (cnn_conf * 100) >= req_conf:
                        classes[i] = cnn_cid
                        confs[i] = cnn_conf

        mb, mc, mcf = merge_boxes_area_rule(boxes, classes, confs, self.class_names)

        # Restore tracking
        new_tracked = []
        unmatched_new = list(range(len(mb)))
        
        for obj in self.tracked_objects:
            best_dist = 100000
            best_idx = -1
            
            # Predict position using current velocity
            vx = obj.get('vx', 0.0)
            vy = obj.get('vy', 0.0)
            pred_box = [
                obj['box'][0] + vx, obj['box'][1] + vy,
                obj['box'][2] + vx, obj['box'][3] + vy
            ]

            for i in unmatched_new:
                new_cx = (mb[i][0] + mb[i][2]) / 2.0
                
                # CRITICAL IDENTITY THEFT FIX:
                # If this object was already fired, it MUST NEVER match an incoming cashew that is still behind the line!
                if line_x and obj.get('fired') and new_cx > (line_x - 10):
                    continue

                dist = centroid_distance(pred_box, mb[i])
                # Ensure they are vertically aligned (same channel/track)
                y_overlap = min(pred_box[3], mb[i][3]) - max(pred_box[1], mb[i][1])
                if dist < 160 and y_overlap > -20: 
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i
            
            if best_idx != -1:
                old_cx = (obj['box'][0] + obj['box'][2]) / 2.0
                old_cy = (obj['box'][1] + obj['box'][3]) / 2.0
                new_cx = (mb[best_idx][0] + mb[best_idx][2]) / 2.0
                new_cy = (mb[best_idx][1] + mb[best_idx][3]) / 2.0
                inst_vx = new_cx - old_cx
                inst_vy = new_cy - old_cy
                
                obj['vx'] = 0.7 * obj.get('vx', 0.0) + 0.3 * inst_vx
                obj['vy'] = 0.7 * obj.get('vy', 0.0) + 0.3 * inst_vy
                
                # Time-based true velocity calculation
                dt = current_time_ms - obj.get('last_time_ms', current_time_ms - 13.3)
                if dt > 0:
                    inst_v_px_per_ms = inst_vx / dt
                    obj['v_px_per_ms'] = 0.7 * obj.get('v_px_per_ms', inst_v_px_per_ms) + 0.3 * inst_v_px_per_ms
                
                obj['last_time_ms'] = current_time_ms
                
                obj['box'] = mb[best_idx]
                obj['cid'] = mc[best_idx]
                obj['conf'] = mcf[best_idx]
                obj['missing_frames'] = 0
                if 'history' not in obj: obj['history'] = []
                obj['history'].append(mc[best_idx])
                new_tracked.append(obj)
                unmatched_new.remove(best_idx)
            else:
                obj['missing_frames'] += 1
                # Drop max_missing from 15 to 3. If a cashew is missing for >3 frames (~40ms), 
                # drop it to prevent the ghost from merging with the next incoming cashew.
                max_missing = 3 if len(obj.get('history', [])) > 2 else 0
                
                if obj['missing_frames'] < max_missing:
                    vx = obj.get('vx', 0)
                    vy = obj.get('vy', 0)
                    if vx == 0 and vy == 0 and line_x:
                        cx = (obj['box'][0] + obj['box'][2]) / 2.0
                        if cx < line_x: vx = 20  
                        else: vx = -20           
                        
                    obj['box'] = [
                        obj['box'][0] + vx, obj['box'][1] + vy,
                        obj['box'][2] + vx, obj['box'][3] + vy
                    ]
                    new_tracked.append(obj)
                    
        for i in unmatched_new:
            cx_init = (mb[i][0] + mb[i][2]) / 2.0
            new_tracked.append({
                'id': getattr(self, 'next_obj_id', 0), 'box': mb[i], 'cid': mc[i], 'conf': mcf[i],
                'fired': False, 'counted': False, 'missing_frames': 0, 'history': [mc[i]],
                'start_x': cx_init,
                'vx': 0.0, 'vy': 0.0, 'track_id': None,
                'last_time_ms': current_time_ms, 'v_px_per_ms': 0.0,
                'cached_label': None, 'cached_act_id': None
            })
            self.next_obj_id = getattr(self, 'next_obj_id', 0) + 1
            
        self.tracked_objects = new_tracked

        px_per_mm = state["size_config"].get("px_per_mm", 10.0)

        for obj in self.tracked_objects:
            x1, y1, x2, y2 = int(obj['box'][0]), int(obj['box'][1]), int(obj['box'][2]), int(obj['box'][3])
            cx, cy = (x1+x2)//2, (y1+y2)//2 
            
            # Count cashew once tracked across at least 2 frames (avoids noise)
            if not obj.get('counted') and len(obj.get('history', [])) >= 2:
                obj['counted'] = True
                self.shared_state.increment_detected_count() 
            
            if 'history' in obj and obj['history']:
                resolved_cid = self._resolve_priority(obj['history'])
                final_cname = self.class_names.get(resolved_cid, "UNK").lower()
                display_cid = resolved_cid
            else:
                final_cname = self.class_names.get(obj['cid'], "UNK").lower()
                display_cid = obj['cid']

            # NEW: Only apply OpenCV sizing logic to 'good' and 'sw' classes per user request.
            # All other defects will directly trigger a valve without OpenCV size operations.
            is_size_class = final_cname in ["good", "good_cashew", "good cashew", "sw", "sw cashew"]
            
            if not is_size_class:
                allowance_mm = state["defect_allowances"].get(final_cname, 0.0)
                if allowance_mm > 0.0:
                    defect_mm_len = max(x2-x1, y2-y1) / px_per_mm
                    if defect_mm_len <= allowance_mm:
                        continue 
            
            c = obj['conf']
            px_length = max(x2 - x1, y2 - y1)
            px_width = min(x2 - x1, y2 - y1)
            color_prefix = "ww" 
            
            try:
                if y2 > y1 and x2 > x1 and obj['missing_frames'] == 0:
                    crop = frame[y1:y2, x1:x2]
                    
                    if is_size_class:
                        # Use CNN classification for WW/SW prefix
                        if "sw" in final_cname:
                            color_prefix = "sw"
                        else:
                            color_prefix = "ww"
                            
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        rect = cv2.minAreaRect(largest_contour)
                        (w, h) = rect[1]
                        
                        if w > 0 and h > 0:
                            px_length = max(w, h)
                            px_width = min(w, h)
                            bx = cv2.boxPoints(rect)
                            bx = np.int32(bx)
                            bx[:, 0] += x1
                            bx[:, 1] += y1
            except Exception as e:
                pass 
                
            mm_length = px_length / px_per_mm
            mm_width = px_width / px_per_mm

            cv2.circle(frame, (int(cx), int(cy)), 5, (255, 255, 255), -1)
            
            line_x = state.get("line_x")
            has_crossed = line_x is not None and cx <= line_x
            
            if self.shared_state.is_golden_capture_active() and obj['missing_frames'] == 0 and not has_crossed:
                pad = 20
                gy1 = max(0, y1 - pad)
                gy2 = min(clean_frame.shape[0], y2 + pad)
                gx1 = max(0, x1 - pad)
                gx2 = min(clean_frame.shape[1], x2 + pad)
                g_crop = clean_frame[gy1:gy2, gx1:gx2]
                if g_crop.size > 0:
                    if self.shared_state.increment_golden_capture():
                        timestamp_ms = int(time.time()*1000)
                        c_type = self.shared_state.golden_capture_type
                        
                        if c_type in ("cropped", "both"):
                            g_filename = os.path.join(GOLDEN_DIR, f"crop_{timestamp_ms}_{obj['id']}.jpg")
                            cv2.imwrite(g_filename, g_crop)
                            
                        if c_type in ("full", "both"):
                            f_filename = os.path.join(FULL_FRAME_DIR, f"full_{timestamp_ms}_{obj['id']}.jpg")
                            cv2.imwrite(f_filename, clean_frame)
                        
                        # Save camera metadata to a text file
                        meta_file = os.path.join(GOLDEN_DIR, "camera_metadata.txt")
                        if not os.path.exists(meta_file):
                            import datetime
                            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            with open(meta_file, "w") as f:
                                f.write(f"Capture Date & Time: {now_str}\n")
                                f.write(f"Exposure: {state.get('exposure', 'N/A')}\n")
                                f.write(f"Gain: {state.get('gain', 'N/A')}\n")
                                f.write(f"R: {state.get('r', 'N/A')}\n")
                                f.write(f"G: {state.get('g', 'N/A')}\n")
                                f.write(f"B: {state.get('b', 'N/A')}\n")

            if obj.get('track_id') is None:
                in_roi1 = (state["roi1"][0] <= cx <= state["roi1"][2]) and (state["roi1"][1] <= cy <= state["roi1"][3]) if state["roi1"] else False
                in_roi2 = (state["roi2"][0] <= cx <= state["roi2"][2]) and (state["roi2"][1] <= cy <= state["roi2"][3]) if state["roi2"] else False
                if in_roi1: obj['track_id'] = 1
                elif in_roi2: obj['track_id'] = 2
                else:
                    mid_y = (state["roi1"][3] + state["roi2"][1]) / 2.0 if state.get("roi1") and state.get("roi2") else frame.shape[0] / 2.0
                    obj['track_id'] = 1 if cy < mid_y else 2
                
            track_id = obj['track_id']
            cls_color = self.get_color_for_class(display_cid)
            
            crossed_line = False
            # Check line crossing: If cashew is past or reaching the decision line, fire it!
            if line_x and not obj['fired']:
                if cx <= line_x and cx >= (line_x - 400):
                    crossed_line = True

            if crossed_line and not obj['fired']:
                # Check if another cashew in the same roller was already fired and is virtually identical (duplicate tracker bounding box)
                too_close = False
                for other_obj in self.tracked_objects:
                    if other_obj['id'] != obj['id'] and other_obj.get('fired'):
                        other_cx = (other_obj['box'][0] + other_obj['box'][2]) / 2.0
                        other_cy = (other_obj['box'][1] + other_obj['box'][3]) / 2.0
                        if abs(cy - other_cy) < 20 and abs(cx - other_cx) < 15:
                            too_close = True
                            break
                            
                if too_close:
                    obj['fired'] = True
                    obj['cached_label'] = "SKIPPED (TOO CLOSE)"
                    obj['cached_act_id'] = None
                    self.shared_state.add_fire_event(track_id, "SKIPPED", c, 0)
                    continue

                act_id = None
                label = ""
                if not is_size_class:
                    map_dict = state["mapping_t1"] if track_id == 1 else state["mapping_t2"]
                    act_id = map_dict.get(normalize_name(final_cname))
                    label = f"DEF: {final_cname.upper()}"
                else:
                    cfg_act_id, s_lbl = self._get_size_data(mm_length, mm_width, state["size_config"], track_id)
                    combined_cname = f"{color_prefix}_{s_lbl}".lower()
                    
                    # Check if user explicitly mapped this size class in the Pneumatics UI
                    map_dict = state["mapping_t1"] if track_id == 1 else state["mapping_t2"]
                    mapped_act = None
                    
                    # 1. Check direct SW size name if scorched (e.g. "SW180")
                    if color_prefix == "sw":
                        sw_name = f"sw{s_lbl[1:]}" if s_lbl.startswith("W") else f"sw_{s_lbl}"
                        mapped_act = map_dict.get(normalize_name(sw_name))
                    
                    # 2. Check general size label (e.g. "W180", "W210", "W240", "W320", "W450", "BUDDS")
                    if mapped_act is None:
                        mapped_act = map_dict.get(normalize_name(s_lbl))
                        
                    # 3. Check combined name (e.g. "ww_w180" or "sw_w180")
                    if mapped_act is None:
                        mapped_act = map_dict.get(normalize_name(combined_cname))
                        
                    # 4. Check color prefix (e.g. "ww" or "sw")
                    if mapped_act is None:
                        mapped_act = map_dict.get(normalize_name(color_prefix))
                    
                    # 5. Check general good cashew class (e.g. "good_cashew" or "good")
                    if mapped_act is None:
                        mapped_act = map_dict.get(normalize_name(final_cname))
                    
                    # Strictly only fire if explicitly mapped in Pneumatics UI (no fallback to default Actuator 1)
                    act_id = mapped_act
                    label = f"{combined_cname.upper()}"
                
                obj['fired'] = True
                obj['cached_label'] = label
                obj['cached_act_id'] = act_id
                
                if act_id is not None and 1 <= act_id <= NUM_ACTUATORS:
                    d_ms, p_ms = self.shared_state.get_timing(act_id)

                    if getattr(self.shared_state, 'master_ejector_enabled', True):
                        self.controller.pulse_actuator(act_id, d_ms, p_ms)
                        print(f"[FIRE] Ejector {act_id} fired for Cashew ID {obj['id']} ({label}) - Delay: {d_ms}ms, Pulse: {p_ms}ms")
                    
                    with self.fire_lock:
                        self.fire_counts[int(display_cid)] = self.fire_counts.get(int(display_cid),0) + 1
                    
                self.shared_state.add_fire_event(track_id, label, c, act_id if act_id else 0)
                
            # Draw persistent ID and status
            if obj['fired']:
                lbl = obj.get('cached_label', final_cname.upper())
                act = obj.get('cached_act_id')
                act_str = f"(A{act})" if act else "(NO ACT)"
                cv2.rectangle(frame, (x1, y1), (x2, y2), FIRE_COLOR, 3)
                cv2.putText(frame, f"ID:{obj['id']} FIRED {lbl[:12]} {act_str}", (x1, max(12, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, FIRE_COLOR, 3)
            else:
                draw_color = cls_color
                thickness = 1 if obj['missing_frames'] > 0 else 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, thickness)
                disp = final_cname.upper()
                if obj['missing_frames'] > 0:
                    disp += " (EXTRAP)"
                cv2.putText(frame, f"ID:{obj['id']} {disp[:12]} {c:.2f}", (x1, max(12, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, draw_color, 3)

        self._draw_arrows(frame, state)
        return frame

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

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Newton Automation | Industrial HMI")
        self.geometry("1800x900+10+10")
        self.attributes("-zoomed", True)
        self.configure(bg="#0F172A")
        
        self.colors = {
            "bg": "#0F172A", "surface": "#1E293B", "surface_light": "#334155",
            "text": "#F8FAFC", "text_muted": "#94A3B8",
            "success": "#10B981", "danger": "#EF4444", "warning": "#F59E0B", "info": "#0EA5E9",
            "accent": "#38BDF8"
        }
        
        self._last_fires_state = []
        
        saved_model = ""
        try:
            import json
            if os.path.exists(CFG_PATH):
                with open(CFG_PATH, "r") as f:
                    cfg = json.load(f)
                    saved_model = str(cfg.get("yolo_model_path", ""))
        except: pass

        try:
            test_path = "/home/phoenix/Documents/1/cashew_gsn/CNN/assets/btn_test.png"
            gear_path = "/home/phoenix/Documents/1/cashew_gsn/CNN/assets/btn_gear.png"
            if os.path.exists(test_path) and os.path.exists(gear_path):
                img_t = Image.open(test_path).resize((22, 22), Image.Resampling.LANCZOS)
                self.img_test = ImageTk.PhotoImage(img_t)
                img_g = Image.open(gear_path).resize((22, 22), Image.Resampling.LANCZOS)
                self.img_gear = ImageTk.PhotoImage(img_g)
            else:
                self.img_test, self.img_gear = None, None
        except Exception as e:
            print(f"[WARN] Failed to load icon images: {e}")
            self.img_test, self.img_gear = None, None
            
        self.nav_icons = {}
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
        for name, file in icon_names.items():
            path = f"/home/phoenix/Documents/1/cashew_gsn/CNN/assets/nav_icons/{file}"
            try:
                if os.path.exists(path):
                    img = Image.open(path).resize((32, 32), Image.Resampling.LANCZOS)
                    self.nav_icons[name] = ImageTk.PhotoImage(img)
                else:
                    self.nav_icons[name] = None
            except:
                self.nav_icons[name] = None
        
        self.model_path = ""
        if saved_model:
            fname = os.path.basename(saved_model)
            p1 = os.path.join(MODEL_ROOT, fname)
            p2 = os.path.join(os.path.dirname(MODEL_ROOT), saved_model)
            if os.path.exists(p1):
                self.model_path = p1
            elif os.path.exists(p2):
                self.model_path = p2
            elif os.path.exists(saved_model):
                self.model_path = saved_model

        if not self.model_path or not os.path.exists(self.model_path):
            best_candidate = MODEL_ROOT / "model_20260913_230121.pt"
            if not best_candidate.exists():
                best_candidate = MODEL_ROOT / "best.pt"
            if best_candidate.exists():
                self.model_path = str(best_candidate)
            else:
                pt_models = list(MODEL_ROOT.glob("*.pt"))
                if pt_models:
                    self.model_path = str(pt_models[0])
                else:
                    self.model_path = str(find_best_weights(MODEL_ROOT))
        print(f"[INFO] YOLO model_path resolved to: {self.model_path}")
        self.class_map = None
        if os.path.exists("cashew_class_mapping.json"):
            try:
                import json
                with open("cashew_class_mapping.json", "r") as f:
                    self.class_map = {int(k): str(v) for k,v in json.load(f).items()}
            except Exception as e:
                print(f"[WARN] Failed to load class mapping JSON: {e}")
                
        if self.class_map is None and self.model_path:
            try:
                from ultralytics import YOLO
                m = YOLO(str(self.model_path))
                self.class_map = {int(k): str(v) for k,v in m.names.items()}
                del m
            except Exception as e:
                print(f"[WARN] Failed to load YOLO model: {e}")
                self.class_map = {0: "Capture_Only_Mode"}
        elif self.class_map is None:
            print("[INFO] No Model Found. Starting in Image Capture ONLY mode.")
            self.class_map = {0: "Capture_Only_Mode"}
            
        self.all_class_names = list(self.class_map.values())

        # Merge additional classes from the authoritative JSON mapping.
        # cashew_class_mapping.json is the single source of truth for
        # user-managed classes. We do NOT auto-scan dataset folders or CNN
        # checkpoints, as that causes deleted classes to reappear on restart.
        _class_mapping_path = Path("cashew_class_mapping.json")
        if _class_mapping_path.exists():
            try:
                with open(_class_mapping_path, "r") as f:
                    _json_map = json.load(f)
                for v in _json_map.values():
                    if v not in self.all_class_names:
                        self.all_class_names.append(v)
            except Exception:
                pass

        # Add hybrid size classes to allow UI mapping for individual valves
        for sz in ["w180", "w210", "w240", "w320", "w450", "budds"]:
            self.all_class_names.extend([f"ww_{sz}", f"sw_{sz}"])
            
        self.shared_state = SharedState(self.all_class_names)
        self.shared_state.load_from_config(CFG_PATH)
        
        # Sync loaded capture_classes with all_class_names
        for v in self.shared_state.capture_classes.values():
            if v and v not in self.all_class_names and not v.startswith("Class_"):
                self.all_class_names.append(v)
                
        if InovancePLCController is not None:
            self.controller = InovancePLCController()
            print("[CONTROLLER] Initialized Inovance EC320 PLC Controller as primary pneumatic driver.")
        else:
            self.controller = ArduinoMegaController()
            print("[CONTROLLER] Fallback to Arduino Mega Controller.")
        
        self.fire_counts = {cid: 0 for cid in self.class_map.keys()}
        self.fire_lock = threading.Lock()
        
        self.stop_event = threading.Event()
        self.camera_queue = queue.Queue(maxsize=2)
        self.display_queue = queue.Queue(maxsize=2)
        
        self.camera_thread = CameraThread(self.camera_queue, self.stop_event, self.shared_state)
        self.processing_thread = ProcessingThread(
            self.model_path, self.class_map, self.camera_queue, self.display_queue, 
            self.stop_event, self.shared_state, self.controller, self.fire_counts, self.fire_lock
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
        style.map("Treeview", background=[('selected', self.colors["surface_light"])])

        style.configure("TEntry", fieldbackground=self.colors["surface_light"], foreground=text, borderwidth=0, padding=5)
        style.configure("TCombobox", fieldbackground=self.colors["surface_light"], foreground="black", borderwidth=0, padding=5)
        self.option_add('*TCombobox*Listbox.foreground', 'black')

    def _build_gui(self):
        header_frm = tk.Frame(self, bg=self.colors["bg"], height=60, bd=0)
        header_frm.pack(side=tk.TOP, fill=tk.X, padx=20, pady=10)
        header_frm.pack_propagate(False)
        
        ttk.Label(header_frm, text="NEWTON AUTOMATION", font=("Roboto", 16, "bold"), foreground=self.colors["text"]).pack(side=tk.LEFT)
        ttk.Label(header_frm, text=" | INSTANT FIRE GRADER", font=("Roboto", 14), foreground=self.colors["text_muted"]).pack(side=tk.LEFT)
        
        self.lbl_clock = tk.Label(header_frm, text="00:00:00", font=("Roboto", 14, "bold"), bg=self.colors["bg"], fg=self.colors["text_muted"])
        self.lbl_clock.pack(side=tk.RIGHT, padx=20)
        
        self.lbl_state_badge = tk.Label(header_frm, text="● IDLE", font=("Roboto", 14, "bold"), bg=self.colors["surface"], fg=self.colors["text_muted"], padx=15, pady=5)
        self.lbl_state_badge.pack(side=tk.RIGHT, padx=20)
        
        body_frm = tk.Frame(self, bg=self.colors["bg"])
        body_frm.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        nav_frm = tk.Frame(body_frm, bg=self.colors["bg"], width=220)
        nav_frm.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        nav_frm.pack_propagate(False)
        
        nav_items = ["Home", "Pneumatic", "Colour Size", "Vision ROI", "Teaching CNN", "Dataset Capture", "Teaching YOLO", "AI Model Select", "Test Console", "Network"]
        self.nav_buttons = {}
        for item in nav_items:
            icon_img = self.nav_icons.get(item)
            btn = NavButton(nav_frm, item, icon_img, self.colors, command=lambda i=item: self._switch_page(i))
            btn.pack(fill=tk.X, pady=2)
            self.nav_buttons[item] = btn
            
        self.center_frm = tk.Frame(body_frm, bg=self.colors["surface"], bd=1, relief=tk.FLAT)
        self.center_frm.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(self.center_frm, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.video_label.bind("<Button-1>", self._on_mouse)
        self.video_label.bind("<B1-Motion>", self._on_mouse)
        self.video_label.bind("<ButtonRelease-1>", self._on_mouse)
        
        self.fps_lbl = tk.Label(self.video_label, text="FPS: 0.0", font=("Roboto", 14, "bold"), bg="black", fg=self.colors["info"])
        self.fps_lbl.place(relx=0.98, rely=0.02, anchor="ne")
        
        self.right_frm = tk.Frame(body_frm, bg=self.colors["bg"], width=480)
        self.right_frm.pack(side=tk.RIGHT, fill=tk.Y, padx=(15, 0))
        self.right_frm.pack_propagate(False)
        
        self.op_frm = tk.Frame(self.right_frm, bg=self.colors["surface"], padx=15, pady=15)
        self.op_frm.pack(fill=tk.X, pady=(0, 15))
        
        self.start_btn = ttk.Button(self.op_frm, text="START RUNNING", style="Success.TButton", command=self._on_start_stop)
        self.start_btn.pack(fill=tk.X, pady=(0, 10))
        
        # Conveyor and Feeder Controls Row
        cf_frm = tk.Frame(self.op_frm, bg=self.colors["surface"])
        cf_frm.pack(fill=tk.X, pady=(0, 10))
        cf_frm.columnconfigure(0, weight=1)
        cf_frm.columnconfigure(1, weight=1)

        self.conv_btn = tk.Button(
            cf_frm, text="CONVEYOR: OFF", font=("Roboto", 10, "bold"),
            bg=self.colors["surface_light"], fg=self.colors["text_muted"],
            activebackground=self.colors["surface_light"], activeforeground="white",
            bd=0, padx=6, pady=8, cursor="hand2", command=self._toggle_conv
        )
        self.conv_btn.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.feed_btn = tk.Button(
            cf_frm, text="FEEDER: OFF", font=("Roboto", 10, "bold"),
            bg=self.colors["surface_light"], fg=self.colors["text_muted"],
            activebackground=self.colors["surface_light"], activeforeground="white",
            bd=0, padx=6, pady=8, cursor="hand2", command=self._toggle_feed
        )
        self.feed_btn.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        health_frm = tk.Frame(self.op_frm, bg=self.colors["surface"])
        health_frm.pack(fill=tk.X, pady=(15, 0))
        
        self.cam_indicator = tk.Label(health_frm, text="● Camera", font=("Roboto", 10, "bold"), bg=self.colors["surface"], fg=self.colors["danger"])
        self.cam_indicator.pack(side=tk.LEFT)
        
        plc_name = "● Inovance PLC" if InovancePLCController and isinstance(self.controller, InovancePLCController) else "● Comm Board"
        self.ard_indicator = tk.Label(health_frm, text=plc_name, font=("Roboto", 10, "bold"), bg=self.colors["surface"], fg=self.colors["danger"])
        self.ard_indicator.pack(side=tk.RIGHT)


        self.page_container = tk.Frame(self.right_frm, bg=self.colors["bg"])
        self.page_container.pack(fill=tk.BOTH, expand=True)
        
        self._build_pages()
        self._build_teaching_console_page()
        self._build_colour_size_page()
        self._build_test_console_page()
        self._switch_page("Home")

    def _build_pages(self):
        # Shared Auto Capture State & Logic for multiple tabs
        self.golden_count_var = tk.StringVar(value="1000")
        self.capture_mode_var = tk.StringVar(value="cropped")
        
        def toggle_golden():
            if not self.shared_state.is_golden_capture_active():
                try:
                    count = int(self.golden_count_var.get())
                except:
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

        p_prod = tk.Frame(self.page_container, bg=self.colors["surface"], padx=15, pady=15)
        self.pages["Home"] = p_prod
        
        prod_hdr = tk.Frame(p_prod, bg=self.colors["surface"])
        prod_hdr.pack(fill="x", pady=(0,10))
        ttk.Label(prod_hdr, text="PRODUCTION HISTORY", style="Title.TLabel").pack(side="left")
        
        self.lbl_prod_count = tk.Label(
            prod_hdr, text="[ Detected: 0 ]",
            bg=self.colors["surface_light"], fg=self.colors["success"],
            font=("Roboto", 10, "bold"), padx=8, pady=2, relief="flat"
        )
        self.lbl_prod_count.pack(side="left", padx=(12, 0))
        
        def clear_history():
            with self.shared_state.lock:
                self.shared_state.recent_fires.clear()
                self.shared_state.reset_detected_count()
            self._last_fires_state = []
            self._last_det_cnt = 0
            for item in self.fire_tree.get_children():
                self.fire_tree.delete(item)
            if hasattr(self, 'lbl_prod_count'):
                self.lbl_prod_count.config(text="[ Detected: 0 ]")
                
        ttk.Button(prod_hdr, text="Clear History", command=clear_history).pack(side="right")
        
        self.golden_pneumatic_btn = tk.Button(prod_hdr, text="Auto Capture", 
                                              bg=self.colors["info"], fg="white", font=("Roboto", 9, "bold"),
                                              bd=0, cursor="hand2", padx=10, pady=5, command=toggle_golden)
        self.golden_pneumatic_btn.pack(side="right", padx=(5, 10))
        
        tree_frame = tk.Frame(p_prod, bg=self.colors["surface"])
        tree_frame.pack(fill=tk.BOTH, expand=True)

        cols = ("time", "track", "label", "conf", "act")
        self.fire_tree = ttk.Treeview(tree_frame, columns=cols, show='headings', height=18)
        self.fire_tree.heading("time", text="TIME"); self.fire_tree.column("time", width=70, anchor='center')
        self.fire_tree.heading("track", text="TRK"); self.fire_tree.column("track", width=50, anchor='center')
        self.fire_tree.heading("label", text="CLASS"); self.fire_tree.column("label", width=120, anchor='center')
        self.fire_tree.heading("conf", text="CONF"); self.fire_tree.column("conf", width=60, anchor='center')
        self.fire_tree.heading("act", text="ACT"); self.fire_tree.column("act", width=50, anchor='center')

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
            except: pass
            
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
                except:
                    self.insight_img_lbl.config(image="")
            else:
                self.insight_img_lbl.config(image="")

        def refresh_models():
            try:
                models = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pt") or f.endswith(".keras")]
                root_models = [f for f in os.listdir(MODEL_ROOT) if f.endswith(".pt") or f.endswith(".keras")]
                models.extend([m for m in root_models if m not in models])
                models.sort()
                
                model_combo['values'] = models
                active_name = os.path.basename(str(self.shared_state.yolo_model_path))
                if active_name in models:
                    self.model_combo_var.set(active_name)
                    load_insights(active_name)
                elif models:
                    self.model_combo_var.set(models[-1])
                    load_insights(models[-1])
            except: pass
            
        def on_model_select(event):
            selected = self.model_combo_var.get()
            if selected:
                load_insights(selected)

        def apply_model():
            selected = self.model_combo_var.get()
            if not selected: return
            
            full_path = os.path.join(MODELS_DIR, selected)
            if not os.path.exists(full_path):
                full_path = os.path.join(MODEL_ROOT, selected)
            
            with self.shared_state.lock:
                self.shared_state.yolo_model_path = full_path
                self.shared_state.save_to_config(CFG_PATH)
            if hasattr(self, 'processing_thread') and self.processing_thread:
                self.processing_thread.reload_model(full_path)
            
            messagebox.showinfo("Model Integrated", f"Successfully integrated model:\\n{selected}")
            
        model_combo.bind("<<ComboboxSelected>>", on_model_select)
        
        btn_apply = ttk.Button(model_frm, text="Apply Model", command=apply_model, style="Success.TButton")
        btn_apply.pack(side="left", padx=10)
        
        def delete_selected_model():
            selected = self.model_combo_var.get()
            if not selected:
                messagebox.showwarning("Warning", "No model selected.")
                return
            if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete {selected}?"):
                try:
                    full_path = os.path.join(MODELS_DIR, selected)
                    if os.path.exists(full_path): os.remove(full_path)
                    
                    base = os.path.splitext(selected)[0]
                    for ext in [".csv", ".png", ".keras", "_results.csv", "_confusion.png"]:
                        p = os.path.join(MODELS_DIR, f"{base}{ext}")
                        if os.path.exists(p): os.remove(p)
                        
                    refresh_models()
                    self.insight_lbl.config(text="Model deleted.")
                    self.insight_img_lbl.config(image="")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to delete model: {e}")

        def clear_all_models():
            if messagebox.askyesno("Confirm Clear All", "Are you sure you want to delete ALL uploaded models?"):
                try:
                    for f in os.listdir(MODELS_DIR):
                        p = os.path.join(MODELS_DIR, f)
                        if os.path.isfile(p): os.remove(p)
                    refresh_models()
                    self.insight_lbl.config(text="All models cleared.")
                    self.insight_img_lbl.config(image="")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to clear models: {e}")

        btn_delete = ttk.Button(upload_frm, text="Delete Selected", command=delete_selected_model, style="Danger.TButton")
        btn_delete.pack(side="left", padx=10)
        
        btn_clear = ttk.Button(upload_frm, text="Clear All", command=clear_all_models, style="Danger.TButton")
        btn_clear.pack(side="left", padx=10)
        
        prep_lbl = tk.Label(model_frm, text="Keras Preprocessing:", bg=self.colors["bg"], fg=self.colors["text"])
        prep_lbl.pack(side="left", padx=(20, 5))
        
        prep_combo_var = tk.StringVar(value=self.shared_state.keras_preprocess)
        prep_combo = ttk.Combobox(model_frm, textvariable=prep_combo_var, state="readonly", width=25)
        prep_combo['values'] = ["MobileNet / TM (-1 to 1)", "Standard (0 to 1)", "None (0 to 255)"]
        prep_combo.pack(side="left")
        
        def on_prep_change(event):
            with self.shared_state.lock:
                self.shared_state.keras_preprocess = prep_combo_var.get()
                self.shared_state.save_to_config(CFG_PATH)
                
        prep_combo.bind("<<ComboboxSelected>>", on_prep_change)
        
        def browse_model():
            path = filedialog.askopenfilename(title="Select Model file", filetypes=[("Model Files", "*.pt *.pth *.keras"), ("All Files", "*.*")])
            if path:
                try:
                    ext = os.path.splitext(path)[1].lower()
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    new_filename = f"model_{ts}{ext}"
                    dest_path = os.path.join(MODELS_DIR, new_filename)
                    shutil.copy2(path, dest_path)
                    
                    if ext == ".keras":
                        map_src = os.path.join(os.path.dirname(path), "cashew_class_mapping.json")
                        if os.path.exists(map_src):
                            shutil.copy2(map_src, os.path.join(MODELS_DIR, "cashew_class_mapping.json"))
                    else:
                        run_dir = os.path.dirname(os.path.dirname(path))
                        res_csv = os.path.join(run_dir, "results.csv")
                        conf_png = os.path.join(run_dir, "confusion_matrix.png")
                        base = f"model_{ts}"
                        if os.path.exists(res_csv):
                            shutil.copy2(res_csv, os.path.join(MODELS_DIR, f"{base}_results.csv"))
                        if os.path.exists(conf_png):
                            shutil.copy2(conf_png, os.path.join(MODELS_DIR, f"{base}_confusion.png"))
                    
                    with self.shared_state.lock:
                        self.shared_state.yolo_model_path = dest_path
                        self.shared_state.save_to_config(CFG_PATH)
                        
                    if hasattr(self, 'processing_thread') and self.processing_thread:
                        self.processing_thread.reload_model(dest_path)
                    
                    refresh_models()
                except Exception as e:
                    messagebox.showerror("Upload Error", f"Failed to upload model: {e}")
                    
        ttk.Button(upload_frm, text="UPLOAD MODEL FILE", style="Primary.TButton", command=browse_model).pack(side="left")
        refresh_models()

        p_vis = tk.Frame(self.page_container, bg=self.colors["surface"], padx=15, pady=15)
        self.pages["Vision ROI"] = p_vis
        ttk.Label(p_vis, text="VISION & ROI SETUP", style="Title.TLabel").pack(anchor="w", pady=(0,15))
        
        roi_grid = tk.Frame(p_vis, bg=self.colors["surface"])
        roi_grid.pack(fill=tk.X, pady=10)
        roi_grid.columnconfigure(0, weight=1); roi_grid.columnconfigure(1, weight=1)
        
        ttk.Button(roi_grid, text="Draw ROI 1", command=lambda: self._set_phase("ROI1_START")).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ttk.Button(roi_grid, text="Reset ROI 1", command=self.shared_state.reset_roi1).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(roi_grid, text="Draw ROI 2", command=lambda: self._set_phase("ROI2_START")).grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        ttk.Button(roi_grid, text="Reset ROI 2", command=self.shared_state.reset_roi2).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(roi_grid, text="Set Decision Line", style="Primary.TButton", command=lambda: self._set_phase("LINE_SET_MODE")).grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        ttk.Button(roi_grid, text="Reset Line", command=self.shared_state.reset_line).grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        
        man_frm = tk.Frame(p_vis, bg=self.colors["surface"])
        man_frm.pack(fill=tk.X, pady=15)
        self.manual_line_var = tk.StringVar()
        ttk.Button(man_frm, text="◀ Shift Left", command=lambda: self._shift_line(-5)).pack(side=tk.LEFT, padx=5)
        tk.Entry(man_frm, textvariable=self.manual_line_var, width=6, bg=self.colors["surface_light"], fg=self.colors["text"], bd=0, font=("Roboto", 12), justify="center").pack(side=tk.LEFT, padx=10, fill=tk.Y)
        ttk.Button(man_frm, text="Shift Right ▶", command=lambda: self._shift_line(5)).pack(side=tk.LEFT, padx=5)
        
        self.lbl_line_pos = ttk.Label(p_vis, text="Line X: Not Set", font=("Roboto", 10, "bold"), foreground=self.colors["warning"])
        self.lbl_line_pos.pack(pady=5)
        self.manual_line_var.trace_add("write", self._on_manual_line_var_write)
        self.shared_state.line_pixel_callback = self._on_line_change_callback
        
        ttk.Label(p_vis, text="IMAGE SENSOR CONTROL", style="Header.TLabel").pack(anchor="w", pady=(20,10))
        self.vars_cam = {
            "gain": tk.StringVar(value=str(self.shared_state.cam_gain)),
            "r": tk.StringVar(value=str(self.shared_state.cam_r)),
            "g": tk.StringVar(value=str(self.shared_state.cam_g)),
            "b": tk.StringVar(value=str(self.shared_state.cam_b)),
            "a_gain": tk.StringVar(value=str(self.shared_state.analog_gain)),
            "d_gain": tk.StringVar(value=str(self.shared_state.digital_gain)),
            "exposure": tk.StringVar(value=str(self.shared_state.exposure))
        }
        cam_grid = tk.Frame(p_vis, bg=self.colors["surface"])
        cam_grid.pack(fill=tk.X)
        labels = [("Global Gain", "gain"), ("R Gain", "r"), ("G Gain", "g"), ("B Gain", "b"), ("Exposure (µs)", "exposure")]
        for idx, (txt, key) in enumerate(labels):
            ttk.Label(cam_grid, text=txt, style="Surface.TLabel").grid(row=idx, column=0, sticky="w", pady=5)
            tk.Entry(cam_grid, textvariable=self.vars_cam[key], width=10, bg=self.colors["surface_light"], fg=self.colors["text"], bd=0).grid(row=idx, column=1, padx=20, pady=5)
            self.vars_cam[key].trace_add("write", self._on_cam_setting_change)
            
        def save_cam_settings():
            self._on_cam_setting_change()
            messagebox.showinfo("Saved", "Camera settings saved successfully.")
            
        ttk.Button(p_vis, text="SAVE CAMERA SETTINGS", style="Primary.TButton", command=save_cam_settings).pack(pady=(15, 0), fill=tk.X)
        
        def trigger_ffc():
            self.shared_state.trigger_ffc_calibration()
            messagebox.showinfo("FFC Calibration", "Captured White Reference.\\nFlat Field Correction is now active and saved.")
            
        ttk.Button(p_vis, text="White Calibration (FFC)", style="Primary.TButton", command=trigger_ffc).pack(pady=(15, 0), fill=tk.X)
        
        ttk.Button(p_vis, text="Test Image Capture", command=self.shared_state.trigger_capture).pack(pady=20, fill=tk.X)

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
        
        ttk.Label(sort_inner, text="MULTI-FRAME PRIORITY", style="Title.TLabel").pack(anchor="w", pady=(0,10))
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
        pn_controls = tk.Frame(p_pneu, bg=self.colors["surface"], width=300, padx=20, pady=20)
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
        
        ttk.Button(pn_controls, text="FIRE ALL", style="Danger.TButton", command=self._fire_all).pack(fill=tk.X, pady=(0, 10))
        ttk.Button(pn_controls, text="SEQUENCE ALL", style="Danger.TButton", command=self._sequence_all).pack(fill=tk.X, pady=(0, 10))
        ttk.Button(pn_controls, text="OPEN PLC DIAGNOSTICS", style="Primary.TButton", command=lambda: self.controller.open_diagnostics_gui() if hasattr(self.controller, 'open_diagnostics_gui') else None).pack(fill=tk.X, pady=(0, 10))

        
        ttk.Separator(pn_controls, orient="horizontal").pack(fill=tk.X, pady=20)
        
        self.master_ejector_var = tk.BooleanVar(value=True)
        self.shared_state.master_ejector_enabled = True
        
        def toggle_ejector():
            state = self.master_ejector_var.get()
            self.shared_state.master_ejector_enabled = state
            color = self.colors["success"] if state else self.colors["danger"]
            text = "MASTER EJECTOR: ON" if state else "MASTER EJECTOR: OFF"
            self.master_ej_btn.config(text=text)
            self.master_ej_led.config(fg=color, text="● Operating" if state else "● Fire Off")
        
        self.master_ej_btn = tk.Checkbutton(pn_controls, text="MASTER EJECTOR: ON", variable=self.master_ejector_var, command=toggle_ejector, bg=self.colors["surface"], fg="white", selectcolor=self.colors["surface_light"], font=("Roboto", 11, "bold"), indicatoron=False)
        self.master_ej_btn.pack(fill=tk.X, pady=(0, 10))
        
        led_frm = tk.Frame(pn_controls, bg=self.colors["surface"])
        led_frm.pack(fill=tk.X, pady=(0, 10))
        self.master_ej_led = tk.Label(led_frm, text="● Operating", font=("Roboto", 14, "bold"), bg=self.colors["surface"], fg=self.colors["success"])
        self.master_ej_led.pack(anchor="center")

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
        
        ttk.Label(p_net, text="Camera Identification", style="Header.TLabel").pack(anchor="w")
        self.lbl_cam_name = ttk.Label(p_net, text="Name: Scanning...", style="Surface.TLabel")
        self.lbl_cam_name.pack(anchor="w", pady=(5,0))
        self.lbl_cam_ip = ttk.Label(p_net, text="IP Address: Unknown", style="Surface.TLabel", foreground=self.colors["info"])
        self.lbl_cam_ip.pack(anchor="w", pady=(2,15))
        ttk.Button(p_net, text="Scan Network", command=self._scan_camera_ip).pack(anchor="w", pady=(0,20))
        
        ttk.Label(p_net, text="Static IP Configuration", style="Header.TLabel").pack(anchor="w")
        ttk.Label(p_net, text="New IP Address:", style="Surface.TLabel").pack(anchor="w", pady=(5,2))
        ttk.Button(p_net, text="Apply Static IP", style="Primary.TButton", command=self._set_camera_ip).pack(anchor="w", pady=(0, 15))

        # --- Inovance PLC (Modbus TCP) Configuration ---
        ttk.Separator(p_net, orient="horizontal").pack(fill=tk.X, pady=(10, 15))
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
        ttk.Button(pr4, text="Open PLC Diagnostics Tool", style="Warning.TButton", command=lambda: self.controller.open_diagnostics_gui() if hasattr(self.controller, 'open_diagnostics_gui') else None).pack(side=tk.LEFT)

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
            
        def launch_tcm_studio():
            import subprocess
            from tkinter import messagebox
            try:
                subprocess.Popen(["python3", "/home/phoenix/Documents/RaniVision_Latest/tcm_studio_pro.py"])
            except Exception as e:
                messagebox.showerror("Launch Error", f"Failed to start TCM Studio Pro: {e}")
                
        def launch_auto_labeler():
            import subprocess
            from tkinter import messagebox
            try:
                subprocess.Popen(["python3", "ai_auto_labeler.py"])
            except Exception as e:
                messagebox.showerror("Launch Error", f"Failed to start AI Auto-Labeler: {e}")
                
        ttk.Button(row2, text="Open Dataset Folders", style="Primary.TButton", command=open_dataset_folders).pack(side=tk.LEFT, padx=(0,10))
        ttk.Button(row2, text="Cross Check Labels (TCM)", style="Warning.TButton", command=launch_tcm_studio).pack(side=tk.LEFT, padx=(0,10))
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

    def _start_cnn_train(self):
        dataset_dir = os.path.abspath("dataset")
        if not os.path.exists(dataset_dir) or not os.listdir(dataset_dir):
            dataset_dir = os.path.abspath("../CNN_Experiment/dataset_crops")
        disabled_dir = os.path.abspath("dataset_disabled")
        
        # In case previous run crashed, restore disabled folders first
        if os.path.exists(disabled_dir):
            for d in os.listdir(disabled_dir):
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
                        
                    import subprocess, sys
                    cmd = ["gnome-terminal", "--wait", "--", "bash", "-c", f"cd ../CNN_Experiment && {sys.executable} train_cnn.py; read -p 'Press Enter to exit...'"]
                    process = subprocess.Popen(cmd)
                    process.wait()
                    
                except Exception as e:
                    print(f"Error starting CNN training: {e}")
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

    def _start_yolo_train(self, arch):
        self.is_training = True
        self.tcm_train_btn.config(state=tk.DISABLED)
        
        def train_thread():
            try:
                import yaml, shutil, glob, sys
                
                # Since we are doing YOLO Classification, the dataset is already in DATASET_DIR (cashew_dataset_images/ClassX/...)
                # YOLO classify requires a specific folder structure (train, val).
                # We will copy the dataset to a temp folder and split into train/val.
                
                temp_dataset_dir = os.path.abspath(os.path.join("runs", "temp_yolo_cls_dataset"))
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
                train_script = os.path.abspath("temp_train.py")
                res_file = os.path.abspath("temp_results.json")
                if os.path.exists(res_file): os.remove(res_file)
                
                # We determine if they picked YOLO11 or YOLOv8 based on the dropdown
                # Assuming 'yolo11n-cls.pt' for YOLO11, and 'yolov8n-cls.pt' for YOLOv8
                base_model = 'yolo11n-cls.pt' if '11' in arch else 'yolov8n-cls.pt'
                
                with open(train_script, "w") as f:
                    f.write(f"""
import sys, json, os
from ultralytics import YOLO

try:
    model = YOLO('{base_model}')
    results = model.train(
        data='{temp_dataset_dir}',
        epochs={self.tcm_epochs.get()},
        imgsz=224,
        verbose=True, workers=0
    )
    # Classification metrics
    if hasattr(results, 'top1'):
        top1 = getattr(results, 'top1', 0.0)
        top5 = getattr(results, 'top5', 0.0)
        with open('{res_file}', 'w') as rf:
            json.dump({{'top1': top1, 'top5': top5}}, rf)
            
except Exception as e:
    print(f"SUBPROCESS ERROR: {{e}}")
    sys.exit(1)
""")
                import subprocess
                cmd = ["gnome-terminal", "--wait", "--", "bash", "-c", f"{sys.executable} {train_script}; read -p 'Press Enter to exit...'"]
                process = subprocess.Popen(cmd)
                process.wait()
                
                if process.returncode == 0 and os.path.exists(res_file):
                    with open(res_file, 'r') as rf:
                        r = json.load(rf)
                    summary = f"\n\n--- FINAL RESULTS ---\nTop-1 Accuracy: {r.get('top1', 0)*100:.2f}%\nTop-5 Accuracy: {r.get('top5', 0)*100:.2f}%\n---------------------\n"
                    print(summary)
                else:
                    print(f"\nTraining Process Exited with Code {process.returncode}!\n")
                
            except Exception as e:
                print(f"\nError: {e}\n")
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
        
        def launch_tcm_studio():
            import subprocess
            try:
                subprocess.Popen(["python3", "/home/phoenix/Documents/RaniVision_Latest/tcm_studio_pro.py"])
            except Exception as e:
                messagebox.showerror("Launch Error", f"Failed to start TCM Studio Pro: {e}")
                
        ttk.Button(row3, text="Cross Check Labels", style="Primary.TButton", command=launch_tcm_studio).pack(side=tk.LEFT, padx=(0,10))
        
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
                import sys
                if "/home/phoenix/Documents/RaniVision_Latest" not in sys.path:
                    sys.path.append("/home/phoenix/Documents/RaniVision_Latest")
                try:
                    import tcm_magic_wand
                    pts = tcm_magic_wand.get_magic_polygon(frame, ox, oy)
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
                    import traceback
                    traceback.print_exc()
                    print(f"[AUTO-TEACH] Error: Magic Wand failed: {e}")
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
            except: pass
            
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
            except: pass
            
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
        
        poly = tcm_magic_wand.get_magic_polygon(self.ann_current_img, orig_x, orig_y)
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
        
        active_model = getattr(self.shared_state, 'yolo_model_path', "yolo11n.pt")
        if not active_model: active_model = "yolo11n.pt"
        self.test_model_path = tk.StringVar(value=active_model)
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

    def _test_load_folder(self):
        from tkinter import filedialog
        import glob
        
        folder = filedialog.askdirectory()
        if not folder: return
        
        model_path = self.test_model_path.get()
        if not os.path.exists(model_path):
            self.test_status_lbl.config(text=f"Model not found: {model_path}")
            return
            
        if self.test_model is None or self.test_current_model_path != model_path:
            try:
                if str(model_path).endswith('.keras'):
                    import keras
                    if not hasattr(keras.layers.Dense, "_patched"):
                        k_original_dense = keras.layers.Dense.from_config
                        @classmethod
                        def k_patched_dense(cls, config):
                            if "quantization_config" in config:
                                del config["quantization_config"]
                            return k_original_dense.__func__(cls, config)
                        keras.layers.Dense.from_config = k_patched_dense
                        keras.layers.Dense._patched = True
                        
                    self.test_model = keras.models.load_model(model_path, compile=False)
                    import json
                    map_path = os.path.join(os.path.dirname(model_path), "cashew_class_mapping.json")
                    if os.path.exists(map_path):
                        with open(map_path, "r") as f:
                            mapping = json.load(f)
                            self.test_class_map = {int(k): str(v) for k, v in mapping.items()}
                    else:
                        self.test_class_map = None
                        
                    if getattr(self, 'test_yolo_detector', None) is None:
                        try:
                            from ultralytics import YOLO
                            best_yolo = find_best_weights(MODEL_ROOT)
                            if best_yolo and str(best_yolo).endswith('.pt'):
                                self.test_yolo_detector = YOLO(str(best_yolo))
                            else:
                                self.test_yolo_detector = YOLO("yolo11n.pt")
                        except:
                            self.test_yolo_detector = None
                else:
                    from ultralytics import YOLO
                    self.test_model = YOLO(model_path)
                    self.test_class_map = {int(k): str(v) for k, v in self.test_model.names.items()}
                    self.test_yolo_detector = None
                self.test_current_model_path = model_path
            except Exception as e:
                self.test_status_lbl.config(text=f"Model error: {e}")
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
        from tkinter import filedialog
        path = filedialog.askopenfilename(filetypes=[("Media Files", "*.jpg *.jpeg *.png *.mp4 *.avi")])
        if not path: return
        
        model_path = self.test_model_path.get()
        if not os.path.exists(model_path):
            self.test_status_lbl.config(text=f"Model not found: {model_path}")
            return
            
        if self.test_model is None or self.test_current_model_path != model_path:
            try:
                if str(model_path).endswith('.keras'):
                    import tensorflow as tf
                    if not hasattr(tf.keras.layers.Dense, "_patched"):
                        original_dense_from_config = tf.keras.layers.Dense.from_config
                        @classmethod
                        def patched_dense_from_config(cls, config):
                            if "quantization_config" in config:
                                del config["quantization_config"]
                            return original_dense_from_config.__func__(cls, config)
                        tf.keras.layers.Dense.from_config = patched_dense_from_config
                        tf.keras.layers.Dense._patched = True
                        
                    import keras
                    if not hasattr(keras.layers.Dense, "_patched"):
                        k_original_dense = keras.layers.Dense.from_config
                        @classmethod
                        def k_patched_dense(cls, config):
                            if "quantization_config" in config:
                                del config["quantization_config"]
                            return k_original_dense.__func__(cls, config)
                        keras.layers.Dense.from_config = k_patched_dense
                        keras.layers.Dense._patched = True
                        
                    self.test_model = tf.keras.models.load_model(model_path, compile=False)
                    import json
                    map_path = os.path.join(os.path.dirname(model_path), "cashew_class_mapping.json")
                    if os.path.exists(map_path):
                        with open(map_path, "r") as f:
                            mapping = json.load(f)
                            self.test_class_map = {int(k): str(v) for k, v in mapping.items()}
                    else:
                        self.test_class_map = None
                        
                    if getattr(self, 'test_yolo_detector', None) is None:
                        try:
                            from ultralytics import YOLO
                            best_yolo = find_best_weights(MODEL_ROOT)
                            if best_yolo and str(best_yolo).endswith('.pt'):
                                self.test_yolo_detector = YOLO(str(best_yolo))
                            else:
                                self.test_yolo_detector = YOLO("yolo11n.pt")
                        except:
                            self.test_yolo_detector = None
                else:
                    from ultralytics import YOLO
                    self.test_model = YOLO(model_path)
                    self.test_class_map = {int(k): str(v) for k, v in self.test_model.names.items()}
                    self.test_yolo_detector = None
                self.test_current_model_path = model_path
            except Exception as e:
                self.test_status_lbl.config(text=f"Model error: {e}")
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
            
        if page_name in ["Teaching YOLO", "Pneumatic", "Teaching CNN", "Colour Size"]:
            self.op_frm.pack_forget()
        else:
            self.op_frm.pack(fill=tk.X, pady=(0, 15), before=self.page_container)
            
        # Full Screen Logic for specific pages
        if page_name in ["Pneumatic", "Teaching CNN", "Colour Size"]:
            if hasattr(self, 'center_frm'):
                self.center_frm.pack_forget()
            if hasattr(self, 'right_frm'):
                self.right_frm.pack_forget()
                self.right_frm.config(width=900)
                self.right_frm.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(15, 0))
        else:
            if hasattr(self, 'center_frm'):
                self.center_frm.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            if hasattr(self, 'right_frm'):
                self.right_frm.pack_forget()
                self.right_frm.config(width=480)
                self.right_frm.pack(side=tk.RIGHT, fill=tk.Y, padx=(15, 0))

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

    def _shift_line(self, amount):
        try:
            val = int(self.manual_line_var.get()) + amount
            self.manual_line_var.set(str(val))
        except: pass

    def _on_manual_line_var_write(self, *args):
        try:
            val = int(self.manual_line_var.get())
            with self.shared_state.lock:
                self.shared_state.line_x_full = val
        except: pass

    def _on_line_change_callback(self, txt, val):
        self.lbl_line_pos.config(text=txt)
        if val is not None and str(val) != self.manual_line_var.get():
            self.manual_line_var.set(str(val))
            
    def _on_cam_setting_change(self, *args):
        try:
            with self.shared_state.lock:
                self.shared_state.cam_gain = float(self.vars_cam["gain"].get())
                self.shared_state.cam_r = float(self.vars_cam["r"].get())
                self.shared_state.cam_g = float(self.vars_cam["g"].get())
                self.shared_state.cam_b = float(self.vars_cam["b"].get())
                self.shared_state.analog_gain = float(self.vars_cam["a_gain"].get())
                self.shared_state.digital_gain = float(self.vars_cam["d_gain"].get())
                self.shared_state.exposure = int(self.vars_cam["exposure"].get())
                self.shared_state.cam_settings_changed = True
            self.shared_state.save_to_config(CFG_PATH)
        except Exception as e:
            print(f"[UI] Error saving camera settings: {e}")
    def _fire_all(self):
        try:
            pulse = int(self.shared_state.common_pulse_ms)
        except Exception:
            pulse = 200

        if hasattr(self.controller, 'fire_all'):
            self.controller.fire_all(pulse)
        else:
            def fire():
                for i in range(1, NUM_ACTUATORS+1):
                    if self.controller and self.controller.connected:
                        self.controller.pulse_actuator(i, 0, pulse) 
            threading.Thread(target=fire, daemon=True).start()
            
    def _sequence_all(self):
        def seq():
            try:
                pulse = int(self.shared_state.common_pulse_ms)
            except Exception:
                pulse = 200
            for i in range(1, NUM_ACTUATORS+1):
                if self.controller and self.controller.connected:
                    self.controller.pulse_actuator(i, 0, pulse)
                if hasattr(self, 'pn_leds') and i in self.pn_leds:
                    try:
                        self.pn_leds[i].config(fg=self.colors["success"])
                        self.after(pulse, lambda ii=i: self.pn_leds[ii].config(fg=self.colors["surface_light"]))
                    except: pass
                time.sleep(0.12)
        import threading
        threading.Thread(target=seq, daemon=True).start()

        
    def _pulse_and_flash(self, act_idx):
        try:
            pulse = int(self.shared_state.common_pulse_ms)
        except:
            pulse = 200

        if self.controller and self.controller.connected:
            self.controller.pulse_actuator(act_idx, 0, pulse)
            
        if hasattr(self, 'pn_leds') and act_idx in self.pn_leds:
            try:
                # Turn ON (Green) when firing
                self.pn_leds[act_idx].config(fg=self.colors["success"]) 
                
                default_color = self.colors["surface_light"] # OFF state
                
                # Turn OFF after pulse duration (min 200ms for visibility)
                delay = max(pulse, 200)
                self.after(delay, lambda a=act_idx, c=default_color: self.pn_leds[a].config(fg=c))
            except Exception as e:
                print(f"[DEBUG] _pulse_and_flash error: {e}")

    def _update_common_pulse(self, *args):
        try:
            pulse = int(self.common_pulse_var.get())
            self.shared_state.set_common_pulse(pulse)
        except: pass

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
            
            # ROI 1 (1..20)
            for act_id in range(1, 21):
                dist = self.shared_state.distances_mm[act_id - 1]
                if dist > 0:
                    if act_id == 1:
                        d = int(round((dist / speed) * 1000.0))
                    else:
                        d = self.shared_state.delays_ms[act_id - 2] + int(round((dist / speed) * 1000.0))
                    self.act_vars[act_id].set(str(d))
                    self.shared_state.set_delay(act_id, d)
            
            # ROI 2 (21..40)
            for act_id in range(21, 41):
                dist = self.shared_state.distances_mm[act_id - 1]
                if dist > 0:
                    if act_id == 21:
                        d = int(round((dist / speed) * 1000.0))
                    else:
                        d = self.shared_state.delays_ms[act_id - 2] + int(round((dist / speed) * 1000.0))
                    self.act_vars[act_id].set(str(d))
                    self.shared_state.set_delay(act_id, d)
            messagebox.showinfo("Recalculate", f"All Delays Recalculated for Speed {speed:.1f} mm/s!")
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
            import traceback
            traceback.print_exc()
            print(f"Error saving config: {e}")

    def _update_class_conf(self, cname, var):
        try:
            val = int(var.get())
            self.shared_state.set_class_conf(cname, val)
        except: pass

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
            if st["roi1"] and st["roi2"] and st["line_x"]:
                self.shared_state.phase = "RUNNING"
            else:
                messagebox.showwarning("Incomplete", "Please draw ROI1, ROI2 and set Decision Line in Vision Setup.")

    def _scan_camera_ip(self):
        if mvsdk is None: return
        devs = mvsdk.CameraEnumerateDevice()
        if devs:
            dev = devs[0]
            try:
                ip_info = mvsdk.CameraGigeGetIp(dev)
                self.lbl_cam_name.config(text=f"Name: {dev.GetFriendlyName()} (SN: {dev.GetSn()})")
                self.lbl_cam_ip.config(text=f"IP Address: {ip_info[0]}")
            except Exception as e:
                self.lbl_cam_ip.config(text=f"Error reading IP: {e}")
        else:
            self.lbl_cam_name.config(text="Name: NOT FOUND")
            self.lbl_cam_ip.config(text="IP Address: ---")
            messagebox.showwarning("Not Found", "No camera detected.")

    def _set_camera_ip(self):
        if mvsdk is None: return
        new_ip = self.new_ip_var.get().strip()
        devs = mvsdk.CameraEnumerateDevice()
        if devs:
            try:
                err = mvsdk.CameraGigeSetIp(devs[0], new_ip, "255.255.255.0", "192.168.3.1", 1)
                if err == 0:
                    messagebox.showinfo("Success", f"IP changed to {new_ip}! REPLUG the camera.")
                    self._scan_camera_ip()
                else:
                    messagebox.showerror("Error", f"Failed. Code: {err}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

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
        
        c_conn = self.camera_thread.is_connected
        a_conn = self.controller.is_connected()
        self.cam_indicator.config(fg=self.colors["success"] if c_conn else self.colors["danger"])
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
        
        self.fps_lbl.config(text=f"FPS: {self.camera_thread.fps:.1f}")

        self.lbl_clock.config(text=datetime.now().strftime("%H:%M:%S"))
        
        current_fires = self.shared_state.get_recent_fires()
        if current_fires != self._last_fires_state:
            self.fire_tree.delete(*self.fire_tree.get_children())
            for row in current_fires:
                self.fire_tree.insert("", "end", values=row)
            self._last_fires_state = list(current_fires)

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
                except: pass
            
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

    def _add_label_class(self):
        import tkinter.simpledialog as sd
        new_cname = sd.askstring("Add Class", "Enter new class name:", parent=self)
        if new_cname:
            new_cname = new_cname.strip()
            if new_cname and new_cname not in self.all_class_names:
                try:
                    keys = [int(k) for k in self.shared_state.capture_classes.keys() if str(k).isdigit()]
                    new_id = str(max(keys + [-1]) + 1)
                    self.shared_state.capture_classes[new_id] = new_cname
                    self.shared_state.save_to_config(CFG_PATH)
                    self.all_class_names.append(new_cname)
                    
                    # Update class_map and JSON
                    c_keys = [int(k) for k in self.class_map.keys()]
                    new_cid = int(max(c_keys + [-1]) + 1)
                    self.class_map[new_cid] = new_cname
                    import json
                    with open("cashew_class_mapping.json", "w") as f:
                        json.dump(self.class_map, f, indent=4)
                        
                    self._refresh_class_grid()
                    messagebox.showinfo("Success", f"Added class '{new_cname}'")
                except Exception as e:
                    messagebox.showerror("Error", f"Could not add class: {e}")
            elif new_cname in self.all_class_names:
                messagebox.showwarning("Warning", f"Class '{new_cname}' already exists!")

    def _rename_label_class(self):
        cname = self.ann_fast_class_var.get()
        if not cname: 
            messagebox.showwarning("Warning", "No class selected!")
            return
        import tkinter.simpledialog as sd
        new_cname = sd.askstring("Rename Class", f"Rename {cname} to:", parent=self)
        if new_cname:
            new_cname = new_cname.strip()
            if new_cname and new_cname not in self.all_class_names:
                try:

                    # Rename in config mapping
                    for k, v in self.shared_state.capture_classes.items():
                        if v == cname:
                            self.shared_state.capture_classes[k] = new_cname
                    # If it wasn't in capture_classes (e.g. YOLO class), we should add it so it persists
                    if new_cname not in self.shared_state.capture_classes.values():
                        keys = [int(k) for k in self.shared_state.capture_classes.keys() if str(k).isdigit()]
                        new_id = str(max(keys + [-1]) + 1)
                        self.shared_state.capture_classes[new_id] = new_cname
                    
                    self.shared_state.save_to_config(CFG_PATH)
                    self.all_class_names = [new_cname if c == cname else c for c in self.all_class_names]
                    
                    # Update class_map and JSON
                    for k, v in self.class_map.items():
                        if v == cname:
                            self.class_map[k] = new_cname
                    import json
                    with open("cashew_class_mapping.json", "w") as f:
                        json.dump(self.class_map, f, indent=4)
                        
                    self.ann_fast_class_var.set(new_cname)
                    self._refresh_class_grid()
                    messagebox.showinfo("Success", f"Renamed to '{new_cname}'")
                except Exception as e:
                    messagebox.showerror("Error", f"Could not rename class: {e}")
            elif new_cname in self.all_class_names:
                messagebox.showwarning("Warning", f"Class '{new_cname}' already exists!")

    def _delete_label_class(self):
        # Batch-delete all checked classes
        to_delete = [cn for cn, var in self._class_check_vars.items() if var.get()]
        if not to_delete:
            messagebox.showwarning("Warning", "No classes selected! Check the boxes first.")
            return
        names_str = ", ".join(to_delete)
        if not messagebox.askyesno("Confirm", f"Delete {len(to_delete)} class(es)?\n\n{names_str}"):
            return
        try:
            for cname in to_delete:
                # Remove from capture_classes
                keys_to_del = [k for k, v in self.shared_state.capture_classes.items() if v == cname]
                for k in keys_to_del:
                    del self.shared_state.capture_classes[k]
                # Remove from all_class_names
                self.all_class_names = [c for c in self.all_class_names if c != cname]
                # Remove from class_map
                del_keys = [k for k, v in self.class_map.items() if v == cname]
                for k in del_keys:
                    del self.class_map[k]

            self.shared_state.save_to_config(CFG_PATH)
            import json
            with open("cashew_class_mapping.json", "w") as f:
                json.dump(self.class_map, f, indent=4)

            self.ann_fast_class_var.set("")
            self._refresh_class_grid()
            messagebox.showinfo("Success", f"Deleted {len(to_delete)} class(es): {names_str}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not delete classes: {e}")

    def _on_close(self):
        print("[SYSTEM] Initiating shutdown...")
        self.stop_event.set()
        self.controller.stop()
        self.shared_state.save_to_config(CFG_PATH)
        
        # Give daemon threads a chance to gracefully release hardware (Aravis/TF)
        if hasattr(self, 'camera_thread') and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1.0)
        if hasattr(self, 'processing_thread') and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
            
        print("[SYSTEM] Shutdown complete.")
        self.destroy()


def main():
    import traceback
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
            except: pass
        traceback.print_exc()
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
