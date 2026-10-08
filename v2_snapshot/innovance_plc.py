#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
INOVANCE EC320 PLC - 40-RELAY MODBUS TCP CONTROLLER GUI
WITH FULL INDIVIDUAL RELAY REGISTER / COIL MAPPING
=============================================================================
Model: Inovance EC320 Series PLC
Protocol: Modbus TCP (Port 502)
Target IP: 192.168.1.88
Total Relays: 40 Channels (Bank A: 1-20, Bank B: 21-40)
Register Mapping: EACH RELAY INDIVIDUALLY CONFIGURABLE (Coil / Holding Register)
=============================================================================
"""

import os
import sys
import json
import time
import threading
import heapq
import subprocess
from datetime import datetime

try:
    import customtkinter as ctk
    from tkinter import messagebox
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
except Exception:
    ctk = None
    messagebox = None

try:
    from pyModbusTCP.client import ModbusClient
except ImportError:
    ModbusClient = None

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inovance_config.json")



# =============================================================================
# THREAD-SAFE MODBUS TCP DRIVER FOR INOVANCE EC320
# =============================================================================
class InovanceModbusDriver:
    def __init__(self, host="192.168.1.88", port=502, unit_id=1, timeout=2.0):
        self.host = host
        self.port = port
        self.unit_id = unit_id
        self.timeout = timeout
        self.client = None
        self.connected = False
        self.lock = threading.Lock()

    def connect(self, host=None, port=None, unit_id=None):
        with self.lock:
            if host: self.host = host
            if port: self.port = int(port)
            if unit_id is not None: self.unit_id = int(unit_id)

            if ModbusClient is None:
                raise RuntimeError("pyModbusTCP library is not installed. Please run: pip install pyModbusTCP")

            if self.client:
                try:
                    self.client.close()
                except Exception:
                    pass

            self.client = ModbusClient(
                host=self.host,
                port=self.port,
                unit_id=self.unit_id,
                auto_open=True,
                auto_close=False,
                timeout=0.5
            )
            ok = self.client.open()
            if ok and hasattr(self.client, '_sock') and self.client._sock:
                import socket
                try:
                    self.client._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                except Exception:
                    pass
            self.connected = bool(ok and self.client.is_open)
            return self.connected

    def disconnect(self):
        with self.lock:
            if self.client:
                try:
                    self.client.close()
                except Exception:
                    pass
            self.connected = False

    def is_connected(self):
        with self.lock:
            return bool(self.client and self.client.is_open)

    # --- COIL OPERATIONS (FC 01 / FC 05 / FC 15) ---
    def write_coil(self, addr: int, state: bool) -> bool:
        with self.lock:
            if not self.client or not self.client.is_open:
                return False
            if hasattr(self.client, '_sock') and self.client._sock:
                import socket
                try:
                    self.client._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                except Exception:
                    pass
            return bool(self.client.write_single_coil(addr, state))

    def write_coils(self, start_addr: int, states: list) -> bool:
        with self.lock:
            if not self.client or not self.client.is_open:
                return False
            return bool(self.client.write_multiple_coils(start_addr, states))

    def read_coils(self, start_addr: int, count: int):
        with self.lock:
            if not self.client or not self.client.is_open:
                return None
            return self.client.read_coils(start_addr, count)

    # --- HOLDING REGISTER OPERATIONS (FC 03 / FC 06 / FC 16) ---
    def write_register(self, addr: int, val: int) -> bool:
        with self.lock:
            if not self.client or not self.client.is_open:
                return False
            if hasattr(self.client, '_sock') and self.client._sock:
                import socket
                try:
                    self.client._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                except Exception:
                    pass
            return bool(self.client.write_single_register(addr, int(val)))

    def write_registers(self, start_addr: int, vals: list) -> bool:
        with self.lock:
            if not self.client or not self.client.is_open:
                return False
            return bool(self.client.write_multiple_registers(start_addr, [int(v) for v in vals]))

    def read_registers(self, start_addr: int, count: int):
        with self.lock:
            if not self.client or not self.client.is_open:
                return None
            return self.client.read_holding_registers(start_addr, count)


# =============================================================================
# INOVANCE PLC CONTROLLER (PRODUCTION GRADE DROP-IN CONTROLLER)
# =============================================================================
class InovancePLCController:
    """
    Industrial Modbus TCP Controller for Inovance EC320 PLC (40 Relays).
    Provides microsecond-accurate time-delayed pulse scheduling for high-speed pneumatic sorting,
    automatic background reconnection, batch controls, and configuration persistence.
    Acts as a direct drop-in replacement for ArduinoMegaController in cashew_cnn_grader.py.
    """
    TOTAL_RELAYS = 48

    def __init__(self, config_file: str = CONFIG_FILE):
        self.config_file = config_file
        self.plc_ip = "192.168.1.88"
        self.plc_port = 502
        self.plc_unit_id = 1
        self.relay1_register = 100  # Default M100 for Channel 1
        self.register_type = "Coil (FC01/05/15)"
        self.pulse_duration = 25  # ms default
        self.relay_addresses = {r: 100 + (r - 1) for r in range(1, self.TOTAL_RELAYS + 1)}

        # Load persisted config
        self.load_config()

        # Thread-safe Modbus Driver
        self.driver = InovanceModbusDriver(self.plc_ip, self.plc_port, self.plc_unit_id)

        # Threading & Event Heap
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.event_cond = threading.Condition()
        self.event_heap = []  # [(timestamp, seq, relay_id, state), ...]
        self._seq_counter = 0

        # High-speed pulse state tracking with automatic gap protection for consecutive cashews
        self.min_gap_sec = 0.025  # 25ms minimum OFF gap between consecutive pulses on same relay
        self.relay_end_times = {}  # {relay_id: (t_on, t_off, seq_off)}
        self.relay_active_count = {}   # {relay_id: int}

        self._conn_thread = None
        self._dispatcher_thread = None
        self._diag_process = None

    @property
    def connected(self) -> bool:
        return self.is_connected()

    def is_connected(self) -> bool:
        return self.driver.is_connected()

    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    data = json.load(f)
                    self.plc_ip = data.get("ip", self.plc_ip)
                    self.plc_port = int(data.get("port", self.plc_port))
                    self.plc_unit_id = int(data.get("unit_id", self.plc_unit_id))
                    self.relay1_register = int(data.get("relay1_register", self.relay1_register))
                    self.register_type = data.get("register_type", self.register_type)
                    self.pulse_duration = int(data.get("pulse_duration", self.pulse_duration))

                    saved_addresses = data.get("relay_addresses", {})
                    for k, v in saved_addresses.items():
                        try:
                            self.relay_addresses[int(k)] = int(v)
                        except Exception:
                            pass
            except Exception as e:
                print(f"[PLC] Error loading config ({self.config_file}): {e}")

    def save_config(self):
        data = {
            "ip": self.plc_ip,
            "port": self.plc_port,
            "unit_id": self.plc_unit_id,
            "relay1_register": self.relay1_register,
            "register_type": self.register_type,
            "pulse_duration": self.pulse_duration,
            "relay_addresses": {str(k): int(v) for k, v in self.relay_addresses.items()}
        }
        try:
            with open(self.config_file, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"[PLC] Error saving config ({self.config_file}): {e}")

    def get_relay_address(self, relay_id: int) -> int:
        return self.relay_addresses.get(relay_id, self.relay1_register + (relay_id - 1))

    def start(self):
        self.stop_event.clear()
        if self._conn_thread is None or not self._conn_thread.is_alive():
            self._conn_thread = threading.Thread(target=self._connection_loop, daemon=True, name="PLC-ConnWatchdog")
            self._conn_thread.start()

        if self._dispatcher_thread is None or not self._dispatcher_thread.is_alive():
            self._dispatcher_thread = threading.Thread(target=self._dispatcher_loop, daemon=True, name="PLC-PulseDispatcher")
            self._dispatcher_thread.start()

        print(f"[PLC] Inovance EC320 Controller Started. Target: {self.plc_ip}:{self.plc_port} (Unit {self.plc_unit_id})")

    def stop(self):
        self.stop_event.set()
        with self.event_cond:
            self.event_cond.notify_all()

        try:
            self.all_off()
        except Exception:
            pass

        if self._conn_thread:
            try: self._conn_thread.join(timeout=0.5)
            except Exception: pass
        if self._dispatcher_thread:
            try: self._dispatcher_thread.join(timeout=0.5)
            except Exception: pass

        self.driver.disconnect()
        print("[PLC] Inovance EC320 Controller Stopped.")

    def _connection_loop(self):
        """Monitors Modbus TCP connection and auto-reconnects in the background."""
        while not self.stop_event.is_set():
            if not self.driver.is_connected():
                try:
                    ok = self.driver.connect(self.plc_ip, self.plc_port, self.plc_unit_id)
                    if ok:
                        print(f"[PLC] Successfully Connected to Inovance PLC at {self.plc_ip}:{self.plc_port}")
                except Exception:
                    pass
            self.stop_event.wait(2.0)

    def _dispatcher_loop(self):
        """High-precision event dispatcher for scheduled ejector pulses."""
        while not self.stop_event.is_set():
            event = None
            with self.event_cond:
                while not self.event_heap and not self.stop_event.is_set():
                    self.event_cond.wait(timeout=0.05)

                if self.stop_event.is_set():
                    break

                now = time.time()
                earliest_time = self.event_heap[0][0]
                if earliest_time > now:
                    wait_sec = min(0.005, earliest_time - now)
                    self.event_cond.wait(timeout=wait_sec)
                    continue

                event = heapq.heappop(self.event_heap)

            if event:
                t_sched, seq, relay_id, state = event
                self._write_relay(relay_id, state)

    def _write_relay(self, relay_id: int, state: bool) -> bool:
        if not self.driver.is_connected():
            return False
        addr = self.get_relay_address(relay_id)
        is_coil = "Coil" in self.register_type
        try:
            if is_coil:
                return self.driver.write_coil(addr, state)
            else:
                return self.driver.write_register(addr, 1 if state else 0)
        except Exception as e:
            print(f"[PLC] Write error for R{relay_id:02d} (Addr {addr}): {e}")
            return False

    def pulse_actuator(self, idx: int, delay_ms: int, pulse_duration_ms: int):
        """
        Schedules a relay pulse for actuator idx (1..40).
        delay_ms: conveyor travel time before solenoid energizes.
        pulse_duration_ms: duration solenoid stays open.
        Enforces a clean OFF gap between consecutive cashews so all pulses fire individually.
        """
        if not (1 <= idx <= self.TOTAL_RELAYS):
            return

        now = time.time()
        d_sec = max(0.0, float(delay_ms) / 1000.0)
        p_sec = max(0.005, float(pulse_duration_ms) / 1000.0)
        t_on = now + d_sec
        t_off = t_on + p_sec

        with self.event_cond:
            # Consecutive cashew protection on same relay:
            # If previous pulse on this relay would overlap or leave no gap,
            # shorten the previous pulse to ensure at least min_gap_sec OFF time before new pulse
            if idx in self.relay_end_times:
                prev_on, prev_off, prev_seq = self.relay_end_times[idx]
                if prev_off > (t_on - self.min_gap_sec):
                    new_prev_off = max(prev_on + 0.015, t_on - self.min_gap_sec)
                    for i, ev in enumerate(self.event_heap):
                        if ev[1] == prev_seq:
                            self.event_heap[i] = (new_prev_off, ev[1], ev[2], False)
                            heapq.heapify(self.event_heap)
                            break

            self._seq_counter += 1
            seq_on = self._seq_counter
            heapq.heappush(self.event_heap, (t_on, seq_on, idx, True))

            self._seq_counter += 1
            seq_off = self._seq_counter
            heapq.heappush(self.event_heap, (t_off, seq_off, idx, False))
            self.relay_end_times[idx] = (t_on, t_off, seq_off)
            self.event_cond.notify()

    def set_relay(self, idx: int, state: bool):
        """Manually sets single relay state immediately."""
        if 1 <= idx <= self.TOTAL_RELAYS:
            with self.event_cond:
                self.relay_end_times.pop(idx, None)
            threading.Thread(target=self._write_relay, args=(idx, state), daemon=True).start()

    def set_conveyor(self, state: bool):
        """Controls Conveyor motor/drive (Relay 41 / Coil M140)."""
        self.set_relay(41, state)

    def set_feeder(self, state: bool):
        """Controls Cashew Feeder vibrator (Relay 42 / Coil M141)."""
        self.set_relay(42, state)

    def all_off(self):
        """Emergency stop: clears pending pulses and sets all 40 relays OFF immediately."""
        with self.event_cond:
            self.event_heap.clear()
            self.relay_end_times.clear()
            self.relay_active_count.clear()

        if not self.driver.is_connected():
            return

        is_coil = "Coil" in self.register_type
        addrs = [self.get_relay_address(r) for r in range(1, self.TOTAL_RELAYS + 1)]
        is_contiguous = (addrs == list(range(addrs[0], addrs[0] + len(addrs))))

        try:
            if is_contiguous:
                start_addr = addrs[0]
                if is_coil:
                    self.driver.write_coils(start_addr, [False] * self.TOTAL_RELAYS)
                else:
                    self.driver.write_registers(start_addr, [0] * self.TOTAL_RELAYS)
            else:
                for r in range(1, self.TOTAL_RELAYS + 1):
                    self._write_relay(r, False)
            print("[PLC] All 40 Relays Switched OFF (Emergency Stop).")
        except Exception as e:
            print(f"[PLC] Error executing all_off: {e}")

    def fire_all(self, pulse_ms: int = 200):
        """Pulses all 40 actuators simultaneously."""
        for r in range(1, self.TOTAL_RELAYS + 1):
            self.pulse_actuator(r, 0, pulse_ms)

    def sequence_all(self, interval_ms: int = 80, pulse_ms: int = 150):
        """Sequences all 40 actuators one after another."""
        def _run_seq():
            for r in range(1, self.TOTAL_RELAYS + 1):
                self.pulse_actuator(r, 0, pulse_ms)
                time.sleep(interval_ms / 1000.0)
        threading.Thread(target=_run_seq, daemon=True).start()

    def update_config(self, ip: str = None, port: int = None, unit_id: int = None,
                      start_reg: int = None, reg_type: str = None, relay_addresses: dict = None):
        """Updates PLC parameters and triggers reconnect if IP/port changed."""
        reconnect_needed = False
        if ip and ip != self.plc_ip:
            self.plc_ip = ip
            reconnect_needed = True
        if port and int(port) != self.plc_port:
            self.plc_port = int(port)
            reconnect_needed = True
        if unit_id is not None and int(unit_id) != self.plc_unit_id:
            self.plc_unit_id = int(unit_id)
            reconnect_needed = True
        if start_reg is not None:
            self.relay1_register = int(start_reg)
        if reg_type:
            self.register_type = reg_type
        if relay_addresses:
            self.relay_addresses.update({int(k): int(v) for k, v in relay_addresses.items()})

        self.save_config()

        if reconnect_needed:
            self.driver.disconnect()
            threading.Thread(target=self.driver.connect, args=(self.plc_ip, self.plc_port, self.plc_unit_id), daemon=True).start()

    def open_diagnostics_gui(self):
        """Launches the dedicated Inovance 40-Relay Diagnostic GUI tool."""
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "innovance_plc.py")
        try:
            self._diag_process = subprocess.Popen([sys.executable, script_path])
            return True
        except Exception as e:
            print(f"[PLC] Failed to launch diagnostic GUI: {e}")
            return False


# =============================================================================
# INDIVIDUAL REGISTER MAPPING MODAL DIALOG
# =============================================================================
class RelayMappingDialog(ctk.CTkToplevel if ctk else object):

    """
    Dedicated Window to configure and visualize all 40 relay Modbus addresses.
    Supports individual editing and batch auto-filling (sequential).
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("Inovance EC320 - Individual 40-Relay Address Mapping")
        self.geometry("920x680")
        self.minsize(800, 600)
        self.transient(parent)
        self.grab_set()

        self.entries = {}  # relay_id -> CTkEntry

        self.build_dialog()

    def build_dialog(self):
        # Header
        top_frame = ctk.CTkFrame(self, fg_color="#0e1626", corner_radius=8)
        top_frame.pack(fill="x", padx=12, pady=10)

        ctk.CTkLabel(
            top_frame,
            text="INDIVIDUAL RELAY REGISTER / COIL MAPPING",
            font=("Arial", 16, "bold"),
            text_color="#00e5ff"
        ).pack(anchor="w", padx=12, pady=(8, 2))

        ctk.CTkLabel(
            top_frame,
            text=f"Current Register Type: {self.parent.register_type} | Configure the exact Modbus address for each relay below.",
            font=("Arial", 11),
            text_color="#94a3b8"
        ).pack(anchor="w", padx=12, pady=(0, 8))

        # Quick Sequential Auto-Fill Bar
        tool_frame = ctk.CTkFrame(self, fg_color="#182238", corner_radius=8)
        tool_frame.pack(fill="x", padx=12, pady=(0, 10))

        ctk.CTkLabel(tool_frame, text="Quick Sequential Fill:", font=("Arial", 12, "bold"), text_color="#f59e0b").pack(side="left", padx=12, pady=8)
        ctk.CTkLabel(tool_frame, text="Start Addr:", font=("Arial", 11), text_color="#cbd5e1").pack(side="left", padx=(4, 2))
        
        self.entry_start_addr = ctk.CTkEntry(tool_frame, width=70, font=("Consolas", 12), justify="center")
        self.entry_start_addr.insert(0, str(self.parent.relay1_register))
        self.entry_start_addr.pack(side="left", padx=4)

        ctk.CTkButton(
            tool_frame,
            text="Fill All (1-40)",
            width=100,
            fg_color="#0284c7",
            hover_color="#0369a1",
            font=("Arial", 11, "bold"),
            command=lambda: self.batch_fill_range(1, 40)
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            tool_frame,
            text="Fill Bank A (1-20)",
            width=120,
            fg_color="#334155",
            hover_color="#475569",
            font=("Arial", 11),
            command=lambda: self.batch_fill_range(1, 20)
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            tool_frame,
            text="Fill Bank B (21-40)",
            width=120,
            fg_color="#334155",
            hover_color="#475569",
            font=("Arial", 11),
            command=lambda: self.batch_fill_range(21, 40)
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            tool_frame,
            text="Reset 0-39",
            width=80,
            fg_color="#475569",
            hover_color="#64748b",
            font=("Arial", 11),
            command=self.reset_default
        ).pack(side="right", padx=12)

        # 2-Column Scrollable Grid for All 40 Relays
        scroll_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll_frame.pack(fill="both", expand=True, padx=12, pady=4)
        scroll_frame.columnconfigure(0, weight=1)
        scroll_frame.columnconfigure(1, weight=1)

        # Bank A (1-20) on Left
        frame_a = ctk.CTkFrame(scroll_frame, fg_color="#111827", corner_radius=8)
        frame_a.grid(row=0, column=0, sticky="nsew", padx=6, pady=4)
        ctk.CTkLabel(frame_a, text="BANK A (Relays 01 - 20)", font=("Arial", 13, "bold"), text_color="#38bdf8").pack(pady=8)

        grid_a = ctk.CTkFrame(frame_a, fg_color="transparent")
        grid_a.pack(fill="both", expand=True, padx=8, pady=4)
        for i in range(1, 21):
            r = (i - 1) % 10
            c = 0 if i <= 10 else 1
            self.create_entry_cell(grid_a, i, r, c)

        # Bank B (21-40) on Right
        frame_b = ctk.CTkFrame(scroll_frame, fg_color="#111827", corner_radius=8)
        frame_b.grid(row=0, column=1, sticky="nsew", padx=6, pady=4)
        ctk.CTkLabel(frame_b, text="BANK B (Relays 21 - 40)", font=("Arial", 13, "bold"), text_color="#38bdf8").pack(pady=8)

        grid_b = ctk.CTkFrame(frame_b, fg_color="transparent")
        grid_b.pack(fill="both", expand=True, padx=8, pady=4)
        for i in range(21, 41):
            r = (i - 21) % 10
            c = 0 if i <= 30 else 1
            self.create_entry_cell(grid_b, i, r, c)

        # Bottom Buttons: Cancel & Save
        bot_bar = ctk.CTkFrame(self, fg_color="#0e1626", corner_radius=8)
        bot_bar.pack(fill="x", padx=12, pady=10)

        ctk.CTkButton(
            bot_bar,
            text="CANCEL",
            width=100,
            fg_color="#4b5563",
            hover_color="#374151",
            command=self.destroy
        ).pack(side="right", padx=8, pady=8)

        ctk.CTkButton(
            bot_bar,
            text="SAVE & APPLY MAPPING",
            width=180,
            fg_color="#16a34a",
            hover_color="#15803d",
            font=("Arial", 12, "bold"),
            command=self.save_and_apply
        ).pack(side="right", padx=8, pady=8)

    def create_entry_cell(self, parent, relay_id: int, row: int, col: int):
        cell = ctk.CTkFrame(parent, fg_color="#1e293b", corner_radius=4)
        cell.grid(row=row, column=col, padx=4, pady=3, sticky="ew")

        ctk.CTkLabel(
            cell,
            text=f"R{relay_id:02d}:",
            font=("Consolas", 11, "bold"),
            text_color="#f8fafc",
            width=40
        ).pack(side="left", padx=(6, 2), pady=3)

        entry = ctk.CTkEntry(cell, width=64, height=24, font=("Consolas", 11), justify="center")
        current_addr = self.parent.get_relay_address(relay_id)
        entry.insert(0, str(current_addr))
        entry.pack(side="left", padx=(2, 6), pady=3)
        self.entries[relay_id] = entry

    def batch_fill_range(self, start_id: int, end_id: int):
        try:
            base = int(self.entry_start_addr.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Invalid starting address.", parent=self)
            return

        for idx, r_id in enumerate(range(start_id, end_id + 1)):
            if r_id in self.entries:
                self.entries[r_id].delete(0, "end")
                self.entries[r_id].insert(0, str(base + idx))

    def reset_default(self):
        for r_id in range(1, 41):
            if r_id in self.entries:
                self.entries[r_id].delete(0, "end")
                self.entries[r_id].insert(0, str(r_id - 1))

    def save_and_apply(self):
        new_map = {}
        for r_id in range(1, 41):
            try:
                addr = int(self.entries[r_id].get().strip())
                if addr < 0 or addr > 65535:
                    raise ValueError(f"Relay {r_id} address out of range (0-65535)")
                new_map[r_id] = addr
            except ValueError as e:
                messagebox.showerror("Validation Error", f"Invalid address for Relay R{r_id:02d}! Error: {e}", parent=self)
                return

        # Apply to parent
        self.parent.apply_full_mapping(new_map)
        self.destroy()


# =============================================================================
# MAIN UI APPLICATION
# =============================================================================
class InovanceEjectorApp(ctk.CTk):
    TOTAL_RELAYS = 40

    def __init__(self):
        super().__init__()

        self.title("Inovance EC320 PLC - 40 Relay Ejector Diagnostics & Control")
        self.geometry("1620x960")
        self.minsize(1360, 820)

        # Default Settings
        self.plc_ip = "192.168.1.88"
        self.plc_port = 502
        self.plc_unit_id = 1
        self.relay1_register = 100  # Starting address reference (M100 for ROI 01)
        self.register_type = "Coil (FC01/05/15)"  # or "Holding Reg (FC03/06/16)"
        self.pulse_duration = 200  # ms
        self.seq_interval = 60  # ms
        self.live_poll_enabled = False

        # Individual Relay Addresses Map: {1: 0, 2: 1, ..., 40: 39}
        self.relay_addresses = {}

        # Load persisted config if available
        self.load_config()

        # Modbus Driver
        self.driver = InovanceModbusDriver(self.plc_ip, self.plc_port, self.plc_unit_id)

        # UI State Tracking
        self.relay_indicators = {}    # relay_id -> CTkLabel (LED)
        self.relay_addr_entries = {}  # relay_id -> CTkEntry (Editable on-card)
        self.relay_type_labels = {}   # relay_id -> CTkLabel (C: / R:)
        self.relay_states = [False] * self.TOTAL_RELAYS
        self.sequence_running = False
        self.poll_thread = None
        self.poll_active = True
        self.pulse_gen = {}           # relay_id -> generation ID to prevent pulse from turning off manual ON
        self.btn_on_map = {}          # relay_id -> CTkButton
        self.btn_off_map = {}         # relay_id -> CTkButton

        # Build GUI
        self.build_gui()

        # Start background polling thread
        self.manual_held_relays = set()
        self.hold_thread = threading.Thread(target=self._hold_alive_worker, daemon=True)
        self.hold_thread.start()

        self.poll_thread = threading.Thread(target=self._polling_worker, daemon=True)
        self.poll_thread.start()

        # Connect automatically in background
        self.after(500, self.connect_plc)

    def _hold_alive_worker(self):
        """Continuously holds True every 120ms on relays that are manually set to ON.
        Overrides the Inovance PLC internal ladder timer auto-reset, ensuring the relay
        stays ON indefinitely with ZERO time gap until the user explicitly clicks OFF."""
        while self.poll_active:
            if self.driver.is_connected() and self.manual_held_relays:
                active_ids = list(self.manual_held_relays)
                is_coil = "Coil" in self.register_type
                for r in active_ids:
                    addr = self.get_relay_address(r)
                    try:
                        if is_coil:
                            self.driver.write_coil(addr, True)
                        else:
                            self.driver.write_register(addr, 1)
                    except Exception:
                        pass
            time.sleep(0.12)

    # =========================================================================
    # CONFIGURATION STORAGE
    # =========================================================================
    def load_config(self):
        # Default initialization (sequential 0..39)
        self.relay_addresses = {r: (self.relay1_register + r - 1) for r in range(1, self.TOTAL_RELAYS + 1)}

        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    data = json.load(f)
                    self.plc_ip = data.get("ip", self.plc_ip)
                    self.plc_port = data.get("port", self.plc_port)
                    self.plc_unit_id = data.get("unit_id", self.plc_unit_id)
                    self.relay1_register = data.get("relay1_register", self.relay1_register)
                    self.register_type = data.get("register_type", self.register_type)
                    self.pulse_duration = data.get("pulse_duration", self.pulse_duration)

                    saved_addresses = data.get("relay_addresses", {})
                    for k, v in saved_addresses.items():
                        try:
                            self.relay_addresses[int(k)] = int(v)
                        except Exception:
                            pass
            except Exception as e:
                print(f"[WARN] Failed to load config: {e}")

    def save_config(self):
        data = {
            "ip": self.plc_ip,
            "port": self.plc_port,
            "unit_id": self.plc_unit_id,
            "relay1_register": self.relay1_register,
            "register_type": self.register_type,
            "pulse_duration": self.pulse_duration,
            "relay_addresses": {str(k): int(v) for k, v in self.relay_addresses.items()}
        }
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"[WARN] Failed to save config: {e}")

    # =========================================================================
    # INDIVIDUAL ADDRESS ACCESSOR & APPLIER
    # =========================================================================
    def get_relay_address(self, relay_id: int) -> int:
        """Returns the specific Modbus address configured for 1-based relay_id"""
        return self.relay_addresses.get(relay_id, self.relay1_register + (relay_id - 1))

    def apply_full_mapping(self, new_map: dict):
        """Updates all 40 relay mappings simultaneously from the modal dialog"""
        self.relay_addresses.update(new_map)
        self.save_config()

        # Update all UI card entry boxes
        for r_id, addr in new_map.items():
            if r_id in self.relay_addr_entries:
                entry = self.relay_addr_entries[r_id]
                entry.delete(0, "end")
                entry.insert(0, str(addr))

        self.log_msg("SUCCESS: Full 40-Relay Address Mapping updated and saved.")

    def on_relay_addr_changed(self, relay_id: int):
        """Called when user edits address directly on an individual relay card"""
        entry = self.relay_addr_entries.get(relay_id)
        if not entry:
            return
        val_str = entry.get().strip()
        try:
            addr = int(val_str)
            if addr < 0 or addr > 65535:
                raise ValueError("Out of range")

            old_addr = self.relay_addresses.get(relay_id)
            if old_addr != addr:
                self.relay_addresses[relay_id] = addr
                self.save_config()
                self.log_msg(f"INDIVIDUAL MAP: R{relay_id:02d} -> Modbus Addr {addr} ({self.register_type})")
        except ValueError:
            # Revert to last valid address
            current = self.get_relay_address(relay_id)
            entry.delete(0, "end")
            entry.insert(0, str(current))
            self.log_msg(f"[ERROR] Invalid address for R{relay_id:02d}. Must be integer 0 to 65535.")

    # =========================================================================
    # GUI LAYOUT
    # =========================================================================
    def build_gui(self):
        # ---------------- 1. TOP HEADER / CONTROL BAR ----------------
        top_bar = ctk.CTkFrame(self, fg_color="#0e1626", corner_radius=10)
        top_bar.pack(fill="x", padx=12, pady=(10, 6))

        # Title & Subtitle
        title_box = ctk.CTkFrame(top_bar, fg_color="transparent")
        title_box.pack(side="left", padx=14, pady=10)
        ctk.CTkLabel(
            title_box,
            text="INOVANCE EC320 PLC",
            font=("Arial", 19, "bold"),
            text_color="#00e5ff"
        ).pack(anchor="w")
        ctk.CTkLabel(
            title_box,
            text="40-Relay Modbus Controller | Individual Register Mapping",
            font=("Arial", 11),
            text_color="#8da2fb"
        ).pack(anchor="w")

        # --- CONNECTION CONTROLS ---
        conn_box = ctk.CTkFrame(top_bar, fg_color="#182238", corner_radius=8)
        conn_box.pack(side="left", padx=8, pady=8)

        # IP
        ctk.CTkLabel(conn_box, text="IP:", font=("Arial", 12, "bold"), text_color="#d1d5db").pack(side="left", padx=(10, 4))
        self.entry_ip = ctk.CTkEntry(conn_box, width=120, font=("Consolas", 12))
        self.entry_ip.insert(0, self.plc_ip)
        self.entry_ip.pack(side="left", padx=3)

        # Port
        ctk.CTkLabel(conn_box, text="Port:", font=("Arial", 12, "bold"), text_color="#d1d5db").pack(side="left", padx=(6, 4))
        self.entry_port = ctk.CTkEntry(conn_box, width=54, font=("Consolas", 12))
        self.entry_port.insert(0, str(self.plc_port))
        self.entry_port.pack(side="left", padx=3)

        # Unit ID
        ctk.CTkLabel(conn_box, text="ID:", font=("Arial", 12, "bold"), text_color="#d1d5db").pack(side="left", padx=(6, 4))
        self.entry_unit = ctk.CTkEntry(conn_box, width=40, font=("Consolas", 12))
        self.entry_unit.insert(0, str(self.plc_unit_id))
        self.entry_unit.pack(side="left", padx=3)

        # Connect Button
        self.btn_connect = ctk.CTkButton(
            conn_box,
            text="CONNECT",
            width=85,
            fg_color="#0284c7",
            hover_color="#0369a1",
            font=("Arial", 11, "bold"),
            command=self.connect_plc
        )
        self.btn_connect.pack(side="left", padx=(8, 4))

        # Status Badge
        self.lbl_status = ctk.CTkLabel(
            conn_box,
            text="DISCONNECTED",
            font=("Arial", 11, "bold"),
            text_color="#ef4444",
            width=110
        )
        self.lbl_status.pack(side="left", padx=(4, 10))

        # --- REGISTER TYPE & GLOBAL MAPPING BUTTON ---
        mapping_box = ctk.CTkFrame(top_bar, fg_color="#182238", corner_radius=8)
        mapping_box.pack(side="left", padx=8, pady=8)

        # Register Type dropdown
        self.combo_reg_type = ctk.CTkComboBox(
            mapping_box,
            values=["Coil (FC01/05/15)", "Holding Reg (FC03/06/16)"],
            width=165,
            font=("Arial", 11),
            command=self.on_reg_type_changed
        )
        self.combo_reg_type.set(self.register_type)
        self.combo_reg_type.pack(side="left", padx=(8, 6))

        # Dedicated "RELAY MAPPING (ALL 40)" Button
        btn_mapping_modal = ctk.CTkButton(
            mapping_box,
            text="MAPPING TABLE (40)",
            width=140,
            fg_color="#7c3aed",
            hover_color="#6d28d9",
            font=("Arial", 11, "bold"),
            command=self.open_mapping_dialog
        )
        btn_mapping_modal.pack(side="left", padx=(2, 8))

        # --- PULSE DURATION SETTING ---
        pulse_box = ctk.CTkFrame(top_bar, fg_color="#182238", corner_radius=8)
        pulse_box.pack(side="left", padx=8, pady=8)

        ctk.CTkLabel(pulse_box, text="Pulse:", font=("Arial", 12, "bold"), text_color="#10b981").pack(side="left", padx=(8, 4))
        self.entry_pulse = ctk.CTkEntry(pulse_box, width=75, font=("Consolas", 12), justify="center")
        self.entry_pulse.insert(0, str(self.pulse_duration))
        self.entry_pulse.pack(side="left", padx=2)
        self.entry_pulse.bind("<Return>", self.update_pulse_from_entry)

        self.slider_pulse = ctk.CTkSlider(
            pulse_box,
            from_=10,
            to=100000,
            width=80,
            command=self.update_pulse_from_slider
        )
        self.slider_pulse.set(self.pulse_duration)
        self.slider_pulse.pack(side="left", padx=4)

        self.lbl_pulse_val = ctk.CTkLabel(pulse_box, text=f"{self.pulse_duration}ms", font=("Arial", 10), width=75)
        self.lbl_pulse_val.pack(side="left", padx=(0, 6))

        # --- GLOBAL EMERGENCY STOP ---
        self.btn_estop = ctk.CTkButton(
            top_bar,
            text="ALL 40 OFF (E-STOP)",
            font=("Arial", 12, "bold"),
            fg_color="#dc2626",
            hover_color="#991b1b",
            width=105,
            height=46,
            command=self.emergency_stop_all
        )
        self.btn_estop.pack(side="right", padx=12, pady=8)

        # ---------------- 2. MAIN RELAY BANKS (BANK A & BANK B) ----------------
        content_frame = ctk.CTkFrame(self, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=12, pady=4)
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)

        # Left: Bank A (Relays 1 - 20)
        self.build_bank_panel(content_frame, "ROI 01 / TRACK 1 (M100 - M119)", start_id=1, end_id=20, col=0)

        # Right: Bank B (Relays 21 - 40)
        self.build_bank_panel(content_frame, "ROI 02 / TRACK 2 (M120 - M139)", start_id=21, end_id=40, col=1)

        # ---------------- 3. FOOTER / LOG CONSOLE ----------------
        footer = ctk.CTkFrame(self, fg_color="#0e1626", height=130, corner_radius=10)
        footer.pack(fill="x", padx=12, pady=(6, 10))

        foot_header = ctk.CTkFrame(footer, fg_color="transparent")
        foot_header.pack(fill="x", padx=12, pady=(6, 2))

        ctk.CTkLabel(foot_header, text="COMMUNICATION LOG & DIAGNOSTICS", font=("Arial", 11, "bold"), text_color="#00e5ff").pack(side="left")

        # Live poll toggle switch
        self.switch_poll = ctk.CTkSwitch(
            foot_header,
            text="Live Read (Auto-Poll PLC)",
            font=("Arial", 11),
            command=self.toggle_live_polling
        )
        self.switch_poll.pack(side="right", padx=10)
        if self.live_poll_enabled:
            self.switch_poll.select()

        # Clear log button
        ctk.CTkButton(
            foot_header,
            text="Clear Log",
            width=70,
            height=22,
            font=("Arial", 10),
            fg_color="#374151",
            hover_color="#4b5563",
            command=self.clear_log
        ).pack(side="right", padx=6)

        # Log Text Box
        self.log_box = ctk.CTkTextbox(footer, height=85, font=("Consolas", 11), fg_color="#0a0f1d", text_color="#93c5fd")
        self.log_box.pack(fill="both", expand=True, padx=12, pady=(2, 8))
        self.log_msg("Ready. Inovance EC320 Controller Initialized with Individual Relay Register Mapping.")

    # =========================================================================
    # BANK PANEL BUILDER
    # =========================================================================
    def build_bank_panel(self, parent, title: str, start_id: int, end_id: int, col: int):
        panel = ctk.CTkFrame(parent, fg_color="#111827", corner_radius=10)
        panel.grid(row=0, column=col, sticky="nsew", padx=6, pady=4)

        # Panel Header with Action Buttons
        head = ctk.CTkFrame(panel, fg_color="#1f2937", corner_radius=8, height=44)
        head.pack(fill="x", padx=8, pady=8)

        ctk.CTkLabel(
            head,
            text=title,
            font=("Arial", 14, "bold"),
            text_color="#38bdf8"
        ).pack(side="left", padx=12)

        # Bank Action Buttons
        ctk.CTkButton(
            head,
            text="ALL OFF",
            width=70,
            height=28,
            fg_color="#ef4444",
            hover_color="#dc2626",
            font=("Arial", 11, "bold"),
            command=lambda: self.bank_action(start_id, end_id, 0)
        ).pack(side="right", padx=4)

        ctk.CTkButton(
            head,
            text="SEQUENCE",
            width=85,
            height=28,
            fg_color="#0284c7",
            hover_color="#0369a1",
            font=("Arial", 11, "bold"),
            command=lambda: self.bank_sequence(start_id, end_id)
        ).pack(side="right", padx=4)

        ctk.CTkButton(
            head,
            text="FIRE ALL",
            width=80,
            height=28,
            fg_color="#f59e0b",
            hover_color="#d97706",
            text_color="#000000",
            font=("Arial", 11, "bold"),
            command=lambda: self.bank_fire_all(start_id, end_id)
        ).pack(side="right", padx=4)

        # Scrollable Grid for 20 Relays (10 rows x 2 cols)
        scroll_grid = ctk.CTkScrollableFrame(panel, fg_color="transparent")
        scroll_grid.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        scroll_grid.columnconfigure(0, weight=1)
        scroll_grid.columnconfigure(1, weight=1)

        for idx, relay_id in enumerate(range(start_id, end_id + 1)):
            grid_col = 0 if idx < 10 else 1
            grid_row = idx if idx < 10 else (idx - 10)
            self.create_relay_card(scroll_grid, relay_id, grid_row, grid_col)

    def create_relay_card(self, parent, relay_id: int, row: int, col: int):
        card = ctk.CTkFrame(parent, fg_color="#182238", corner_radius=6, height=48)
        card.grid(row=row, column=col, sticky="ew", padx=4, pady=3)

        # Relay Title Badge
        ctk.CTkLabel(
            card,
            text=f"R{relay_id:02d}",
            font=("Arial", 12, "bold"),
            text_color="#f3f4f6",
            width=36
        ).pack(side="left", padx=(6, 2))

        # Status Indicator LED
        ind = ctk.CTkLabel(
            card,
            text="●",
            font=("Arial", 22),
            text_color="#374151",  # Off
            width=20
        )
        ind.pack(side="left", padx=2)
        self.relay_indicators[relay_id] = ind

        # Register Prefix Label (C: or R:)
        prefix = "C:" if "Coil" in self.register_type else "R:"
        lbl_type = ctk.CTkLabel(
            card,
            text=prefix,
            font=("Consolas", 10, "bold"),
            text_color="#38bdf8",
            width=16
        )
        lbl_type.pack(side="left", padx=(2, 0))
        self.relay_type_labels[relay_id] = lbl_type

        # Modbus Address Entry (INDIVIDUALLY EDITABLE ON-CARD)
        modbus_addr = self.get_relay_address(relay_id)
        entry_addr = ctk.CTkEntry(
            card,
            width=54,
            height=26,
            font=("Consolas", 11, "bold"),
            justify="center",
            fg_color="#0a0f1d",
            border_color="#334155",
            border_width=1
        )
        entry_addr.insert(0, str(modbus_addr))
        entry_addr.pack(side="left", padx=(1, 4))
        self.relay_addr_entries[relay_id] = entry_addr

        # Update address immediately when user presses Enter or clicks away
        entry_addr.bind("<Return>", lambda e, r=relay_id: self.on_relay_addr_changed(r))
        entry_addr.bind("<FocusOut>", lambda e, r=relay_id: self.on_relay_addr_changed(r))

        # Action Buttons: ON (Stays ON until OFF clicked), OFF, PULSE
        b_on = ctk.CTkButton(
            card,
            text="ON",
            width=38,
            height=26,
            fg_color="#16a34a",
            hover_color="#15803d",
            font=("Arial", 10, "bold"),
            command=lambda r=relay_id: self.set_relay(r, True)
        )
        b_on.pack(side="left", padx=2)
        self.btn_on_map[relay_id] = b_on

        b_off = ctk.CTkButton(
            card,
            text="OFF",
            width=38,
            height=26,
            fg_color="#4b5563",
            hover_color="#374151",
            font=("Arial", 10, "bold"),
            command=lambda r=relay_id: self.set_relay(r, False)
        )
        b_off.pack(side="left", padx=2)
        self.btn_off_map[relay_id] = b_off

        ctk.CTkButton(
            card,
            text="PULSE",
            width=46,
            height=26,
            fg_color="#ea580c",
            hover_color="#c2410c",
            font=("Arial", 10, "bold"),
            command=lambda r=relay_id: self.pulse_relay(r)
        ).pack(side="left", padx=(2, 4))

    # =========================================================================
    # UI EVENT HANDLERS
    # =========================================================================
    def open_mapping_dialog(self):
        RelayMappingDialog(self)

    def on_reg_type_changed(self, choice):
        self.register_type = choice
        self.save_config()
        prefix = "C:" if "Coil" in self.register_type else "R:"
        for r_id in range(1, self.TOTAL_RELAYS + 1):
            if r_id in self.relay_type_labels:
                self.relay_type_labels[r_id].configure(text=prefix)
        self.log_msg(f"Modbus Register Type switched to: {choice}")

    def update_pulse_from_slider(self, val):
        ms = int(val)
        self.pulse_duration = ms
        self.entry_pulse.delete(0, "end")
        self.entry_pulse.insert(0, str(ms))
        self.lbl_pulse_val.configure(text=f"{ms}ms")

    def update_pulse_from_entry(self, event=None):
        try:
            ms = int(self.entry_pulse.get().strip())
            ms = max(10, min(ms, 100000))
            self.pulse_duration = ms
            self.slider_pulse.set(ms)
            self.lbl_pulse_val.configure(text=f"{ms}ms")
            self.save_config()
        except Exception:
            self.entry_pulse.delete(0, "end")
            self.entry_pulse.insert(0, str(self.pulse_duration))

    def toggle_live_polling(self):
        self.live_poll_enabled = bool(self.switch_poll.get())
        state = "ENABLED" if self.live_poll_enabled else "DISABLED"
        self.log_msg(f"Live status polling {state}.")

    def clear_log(self):
        self.log_box.delete("1.0", "end")

    def log_msg(self, msg: str):
        now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        line = f"[{now}] {msg}\n"
        self.log_box.insert("end", line)
        self.log_box.see("end")

    def update_indicator(self, relay_id: int, active: bool):
        if relay_id in self.relay_indicators:
            color = "#00e676" if active else "#374151"
            try:
                self.relay_indicators[relay_id].configure(text_color=color)
            except Exception:
                pass
        if 1 <= relay_id <= self.TOTAL_RELAYS:
            self.relay_states[relay_id - 1] = active

    # =========================================================================
    # PLC CONNECTION LOGIC
    # =========================================================================
    def connect_plc(self):
        threading.Thread(target=self._connect_worker, daemon=True).start()

    def _connect_worker(self):
        ip = self.entry_ip.get().strip()
        try:
            port = int(self.entry_port.get().strip())
            unit = int(self.entry_unit.get().strip())
        except ValueError:
            self.log_msg("Invalid Port or Unit ID.")
            return

        self.lbl_status.configure(text="CONNECTING...", text_color="#f59e0b")
        self.btn_connect.configure(state="disabled")

        try:
            connected = self.driver.connect(host=ip, port=port, unit_id=unit)
            if connected:
                self.plc_ip = ip
                self.plc_port = port
                self.plc_unit_id = unit
                self.save_config()
                self.lbl_status.configure(text="CONNECTED", text_color="#10b981")
                self.btn_connect.configure(text="DISCONNECT", fg_color="#dc2626", hover_color="#b91c1c", command=self.disconnect_plc)
                self.log_msg(f"CONNECTED to Inovance EC320 at {ip}:{port} (Unit ID: {unit})")
            else:
                self.lbl_status.configure(text="FAILED", text_color="#ef4444")
                self.log_msg(f"Connection FAILED to {ip}:{port}. Check network/cable.")
        except Exception as e:
            self.lbl_status.configure(text="ERROR", text_color="#ef4444")
            self.log_msg(f"Connection Exception: {e}")
        finally:
            self.btn_connect.configure(state="normal")

    def disconnect_plc(self):
        self.driver.disconnect()
        self.lbl_status.configure(text="DISCONNECTED", text_color="#ef4444")
        self.btn_connect.configure(text="CONNECT", fg_color="#0284c7", hover_color="#0369a1", command=self.connect_plc)
        self.log_msg("Disconnected from PLC.")

    # =========================================================================
    # RELAY CONTROL WORKERS
    # =========================================================================
    def set_relay(self, relay_id: int, state: bool):
        """Sets an individual relay state ON or OFF using its mapped address.
        When state is True (ON), it stays ON indefinitely with NO time gap until state is False (OFF).
        Cancels any active pulse on this relay."""
        self.pulse_gen[relay_id] = self.pulse_gen.get(relay_id, 0) + 1  # Cancel any sleeping pulse
        if state:
            self.manual_held_relays.add(relay_id)
        else:
            self.manual_held_relays.discard(relay_id)
        self.update_indicator(relay_id, state)
        
        # Visual highlight on ON/OFF buttons
        if relay_id in self.btn_on_map and relay_id in self.btn_off_map:
            if state:
                self.btn_on_map[relay_id].configure(fg_color="#22c55e", text="ON ✓")
                self.btn_off_map[relay_id].configure(fg_color="#374151")
            else:
                self.btn_on_map[relay_id].configure(fg_color="#16a34a", text="ON")
                self.btn_off_map[relay_id].configure(fg_color="#ef4444")
                self.after(600, lambda r=relay_id: self.btn_off_map[r].configure(fg_color="#4b5563") if r in self.btn_off_map else None)

        threading.Thread(target=self._write_relay_worker, args=(relay_id, state), daemon=True).start()

    def _write_relay_worker(self, relay_id: int, state: bool):
        if not self.driver.is_connected():
            self.log_msg(f"Cannot send R{relay_id:02d}: PLC Not Connected.")
            return

        addr = self.get_relay_address(relay_id)
        is_coil = "Coil" in self.register_type

        try:
            if is_coil:
                ok = self.driver.write_coil(addr, state)
            else:
                ok = self.driver.write_register(addr, 1 if state else 0)

            res = "OK" if ok else "FAIL"
            cmd_name = "ON" if state else "OFF"
            self.log_msg(f"WRITE R{relay_id:02d} (Addr {addr}) -> {cmd_name} [{res}]")
            if not ok:
                self.update_indicator(relay_id, not state)
        except Exception as e:
            self.log_msg(f"Error writing R{relay_id:02d}: {e}")

    def pulse_relay(self, relay_id: int):
        """Fires relay for configured pulse duration (up to 100000ms) then turns OFF"""
        gen = self.pulse_gen.get(relay_id, 0) + 1
        self.pulse_gen[relay_id] = gen
        threading.Thread(target=self._pulse_worker, args=(relay_id, gen), daemon=True).start()

    def _pulse_worker(self, relay_id: int, gen: int):
        addr = self.get_relay_address(relay_id)
        is_coil = "Coil" in self.register_type
        pulse_s = self.pulse_duration / 1000.0

        self.update_indicator(relay_id, True)
        t_end = time.time() + pulse_s

        while time.time() < t_end:
            if self.pulse_gen.get(relay_id) != gen or relay_id in self.manual_held_relays:
                return  # Aborted or user clicked manual ON
            if self.driver.is_connected():
                try:
                    if is_coil:
                        self.driver.write_coil(addr, True)
                    else:
                        self.driver.write_register(addr, 1)
                except Exception as e:
                    self.log_msg(f"Pulse ON err R{relay_id:02d}: {e}")
            rem = t_end - time.time()
            if rem <= 0:
                break
            time.sleep(min(0.12, max(0.01, rem)))

        # Only turn off if this pulse is still the active command and not manually held
        if self.pulse_gen.get(relay_id) == gen and relay_id not in self.manual_held_relays:
            self.update_indicator(relay_id, False)
            if relay_id in self.btn_on_map:
                self.btn_on_map[relay_id].configure(fg_color="#16a34a", text="ON")
            if self.driver.is_connected():
                try:
                    if is_coil:
                        self.driver.write_coil(addr, False)
                    else:
                        self.driver.write_register(addr, 0)
                except Exception as e:
                    self.log_msg(f"Pulse OFF err R{relay_id:02d}: {e}")

    # =========================================================================
    # BANK BATCH OPERATIONS (SUPPORTS CONTIGUOUS AND INDIVIDUAL MAPPING)
    # =========================================================================
    def bank_action(self, start_id: int, end_id: int, state_val: int):
        """Turn entire bank ON (1) or OFF (0)"""
        threading.Thread(target=self._bank_action_worker, args=(start_id, end_id, bool(state_val)), daemon=True).start()

    def _bank_action_worker(self, start_id: int, end_id: int, state: bool):
        for r in range(start_id, end_id + 1):
            if state: self.manual_held_relays.add(r)
            else: self.manual_held_relays.discard(r)
        # Update all indicators immediately
        for r in range(start_id, end_id + 1):
            self.update_indicator(r, state)

        if not self.driver.is_connected():
            self.log_msg("Batch Write Aborted: PLC Not Connected.")
            return

        is_coil = "Coil" in self.register_type
        addrs = [self.get_relay_address(r) for r in range(start_id, end_id + 1)]
        is_contiguous = (addrs == list(range(addrs[0], addrs[0] + len(addrs))))

        try:
            if is_contiguous:
                count = len(addrs)
                start_addr = addrs[0]
                if is_coil:
                    ok = self.driver.write_coils(start_addr, [state] * count)
                else:
                    ok = self.driver.write_registers(start_addr, [1 if state else 0] * count)
                res = "OK" if ok else "FAIL"
                action_name = "ON" if state else "OFF"
                self.log_msg(f"BANK [{start_id}-{end_id}] (Addr {start_addr}-{start_addr+count-1}) -> {action_name} [{res}]")
            else:
                # Individual writes for custom-mapped relays
                all_ok = True
                for r in range(start_id, end_id + 1):
                    addr = self.get_relay_address(r)
                    if is_coil:
                        ok = self.driver.write_coil(addr, state)
                    else:
                        ok = self.driver.write_register(addr, 1 if state else 0)
                    if not ok: all_ok = False
                res = "OK" if all_ok else "PARTIAL/FAIL"
                action_name = "ON" if state else "OFF"
                self.log_msg(f"BANK [{start_id}-{end_id}] (Individual Addrs) -> {action_name} [{res}]")
        except Exception as e:
            self.log_msg(f"Batch Bank Error: {e}")

    def bank_fire_all(self, start_id: int, end_id: int):
        """Fires all relays in bank as pulse"""
        threading.Thread(target=self._bank_fire_worker, args=(start_id, end_id), daemon=True).start()

    def _bank_fire_worker(self, start_id: int, end_id: int):
        self._bank_action_worker(start_id, end_id, True)
        time.sleep(self.pulse_duration / 1000.0)
        self._bank_action_worker(start_id, end_id, False)

    def bank_sequence(self, start_id: int, end_id: int):
        """Sequentially triggers each relay one after another"""
        if self.sequence_running:
            self.log_msg("A sequence is already executing.")
            return
        threading.Thread(target=self._sequence_worker, args=(start_id, end_id), daemon=True).start()

    def _sequence_worker(self, start_id: int, end_id: int):
        self.sequence_running = True
        self.log_msg(f"Starting Sequence: Relays {start_id} to {end_id}...")
        try:
            for r in range(start_id, end_id + 1):
                self._pulse_worker(r)
                time.sleep(self.seq_interval / 1000.0)
            self.log_msg(f"Sequence complete for Relays {start_id}-{end_id}.")
        finally:
            self.sequence_running = False

    def emergency_stop_all(self):
        """Instantly turns off all 40 relays"""
        threading.Thread(target=self._estop_worker, daemon=True).start()

    def _estop_worker(self):
        self.manual_held_relays.clear()
        self.log_msg(">>> EMERGENCY STOP ACTIVATED: ALL 40 RELAYS OFF <<<")
        for r in range(1, self.TOTAL_RELAYS + 1):
            self.update_indicator(r, False)

        if not self.driver.is_connected():
            return

        is_coil = "Coil" in self.register_type
        addrs = [self.get_relay_address(r) for r in range(1, self.TOTAL_RELAYS + 1)]
        is_contiguous = (addrs == list(range(addrs[0], addrs[0] + len(addrs))))

        try:
            if is_contiguous:
                start_addr = addrs[0]
                if is_coil:
                    self.driver.write_coils(start_addr, [False] * self.TOTAL_RELAYS)
                else:
                    self.driver.write_registers(start_addr, [0] * self.TOTAL_RELAYS)
            else:
                for r in range(1, self.TOTAL_RELAYS + 1):
                    addr = self.get_relay_address(r)
                    if is_coil:
                        self.driver.write_coil(addr, False)
                    else:
                        self.driver.write_register(addr, 0)
            self.log_msg("Emergency Stop: All 40 coils set to 0.")
        except Exception as e:
            self.log_msg(f"E-Stop write error: {e}")

    # =========================================================================
    # LIVE POLLING WORKER (BACKGROUND)
    # =========================================================================
    def _polling_worker(self):
        while self.poll_active:
            if self.live_poll_enabled and self.driver.is_connected():
                try:
                    is_coil = "Coil" in self.register_type
                    addrs = [self.get_relay_address(r) for r in range(1, self.TOTAL_RELAYS + 1)]
                    is_contiguous = (addrs == list(range(addrs[0], addrs[0] + len(addrs))))

                    if is_contiguous:
                        start_addr = addrs[0]
                        if is_coil:
                            data = self.driver.read_coils(start_addr, self.TOTAL_RELAYS)
                            if data is not None and len(data) == self.TOTAL_RELAYS:
                                for idx, val in enumerate(data):
                                    self.update_indicator(idx + 1, bool(val))
                        else:
                            data = self.driver.read_registers(start_addr, self.TOTAL_RELAYS)
                            if data is not None and len(data) == self.TOTAL_RELAYS:
                                for idx, val in enumerate(data):
                                    self.update_indicator(idx + 1, bool(val != 0))
                    else:
                        # Non-contiguous polling
                        for r in range(1, self.TOTAL_RELAYS + 1):
                            addr = self.get_relay_address(r)
                            if is_coil:
                                res = self.driver.read_coils(addr, 1)
                                if res: self.update_indicator(r, bool(res[0]))
                            else:
                                res = self.driver.read_registers(addr, 1)
                                if res: self.update_indicator(r, bool(res[0] != 0))
                except Exception:
                    pass
            time.sleep(0.3)


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    app = InovanceEjectorApp()
    app.mainloop()
