"""Self-test of the whole program WITHOUT the machine: fake PLC (coil memory), simulated belt, temporary settings
and logs, window outside the visible screen. Photographs every page and tool window.

    python3 tools/ui_selftest.py [output folder for the PNG files]

Nothing real is touched: no camera, no PLC, the real settings file and saved_models are not written.
"""
import os
import sys

os.environ["GSN_UI_OFFSCREEN"] = "1"
import glob
import json
import shutil
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "design" / "pages"
OUT.mkdir(parents=True, exist_ok=True)
T = Path(tempfile.mkdtemp(prefix="gsn_selftest_"))


def make_belt(folder, frames=70):
    """Blue striped belt with the four sample cashew views moving right to left through both ROIs."""
    import cv2
    import numpy as np
    samples = [cv2.imread(f) for f in sorted(glob.glob(str(ROOT / "samples" / "rolling_example" / "*.jpg")))]
    folder.mkdir(parents=True)
    cashews = [(1500, 300, 0), (1760, 310, 1), (2300, 290, 2), (1400, 1750, 3), (2000, 1760, 0)]
    if os.environ.get("GSN_SELFTEST_LOAD"):     # a full belt for speed measurements: 7 cashews per ROI
        cashews = [(1300 + 330 * k, y + 14 * (k % 3), k) for k in range(7) for y in (300, 1750)]
    for n in range(frames):
        img = np.zeros((2048, 2448, 3), np.uint8)
        img[:] = (120, 40, 15)
        img[:, ::90] = (8, 8, 8)
        for x0, y, phase in cashews:
            crop = samples[(n // 3 + phase) % 4]
            h, w = crop.shape[:2]
            x, y0 = int(x0 - 45 * n - w / 2), int(y + 6 * ((n + phase) % 3 - 1) - h / 2)
            x1, y1, x2, y2 = max(0, x), max(0, y0), min(2448, x + w), min(2048, y0 + h)
            if x2 > x1 and y2 > y1:
                img[y1:y2, x1:x2] = crop[y1 - y0:y2 - y0, x1 - x:x2 - x]
        cv2.imwrite(str(folder / f"f{n:03d}.png"), img)


make_belt(T / "belt")
cfg = json.load(open(ROOT / "pneumatic_config.json"))
cfg["camera_source"] = {"source": "file", "test_path": str(T / "belt"), "test_loop": True,
                        "test_fps": float(os.environ.get("GSN_SELFTEST_FPS", 25.0))}
cfg["production_capture"] = {"enabled": False}
json.dump(cfg, open(T / "pneumatic_config.json", "w"), indent=2)

import app_paths
app_paths.LOGS_DIR, app_paths.CFG_PATH, app_paths.INBOX_DIR = T / "logs", T / "pneumatic_config.json", T / "inbox"
app_paths.CAPTURES_DIR = T / "captures"
app_paths.MACHINE_CONFIG_PATH, app_paths.ACTIVE_MODELS_PATH = T / "machine_config.json", T / "active_models.json"
import cashew_cnn_grader as g
import ui_debug
from test_relay_output import FakePLC

g.CFG_PATH = str(T / "pneumatic_config.json")
plc = None


class FakeController(FakePLC):
    plc_ip, plc_port, plc_unit_id = "192.168.1.88", 502, 1

    def __init__(self):
        global plc
        super().__init__()
        plc = self
        self.relay_addresses = {r: 100 + r - 1 for r in range(1, 49)}

    def get_relay_address(self, relay):
        return self.relay_addresses.get(relay, 100 + relay - 1)

    def update_config(self, relay_addresses=None, **kw):
        self.relay_addresses.update(relay_addresses or {})


g.InovancePLCController = FakeController
asked = []
g.messagebox.askyesno = lambda title, text="", **k: (asked.append(str(text).split("\n")[0]), True)[1]
for name in ("showinfo", "showerror", "showwarning"):
    setattr(g.messagebox, name, lambda *a, **k: None)
app = g.App()
ui_debug.place_offscreen(app, 1920, 1080)


def pump(seconds):
    end = time.time() + seconds
    while time.time() < end:
        app.update()
        time.sleep(0.02)


def shot(widget, name):
    pump(0.5)
    print(f"  {name}.png", "ok" if ui_debug.capture_window(widget, str(OUT / f"{name}.png")) else "FAILED")


def on_coils():
    return sorted(a for a, v in plc.coils.items() if v)


out = app.relay_output
print("== start")
pump(1.0)
print("  state:", app.shared_state.phase, "| pills:", list(app.pills), "| DRY-RUN / MASTER left:", hasattr(out, "dry_run") or hasattr(out, "master_enabled"))
print("== manual fire while STOPPED (fake PLC)")
app._switch_page("Pneumatic")
app._pulse_and_flash(7)
pump(0.03)
print("  lightning A07: coils ON now", on_coils())
pump(0.4)
print("  after the pulse: coils ON", on_coils(), "| last:", out.last_result.get(7, ("", "-"))[1], "| page:", app.lbl_test_relay_result.cget("text")[10:70])
n = len(plc.writes)
app._fire_all()
pump(0.6)
print("  FIRE ALL: asked", asked[-1:], "| ejectors fired", sorted(set(plc.pulses[-40:])) == list(range(1, 41)), "| coils ON after", on_coils())
app._sequence_all()
pump(1.2)
app._all_relays_off()
pump(0.3)
print("  SEQUENCE ALL then ALL RELAYS OFF: sequence running", app._sequence_running, "| coils ON", on_coils(), "|", app.lbl_test_relay_result.cget("text")[10:60])
print("  addresses written:", min(plc.get_relay_address(r) for r, _ in plc.writes), "-", max(plc.get_relay_address(r) for r, _ in plc.writes), "| forbidden PLC calls:", sorted({f[0] for f in plc.forbidden} - {"start", "stop"}))
print("== RUNNING: grading fires, manual is refused")
app._switch_page("Home")
writes_before = len(plc.writes)
app._on_start_stop()
pump(7.0)
shot(app, "01_home_running")
fired = len(plc.writes) - writes_before
result = out.manual_pulse(3, 100)
print("  decisions:", app.shared_state.get_result_counts(), "| coil writes by grading:", fired, "| waiting pulses:", len(out.pending()), "| manual while RUNNING:", result)
app._on_start_stop()
pump(0.5)
print("  STOP: state", app.shared_state.phase, "| waiting pulses", len(out.pending()), "| coils ON", on_coils(), "| cancelled", out.counts()[2]["CANCELLED"])
print("== pages")
names = ["Pneumatic", "Sorting Logic", "Colour Size", "Vision ROI", "Teaching CNN", "Dataset Capture", "Teaching YOLO", "AI Model Select", "Test Console", "Network", "Info"]
names = [n for n in names if n in app.pages] + [n for n in app.nav_buttons if n not in names and n != "Home"]
if os.environ.get("GSN_SELFTEST_QUICK"):      # only the checks, no page pictures (faster)
    names = []
for number, page in enumerate(names, start=2):
    app._switch_page(page)
    pump(2.5 if page in ("AI Model Select", "Info", "Teaching CNN") else 0.8)
    if page == "Teaching CNN":
        app.teach_page.refresh(force=True)
    if page == "AI Model Select":
        app.model_page.refresh()
        pump(3.0)
        app.model_page.refresh()
    shot(app, f"{number:02d}_" + page.lower().replace(" ", "_"))
print("== tool windows")
if not os.environ.get("GSN_SELFTEST_QUICK"):
    app._switch_page("Pneumatic")
    app._open_plc_diagnostics()
    win = app._plc_diag_window
    pump(0.5)
    win._test(5)
    pump(0.5)
    print("  mapping window lightning A05:", win.row_status[5].cget("text"), "|", win.lbl_result.cget("text"))
    shot(win, "20_window_plc_mapping")
    win.withdraw()          # not destroyed: without a Tk main loop that can stall
    app._open_ejector_check()
    w = app._ejector_check_window
    w.overrideredirect(True)
    w.geometry(f"1180x980+{app.winfo_screenwidth() + 4500}+0")
    shot(w, "21_window_ejector_check")
    w.withdraw()          # not destroyed: without a Tk main loop that can stall
    app._popup_mapper(7)
    shot(app._mapper_window[0], "22_window_ejector_picker")
    app._mapper_window[0].withdraw()          # not destroyed: without a Tk main loop that can stall
if len(sys.argv) > 2:
    exec(open(sys.argv[2]).read())          # extra checks of a development step
print("  menu:", list(app.nav_buttons))
g.App.HARD_EXIT = False
app.destroy = lambda: None          # without a Tk main loop, destroying the window can stall; the process ends below
app._on_close()
shutil.rmtree(T, ignore_errors=True)
print("done ->", OUT)
sys.stdout.flush()
os._exit(0)            # worker threads of the AI libraries can keep the interpreter from ending
