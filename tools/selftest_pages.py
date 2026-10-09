# Extra part of tools/ui_selftest.py: Vision ROI calibration, Colour Size (manual size table, colour split), PASS classes.
#     python3 tools/ui_selftest.py design/pages tools/selftest_pages.py
import csv
import random
import colour_size
cfg_file = lambda: json.load(open(g.CFG_PATH))
changes_log = lambda: (T / "logs" / "settings_changes.log").read_text().splitlines() if (T / "logs" / "settings_changes.log").exists() else []

print("== Vision ROI: calibration of both ROIs, MEASURE OBJECT")
app._switch_page("Vision ROI")
pump(1.5)
print("  before:", app.lbl_calibration.cget("text")[-150:])
print("  MEASURE OBJECT (not calibrated):", app._measure_object(), "|", app.lbl_measure.cget("text")[:110])
app.roi_length_vars[1].set("300")
app.roi_length_vars[2].set("290")
pump(0.3)
app._save_vision_settings()
pump(0.5)
print("  saved:", {k: cfg_file()["size_config"].get(k) for k in ("roi_length_mm", "px_per_mm", "roi2_length_mm", "px_per_mm_2")})
app._measure_object()
print("  MEASURE OBJECT:", app.lbl_measure.cget("text")[:120])
shot(app, "05_vision_roi")

print("== Colour Size: start table, ejector column")
app._switch_page("Colour Size")
pump(0.8)
print("  table:", {n: (app.cs_vars[("grade", n, "l_min")].get() or "-", app.cs_vars[("grade", n, "l_max")].get() or "-") for n in ("W180", "W210", "W450", "BUDDS", "SW240")})
print("  ejectors:", {n: app.cs_grade_ejector[n].cget("text") for n in ("W180", "W240", "BUDDS", "SW240", "SW_BUDDS")})
print("  split:", app.lbl_cs_split.cget("text"), "| ready:", app.lbl_cs_ready.cget("text"), "|", app.lbl_cs_calibration.cget("text"))

real_decide = g.decision.decide
def all_good(*a, **k):                       # the simulated cashews are REJECTION for YOLO: the test turns them into GOOD
    d = real_decide(*a, **k)
    d.result, d.reason, d.reject = g.decision.GOOD, "forced GOOD by the self-test", False
    return d
g.decision.decide = all_good

print("== SHADOW (READY off): grade is only shown")
app._switch_page("Home")
app._on_start_stop(); pump(7.0); app._on_start_stop(); pump(0.8)
rows = app.shared_state.get_recent_fires()
print("  RECENT RESULTS:", rows[0], "| all ejectors:", sorted({r[-1] for r in rows}))

print("== change the table, SAVE (asks with old -> new), SIZE GRADES READY on")
app._switch_page("Colour Size")
app.cs_vars[("grade", "W320", "l_max")].set("26.5")
app.cs_vars[("grade", "W240", "l_min")].set("26.5")
app._copy_w_to_sw()
print("  SAVE:", app._save_colour_size(), "| asked:", asked[-1][:20], "|", app.lbl_cs_saved.cget("text"))
print("  file:", {n: cfg_file()["size_config"]["grades"][n] for n in ("W320", "SW240", "SW_BUDDS")})
app._set_size_ready(True)
print("  READY:", cfg_file()["size_config"]["size_ready"], "| confirmation started:", asked[-1][:30])
app._switch_page("Home")
before = len(plc.writes)
app._on_start_stop(); pump(7.0)
shot(app, "01_home_running")
pump(6.0)
app._on_start_stop(); pump(0.8)
with app.shared_state.lock:
    live = [s for s in app.shared_state.size_samples if s.get("grade") and s["label"].startswith("GOOD")]
for s in live[-4:]:
    print("   ", s["id"], s["shown"], "| L", round(s["l"], 1), "mm | b*", round(s["lab"]["b"], 1), "|", s["colour"], "| grade", s["grade"], "| ejector", s["relay"])
start = {"W180": (11, 31), "W210": (12, 32), "W240": (13, 33), "W320": (14, 34), "W450": (15, 35)}
print("  cashews:", len(live), "| ejector = mapping of the grade:", all(s["relay"] in start.get(s["grade"], ()) for s in live), "| grades:", sorted({s["grade"] for s in live}),
      "| coils written:", sorted({plc.get_relay_address(r) for r, _ in plc.writes[before:]}))

print("== COLOUR SPLIT: threshold, switch on, SW cashews use the SW ejectors")
app._switch_page("Colour Size")
app._set_colour_split(True)
print("  without a saved threshold:", cfg_file()["size_config"].get("colour", {}).get("enabled"))
b_values = sorted(s["lab"]["b"] for s in live)
app.cs_threshold_var.set(f"{(b_values[0] + b_values[-1]) / 2 + 30:.1f}")       # above every simulated cashew ...
app.cs_side.set("BELOW")                                                       # ... and SW is below it: all become SW
app._save_colour_size()
app._set_colour_split(True)
print("  with it:", cfg_file()["size_config"]["colour"])
app._switch_page("Home")
with app.shared_state.lock:
    app.shared_state.size_samples.clear()
app._on_start_stop(); pump(6.0); app._on_start_stop(); pump(0.8)
with app.shared_state.lock:
    sw = [s for s in app.shared_state.size_samples if s.get("grade")]
print("  cashews:", len(sw), "| all SW:", {s["colour"] for s in sw}, "| grades:", sorted({s["grade"] for s in sw}), "| ejectors:", sorted({s["relay"] for s in sw}))
app._switch_page("Colour Size"); pump(0.8)
shot(app, "04_colour_size")

print("== PASS class: TESTA SMALL (the test now lets the cashews be TESTA SMALL)")
def all_testa(votes, voting=None, priority=None, good_classes=None, *a, **k):
    for v in votes:
        v.cnn_class, v.cnn_conf, v.yolo_class, v.yolo_conf = "TESTA SMALL", 0.95, None, 0.0
    return real_decide(votes, voting, priority, good_classes, *a, **k)
g.decision.decide = all_testa
app._set_colour_split(False)
def run_once():
    with app.shared_state.lock:
        app.shared_state.recent_fires.clear()
    app._switch_page("Home")
    app._on_start_stop(); pump(6.0); app._on_start_stop(); pump(0.8)
    return [r for r in app.shared_state.get_recent_fires() if "TESTA" in r[2]]
fire = run_once()
print("  as FIRE:", fire[0][2:] if fire else fire, "| ejectors:", sorted({r[-1] for r in fire}))
app._toggle_class_pass("REJECTION"); app._toggle_class_pass("GOOD_CASHEW")
app._toggle_class_pass("TESTA SMALL")
print("  chip changed, PASS list:", cfg_file()["grading"]["good_classes"], "| asked:", asked[-1][:26])
passed = run_once()
print("  as PASS:", passed[0][2:] if passed else passed, "| ejectors:", sorted({r[-1] for r in passed}))
app._switch_page("Pneumatic"); pump(1.0)
shot(app, "02_pneumatic")
app._toggle_class_pass("TESTA SMALL")
g.decision.decide = real_decide
print("  back to FIRE:", "TESTA SMALL" not in cfg_file()["grading"]["good_classes"])
app._switch_page("Colour Size")
app._set_size_ready(False)
print("== logs/settings_changes.log:", len(changes_log()), "lines")
for line in changes_log():
    print("   ", line[20:150])
