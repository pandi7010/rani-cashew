# Extra part of tools/ui_selftest.py for Vision ROI and Colour Size, step (c2) (temporary settings file, fake PLC):
#     python3 tools/ui_selftest.py design/pages tools/selftest_pages.py
import csv
import random
import colour_size
cfg_file = lambda: json.load(open(g.CFG_PATH))
print("== Colour Size: SIZE GRADES READY is refused before calibration, threshold and grades")
app._switch_page("Colour Size")
pump(0.5)
print("  missing:", len(app._size_ready_blockers()), "things;", [m.split(" (")[0] for m in app._size_ready_blockers()])
app._set_size_ready(True)
app._size_teach_toggle("W320")
print("  ready:", app.shared_state.size_config.get("size_ready"), "| teach started without calibration:", app._size_teach is not None)

print("== Vision ROI: calibration")
app._switch_page("Vision ROI")
app.roi_length_var.set("300")
pump(0.3)
app._save_vision_settings()
pump(0.5)
shot(app, "05_vision_roi")
print("  300 mm ->", app.lbl_px_per_mm.cget("text"), "| saved:", cfg_file()["size_config"]["px_per_mm"])

print("== SHADOW run: every cashew forced to GOOD by the test, so colour and size are measured")
real_decide = g.decision.decide
def all_good(*a, **k):
    d = real_decide(*a, **k)
    d.result, d.reason, d.reject = g.decision.GOOD, "forced GOOD by the self-test", False
    return d
g.decision.decide = all_good
app.shared_state.reset_counts() if hasattr(app.shared_state, "reset_counts") else None
app._switch_page("Home")
app._on_start_stop(); pump(7.0); app._on_start_stop(); pump(0.8)
with app.shared_state.lock:
    shadow = [s for s in app.shared_state.size_samples if s.get("lab")]
print("  measured:", len(shadow), "cashews, e.g.", [(s["id"], round(s["l"], 1), round(s["lab"]["b"], 1), s["grade"], s["relay"]) for s in shadow[:3]])
print("  all to the reject ejector:", {s["relay"] for s in shadow} <= {1, 21}, "| RECENT RESULTS row:", app.shared_state.get_recent_fires()[0])
lengths = sorted(s["l"] for s in shadow)

print("== Colour Size: TEACH WW / TEACH SW with simulated cashews, APPLY")
app._switch_page("Colour Size")
random.seed(5)
def feed(n, L, a, b, length=24.5):
    with app.shared_state.lock:
        for i in range(n):
            app.shared_state.size_samples.append({"id": f"T{random.random():.6f}", "t": time.time(), "l": random.gauss(length, 1.0), "w": random.gauss(12.4, 0.7),
                "a": random.gauss(226, 19), "solidity": 0.9, "double": False, "label": "UNDEF SIZE_NOT_READY (would be ? not taught)",
                "lab": {"L": random.gauss(L, 2.5), "a": random.gauss(a, 0.8), "b": random.gauss(b, 2.5)}, "colour": "", "colour_value": None, "grade": "", "relay": 1})
app._colour_teach_toggle(colour_size.WW); feed(45, 68, -6, 22); app._refresh_colour_size(force=True); app._colour_teach_toggle(colour_size.WW)
app._colour_teach_toggle(colour_size.SW); time.sleep(0.05); app._colour_teach["since"] = time.time(); feed(40, 58, -3, 24); app._refresh_colour_size(force=True)
pump(0.4)
print("  taught WW", len(app._colour_teach["WW"]), "SW", len(app._colour_teach["SW"]), "|", app.lbl_cs_suggest.cget("text").replace("\n", " | ")[:260])
shot(app, "04_colour_size_teach_colour")
app._colour_teach_apply()
print("  APPLY ->", cfg_file()["size_config"]["colour"])

print("== Colour Size: TEACH a grade, COPY W LIMITS TO SW, SAVE")
app._size_teach_toggle("W320"); time.sleep(0.05); app._size_teach["since"] = time.time(); feed(44, 68, -6, 22, length=24.5)
app._refresh_colour_size(force=True); pump(0.4)
shot(app, "04_colour_size_teach_grade")
app._size_teach_apply()
lo, hi = lengths[0] - 1, lengths[-1] + 1                     # a second grade that holds the simulated belt cashews
for key, value in (("l_min", lo), ("l_max", hi)):
    app.cs_vars[("grade", "W180", key)].set(f"{value:.1f}")
app._copy_w_to_sw()
print("  SAVE:", app._save_colour_size(), "| grades saved:", sorted(cfg_file()["size_config"]["grades"]), "| W320:", cfg_file()["size_config"]["grades"]["W320"])
print("  SW320 == W320 limits:", all(cfg_file()["size_config"]["grades"]["SW320"][k] == cfg_file()["size_config"]["grades"]["W320"][k] for k in colour_size.LIMIT_KEYS),
      "|", cfg_file()["size_config"]["grades"]["SW320"].get("taught"))

print("== SIZE GRADES READY on: GOOD -> colour -> grade -> ejector of the mapping")
app._set_size_ready(True)
print("  ready:", cfg_file()["size_config"]["size_ready"], "| asked:", asked[-1][:32], "| engines:", {k: v for k, v in cfg_file()["grading"]["engines"].items() if k in ("sw_ww", "size")})
with app.shared_state.lock:
    app.shared_state.size_samples.clear()
table = app.shared_state.get_mapping() if hasattr(app.shared_state, "get_mapping") else None
app._switch_page("Home")
before = len(plc.writes)
app._on_start_stop(); pump(7.0)
shot(app, "01_home_running")
waiting = len(app.relay_output.pending())
pump(6.0)                                    # the grade ejectors are far down the belt: let their delays run out
app._on_start_stop(); pump(0.8)
print("  pulses waiting after 7 s:", waiting, "| after 13 s the coils written:", sorted({plc.get_relay_address(r) for r, _ in plc.writes[before:]}))
with app.shared_state.lock:
    live = [s for s in app.shared_state.size_samples if s.get("lab")]
for s in live[:6]:
    print("   ", s["id"], s["label"], "| colour", s["colour"], round(s["colour_value"], 1), "| L", round(s["l"], 1), "mm | grade", s["grade"], "| ejector", s["relay"])
expected = {"W180": (11, 31), "SW180": (16, 36), "W320": (14, 34), "SW320": (19, 39)}
ok = all(s["relay"] in expected.get(s["grade"], (1, 21)) for s in live)
print("  cashews:", len(live), "| every ejector matches the mapping of its grade:", ok, "| grades seen:", sorted({s["grade"] for s in live}),
      "| coil writes:", len(plc.writes) - before)
rows = [r for f in glob.glob(str(T / "logs" / "cashews_*.csv")) for r in csv.DictReader(open(f))]
print("  log rows (last 3):", [(r["id"], r["result"], r["reason"][:28], r["grade"], r["relay"], r["b"][:5]) for r in rows[-3:]])
g.decision.decide = real_decide
app._switch_page("Colour Size"); pump(0.8)
shot(app, "04_colour_size")
app._set_size_ready(False)
print("  switched off again:", cfg_file()["size_config"]["size_ready"])
