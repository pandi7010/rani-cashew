# Speed measurement of the grading loop and the UI (extra part of tools/ui_selftest.py; fake PLC, simulated full belt):
#     GSN_SELFTEST_LOAD=1 GSN_SELFTEST_FPS=70 GSN_SELFTEST_QUICK=1 python3 tools/ui_selftest.py /tmp/shots tools/profile_grader.py
# Prints the AI frames per second, the milliseconds per frame of every stage, the time of one UI refresh, the time to
# open each page and the time of a label click on the Teaching CNN page.
import statistics
import perf as perf_module

SECONDS = float(os.environ.get("GSN_PROFILE_SECONDS", 12))
ui_times = []
original_loop = app._update_loop
def timed_loop():
    t0 = time.perf_counter()
    try:
        original_loop()
    finally:
        ui_times.append(time.perf_counter() - t0)
app._update_loop = timed_loop

print("== grading loop,", SECONDS, "s RUNNING on the simulated full belt")
app._switch_page("Home")
app._on_start_stop()
pump(3.0)                                   # warm-up (first frames load the GPU kernels)
stages = app.processing_thread.perf
app.processing_thread._perf_logged = time.time() + 3600      # no 30 s report during the measurement
stages.reset()
ui_times.clear()
t0 = time.time()
pump(SECONDS)
report = stages.report()
elapsed = time.time() - t0
live = sum(len(t.tracks) for t in app.processing_thread.trackers.values())
print(f"  AI fps {report['fps']:.1f}   camera fps {app.camera_thread.fps:.1f}   cashews in view now {live}   decided {sum(app.shared_state.get_result_counts().values())}")
for name, value in sorted(report["ms"].items(), key=lambda item: -item[1]):
    print(f"    {name:32s} {value:6.2f} ms per frame")
print(f"    {'TOTAL of the stages':32s} {report['total_ms']:6.2f} ms per frame  (= at most {1000 / max(report['total_ms'], 0.001):.0f} fps)")
if ui_times:
    print(f"  UI refresh: {len(ui_times) / elapsed:.1f} per second, {1000 * statistics.mean(ui_times):.1f} ms each (max {1000 * max(ui_times):.0f} ms): "
          f"{100 * sum(ui_times) / elapsed:.0f} % of the UI thread")
app._on_start_stop()
pump(0.5)

print("== time to open each page (STOPPED)")
def open_time(page):
    t0 = time.perf_counter()
    app._switch_page(page)
    app.update_idletasks()
    app.update()
    return 1000 * (time.perf_counter() - t0)
for page in ("Pneumatic", "Sorting Logic", "Colour Size", "Vision ROI", "Teaching CNN", "Dataset Capture", "Teaching YOLO", "AI Model Select", "Info", "Home"):
    first = open_time(page)
    app._switch_page("Home"); app.update()
    print(f"    {page:18s} first {first:7.0f} ms    again {open_time(page):7.0f} ms")

print("== Teaching CNN page with a temporary copy of 1,200 captures and 800 dataset images")
import app_paths
import dataset_ops as ops
import teach_ui
real_captures = sorted(Path(app_paths.GOLDEN_DIR).glob("*.jpg"))[:1200]
real_dataset = Path(app_paths.CLEAN_DATASET_DIR)
captures, data = T / "captures_copy", T / "prog" / "dataset"
captures.mkdir(parents=True)
for f in real_captures:
    shutil.copy(f, captures / f.name)
for name in ("GOOD_CASHEW", "UPW", "TESTA", "PKW2"):
    (data / name).mkdir(parents=True, exist_ok=True)
    for f in sorted((real_dataset / name).glob("*.jpg"))[:200]:
        shutil.copy(f, data / name / f.name)
app_paths.CLEAN_DATASET_DIR = data
page = app.teach_page
page.golden_dir = str(captures)
page.changes = ops.Changes(str(T / "logs"))
def timed(action):
    t0 = time.perf_counter(); action(); app.update_idletasks(); app.update(); return 1000 * (time.perf_counter() - t0)
app._switch_page("Home"); app.update()
page.invalidate()
t_open = timed(lambda: app._switch_page("Teaching CNN"))
t0 = time.perf_counter(); page.wait_idle(); app.update(); t_loaded = 1000 * (time.perf_counter() - t0)
print(f"    open the page: {t_open:.0f} ms until it is on screen, {t_loaded:.0f} ms more until {len(page.items)} captures and the counts are listed (in the background)")
print(f"    next page (25 new thumbnails): {timed(lambda: page.turn(1)):.0f} ms    back (from memory): {timed(lambda: page.turn(-1)):.0f} ms")
def label(n, name):
    page.selected = {item['path'] for item in page._page_items()[:n]}
    before = ops.class_counts(str(data)).get(name, 0)
    ui = timed(lambda: page.label_selected(name))
    t0 = time.perf_counter(); page.wait_idle(); files = 1000 * (time.perf_counter() - t0)
    return ui, files, ops.class_counts(str(data)).get(name, 0) - before
ui1, files1, moved1 = label(1, "UPW")
ui25, files25, moved25 = label(25, "TESTA")
print(f"    label 1 image: UI {ui1:.0f} ms (files moved in the background within {files1:.0f} ms, {moved1} moved)")
print(f"    label a page of 25: UI {ui25:.0f} ms (files within {files25:.0f} ms, {moved25} moved)")
left = len(page.items)
page.undo(); page.undo(); app.update()
print(f"    UNDO x 2: captures {left} -> {len(page.items)}, UPW {ops.class_counts(str(data))['UPW']}, TESTA {ops.class_counts(str(data))['TESTA']} (200 each before)")
