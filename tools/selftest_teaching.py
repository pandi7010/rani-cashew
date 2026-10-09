# Extra part of tools/ui_selftest.py for the Teaching CNN confusion tools (run by it with exec):
#     python3 tools/ui_selftest.py design/pages tools/selftest_teaching.py
# Works on a COPY of a few dataset images in the temporary folder; the real dataset, saved_models and logs are not written.
import random
import teach_ui
import training
import training_report
import dataset_ops as ops

print("== Teaching CNN: confusion tools (temporary copy of the dataset)")
real = Path(app_paths.CLEAN_DATASET_DIR)
data, models = T / "prog" / "dataset", T / "models"
random.seed(1)
for name in ("GOOD_CASHEW", "UPW", "TESTA", "PKW2"):
    files = sorted((real / name).glob("*.jpg"))[:400]
    (data / name).mkdir(parents=True, exist_ok=True)
    (T / "prog" / "testset" / name).mkdir(parents=True, exist_ok=True)
    picks = random.sample(files, min(70, len(files)))
    for i, f in enumerate(picks):
        shutil.copy(f, (T / "prog" / "testset" / name if i >= 60 else data / name) / f.name)
for f in sorted((data / "UPW").glob("*.jpg"))[:4]:          # four wrong labels on purpose
    shutil.move(str(f), str(data / "GOOD_CASHEW" / ("wronglabel_" + f.name)))
app_paths.CLEAN_DATASET_DIR = data
opened = []
teach_ui.subprocess.Popen = lambda cmd, *a, **k: opened.append(cmd)
result = training.train_cnn(str(data), ["GOOD_CASHEW", "UPW", "TESTA", "PKW2"], 6, str(models), name_prefix="cnn_selftest", pass_classes=["GOOD_CASHEW"])
model_path = result["model_path"]
print("  trained:", os.path.basename(model_path), f"val {result['val_accuracy'] * 100:.0f} %", "| history keys", sorted(result["history"][0]),
      "| pictures", sorted(result["reports"]), "| still wrong", result["wrong_images_total"])
page = app.teach_page
page.model_root = str(models)
page.changes = ops.Changes(str(T / "logs"))
app._active_cnn_path = lambda: model_path
app._switch_page("Teaching CNN")
page._result_key = "-"
page._class_signature = None
page.refresh(force=True)
pump(1.0)
print("  panel:", page.lbl_fire.cget("text")[:90], "|", page.btn_wrong.text if hasattr(page.btn_wrong, "text") else "")
shot(app, "30_teaching_cnn_after_training")

# 1. CHECK CONFUSION before training
windows_before = set(app.winfo_children())
for kind_ in ("showinfo", "showerror", "showwarning"):
    setattr(g.messagebox, kind_, lambda *a_, **k_: print("  MESSAGE:", *[str(x)[:300] for x in a_]))
class InlineThread:                     # the self-test has no Tk main loop, so the check runs in this thread here
    def __init__(self, target=None, **kw):
        self.target = target

    def start(self):
        self.target()


real_thread, g.threading.Thread = g.threading.Thread, InlineThread
app._verify_mismatches()
g.threading.Thread = real_thread
end = time.time() + 90
win = None
while time.time() < end and win is None:
    pump(0.5)
    win = next((w for w in set(app.winfo_children()) - windows_before if hasattr(w, "review")), None)
r = win.review
rows = [r["tree"].item(i, "values") for i in r["tree"].get_children()]
print("  CHECK CONFUSION: listed", len(rows), "| kinds", sorted({v[4] for v in rows}), "| planted wrong labels found",
      sum(1 for v in rows if v[0].startswith("wronglabel_")), "of 4")
counts0 = ops.class_counts(str(data))
r["move"](); pump(0.2)                  # to the model's class
r["to_confusion"](); pump(0.2)
r["delete"](); pump(0.2)
r["keep"](); pump(0.2)
print("  after move / confusion / delete / keep: dataset", sum(ops.class_counts(str(data)).values()), "of", sum(counts0.values()),
      "| CONFUSION", ops.confusion_count(str(data)), "| list left", len(r["tree"].get_children()), "of", len(rows))
shot(win, "31_window_check_confusion")
r["undo"](); r["undo"](); r["undo"](); pump(0.2)
print("  3 x UNDO: dataset", sum(ops.class_counts(str(data)).values()), "| counts equal", ops.class_counts(str(data)) == counts0, "| CONFUSION", ops.confusion_count(str(data)),
      "| list", len(r["tree"].get_children()))
win.destroy()

# 2. after training: still wrong -> CONFUSION in one click
win = app._review_wrong_images()
pump(0.5)
if win is None:
    print("  STILL WRONG: nothing listed")
else:
    n = len(win.review["tree"].get_children())
    shot(win, "32_window_still_wrong")
    win.review["all_to_confusion"](); pump(0.3)
    print("  STILL WRONG:", n, "listed -> MOVE ALL: CONFUSION", ops.confusion_count(str(data)), "| asked", asked[-1][:40], "| training sees",
          sum(len(v) for v in training.list_class_images(str(data)).values()), "of", sum(counts0.values()))
    win.destroy()
if not ops.confusion_count(str(data)):
    page.changes.add("MOVE TO CONFUSION", ops.move_to_confusion([str(p) for p in sorted((data / "TESTA").glob("*.jpg"))[:3]], str(data)))

# 3. CONFUSION tab: grid, large view, re-label, delete, undo
page._class_signature = None
page.set_source(teach_ui.CONFUSION)
pump(1.0)
shot(app, "33_teaching_cnn_confusion_tab")
items = list(page.items)
viewer = page.view_large(items[0]["path"])
shot(viewer, "34_window_large_image")
viewer.destroy()
before = ops.confusion_count(str(data))
page.selected = {items[0]["path"]}
page.label_selected("TESTA"); pump(0.3)
after_label = ops.confusion_count(str(data))
if len(items) > 1:
    page.selected = {items[1]["path"]}
    page.delete_selected(); pump(0.3)
print("  CONFUSION tab:", before, "images | re-label ->", after_label, "| delete ->", ops.confusion_count(str(data)), "| last:", page.changes.last_action())
steps = 0
while page.changes.can_undo():
    page.undo(); steps += 1
pump(0.3)
print("  UNDO x", steps, ": counts equal to the start", ops.class_counts(str(data)) == counts0, "| CONFUSION", ops.confusion_count(str(data)),
      "| trash empty of images", not any(Path(ops.trash_dir(str(data))).rglob("*.jpg")))
log = (T / "logs" / ops.CHANGES_LOG).read_text().splitlines()
print("  dataset_changes.log:", len(log), "lines, e.g.", log[0][20:60], "...")

# 4. folders
page.open_source_folder()
page.open_folder(os.path.join(page.dataset_dir(), "UPW"))
print("  OPEN FOLDER:", [(c[0], os.path.relpath(c[1], T)) for c in opened])
page.set_source(teach_ui.CAPTURES)

# 5. graphs
viewer = page.show_report(model_path)
for kind in ("VALIDATION: CURVES", "VALIDATION: CLASSES", "VALIDATION: CONFUSION"):
    viewer.viewer_show(kind)
    shot(viewer, "35_graph_" + kind.split(": ")[1].lower())
viewer.destroy()
mp = app.model_page
mp.model_root, mp.testset_dir = str(models), (lambda: str(T / "prog" / "testset"))
stamp = os.path.getmtime(model_path), os.path.getmtime(training.metrics_path(model_path))
viewer = mp._testset_report(os.path.basename(model_path), model_path)
print("  TEST SET REPORT:", [t for t, _ in training_report.existing_reports(model_path)][-2:], "| model and result file unchanged",
      stamp == (os.path.getmtime(model_path), os.path.getmtime(training.metrics_path(model_path))))
viewer.destroy()
app._switch_page("AI Model Select")
mp._signature = None
mp.refresh(); pump(3.0); mp.refresh()
shot(app, "36_ai_model_select_with_graphs")
