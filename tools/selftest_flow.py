# Extra part of tools/ui_selftest.py: what the simulated run left behind (decisions, LOST notes, Undefined captures).
import copy
import csv
# Second run with a rule that makes every cashew Undefined (longest allowed gap 1 ms), to see the capture work.
forced = copy.deepcopy(app.shared_state.grading)
forced["rules"]["max_lost_ms"] = 1.0
app.shared_state.set_grading(forced)
app._switch_page("Home")
app._on_start_stop()
pump(6.0)
app._refresh_status()
print("== Home line while capturing:", app.lbl_home_small.cget("text") if hasattr(app, "lbl_home_small") else "")
app._on_start_stop()
pump(1.5)
app.processing_thread.undefined_capture.drain()
rows = [r for f in glob.glob(str(T / "logs" / "cashews_*.csv")) for r in csv.DictReader(open(f))]
print("== flow check on the simulated belt:", len(rows), "cashews in the log")
for r in rows:
    print("  ", r["id"], "frames", r["frames"], "|", r["result"], r["reason"], "| relay", r["relay"] or "-", r["fire"], "|", r["shadow"][:110])
files = sorted(str(p.relative_to(T)) for p in (T / "captures").rglob("*") if p.is_file())
ids = sorted({os.path.basename(f).split("_")[1] for f in files})
print("  captures:", len(files), "files for", len(ids), "cashews:", ids)
for f in [f for f in files if ids and ids[0] in f][:8]:
    print("     ", f)
wanted = {r["id"] for r in rows if r["result"] == "LOST" or (r["result"] == "UNDEFINED" and not r["reason"].startswith("SIZE_NOT_READY"))}
print("  every Undefined / lost cashew captured:", wanted <= set(ids), "| missing:", sorted(wanted - set(ids)), "| others captured:", sorted(set(ids) - wanted))
print("  status:", app.processing_thread.undefined_capture.status())
