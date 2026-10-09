# GSN_V21 cashew grader: guide for Claude

## Language

- **Reply only in English.** The owner's terminal cannot render Tamil.
- New plans and reports are written in English (`REPORT.md`, newest entry on top).
  `PLAN_TAMIL.md` stays as it is (Tamil); do not rewrite it. Code, comments and commit messages: English.

## How to work

- Big change: write the plan first, commit it, wait for the owner's approval.
- One commit per finished step. After each step: `ast` syntax check, run the tests, say briefly what changed
  and how the owner can test it, then stop.
- Long reports also go into `REPORT.md`; keep the terminal reply short.
- Do not guess machine values (relay mapping, voting N and thresholds, valve delay, belt speed, SW/WW threshold,
  size grade limits, double-kernel sizes, what the cashew "top side" looks like). Ask.
- Say plainly what was tested and what was not. Real valves, real cashews and timing can only be checked on the machine.

## Safety

- **My own tests never contact the real PLC.** Tests and the self-test (`tools/ui_selftest.py`) use a fake PLC.
- START / STOP is the only control (owner, 2026-10-10: DRY-RUN and MASTER EJECTOR were removed on purpose; do not
  bring them back). RUNNING = grading fires the valves. STOPPED = nothing fires from grading (waiting pulses are
  cancelled) and the manual buttons work. The program always opens STOPPED and never starts RUNNING by itself.
- All ejector driving goes through `relay_output.py`: relays 1-40 only, never an address of M140 (conveyor) /
  M141 (feeder), never the PLC's `fire_all()` / `sequence_all()` / `all_off()`. A pulse = coil written ON, written
  ON again while it lasts, then OFF.
- Manual fire (row lightning buttons, TEST FIRE, FIRE ALL, SEQUENCE ALL, ejector check, mapping window) writes to
  the real PLC, only when STOPPED, and is logged in `logs/manual_fire_*.csv`. FIRE ALL, SEQUENCE ALL and the
  ejector check ask "Make sure nobody is at the nozzles" first; a single pulse fires at once.
- Class / grade -> ejector comes only from the mapping (`relay_mapping.py`): never tie a class name to an ejector
  number in code. Several classes per ejector are allowed; a class is on one ejector per ROI; none -> reject 1 / 21.
- Do not change `innovance_plc.py` without asking first.

## Machine facts

- Camera: HIFLY MV518UC-O2C, USB3 Vision, 2448 x 2048, BayerGB8, about 70 fps free-running. Driven through
  Aravis (`gi`). It does not appear under `/dev/video*`. Do not replace it with `cv2.VideoCapture`.
  It cannot be opened while the grader is running.
- BayerGB8 is converted with `cv2.COLOR_BayerGR2BGR` (OpenCV's Bayer names are shifted by one position).
- PLC: Inovance EC320, Modbus TCP 192.168.1.88:502, coils. Relay r -> coil address 100 + (r - 1) (M100...).
  ROI1 = relays 1-20, ROI2 = relays 21-40. Relay 1 and relay 21 are the reject relays.
- An encoder will be added later; timing is "time mode" (belt speed) for now.

## Files

- Entry point: `cashew_cnn_grader.py`. Run from anywhere; all paths come from `app_paths.py`.
- Tests: `python3 -m unittest discover -s tests` (no camera or PLC needed).
- Read-only references on the external drive (`/media/phoenix/00A46655A4664D6A`): the 5-track project
  `TCM_CASHEWXAI_2026-10-06/` (the owner's UI reference; theme in `source/home_ui.py`, screenshots in
  `source/design/`) and `CNN/CNN/app/` (the same project, 11 commits newer). Also
  `/home/phoenix/Documents/RaniVision_Latest`, `v2_snapshot/`. Never modify any of them.
- Plan: `PLAN.md` (English, current). Owner's requirements: `FINAL_PROMPT.txt`. Step order: (c1) done, then
  (e) UI, (c2), (c3), (d), (f).
- `external_tools/` holds unmodified copies of the TCM tools; do not edit them.
- Ported from the 5-track project (keep them close to the original): `ui_theme.py`, `teach_ui.py`, `dataset_ops.py`,
  `production_capture.py`, `training.py`, `training_ui.py`. Training data: `programs/Default/{dataset,testset,inbox}`
  (not in git), built by `dataset_import.py`; old `dataset/` folders are never moved or deleted.
- Teaching CNN additions that are NOT in the 5-track files (marked `GSN_V21` in the code): CONFUSION folder
  (`programs/<name>/confusion`, never trained), trash (`programs/<name>/trash`, never emptied by the program), multi-step
  undo (`dataset_ops.Changes`, `logs/dataset_changes.log`), `training_report.py` (graphs next to the model).
- Whole-program self-test with a fake PLC: `python3 tools/ui_selftest.py design/pages [tools/selftest_teaching.py]`.
- UI checks without touching the owner's screen: start with `GSN_UI_OFFSCREEN=1` and photograph with `ui_debug.py`.
