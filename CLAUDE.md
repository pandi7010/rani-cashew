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

- Never fire real relays while testing. MASTER EJECTOR is OFF and DRY-RUN is ON at every start.
- All relay firing goes through `relay_output.py`. Only relays 1-40 may be fired. It never calls the PLC's
  `fire_all()` / `sequence_all()` (they pulse 48 coils, including conveyor M140 and feeder M141).
- Grading logic never touches relay 41 (conveyor, M140) or 42 (feeder, M141).
- Do not change `innovance_plc.py` without asking first.
- Every manual fire button is `RelayOutput.test_fire()` (DRY-RUN checked, logged in `logs/manual_fire_*.csv`).
  The PLC diagnostics tool (`innovance_plc.py` run as a program) writes directly to the PLC; the grader only
  opens it with DRY-RUN off, STOPPED and after a confirmation.
- Ejector check (`ejector_check.py`, window `ejector_check_ui.py`): `RelayOutput.check_pulse()` with read-back;
  STOPPED and conveyor OFF only. `RelayOutput.all_ejectors_off()` writes OFF to relays 1-40 (allowed in DRY-RUN).

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
