# PLAN v3: GSN_V21 cashew grader

Date: 2026-10-09
Status: steps (a), (b), (c1) done. **Waiting for approval of step (e) (section 7), which now comes before (c2).**
Source: `FINAL_PROMPT.txt` (copied into this folder). It replaces every earlier prompt.
`PLAN_TAMIL.md` (plan v2) is kept unchanged for history; where the two differ, this file wins.

---

## 0. Current state (checked 2026-10-09, after the power cut)

| Check | Result |
|---|---|
| `git fsck` | clean, no errors |
| Working tree | clean; only `samples/` was untracked |
| `12792a7` baseline, `b980fb7` step (a) | committed |
| Step (b) | committed as `8ff570d` (Aravis backend, reconnect, file source, recorder, FFC, `relay_output.py`). `b910963` was the first, wrong attempt (cv2.VideoCapture) and is superseded. |
| `.py` files | all pass the `ast` syntax check |
| `.json` files | `pneumatic_config.json`, `cashew_class_mapping.json`, `inovance_config.json` all parse |
| Unit tests | 38 pass |
| `samples/rolling_example/frame1..4.jpg` | present, identical to the zip in Downloads |

Steps (a) and (b) are not redone. Already done from section 1 of the prompt: Aravis + file source, no webcam
backend, auto reconnect, exposure / gain / FPS in the UI, FFC white calibration, recorder with replay at
recorded timing, Bayer check (on rollers only, see question 14).

---

## 1. Data per class

Image counts. "Old" = `/home/phoenix/Documents/CNN_final/CNN/dataset` (read only). "Project" = `dataset/` here.

| Class (prompt) | Old | Project | In the CNN checkpoint | In the YOLO model | Verdict |
|---|---|---|---|---|---|
| GOOD | 515 | 105 | yes (GOOD_CASHEW) | yes | ok |
| TESTA | 249 | 17 | yes | yes | ok |
| TESTA SMALL | 244 | 12 | yes | yes | ok |
| UPW | 244 | 0 | yes | yes | ok |
| PKW2 | 226 | 0 | yes | yes | ok |
| PKW1 | 206 | 0 | yes | yes | ok |
| BROWN | 117 | 0 | yes | yes | low |
| REJECTION | 93 | 0 | **no** | yes | low, and the CNN cannot output it |
| DW2 | 41 | 0 | yes | yes | **too few** |
| WHITE_DW | 21 | 0 | yes | yes | **too few** |
| OIL | 0 (1 in `dataset_disabled`) | 0 | yes | **no** | **no images** |
| DW1 | 0 | 0 | **no** | yes | **no images**, the CNN cannot output it |
| SW | 0 (15 in `dataset_disabled`, 15 in `dataset_backup_old`) | 0 | yes | yes | not needed if colour decides SW/WW |
| DOUBLE_KERNELS | 52 | 0 | no | no | too few for a class; fine for tuning box limits |
| BACK_CUT_OIL / OIL_BACKCUT | 56 | 13 | no | no | not in the prompt's class list (question 9) |
| EDGECUT | 0 | 2 | no | no | not in the prompt's class list (question 9) |

Other findings:

- The CNN in use (`cashew_color_cnn_new.pth`) has 11 classes. It has **no REJECTION and no DW1**. Today those
  can only come from the YOLO class when the CNN is unsure. "REJECTION is always REJECTION" cannot be
  guaranteed until the CNN is retrained with REJECTION (question 8).
- `dataset/Class_1` and one more folder hold full camera frames (2448 x 2048), not crops. The training script
  will skip anything that is not a crop and list it.
- Rough guide I will use in the training report: under 100 images = low, under 50 = too few. Classes that
  matter and are short: **OIL, DW1, DW2, WHITE_DW, REJECTION, BROWN**. More images are needed before step (f);
  the auto-capture of step (d) plus the labeler is the intended way to collect them.
- No view labels (TOP / BACK / NECK) exist anywhere yet.
- No recording with cashews exists yet (`recordings/` is empty). Steps (c1) and (c3) need one (question 15).

---

## 2. What the four sample frames say (measured, not guessed)

Blue-background removal worked on all four frames and on 1,300+ old crops (only 14 very dark REJECTION crops
failed, because a black cashew looks like the dark belt gaps; handled in 3.1).

| Frame | View (owner) | Aspect ratio | Solidity | Deepest defect / width | L* | b* |
|---|---|---|---|---|---|---|
| frame1 | TOP | 1.52 | 0.85 | 0.35 | 63.5 | 19 |
| frame2 | BACK | 2.05 | 0.96 | 0.11 | 51.8 | 13 |
| frame3 | inner curve + TESTA | 1.52 | 0.84 | 0.32 | 63.5 | 17 |
| frame4 | NECK + TESTA | 1.84 | 0.95 | 0.11 | 53.7 | 13 |

Two things follow, both from four images only, so they must be confirmed on a recording:

1. **The outline alone separates two groups, not four views.** TOP and "inner curve" look the same from the
   outline (kidney shape); BACK and NECK look the same (narrow, convex). Aspect ratio, solidity and
   convexity-defect depth **cannot tell NECK from BACK**. The difference is inside the mask: the NECK view
   shows a hollow (darker / pinker region with a strong edge), the BACK view shows a thin centre line.
   So for stage (a) I add interior features to the outline features (see 3.3).
2. **Colour changes with the view.** The same cashew measures b* 13 in the narrow views and 17-19 in the
   kidney views (L* 52 against 64). SW/WW must therefore use the median over the track, and I will show
   b* per view so we can decide whether to use only one view type for colour.

---

## 3. Step (c1): tracking + views + voting + grading rules

New files: `segmentation.py`, `lane_tracker.py`, `view_features.py`, `decision.py`, `cashew_log.py`, tests.
All pure Python without GUI, so they are unit-tested without camera or PLC.

### 3.1 Segmentation (blue-background removal)

- Mask = not blue (HSV hue band of the belt) and not belt-gap dark, largest blob, small holes closed.
  The blue band is taught from an empty-belt frame (button "TEACH BELT COLOUR"), not hard-coded.
- Very dark cashews: if the colour mask is empty or tiny inside a YOLO box, fall back to "differs from the
  taught empty belt" so black REJECTION pieces still get a mask.
- Everything downstream (centroid, view, colour, size, dark / skin measurement) uses only this mask.
- Detections whose box touches the ROI or frame edge (margin in px, UI) are tracked for identity but do not
  vote, are not measured and are not captured.

### 3.2 Tracker (one per ROI)

Problems in the current tracker: greedy nearest-box matching, box centre as position, speed in px per frame,
accepts backward moves, one ID list for both ROIs, track dies after 3 missed frames and comes back as a new ID.

Proposal: improved own tracker.

- IDs `R1-000123`, `R2-000045`.
- Position = mask centroid (box centre jumps when the cashew rolls).
- Motion prediction in px/ms along the belt direction only; a new track starts with the lane's recent speed.
- Global assignment (Hungarian, `scipy`), cost = distance from the predicted position.
- Gates: no backward move, maximum forward step from speed x time, sideways limit (same roller channel).
- A missed track is kept by time and continues with the **same ID** when it reappears at the predicted place.
- A track that has been decided never matches a new cashew behind the line.

Against ultralytics ByteTrack / BoT-SORT: their Kalman state includes box aspect ratio and height and they
match on IoU, which is weak exactly when the box changes shape every frame; belt rules (one direction, lane,
known speed) cannot be put inside them; BoT-SORT's camera-motion compensation and ReID are cost without use on
a fixed camera. **Recommendation: own tracker.** This is a judgement, not a measurement. In (c1) I run both on
a recording and report: number of IDs against the real cashew count, tracks born or lost in the middle of the
ROI, track length distribution. The final choice follows the numbers.

### 3.3 View per frame: TOP / BACK / NECK / OTHER

Stage (a), now, values shown live on each cashew and on a "View tuning" panel:

- Outline: aspect ratio, solidity, deepest convexity defect / width.
- Interior (new, because of finding 2.1): dark / pink area fraction inside the mask, its position relative to
  the long axis, edge strength inside the mask, left-right symmetry about the long axis.
- Rules with thresholds in the UI. **I set no thresholds.** At first every frame shows its numbers and the
  view is OTHER until the owner sets the limits (or loads the ones I propose from a labelled recording).
- Honest limit: interior features may confuse a NECK hollow with a large dark defect. If stage (a) is not
  reliable on the recording, I say so with numbers and we go to stage (b) earlier.

Stage (b), step (f): a small view classifier trained on crops, and a view-labelling mode in the auto-labeler.

### 3.4 Voting and grading rules (`decision.py`)

One decision per ID, when the centroid crosses that ROI's Decision line. Starting values from the prompt:
defect = confidence >= 0.50 in at least 2 frames; GOOD = at least 3 frames at >= 0.80; NECK needed in M = 1
frames. All in the UI.

A class is "seen" when it has enough qualifying frames. Then, in this order:

| # | Condition | Result | Relay |
|---|---|---|---|
| 1 | Double kernel (step d, default OFF) | DOUBLE KERNEL | reject |
| 2 | REJECTION seen (from the YOLO REJECTION class, see below) | REJECTION | reject |
| 3 | UPW seen, or skin (TESTA / TESTA SMALL) seen together with any other defect | UPW | mapped |
| 4 | PKW2 seen | PKW2 | mapped |
| 5 | PKW1 seen | PKW1 | mapped |
| 6 | OIL, BROWN, DW2, DW1 seen (in this order) | that class | mapped |
| 7 | TESTA seen, then TESTA SMALL seen | that class | mapped |
| 8 | WHITE_DW seen | WHITE_DW | mapped |
| 9 | No defect, GOOD votes enough, NECK seen >= M | GOOD -> colour -> size (step c2) | grade relay |
| 10 | No defect, GOOD votes enough, NECK not seen | Undefined (`not_rolled`) | reject |
| 11 | Anything else: too few frames, low confidence, class not mapped, no clear vote, size outside every grade | Undefined | reject |

- **REJECTION comes from YOLO until the CNN is retrained (owner's decision, 2026-10-09).** The CNN in use has
  no REJECTION class; the YOLO model has. Every frame keeps both results: the YOLO class and the CNN class.
  A frame in which YOLO says REJECTION is a REJECTION vote whatever the CNN says for that frame (today the
  CNN result overwrites the YOLO class; that stops). REJECTION is "seen" with the defect rule (>= 0.50 in at
  least 2 frames), as its own pair of values in the UI. It is checked before every other class, so no other
  vote can override it. The measured dark area % is logged beside it but does not decide yet. When a
  retrained CNN has REJECTION, its votes are added to the same rule.
- The order of rows 3-8 is the priority list, editable in the UI and stored in `priority_rules` (#22).
- NECK weight: a TESTA / TESTA SMALL vote from a NECK frame counts as W frames (UI). Start value: question 6.
- Skin %, dark % and spot count come from CNN votes for now (TESTA SMALL / TESTA / UPW, REJECTION, PKW1 / PKW2).
- CNN classes `SW` and `GOOD_CASHEW` both count as "no defect" votes; SW/WW is decided by colour (question 7).
- The old `opencv_filter` dark-spot filter stays in the code, default OFF.

### 3.5 Would image processing be more reliable than CNN votes for skin %, dark %, spot count?

- **Dark area %: yes.** Dark pixels inside the mask / mask area is a simple, stable measurement once the
  exposure is fixed. I recommend it for the "above 30 % dark" REJECTION rule, also because the CNN has no
  REJECTION class today. Data needed: 30-50 crops each of clearly rejected and clearly acceptable dark pieces.
- **Skin %: likely yes, to be proven.** Testa is red-brown, the kernel is cream, so a colour band in LAB
  inside the mask gives an area. Per ID the **maximum over frames** is used (skin hides under the neck). The
  measured value is % of the visible side, not of the whole surface; the 1 / 5 / 30 % limits would need
  re-tuning against the owner's judgement. Risk: confusion with BROWN and with the shadow in the neck hollow.
  Data needed: about 50 crops each of TESTA SMALL, TESTA, UPW, plus 50 clean NECK views, with the owner's label.
- **Spot count: frame count yes, across the rotation only approximately.** Counting dark blobs in one frame
  is easy. Not counting the same spot twice across frames needs the spot's position on a rolling body; I can
  only approximate it (position along the long axis + continuity between neighbouring frames). I would use
  "largest number of spots seen in one frame" plus "spots at clearly different positions along the length".
  Data needed: recordings of 20+ PKW1 and 20+ PKW2 cashews with the true spot count.
- Proposal: in (c1) these three values are measured and **logged and shown only**; the decision stays on CNN
  votes. After comparing them with the owner's labels we switch each rule separately.

### 3.6 Log and history

- `logs/cashews_YYYYMMDD.csv`: one row per ID: time, ID, ROI, frames, votes per class, views seen, final
  class, reason, colour value, SW/WW, size mm, grade, relay, fire time, dry-run / fired / LATE.
- `logs/frames_YYYYMMDD.jsonl`: per frame: class, confidence, view, features. Kept 7 days (UI).
- Click a history row to see the frame-by-frame detail.

---

## 4. Step (c2): colour SW/WW + size + px_per_mm

File: `colour_size.py`.

- Colour metric proposal: **median CIE LAB b\*** inside the eroded mask, after colour gain and FFC, glare and
  very dark pixels excluded; per ID the median over frames. b\* is the yellow-blue axis and is less tied to
  brightness than HSV saturation, which is unstable near white. L\*, a\* and HSV S are shown beside it.
- UI: SW/WW threshold slider with the live value; **teach helper**: run known WW, press "WW sample", run known
  SW, press "SW sample"; it shows both histograms and suggests the midpoint. The owner sets the final value.
- Warning from the old dataset: GOOD crops spread from b\* 13 to 46. That is far wider than one cashew type,
  so those crops were taken under different light or gain. The threshold only holds while exposure, gain,
  white balance, the software B gain (now 1.4) and the lamp stay fixed. White calibration first, then teach.
- Size: `minAreaRect` length and width, contour area, converted with px_per_mm. Per ID: length = median,
  width = max, area = max (proposal). Which one the grades use: question 2.
- px_per_mm calibration: click two ends of a ruler on the live image, type the mm. One value per ROI.
- Grade tables: one for SW, one for WW. Row = grade name, min, max, ROI1 relay, ROI2 relay. Overlap warning.
  Size outside every grade -> Undefined -> reject relay.

---

## 5. Step (c3): relay mapping + speed + match + centre fire + LATE

Files: `fire_timing.py`, `speed_sources.py`; `relay_output.py` stays the only path to a relay.

- **Relays 1 and 21 are reject only** (owner's decision, 2026-10-09): Double, Undefined, not rolled and
  REJECTION, nothing else. UPW was removed from relays 1 and 21 in `pneumatic_config.json`. The mapping page
  shows them as "REJECT - FIXED" and no class can be assigned to them.
- Class -> relay mapping page: ROI1 relays 2-20, ROI2 relays 22-40. The owner gives the full list later.
  Until then UPW has no relay; a class without a relay is Undefined -> reject relay (rule 11 in 3.4).
- Relay distance from its Decision line per relay (mm), "fill from first distance + spacing" button.
- **Three speed sources, shown side by side:**
  1. ENCODER: interface `ConveyorClock` -> `EncoderClock` (position counter from the PLC, mm/pulse). Designed
     now, shown as NOT AVAILABLE; implemented when the hardware details arrive (question 5). Reading the
     counter needs a read call on the PLC; if that means changing `innovance_plc.py`, I ask first.
  2. VISION: per-cashew speed = centroid travel in mm / frame-timestamp difference, fitted over the whole
     track; lane speed = smoothed median of recent cashews.
  3. MANUAL: belt speed from the UI (300 mm/s), fallback.
- MATCH: vision against encoder, continuously; warning and log line above X % (start 5 %). Until the encoder
  exists, the same check runs vision against manual. Calibration helper: from the comparison it suggests
  mm/pulse (trusting px_per_mm) or px_per_mm (trusting mm/pulse); the owner accepts or not.
- Scheduling: position-based with the encoder; otherwise time-based from the vision lane speed, manual speed
  if vision has too few cashews.
- Centre fire:
  `t_cross` interpolated between the last frame before and the first frame after the line (capture
  timestamps); `start = t_cross + (relay distance + offset mm) / speed - pulse/2 - valve delay + offset ms`.
  Pulse fixed (50 ms) or from cashew length and speed with min / max. Valve delay starts at 10 ms.
- Fire time already passed: **no fire**, logged LATE, LATE counter.
- Honest limit: at relay 20 / 40 (about 4.4 m) a 1 % speed error is about 44 mm. Without the encoder the far
  relays will not be centred reliably.

---

## 6. Step (d): double kernel + auto capture

- Double kernel: contour length, aspect ratio, area and two-box overlap shown live in px and mm; each test has
  its own ON/OFF and limit; **all OFF by default**. For reference only, the 52 old DOUBLE_KERNELS crops have a
  median aspect ratio of 2.39 against 1.74 for GOOD, with heavy overlap, so aspect ratio alone will not do it.
- Auto capture (own thread): `captures/<undefined|double|not_rolled|low_conf>/<date>/`, crop + full frame,
  file name `time_ID_ROI_class_conf_view`, saved after colour gain, 500 per category per day (UI), stops with
  "LIMIT REACHED", nothing is deleted automatically.

## 7. Step (e): UI rebuild from the 5-track project (CASHEW X AI)

**Order changed by the owner (2026-10-09): step (e) comes right after (c1), before (c2) and (c3).**
New order: (c1) done -> **(e)** -> (c2) -> (c3) -> (d) -> (f). Pages whose logic comes later are built in the
new look with what exists now; controls that do not work yet are shown greyed as `NOT AVAILABLE` (5-track rule:
never fake a feature).

### 7.1 Which copy is the real 5-track UI

Both folders on the external drive are the **same project** (same git history):

| Folder | Last commit | What it is |
|---|---|---|
| `TCM_CASHEWXAI_2026-10-06/source/` | `e1522f0`, 2026-10-07 00:54 | The backup, git tag `TCM_CASHEWXAI_v1.0`, with its documents, venv, wheels, drivers. |
| `CNN/CNN/app/` | `f78c172`, 2026-10-07 11:27 | The same project **11 commits later** (the working folder after the backup). |

The 11 later commits: Home clock + date box and pills that always fit, CLEAR HISTORY without password, bigger
customer logo (131 x 74), FIRE ALL / SEQUENCE ALL without popup, **relay mapping in the ejector test window**
(firmware 2.1.0, `machine_config.py`), user manual (EN / TA / HI), restore notes. Nothing was removed.
The theme module `home_ui.py` has the same colours, fonts and sizes in both; pages are the same.

So: `CNN/CNN/app/` is the **newest**; `TCM_CASHEWXAI_2026-10-06` is the documented backup one half-day older.
Both are the real 5-track UI. **I use `TCM_CASHEWXAI_2026-10-06` as the reference, as you said, and take the
later improvements from `CNN/CNN/app/` where they apply here** (clock/date box, pills that fit, relay mapping
view). Tell me if you want strictly the backup version only. Both stay read only.

### 7.2 Look to copy (from `home_ui.py`)

- Colours: background `#0E1420`, panel `#161E2E`, border `#263044`, text `#E6EAF2`, muted `#8A94A8`, accent cyan
  `#22B8CF`, good `#2FBF71`, warning `#F2A93B`, bad `#E5484D`, idle `#3A4558`, input field `#0B111B`.
- Four text sizes (13 / 16 / 22 / 34 px), 8 px grid. Font: Segoe UI there; not installed here, **Roboto** is.
- Layout: top bar 64 px (customer logo, PROGRAM box, status pills, clock + date, EXIT, SHUTDOWN), left menu
  176 px with drawn icons and the product logo at the bottom, camera in the centre, right column 696 px.
- Drawn widgets (Pillow, no image files): round chrome MACHINE START / STOP button with red / green glow,
  EXIT / SHUTDOWN buttons, LEDs, status pills, KPI tiles, flat outlined buttons, toggle switch, PASS / FIRE
  chips, OFF / SHADOW / ACTIVE segmented switch, sliders, histograms, progress bar, line chart.
- I port `home_ui.py` into this project as `ui_theme.py` (a copy adapted for Linux fonts; the source stays
  untouched) and rebuild every page with it. Design work does not change grading, tracking or relay logic.

### 7.3 5-track pages and features against this grader (today, after step c1)

| 5-track page / feature | GSN_V21 now | Plan for step (e) |
|---|---|---|
| **Look**: dark blue theme, drawn widgets, 176 px menu with icons, 64 px top bar | different (older slate theme, plain ttk buttons, PNG icons, 220 px menu) | rebuild all pages with the 5-track theme |
| **Top bar**: customer logo | missing | text only until you give the logo / name (question 16) |
| Top bar: PROGRAM box, NEW / COPY / DELETE (recipes: config + dataset per program) | missing | include (programs = full settings sets); question E1 |
| Top bar pills: TCM VISION, TCM32, SAFETY, CONVEYOR (mm/s), STATE, MODE, EJECTOR CHECK | different (camera + PLC health text, state badge) | pills: CAMERA, PLC, MASTER EJECTOR, MODE (LIVE / DRY-RUN), CONVEYOR (speed), STATE; question E2 |
| Top bar: clock + date, EXIT, SHUTDOWN buttons | clock only | include (SHUTDOWN asks for confirmation) |
| Welcome screen with startup ejector check (fires every lane) | missing | **not copied as it is**: firing 40 valves at start breaks "never fire at start". Offer a manual, DRY-RUN-aware check instead; question E3 |
| **Home**: camera with lane ROIs, rulers, decision line, flow arrows, CAM / AI fps | have (2 ROIs, line per ROI, IDs, view values) | same, new style |
| Home: lane LED strip under the camera | missing | LED per ROI (last result); per-relay strip on the Relay page |
| Home: round MACHINE START / STOP, state text | different (plain button) | include |
| Home: KPI tiles TOTAL, GOOD, BAD, REJECT %, THROUGHPUT kg/hr + pcs/min, RESET COUNTERS (password) | different (detected count only; counters on Sorting Logic) | tiles: TOTAL, GOOD, UNDEFINED, REJECTION, DOUBLE, NOT ROLLED, LATE, throughput; counts per grade and per relay; question E4 |
| Home: RECENT RESULTS (time, ID, lane, result / class, ejector result), alarms | have (ID, result, frames, relay; double click = frame detail) | same, new style, add SW/WW, size, grade columns when c2 is done |
| **Pneumatic**: lane table (relay, delay, pulse, TEST, status, fires today) | different (40-relay table with class mapping, distance, delay; single-relay test picker) | **Relay Mapping** page: 2 x 20 relay table in the same style (class / grade, distance, offset, pulse, TEST, status, count), relay 1 / 21 "REJECT - FIXED" |
| Pneumatic: REJECT CLASSES chips PASS / FIRE (one valve per lane) | different by design (class -> relay, many relays) | class -> relay chips per ROI replace PASS / FIRE |
| Pneumatic: MASTER EJECTOR switch, FIRE ALL, SEQUENCE ALL, SAVE | have (plus DRY-RUN; 1-40 only) | same, new style; DRY-RUN switch beside MASTER EJECTOR |
| Pneumatic: INSPECTION ENGINES (colour, shape, size, hybrid) OFF / SHADOW / ACTIVE, decision mode AI / COLOR ONLY, TEST ENGINES ON DATASET | missing (only the old dark-spot filter, default OFF) | engine rows shown: dark %, skin %, spots as SHADOW (logged only, as now); AI / COLOR ONLY mode `NOT AVAILABLE`; question E5 |
| Pneumatic: RECENT EJECTOR EVENTS (time, lane, source, delay, pulse, result) | partly (result line of the last test) | include, from `relay_output` events (DRY-RUN shown as such) |
| Pneumatic: AVG CASHEW WEIGHT for kg/hr | missing | include |
| **Ejector hardware test window**: connection, diagnostic counters, relay mapping, ALL RELAYS OFF NOW | partly (relay test on the Pneumatic page) | PLC test window: connection, counters (sent / dry-run / offline / refused), relay 1-40 test, sequence; ALL OFF for relays 1-40 only (never conveyor / feeder) |
| **Colour Size**: colour engine sliders, live preview with outline and spots, HOLD | missing (old threshold sliders) | page built; colour part filled by c2 (SW/WW b* slider, teach, live value) |
| Colour Size: size grades table (L / W / AREA min-max), TEACH from camera with histograms, shape limits, LAST CASHEWS | different (old L / W numbers, auto teach) | grade tables for SW and WW with relay columns; teach with histograms; filled by c2 |
| **Vision ROI**: lane ROIs DRAW / RESET, decision line SET / shift / orientation, RESET ALL | have (2 ROIs, line per ROI or linked, flow direction) | same, new style |
| Vision ROI: calibration (ROI length mm -> px per mm) | missing (px_per_mm typed) | include; two-point ruler calibration comes with c2 |
| Vision ROI: camera rotation 0 / 90 / 180 / 270 | missing | question E6 (do you need it here?) |
| Vision ROI: gain, exposure, colour gain R / G / B, TEST IMAGE CAPTURE, FFC `NOT AVAILABLE` | have, and more (FPS limit, FFC works, recorder, test source) | same, new style |
| **Teaching CNN**: CAPTURE / LABEL / TRAIN steps, image grid, classes with target counts, dataset warnings, inbox review, loss / accuracy charts, ACTIVATE A MODEL | partly (older page, external labeler, training without validation charts) | new layout now; working training with validation = step (f); `NOT AVAILABLE` until then where it does not work |
| Production capture into an inbox (quotas, never deletes) | missing | this is step (d) auto capture; inbox view added then |
| **Dataset Capture**: 20 class slots with counts, CAPTURE / VIEW, auto capture (crop / full / both) | have (older look) | same, new style |
| **Teaching YOLO**: annotate folder, class list, dataset numbers, training, one-class option | have (older look; TCM Studio copy) | same, new style |
| **AI Model Select**: YOLO and CNN cards with classes and accuracy, SET ACTIVE, class-change popup | partly (model list + upload) | include cards; accuracy shown only when recorded |
| **Network**: AnyDesk, camera network, this PC and ports, connection check table | different (PLC network settings) | PLC (Modbus IP / port, state), camera (USB3, model, serial), connection check; AnyDesk block; question E7 |
| **Info**: product, versions, hardware, software, today's numbers | missing | include, only facts the software knows |
| Hardware lock to one PC | missing | not copied unless you ask |
| Self-screenshot hook for checking pages without clicking | missing | include (`ui_debug.py` idea); I use it to show you each page |
| Daily summary CSV | missing (per-cashew CSV exists) | include with the Home counters |
| User manual, .exe build | missing | not in step (e) |
| -- **Only this machine** -- | | |
| Sorting Logic: voting limits, priority, view limits, live values, belt colour, tracker | have (plain look) | new style |
| Speed panel: encoder / vision / manual side by side, mismatch warning | missing (lane speed number only) | page built in (e) with vision + manual; encoder `NOT AVAILABLE`; logic in c3 |
| DRY-RUN, single-relay test fire | have | new style |
| Recording and replay (test source) | have | new style |
| Test Console page | have | kept, new style |

### 7.4 How step (e) is done (several commits, stop after each)

Feature reference for (e2)-(e4): the 5-track user manual v4.2, `docs/CASHEW_X_AI_Manual_English.pdf` (copied
from `CNN/CNN/app/manual/` on the drive; not committed, 15 MB), adapted to this machine: Inovance PLC instead
of the TCM32 board, 2 ROIs, 40 relays. Each sub-step: syntax check, tests, real screenshots taken by the
program itself, stop for the owner's OK.

| Sub-step | Content |
|---|---|
| (e1) **done** | `ui_theme.py`, top bar, pills, clock, EXIT / SHUTDOWN, left menu, **Home** |
| (e2a) **done** | **Logic behind the Pneumatic and Home pages** (no new page yet): delay formula `delay = distance / speed - pulse / 2` and the delay-table bug; always-reject reasons of manual B14 (LOST > 300 ms, APPEARED_AFTER_LINE, STUCK, DOUBLE longer than 45 mm); inspection engines OFF / SHADOW / ACTIVE with "would reject" logging (view rule and shape start in SHADOW); Home **ALARMS** box, **SAFETY** pill, ACKNOWLEDGE; AVG CASHEW WEIGHT (1.8 g) and kg/hr; fires today per relay |
| (e2b) | **Branding** (owner, 2026-10-09): `assets/cashew_x_ai_logo_clean.png` at the bottom of the left menu, scaled to the menu width, "GSN V21" and the software commit under it; top-left text and window title "TCM CASHEW X AI"; no Gulf logo. **Pneumatic page** in the manual's layout: relay table 2 x 20 (relay -> M coil, class / grade, distance, delay, pulse, TEST, status, fires today); class chips PASS / FIRE; engine rows OFF / SHADOW / ACTIVE with "would reject today"; RECENT EJECTOR EVENTS; AVG CASHEW WEIGHT; MASTER EJECTOR; DRY-RUN; FIRE 1-40 / SEQUENCE 1-40; SAVE PNEUMATICS. **PLC diagnostics window** like the Ejector Hardware Test: connection, counters (reconnects, write timeouts, lost, blocked, dry-run, refused), relay mapping with SAVE MAPPING (password), `logs/mapping_changes.log`, COPY DIAGNOSTICS, ALL EJECTORS OFF NOW (relays 1-40 only) |
| (e3) | **Vision ROI**: ROIs, line per ROI, **ROI length in mm -> px per mm**, gain / exposure / colour gain R G B, FFC, recorder. **Colour Size**: colour engine limits (dark spot level 0.43, min spot 0.2 mm2, biggest spot 0.7 mm2, max count 8, max dark area 4 %, views needed 2), live preview, **histogram TEACH** (mean +- k x std, k = 3), shape limits (solidity 0.75, area 100 mm2, length 16 mm, length / width 1.15), SW/WW and grade tables as shells until (c2). Sorting Logic is **done** (full width, SAVE bar, 2026-10-09). **Speed** shell. **Programs** (NEW / COPY / DELETE) |
| (e4) | **Teaching CNN**: CAPTURE / LABEL / TRAIN, **INBOX** production capture (60 % uncertain / 30 % reject / 10 % good, 5,000 per day, never deletes), dataset warnings, validation accuracy, test set (every 10th image). **AI Model Select**: SET ACTIVE with test-set comparison and PASS / FIRE popup. **Info**, Network, Dataset Capture, Teaching YOLO, Test Console. Daily summary CSV |

How the manual's one-valve ideas are adapted to 40 relays:

- **PASS / FIRE chips.** PASS = the class counts as "no defect" (today: GOOD_CASHEW, GOOD, SW). FIRE = the class
  is a defect; the chip shows where it goes: its relay (`R03`), or `REJECT` when it has no relay. REJECTION and
  DOUBLE KERNEL are fixed chips to the reject relay.
- **Inspection engines.** OFF = not measured; SHADOW = measured, "would reject" written to the log and shown,
  **never fires**; ACTIVE = can turn a GOOD into a reject (never the other way). Engines: View (NECK seen),
  Shape, Colour (dark spots / area), SW/WW colour, Size. All new rules start in SHADOW.
  **Consequence for the NECK rule:** with View in SHADOW a clean cashew is GOOD again and "not rolled" is only
  logged; it rejects only after you switch View to ACTIVE.
- **Relay mapping in the diagnostics window** = relay number -> PLC coil (M address), the table of
  `inovance_config.json`. M140 / M141 (conveyor, feeder) can never be given to an ejector relay. Saving goes
  through the PLC controller's own `update_config()`; `innovance_plc.py` is not changed.
- **Startup ejector check** stays a button (RUN RELAY CHECK, obeys DRY-RUN), not automatic (E3).
- **Password.** The 5-track code has its supervisor password written in the source. I do not copy a password
  into this code. The first time a protected action is used, the program asks you to set the supervisor
  password; only a salted hash is stored in `machine_config.json` (not committed).

### 7.4a Current relay mapping against the grading rules (checked 2026-10-09)

Kept as it is (A02 REJECTION ... A17 BUDDS, A22-A37 the same; relays 1 and 21 reject only). Conflicts:

| Mapping | Conflict | What happens today |
|---|---|---|
| A02 / A22 = REJECTION | The rules send REJECTION to the reject relay 1 / 21. | A02 / A22 never fire. Tell me if REJECTION should go to A02 / A22 instead. |
| UPW has no relay | UPW is the highest defect in the priority list. | UPW goes to the reject relay as "UNDEF no relay (UPW)". |
| A11 / A31 = SW | CNN `SW` is a "no defect" vote; SW or WW is decided by colour in (c2). | A11 / A31 fire only for a GOOD cashew with mostly SW votes whose size name has no relay. In practice: never. |
| A12-A16 = "SW180, W180" ... | One relay for the SW and the WW grade of a size. | SW and WW of one size go to the same relay; separate SW / WW tables come in (c2). |
| A10 / A30 = DW1 + DW2 | none (two classes may share a relay) | works; DW1 can only come from YOLO until retraining. |
| `size_config` T1/T2 relay numbers are all 1 / 21 | Old size table pointed every grade at relay 1 / 21. | Ignored: the class -> relay mapping (A12-A17) is used. |
| EDGECUT, DOUBLE KERNEL have no relay | intended | reject relay. |

### 7.5 Questions for step (e)

| # | Question | My proposal |
|---|---|---|
| E1 | Programs (recipes) in the top bar: wanted here? | Yes, as full settings sets (`programs/<name>/`), start with one program "Default" made from today's config. |
| E2 | Pill names: the 5-track screen says TCM VISION / TCM32. Here? | CAMERA and PLC until you give the branding. |
| E3 | Startup ejector check: it fires every valve at start. | No automatic firing. A "RUN RELAY CHECK" button that obeys DRY-RUN. |
| E4 | Home KPI tiles: which numbers do you want biggest? | TOTAL, GOOD, UNDEFINED, REJECTION, throughput; the rest smaller. Average cashew weight for kg/hr: you give it. |
| E5 | Inspection engines and COLOR ONLY mode from the 5-track machine: wanted here? | Not now: show measured values as SHADOW only. |
| E6 | Camera rotation control: needed? | No (the camera is fixed and the flow direction is already settable). |
| E7 | AnyDesk block on the Network page: wanted? Screen resolution of this machine's monitor? | Yes if AnyDesk is installed. I read the resolution from the running program. |

## 8. Step (f): training

`train_cnn.py`: reads the dataset folders, skips empty folders and non-crops, splits train / val / test so
that frames of one cashew stay on one side, same `preprocess.py` as inference, class weights, best-val
checkpoint, accuracy + confusion matrix saved. A new model is never activated automatically.
View classifier (TOP / BACK / NECK / OTHER) trained the same way from the labeler's view mode.

### Dataset for the retraining (owner, 2026-10-09)

The owner retrains the CNN with all classes. Step (f) delivers **one clean dataset folder per final class
name**, with exactly the names the grading rules use: GOOD, SW (only if still wanted), UPW, PKW2, PKW1, OIL
(including BACK_CUT_OIL / OIL_BACKCUT), BROWN, DW2, DW1, TESTA, TESTA SMALL, WHITE_DW, REJECTION, EDGECUT,
DOUBLE KERNEL (if it becomes a class). Built by **copying** from the old and the project datasets (nothing is
moved or deleted; full frames and duplicates are left out and listed). With it comes a table: images per
class now, and how many each class still needs.

## 9. Testing

Each step: `ast` check, unit tests, replay of a recording through the file backend in DRY-RUN.
Only testable on the machine: valve timing, centre-fire accuracy, real voting / colour / size values,
cable-pull reconnect, AI FPS under load. MASTER EJECTOR OFF and DRY-RUN ON at every start; no real relay is
fired in any test.

How to record (already works): Vision ROI page, RECORD button with the duration in seconds, frames go to
`recordings/<date_time>/` as JPEG with the capture time in the file name. Replay: choose "Test source", pick
that folder, APPLY.

---

## 10. Open questions

Answered by the final prompt and no longer open: view definitions, voting start values, priority order,
LATE = no fire, capture limit 500, frame logs 7 days, old dark-spot filter kept OFF, valve delay 10 ms,
SW/WW by colour with teach helper, double kernel OFF with live values.

| # | Question | Needed before |
|---|---|---|
| 1 | **Class -> relay mapping** for relays 2-20 (relays 1 / 21 are now reject only, UPW removed, REJECTION goes to 1 / 21). The config still has 2 = REJECTION, 3 = TESTA, 4 = TESTA SMALL, 5 = PKW2, 6 = PKW1, 7 = OIL, 8 = BROWN, 9 = WHITE_DW, 10 = DW1 + DW2, 11 = SW, 12-16 = size grades, 17 = BUDDS (ROI2 the same + 20). The owner will give the full list later. | c3 |
| 2 | **Grade tables** for SW and for WW: grade names, min / max, relay. Limits in length, width or area? The config has length limits W180 30, W210 28, W240 26, W320 24.5, W450 18, BUDDS 16 mm: is that the starting table, and are SW and WW limits the same? | c2 |
| 3 | **Relay distances.** Distance from the Decision line to relay 1 and to relay 21? Spacing 220 mm for all? The config has 393.7 mm then about 195 mm each: from which machine? | c3 |
| 4 | `px_per_mm` in the config is 10.0. Real or placeholder? (It will be calibrated in c2 either way.) | c2 |
| 5 | **Encoder:** pulses per revolution, mm per pulse, which PLC input / which register holds the count, counter size and reset behaviour. | when available |
| 6 | NECK weight for TESTA / TESTA SMALL: start value W (my proposal: a NECK frame counts as 2 frames)? | c1 |
| 7 | CNN says `SW` with no defect: treat it as a "no defect" vote like GOOD (my proposal), so colour alone decides SW/WW? | c1 |
| 8 | **Answered 2026-10-09:** REJECTION is triggered by the YOLO REJECTION class until the CNN is retrained (section 3.4). Still open: should the measured dark area above 30 % also trigger it later, after you have seen the logged values? | later |
| 9 | BACK_CUT_OIL / OIL_BACKCUT (69 images) and EDGECUT: which of your classes do they belong to (OIL? their own class? drop)? Also: does WHITE_DW count as "another defect" for the "skin + any other defect = UPW" rule? | c1 / f |
| 10 | "GOOD >= 3 views at >= 0.80": 3 frames, or 3 different view types? I assume 3 frames. | c1 |
| 11 | One Decision line per ROI: independent positions, or an option to link them (my proposal: both, linked by default as now)? | c1 |
| 12 | Pulse: fixed 50 ms as the start, with "from cashew length" as an option? | c3 |
| 13 | Double kernel as a CNN class too (52 images exist), or box measurements only? | d |
| 14 | Bayer check needs a frame with white cashews on the blue belt (so far only rollers). Were the four sample frames taken by this camera through this program? If yes, the colours are confirmed natural. | c2 |
| 15 | **A recording with cashews** (10-20 s, both lanes, known cashew count) is needed for the tracker comparison and view tuning. | c1 |
| 16 | Branding name and logo. | e |

---

### Answers received 2026-10-09

| # | Answer |
|---|---|
| 6 | NECK weight 2 (a TESTA / TESTA SMALL vote from a NECK frame counts as 2 frames). |
| 7 | CNN `SW` with no defect is a "no defect" vote like GOOD; colour alone decides SW / WW. |
| 8 | REJECTION from the YOLO class, confidence >= 0.50 in at least 2 frames. Dark area % is only logged. |
| 9a | BACK_CUT_OIL / OIL_BACKCUT = **OIL** (merged for grading and training). **EDGECUT** = broken-edge cashew, its own class, reject relay until the relay list is given; only 2 images, more to be collected. |
| 9b | WHITE_DW counts as "another defect": skin + WHITE_DW = UPW. |
| 10 | GOOD needs 3 frames at >= 0.80 (frames, not view types). |
| 11 | One Decision line per ROI with a "one line for both" switch, ON at start. |

## 11. Approval

Steps (c1) and (e1) are done. (e2a) is done. Next: (e2b), the Pneumatic page and the PLC diagnostics window. Questions E1-E7 have proposals; without answers I follow the proposals.
The recording (question 15) is still needed for tracker tuning and view limits.
