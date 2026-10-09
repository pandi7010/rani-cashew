# PLAN v3: GSN_V21 cashew grader

Date: 2026-10-09
Status: **waiting for the owner's approval.** No code was changed for this plan.
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
| 2 | REJECTION seen | REJECTION | reject |
| 3 | UPW seen, or skin (TESTA / TESTA SMALL) seen together with any other defect | UPW | mapped |
| 4 | PKW2 seen | PKW2 | mapped |
| 5 | PKW1 seen | PKW1 | mapped |
| 6 | OIL, BROWN, DW2, DW1 seen (in this order) | that class | mapped |
| 7 | TESTA seen, then TESTA SMALL seen | that class | mapped |
| 8 | WHITE_DW seen | WHITE_DW | mapped |
| 9 | No defect, GOOD votes enough, NECK seen >= M | GOOD -> colour -> size (step c2) | grade relay |
| 10 | No defect, GOOD votes enough, NECK not seen | Undefined (`not_rolled`) | reject |
| 11 | Anything else: too few frames, low confidence, class not mapped, no clear vote, size outside every grade | Undefined | reject |

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

- Class -> relay mapping page: ROI1 relays 2-20, ROI2 relays 22-40; relays 1 and 21 fixed as reject.
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

## 7. Step (e): UI rebuild

Style of `CNN/CNN/app/` on the external drive (read only; it is mounted): dark blue, cyan accent, top status
pills (CAMERA, PLC, MASTER EJECTOR, MODE, STATE), left menu, camera in the centre, START/STOP and KPI tiles on
the right, history. Pages: Home; Relay Mapping (mapping, distance, offset, pulse, valve delay, MASTER EJECTOR,
DRY-RUN, single-relay test); Sorting Logic (#16: priority, voting, view tuning, SW/WW + teach, grade tables,
px_per_mm, double limits, capture limit); Speed (encoder / vision / manual, mismatch); Vision ROI (ROI1, ROI2,
one Decision line per ROI, camera settings, recording); the existing teaching pages. Counters per relay and
per grade; Undefined / Double / not_rolled / LATE. Branding later.

## 8. Step (f): training

`train_cnn.py`: reads the dataset folders, skips empty folders and non-crops, splits train / val / test so
that frames of one cashew stay on one side, same `preprocess.py` as inference, class weights, best-val
checkpoint, accuracy + confusion matrix saved. A new model is never activated automatically.
View classifier (TOP / BACK / NECK / OTHER) trained the same way from the labeler's view mode.

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
| 1 | **Class -> relay mapping.** The config now has relay 1 = UPW, 2 = REJECTION, 3 = TESTA, 4 = TESTA SMALL, 5 = PKW2, 6 = PKW1, 7 = OIL, 8 = BROWN, 9 = WHITE_DW, 10 = DW1 + DW2, 11 = SW, 12-16 = size grades, 17 = BUDDS (ROI2 the same + 20). Relay 1 / 21 must be reject, so this must change. Please give the mapping for relays 2-20. | c3 |
| 2 | **Grade tables** for SW and for WW: grade names, min / max, relay. Limits in length, width or area? The config has length limits W180 30, W210 28, W240 26, W320 24.5, W450 18, BUDDS 16 mm: is that the starting table, and are SW and WW limits the same? | c2 |
| 3 | **Relay distances.** Distance from the Decision line to relay 1 and to relay 21? Spacing 220 mm for all? The config has 393.7 mm then about 195 mm each: from which machine? | c3 |
| 4 | `px_per_mm` in the config is 10.0. Real or placeholder? (It will be calibrated in c2 either way.) | c2 |
| 5 | **Encoder:** pulses per revolution, mm per pulse, which PLC input / which register holds the count, counter size and reset behaviour. | when available |
| 6 | NECK weight for TESTA / TESTA SMALL: start value W (my proposal: a NECK frame counts as 2 frames)? | c1 |
| 7 | CNN says `SW` with no defect: treat it as a "no defect" vote like GOOD (my proposal), so colour alone decides SW/WW? | c1 |
| 8 | REJECTION: the CNN in use cannot output it. Until retraining, use YOLO's REJECTION class plus the measured dark area above 30 % (my proposal)? | c1 |
| 9 | BACK_CUT_OIL / OIL_BACKCUT (69 images) and EDGECUT: which of your classes do they belong to (OIL? their own class? drop)? Also: does WHITE_DW count as "another defect" for the "skin + any other defect = UPW" rule? | c1 / f |
| 10 | "GOOD >= 3 views at >= 0.80": 3 frames, or 3 different view types? I assume 3 frames. | c1 |
| 11 | One Decision line per ROI: independent positions, or an option to link them (my proposal: both, linked by default as now)? | c1 |
| 12 | Pulse: fixed 50 ms as the start, with "from cashew length" as an option? | c3 |
| 13 | Double kernel as a CNN class too (52 images exist), or box measurements only? | d |
| 14 | Bayer check needs a frame with white cashews on the blue belt (so far only rollers). Were the four sample frames taken by this camera through this program? If yes, the colours are confirmed natural. | c2 |
| 15 | **A recording with cashews** (10-20 s, both lanes, known cashew count) is needed for the tracker comparison and view tuning. | c1 |
| 16 | Branding name and logo. | e |

---

## 11. Approval

On "approved, start (c1)" I do step (c1) only, commit, report, and stop.
Step (c1) needs answers 6-11 and the recording (15). Without the recording I can build and unit-test (c1),
but tracker choice and view thresholds stay open.
