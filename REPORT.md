# Reports (newest on top)

New entries are in English. Older entries below are in Tamil.

---

## 2026-10-13 (9): Dataset cleaned (yellow light, white dots, test set); steadier training; START blocked while training

Owner: "check and solve all the issues, you can delete the yellow lighting images". The real PLC and the camera were
not contacted. Nothing was deleted for good: every removed image is in `programs/Default/trash/20261013_134423_cleanup/`,
and `changes.csv` in that folder lists every file (from, to, why).

### Dataset (`programs/Default`, not in git)

| # | What | Count |
|---|---|---|
| 1 | Yellow-light images (captured 3 to 7 Sept, names with `_cashew_`, and the yellow ones of `old_crop_NNNN`) taken out of the **dataset** | 357 (PKW2 183, PKW1 147, BROWN 27) |
| 2 | The same images put back **colour corrected** to today's light, as `wb_<old name>.jpg` | 357 |
| 3 | Yellow-light images taken out of the **test set** (no corrected copy there) | 103 |
| 4 | Images with a white dot drawn on the cashew taken out | 46 |
| 5 | Test images whose cashew also had frames in the dataset moved into the dataset | 68 |
| 6 | New test set: whole cashews held back, about 10 % per class, never more than 30 % of the real images of a class | 229 images, 14 classes |

Why corrected and not only deleted: without the yellow images only 51 PKW1 and 27 PKW2 images are left. I trained both
ways and tested on real white-light PKW1 cashews that neither model had seen: yellow images deleted = 12 of 23 found,
yellow images colour corrected = 20 of 23 found, and no more GOOD cashews called PKW (0 of 37 against 2 of 37).
The correction is one fixed gain per colour channel (blue x 1.56, green x 1.19, red x 1.17 for the dark yellow session;
1.08 / 0.82 / 0.81 for the bright one), taken from GOOD cashews of both sessions. It is an approximation: real PKW images
with today's settings are still better. To undo: delete the `wb_*.jpg` files and move the trash images back.

After the clean-up (dataset / test set): GOOD_CASHEW 670 / 74, UPW 379 / 42, PKW2 202 / 8, TESTA 197 / 23, PKW1 189 / 9,
TESTA SMALL 108 / 12, BROWN 97 / 11, OIL 85 / 9, REJECTION 81 / 9, SW 67 / 9, KNIEF_MARK 59 / 6, DOUBLE 50 / 6,
WHITE_DW 43 / 5, DW2 35 / 6, DW1 8 / 0, EDGECUT 2 / 0, JH 1 / 0. No cashew is in both parts.

### Code (one commit)

1. `training.py`: label smoothing 0.1, AdamW with weight decay, learning rate that rises and then falls to zero
   (OneCycle), fixed random seed. Before, the validation accuracy jumped between 60 % and 82 % from epoch to epoch and
   the "best epoch" was luck; now the last 10 epochs stay within 1 point.
2. `cashew_cnn_grader.py`: START is refused while a training runs (message: wait or stop the training). Only this
   change of that file is in the commit; the other uncommitted edits of that file are untouched.

### Measured on the new test set (229 real white-light images, cashews never trained on)

| Training (30 epochs, all 14 classes with 15+ images) | validation | test set | GOOD_CASHEW found |
|---|---|---|---|
| old recipe | 82.2 % | 64 % | 38 of 74 |
| new recipe (in the code now) | 79.8 % | 69 % | 49 of 74 |
| new recipe, **SW not ticked** | 85.7 % | 77 % | 60 of 74 |

One run each, so a few points up or down is noise. The old models of 10 Oct score 80-84 % on this test set, but they
were trained on most of these images, so that number does not count.

**SW is the biggest loss:** with the SW class 16-17 of 74 GOOD cashews are called SW. In the pictures SW and GOOD look
the same. I did not move the SW folder (yours, made on 10 Oct); in the class list before training you can untick SW.

Still weak in every run: TESTA against TESTA SMALL (3 to 7 of 23 mixed up), PKW2 (3 of 8; 3 more called PKW1),
WHITE_DW (1 of 5), OIL against KNIEF_MARK.

### Not solved, and why

- The 7 failing decision tests and L1 / L2 / L3 of entry (7): they depend on voting values and rules that are yours
  (defect / GOOD / REJECTION confidence and frames, the UPW rule, what a low-confidence frame counts as).
- SW as a class, OIL / KNIEF_MARK as one or two classes, TESTA / TESTA SMALL limit: your grading decisions.
- DW1 (8), EDGECUT (2), JH (1): too few images to train; only new captures help.
- `review_not_good_from_GOOD_CASHEW/` (107 images I sorted out of GOOD by eye): still waits for your look.

### Tests

`ast` clean. Unit tests 198 pass, 7 fail (the same 7 as before). Whole-program self-test with the fake PLC
(`tools/ui_selftest.py` + `tools/selftest_teaching.py`): passes, including START / STOP and a Teaching CNN training.
Not tested: the new START message on the real screen during a real training; any model on real cashews.
No model was put into `saved_models`; the active CNN is unchanged.

---

## 2026-10-13 (8): Dataset and CNN training check; validation split fixed

Looked at all images of `programs/Default/dataset` and `testset` (class sample sheets, every GOOD_CASHEW image), read
`training.py`, `preprocess.py`, `training_ui.py`, `training_report.py`, `dataset_import.py` and the CNN part of the grader,
and ran comparison trainings on the PC (30 epochs, 14 classes, results in a scratch folder, nothing in `saved_models`).
The real PLC and the camera were not contacted. The rest of the program was checked this morning in entry (7); not repeated.

### Changed (one commit: `training.py`, `tests/test_teaching.py`)

1. **Validation split by cashew.** A cashew is photographed many times while it rolls (UPW: 408 images of about 55
   cashews; GOOD_CASHEW: 3 cashews have 91 frames each = 273 of 747 images). The old split shuffled single images, so
   342 of 486 validation images had frames of the same cashew in the training part. The validation accuracy and the
   "best epoch" were measured on cashews the model had already seen. Now all frames of one cashew go to the same part
   (`training.cashew_groups`, from the file name: `crop_<time>_<track id>` and `<HHMMSS>_<id>_v<n>`). The result file
   has the new field `val_cashews` (validation cashews per class).
2. **An unreadable image no longer ends a training run**; it is left out and listed in `unreadable_images`.

Limit: when the tracker gave one cashew two ids, its frames can still be on both sides. Names without time and id
(`old_crop_0002.jpg`) count as one cashew each.

### Measured with the new split (30 epochs, pre-trained MobileNetV2, RTX 3050, about 7 minutes)

| | validation | test set (215 images) |
|---|---|---|
| new split, training unchanged | 81.5 % (balanced 81.3 %), best epoch 8 | 55 % |
| new split + milder crop / no saturation change | 84.3 % (balanced 81.2 %), best epoch 16 | 60 % |
| old model `cnn_Default_20261010_195822` (old split, as reported then) | 87.3 % | 74 % (DOUBLE, DW2 not known) |
| old model `cnn_Default_20261010_162504` | 84.2 % | 81 % |

- Validation accuracy jumps between 61 % and 85 % from epoch to epoch: with whole cashews the validation part has few
  cashews (SW 2, WHITE_DW 1, KNIEF_MARK 3, DW2 3), so one cashew moves the number a lot. The number is honest now, but coarse.
- The milder augmentation is within that noise. It is **not** in the code. A third run (label smoothing, cosine learning
  rate) was lost when the session restarted; no result.
- The test-set numbers are low for a reason that is in the data, see D1.

### Dataset findings, most important first

**D1. Classes are tied to capture sessions with different lighting.** Cashew colour b* / background blue:
GOOD_CASHEW in the dataset 15-22 / 48-50; PKW1 31 / 32; PKW2 28 / 26 (180 of 198 PKW1 and 201 of 208 PKW2 images are from
the old yellow-looking session). The test set still has yellow-session GOOD images (b* 27.5): the new model calls 28 of
63 of them PKW1. So the model partly learns "yellow picture = PKW". With today's camera settings only 18 PKW1 and 7 PKW2
images exist. Whether PKW is found on the machine today cannot be judged from these numbers.

**D2. Classes that look the same in the pictures.** SW and GOOD_CASHEW (b* 21.6 vs 20.8; SW 53-77 % in every training);
OIL and KNIEF_MARK (both a brown line along the seam); TESTA and TESTA SMALL (no clear size limit in the images).
Owner's rule of 2026-10-09 was "no SW class", but a `SW` folder with 76 images exists and was trained (see L6 in entry 7).

**D3. Too few images:** JH 1, EDGECUT 2 (both show the same cashew), DW1 8 (about 5 cashews), DW2 37 (17 cashews),
WHITE_DW 46 (17 cashews).

**D4. 46 images have a white dot drawn on the cashew** (GOOD_CASHEW 30 + 3 in the test set, TESTA SMALL 8, PKW1 3 + 1,
OIL 1). They come from an older program version; today's capture code saves the clean frame.

**D5. Test set:** about a third of its images have frames of the same cashew in the training dataset, and it has no
SW, KNIEF_MARK, DW1 images. Its numbers on the AI Model Select page are for the old sessions.

**D6. GOOD_CASHEW was sorted by eye today** (owner's request): 107 images that did not look like a clean whole kernel
were moved to `programs/Default/review_not_good_from_GOOD_CASHEW/<type>/` (outside the dataset, so they are not a
class). `moved_files.csv` in that folder lists them. Owner's correction applied: small specks on a light kernel are GOOD.

### Other things seen in the training code (not changed)

- START is not blocked while a training runs (training is blocked while RUNNING, not the other way round). Grading
  and training would share the graphics card.
- A low-confidence defect becomes a GOOD vote (L2 in entry 7): still open.
- No fixed random seed: two trainings on the same data give slightly different models.

### Tests

`ast` clean. Unit tests: 198 pass, 7 fail; the 7 are the decision-rule tests of L1 in entry (7), unchanged. 2 new tests
(split by cashew, unreadable image) pass. Not tested: the Teaching CNN page on the machine screen with the new split,
and any new model on real cashews.

### Questions for you

1. D1: can you capture PKW1, PKW2 (and DW1, DW2, WHITE_DW) again with today's camera settings? Or shall I plan a colour
   correction from the blue roller (same step in training and in live grading; a big change, plan first)?
2. D2: SW as a CNN class, yes or no? Keep OIL and KNIEF_MARK apart, or one class?
3. Rebuild the test set by whole cashews (moves files between `dataset` and `testset`)?
4. Block START while a training runs?

---

## 2026-10-13 (7): Health and logic check of the whole program

Read-only check. Nothing was changed in code or settings, the real PLC and the camera were not contacted.

### Health: good

| Check | Result |
|---|---|
| Syntax (`ast`), undefined names, attributes never set | clean in all 41 files |
| Whole-program self-test, fake PLC (`tools/ui_selftest.py` + pages + teaching + speed) | passes: all 12 pages and tool windows open, START / STOP, manual fire only when STOPPED, addresses written 100-139 only, no forbidden PLC call, STOP cancels waiting pulses, size table / COLOUR SPLIT / PASS chips / change log, Teaching CNN (train, confusion, undo, graphs) |
| `logs/grader.log` (3,565 lines, 35 program starts on 10 and 11 Oct) | 0 errors, 0 tracebacks. 2,350 pulses OK, 2 not fired (PLC offline for a moment: 10 Oct 18:37, 11 Oct 01:10) |
| Speed on the machine (LOOP lines in the log) | 63-64 fps, 11.7 ms per frame (YOLO 8.6 ms) |
| Disk / GPU | 29 GB free (75 % used), RTX 3050 idle, 40 C |
| Unit tests | **196 pass, 7 fail**, all 7 in the decision rules, see L1 |
| Safety rules in code | hold: ejectors only through `relay_output.py`, relays 1-40, M140 / M141 refused, opens STOPPED |

Smaller health notes:
- `logs/cashews_20261010.csv` has 205 empty (NUL) bytes in the middle: the PC was switched off or the program was killed while writing. The rows around it are readable.
- Simulated full belt (14 cashews in view): the loop needs 42 ms per frame (CNN 17, YOLO 9, features 6, screen picture 5) = at most 24 fps, i.e. about 9 frames per cashew instead of 25. Simulation only; the real limit must be measured on a full belt.
- `cashew_class_mapping.json` differs from the YOLO model for PKW1 / PKW2 / TESTA SMALL (warning at every start; the model's numbers are used, no effect on grading).
- A lot of work is not committed: the speed step (`perf.py`, batch CNN, faster Teaching page, `tools/profile_grader.py`) plus later edits in `cashew_cnn_grader.py` (+686 lines), `decision.py`, `teach_ui.py`. 1,370 files of `cnn_final/golden_crops` are deleted in the working tree (still in git).

### Logic: problems found, most important first

**L1. Decision rules were changed after the last commit; tests fail.** `decision.py`, 10 Oct 19:20:
- the rule "skin (TESTA / TESTA SMALL) + any other defect = UPW" is removed (it decided 16 cashews on 10 Oct);
- start values changed to defect 0.60, GOOD 0.70, REJECTION 0.75 / 3 frames (were 0.50 / 0.80 / 0.50 / 2). These do **not** act on the machine: the saved settings (0.50 / 0.80 / 0.50 / 2) win.
I do not know whether you asked for these. Tests and code must agree again, in whichever direction you say.

**L2. A defect below its "class confidence" is counted as GOOD.** New uncommitted code (`cashew_cnn_grader.py:1559-1563`): a CNN result below the per-class confidence is renamed GOOD_CASHEW and keeps its confidence. Saved now: **UPW 95 %, REJECTION 85 %**, others 50 %. So a UPW frame seen at 80-94 % becomes a GOOD vote (GOOD needs 80 %). On 10 Oct 17 % of all UPW frames were below 95 %. A real UPW can now leave as a good grade. Safer: such a frame gives no vote at all.

**L3. Too many UPW: 2 frames out of about 25 are enough.** 10 Oct after 16:00: 641 of about 1,300 decided cashews were UPW. In 236 of them most frames said GOOD_CASHEW; in 193 fewer than a quarter of the frames said UPW. The rule "defect in 2 frames" was set when a cashew had few frames; at 63 fps it has about 25, so two wrong frames decide it. Raising the UPW confidence does not help (83 % of UPW frames are above 95 %). What would help: a limit as a share of the frames (for example "UPW in at least 30 % of the frames") or a higher N. The value is yours.

**L4. SIZE GRADES READY is ON, but px per mm is not calibrated.** `size_config` has no `roi_length_mm`; the program uses the placeholder 10 px per mm. All lengths (median 23 mm in the log) and all W grades come from that guess. Hint only: belt speed in the log is 2.48 px/ms; if the belt really runs 300 mm/s that is about 8.3 px per mm, so real lengths would be about 20 % larger than shown. Calibrate on Vision ROI (ROI 1 real length) before trusting any grade.

**L5. The CNN in use is the old one.** Active: `cashew_color_cnn_new.pth` of 16 Sept (11 classes). The five models trained on 10 Oct are not activated (no `active_models.json`). Their validation accuracy is 83-91 %, the last one (19:58, 12 classes) 87 %, GOOD_CASHEW recall 88 %: about 1 good cashew in 8 is called something else per frame. That is too low for N = 2 voting (see L3).

**L6. SW from the CNN would go to the reject ejector.** Saved PASS list is GOOD, GOOD_CASHEW (SW is not in it) and "SW" stands in the priority list. A cashew with 2 CNN frames "SW" becomes result SW, no ejector is mapped for "SW", so it is rejected as "no ejector". Not seen in the logs yet (0 SW frames), but it will happen with a model that outputs SW.

**L7. Many LOST rows.** 10 Oct: 344 of 2,159 rows; 11 Oct: 79 of 169. LOST = seen but gone before the line, nothing fired. Part of it is not cashews: the same spots repeat (ROI 2 near x 2231 / y 1988, ROI 1 near x 2018 / y 92; standing objects at the ROI edge) and tiny slow boxes. Four objects stood in ROI 2 for 1,500 frames (STUCK -> reject pulse). Whether real cashews are lost can only be seen on the machine: compare the cashews fed with the TOTAL counter.

Smaller:
- COLOUR SPLIT is OFF and no threshold is saved: all good cashews use the W grades (as intended for now).
- Engines view / shape / colour / size are all SHADOW: NECK view and shape never reject.
- Delays of ejectors 19, 20, 39, 40 (about 12.9 / 13.6 s) are still the calculated ones and smaller than ejector 18 (14.3 s). They have no class, so nothing fires there now.
- CONVEYOR / FEEDER buttons write relay 41 / 42 (M140 / M141) through the PLC controller, not through `relay_output.py`. This is their purpose; noted only because all other outputs go through one place.
- `decision.py:198` leaves an unused line (`others = ...`) from the edit in L1.

### Not tested

Real valves, real cashews, timing, the camera, the real PLC. Accuracy numbers come from the logs of 10 / 11 Oct and from the model reports, not from a new run.

### Questions for you

1. L1: keep the removed UPW rule removed? Which start values are right?
2. L2: a result below the class confidence: no vote (my proposal), GOOD as now, or remove the rule?
3. L3: which rule for a defect: N frames, or a share of the frames, and what value?
4. Then: finish and commit the speed step?

---

## 2026-10-10 (6): Manual size table, SIZE GRADES READY, COLOUR SPLIT, PASS classes to size sorting

This replaces parts of entry (5): the size table is now MIN / MAX length only, and SIZE GRADES READY no longer needs
teaching. Where (5) and this entry differ, this one is right.

### What changed

| # | Asked | Now |
|---|---|---|
| 1 | Editable size table | Colour Size page: rows W180, W210, W240, W320, W450, BUDDS, SW180 ... SW450, SW_BUDDS with **MIN length mm, MAX length mm, EJECTOR ROI 1 / ROI 2** (read-only, from the ejector mapping; "no fire" for BUDDS). Start values as you gave them: W180 >= 30, W210 28-30, W240 26-28, W320 24.5-26, W450 18-24.5, BUDDS < 18; SW the same. Rule: MIN <= length < MAX, empty = no limit. **COPY WW -> SW**. **SAVE COLOUR & SIZE** asks with the list old -> new. TEACH is still there, optional (sets MIN / MAX of one grade). |
| 2 | Big switch | **SIZE GRADES READY: ON / OFF** at the top of the size card. ON: GOOD cashews go to their grade ejector from this table (no SIZE_NOT_READY). It works with the start table at once. A red warning stays on the page while px per mm is not calibrated, and the confirmation says so; it is not blocked. |
| 3 | Calibration | Vision ROI: **ROI 1 real length (mm)** and **ROI 2 real length (mm)**, each shows its px per mm (ROI 2 empty = same as ROI 1). **MEASURE OBJECT**: measures the largest object in each ROI of the picture on screen and shows its length and width in mm (works STOPPED, nothing fires). |
| 4 | SW / WW | Threshold field + slider, live value of the current cashew, **COLOUR SPLIT: ON / OFF**. OFF (start state): all GOOD cashews use the W grades. Start value in the field: **b\* 23, SW BELOW** - see the note. |
| 5 | Live view / results | Under every cashew: `24.6 mm  b* 23.1  W320`. RECENT RESULTS: CLASS / RESULT, SW/WW, COLOUR, L mm, GRADE, EJECTOR. Double click: median length and the length of every frame, and per frame L, W, b\*. |
| 6 | Change log | `logs/settings_changes.log`: every change of the size table, threshold, value, side, shape limits, calibration, SIZE GRADES READY, COLOUR SPLIT and PASS / FIRE, each line `old -> new` with the time. |
| - | PASS classes | Pneumatic page chips. A class set to PASS (e.g. TESTA SMALL) is treated like GOOD: colour -> size grade -> grade ejector, **not** its own ejector. RECENT RESULTS: `TESTA SMALL (PASS) -> W240 -> A13`. GOOD_CASHEW is always PASS; REJECTION, DOUBLE, Undefined and the reject classes (EDGECUT) can never be PASS. Every chip change asks first, is logged and saved in the program's settings. |

**Note on the b\* start value.** In entry (5) I proposed a method, not a number. The 23 / BELOW comes from your dataset (SW b\* 22.4,
GOOD 24.8) and separates those images badly, which is why COLOUR SPLIT starts OFF. Set the real value on the machine
(watch the live b\* of known WW and SW cashews, or use TEACH WW / TEACH SW).

**Found by the new tests and fixed:** if DOUBLE had been written into the PASS list, the decision would have treated a
double as GOOD. REJECTION, DOUBLE and the reject classes are now ignored in the PASS list by the decision itself.

### Tested (fake PLC, simulated belt, temporary settings)

- `ast` check; **203 unit tests pass** (`tests/test_colour_size.py` start table and boundaries, gaps, changes old -> new;
  `tests/test_pass_classes.py`: TESTA SMALL as FIRE -> ejector 9 / 29, as PASS -> W240 -> 13 / 33, SW240 -> 18 with the split on,
  BUDDS size not fired, other defects unchanged, REJECTION / DOUBLE never PASS).
- Self-test `tools/selftest_pages.py`: calibration of both ROIs and MEASURE OBJECT; SHADOW shows "would be"; table change + COPY +
  SAVE (6 changes logged); READY on -> every cashew at the ejector of its grade (W240 -> A13, W320 -> A34); COLOUR SPLIT on -> SW
  grades and SW ejectors (A18, A19, A39); TESTA SMALL as FIRE -> A09 / A29, as PASS -> `TESTA SMALL (PASS) -> W240 -> A13`; 17 log lines.
  (The test forces the class of the simulated cashews.) Pictures: `design/pages/04_colour_size.png`, `01_home_live.png`, `05_vision_roi.png`, `02_pneumatic.png`.

### Not tested

- The real machine: lengths in mm against real cashews, the b\* of real SW / WW, timing of the grade ejectors A11-A20 / A31-A40
  (their delays are still the ones calculated for 300 mm/s).

### Exact buttons

**Calibrate (once):** STOP -> **Vision ROI** -> type the real length of the yellow ROI 1 box into **ROI 1 real length** (and ROI 2 if
it differs) -> **SAVE VISION SETTINGS**. Check: lay an object of known length on the belt -> **MEASURE OBJECT** -> read "ROI 1: xx.x mm long".

**Tune the size table:** **Colour Size** -> edit **MIN length mm / MAX length mm** -> (**COPY WW -> SW** if SW should be the same) ->
**SAVE COLOUR & SIZE** -> read the old -> new list -> Yes.

**Switch size sorting on:** STOP -> **Colour Size** -> switch **SIZE GRADES READY** -> read the table in the question -> Yes -> **START**.
Watch RECENT RESULTS: L mm, GRADE, EJECTOR. To go back: STOP -> switch it OFF.

**Colour split:** STOP -> **Colour Size** -> set **Threshold** (and **SW is ABOVE / BELOW**) -> **SAVE COLOUR & SIZE** -> switch
**COLOUR SPLIT** -> Yes. While OFF, watch the live "CURRENT CASHEW b\*" and the b\* under each cashew to find the value.

**PASS a class:** STOP -> **Pneumatic** -> click the chip of the class (e.g. TESTA SMALL) -> Yes. Click again for FIRE.

The machine must be STOPPED for SAVE and for all three switches.

---

## 2026-10-10 (5): Step (c2) - SW / WW colour and size grades; machine instructions

### What works now

| Part | How |
|---|---|
| Colour value | Inside the cashew mask (blue belt removed, after FFC / colour gain), per frame; one value per cashew = **median over its frames**. |
| Which value | **Selectable: b\* (yellow), a\* (red) or L\* (lightness).** Default b\* as you asked. See "Why not only b\*" below. |
| SW / WW | Threshold + side (SW is ABOVE / BELOW). Colour Size page: slider, live value of the current cashew (also L\*, a\*, b\*), TEACH WW, TEACH SW, both histograms in one picture, suggestion for all three values with "% of the taught cashews on the wrong side", APPLY SUGGESTION (saves at once). |
| Size | Length, width, area from the contour, **median over the frames**, in mm with the px per mm of the Vision ROI calibration. |
| Grades | W180, W210, W240, W320, W450, BUDDS, SW180 ... SW450. Per grade L / W / AREA min-max; TEACH = mean -/+ k x std (k = 3, histograms, 30 cashews minimum). COPY W LIMITS TO SW. A cashew that fits two overlapping grades gets the one whose middle is nearest. |
| Flow | GOOD -> SW or WW -> grade of that colour -> ejector of that grade **from the ejector mapping** (W180 = A11 / A31 ... SW450 = A20 / A40 in your mapping). Outside every grade of its colour -> reject ejector 1 / 21 (`SIZE_OUT_OF_RANGE`). BUDDS size -> no fire. |
| SHADOW | While SIZE GRADES READY is off: everything is measured and logged as "would be SW210"; the GOOD cashew still goes to 1 / 21. |
| SIZE GRADES READY | Refused until px per mm is calibrated, the threshold is set and at least one W and one SW grade has limits; asks before switching on. |
| RECENT RESULTS (Home) | TIME, ID, CLASS / RESULT, SW/WW, COLOUR value, L mm, GRADE, EJECTOR. |
| Log | `logs/cashews_*.csv`: columns `L`, `a`, `b` (medians), `length_mm`, `width_mm`, `grade` (also the "would be"), `relay`. |

The old "L below / W below" table and the old OpenCV filter sliders are no longer on the page (the old table is not used
any more; their values stay in the settings file).

### Why not only b\*

I measured your own dataset (65 SW images, 400 GOOD_CASHEW images) with the same mask and colour code: b\* hardly separates
them (SW 22.4 +/- 2.9, GOOD 24.8 +/- 9.8), a\* (SW -4.0, GOOD -5.7) and L\* (SW 59.6, GOOD 65.2) do better, none is clean.
That folder is mixed (old images, other lighting), so it does not decide the question. Therefore TEACH compares all three on
the cashews you run and marks the best one; you can still choose b\* yourself.

### Tested (fake PLC, simulated belt, temporary settings; nothing real was touched)

- `ast` check, **194 unit tests pass** (`tests/test_colour_size.py`: median colour, threshold and side, suggestion, grade from
  limits per colour, overlap, out of range, BUDDS, copy W to SW).
- Self-test `tools/selftest_pages.py`: READY refused with the 4 missing things named; calibration saved; SHADOW run: all GOOD to
  1 / 21 with "would be"; TEACH WW / SW with simulated values -> suggestion -> APPLY saved; TEACH W320, COPY, SAVE; READY on:
  every cashew fired by the ejector of its grade in the mapping (coils M110, M118 ... written). Pictures: `design/pages/04_*`, `01_home_running.png`.
- In that run the test forces every cashew to GOOD (the simulated cashews are otherwise REJECTION).

### Not tested

- Real cashews: whether one LAB value separates your SW from your WW, and whether taught grade ranges overlap too much.
  This can only be seen on the machine (the suggestion shows the "wrong side" percentage; the SHADOW run shows the grades).

### Machine instructions

**A. px per mm (once, and again whenever the camera or the ROI is moved)**
1. Machine STOPPED. Vision ROI page. ROI 1 must be drawn.
2. Put a ruler on the rollers inside ROI 1, along the direction the cashews travel. Read the real distance between the two
   ends of the yellow ROI 1 box (its long side for your right-to-left belt), in mm.
3. Type it into "ROI 1 length", check the "px per mm" shown, press SAVE VISION SETTINGS. The yellow NOT CALIBRATED text goes away.
4. Check: put a cashew of known length under the camera, START, read "L mm" in RECENT RESULTS. If it is off by the same
   factor for every cashew, correct the ROI length by that factor.

**B. TEACH the SW / WW colour** (machine RUNNING: GOOD cashews go to the reject ejectors 1 / 21 during teaching)
1. Colour Size page. Press CLEAR, then TEACH WW. START. Run only WW cashews (at least 20; 50-100 is better). The counter on
   the button rises. Press the button again (STOP WW).
2. Press TEACH SW. Run only SW cashews. Press STOP SW.
3. Look at the histogram (WW blue, SW yellow) and at the three suggestion lines. Use the value buttons b\* / a\* / L\* to see each
   histogram. The best line has the smallest "% on the wrong side"; if that number is high (say above 10 %), colour alone will
   mix the two - tell me the three numbers.
4. STOP the machine. APPLY SUGGESTION (it sets value, threshold and side and saves). Or type a threshold and SAVE COLOUR & SIZE.
5. Check: START, run a few known WW and SW cashews, watch SW/WW and COLOUR in RECENT RESULTS.

**C. TEACH each size grade** (px per mm must be calibrated first)
1. Colour Size page, row W180: press TEACH. START. Run only W180 cashews, 50 if possible (30 minimum). The three histograms
   fill; the yellow lines are the limits that will be set.
2. STOP the machine. APPLY TO W180. The six limit fields of the row are filled ("not saved yet").
3. Repeat for W210, W240, W320, W450 and BUDDS (BUDDS = pieces that must NOT be fired).
4. SW grades: either teach SW180 ... SW450 the same way with SW cashews, or press COPY W LIMITS TO SW.
5. SAVE COLOUR & SIZE. k = 3 gives wide limits; if neighbouring grades overlap a lot, set k to 2 and APPLY again, or edit the fields.

**D. Check in SHADOW, then switch on**
1. With SIZE GRADES READY still off: START and run known cashews of each grade. RECENT RESULTS shows
   "GOOD (would be W240)" etc. All of them still go to 1 / 21. Check that the "would be" is right.
2. Check the ejector mapping (Pneumatic page): W180 ... SW450 on the ejectors you want; BUDDS / SW_BUDDS in "no fire".
3. STOP. Colour Size: switch SIZE GRADES READY on and confirm. From now on GOOD cashews are fired by their grade ejector.
4. First run with few cashews and watch the EJECTOR column. The delays of the grade ejectors (A11 ... A20) were calculated for
   300 mm/s and are not verified: check that the right valve blows at the right place.

---

## 2026-10-10 (4): URGENT tracking fix - one cashew = one ID = one decision = at most one fire

### Analysis of the 03:15 run (20 cashews, all in ROI 2; from `logs/cashews_20261010.csv`)

No camera recording of that run exists and the frames log had no positions then, so the analysis is from the cashew log and
the LOST notes (first / last position, box, speed) that were added before that run.

| What the log shows | IDs | Explanation |
|---|---|---|
| 20 tracks with 10-23 frames, decided | the 20 real cashews (two of them carried a second ID, see below) | |
| Seen 2x at the ROI entry (x 2319-2420, edge at 2447), full-size box, measured speed 0.94-1.48 px/ms, then LOST; a normal track follows | R2-53, 65, 69, 71, 74 | **Entry split.** While a cashew is only partly inside the ROI its box centre moves at about half the belt speed (2.46). The track predicted with that speed; after 1-2 missed YOLO frames the cashew was outside its gate -> new ID. |
| The same in the middle of the ROI | R2-54 (53 -> 54 -> 55: one cashew, three IDs), R2-61 (14 frames, lost at x 1143; R2-63 continues 10 px away), R2-76 (5 detections in 275 ms = every second frame missed; R2-78 continues) | **Mid-ROI split** after missed frames with a speed estimate that was too low. YOLO did miss frames while the cashew rolled: 5 detections in 275 ms instead of about 9. |
| Tiny boxes (19x57, 22x55, 45x27 px; a cashew is about 160x250), speed 0.2-0.6, 1-4 frames | R2-46, 64, 79, 80 and others, also every 46 s while the belt was empty | **Not cashews**: a mark that comes round with the belt / rollers. Counted as LOST. |
| 1-2 frame tracks decided at the line and FIRED | R1-14, R2-45, R2-47, R2-66 (length 3.2 mm), R2-78 | Fragments and tiny boxes got their own decision and fired the reject valve. |

So: 20 cashews + 8 fragment IDs + tiny boxes = the TOTAL 33 / LOST 12 you saw.

### Fix

| # | Your point | Done |
|---|---|---|
| 2 | Prediction along the lane, bridge missed frames, re-attach near the predicted position, belt speed | A detection is matched from where the cashew was **last seen**, with the **lane (belt) speed** for a track seen fewer than 4 times, and a tolerance that grows with the time since it was seen (up to the 600 ms hold time). The lane speed is kept over STOP and is no longer taken from marks that hardly move. First cashew after program start (no lane speed yet): allowance for the half-speed entry. |
| 2 | ByteTrack-style low confidence | YOLO boxes down to confidence 0.10 are kept as "weak": they may continue a cashew that already has an ID, never start one. |
| 3 | Fragments are merged | A second box lying inside a tracked cashew's box gets no ID. A track that ends or reaches the line at the same place as another live cashew is **MERGED**: its frames go to that cashew, it is not counted, decided or fired. A tiny short-lived box (under a quarter of the usual cashew box, at most 8 detections) is **NOISE**: logged only. A box much narrower across the lane cannot take over a cashew's ID. |
| 4 | One cashew = one decision = one fire | Follows from 2 and 3; Home shows `FRAGMENTS / NOISE IGNORED n` separately, TOTAL and LOST no longer contain them. |
| 5 | Replay test | **A replay of the real run is not possible** (no recording, no positions in the old frames log). Instead: `tests/replay_model.py` builds runs with the measured numbers of the 03:15 run (2.46 px/ms, frames 26-53 ms apart, 165x250 boxes, half-speed entry, misses after the entry and while rolling, tiny slow boxes); `tests/test_tracking_replay.py` checks 20 in -> 20 IDs on 40 such runs and that the tracker of commit `3618f26` splits the same frames. From now on the frames log holds time and box of every detection (also of LOST / MERGED tracks), and `python3 tools/replay_tracks.py logs/frames_<date>.jsonl 03:15:00 03:16:00` replays a real run through the tracker. |

### Before / after

| | Before (`3618f26`) | After |
|---|---|---|
| Real run 03:15, 20 cashews | TOTAL 33, LOST 12 | not measured yet - needs your run |
| Modelled run, 60 x 20 = 1,200 cashews | 1,587 IDs (about 26 per 20) | 1,200 IDs |
| Modelled run, 200 x 20 = 4,000 cashews | - | 4,000 IDs, no ID shared by two cashews (also with no weak boxes at all) |
| Simulated belt in the program (fake PLC) | - | every cashew one ID, no LOST, no error |

**Honest limit:** the model is built from the log, not from pictures. Please repeat the 20-cashew test. If TOTAL is still
not 20, send nothing by hand: the new frames log has the positions, and `tools/replay_tracks.py` shows it. For a full
picture replay press RECORD on the Vision ROI page (10 s) during such a run.

### Also in this commit range

- **EXIT freeze:** closing could stall while Tk destroyed widgets of pages that were never shown. EXIT / SHUTDOWN now stop
  everything, save, hide the window and end the process directly. Tested: closes in all three cases that stalled before.
  The cause inside Tk was not found; this goes around it.
  **Open risk:** in my self-test (no Tk main loop) destroying a tool window (PLC mapping, ejector check) also stalled a few
  times. I could not reproduce it on purpose. If closing one of those windows ever freezes the program on the machine, tell me.
- Assumptions: weak boxes from confidence 0.10; "tiny" = under 25 % of the usual box area and at most 8 detections; fragment =
  within 0.6 box lengths of another live cashew; a cashew seen only 1-2 times, full size, alone, still gets "too few frames"
  -> reject (so a real cashew cannot slip through).

---

## HANDOFF - paused 2026-10-10 about 03:00 on your instruction

### Done and committed

| Commit | Content |
|---|---|
| `7b0e2c3` | Home: ROI boxes (ROI 1 yellow, ROI 2 magenta) drawn again while RUNNING. LOST rows in `logs/cashews_*.csv` now say where the cashew was seen (column `shadow`). |
| this commit | PART 1 (part): automatic capture of Undefined / lost / double cashews. |

### PART 1 - what I can say so far

| Step of the flow | Result |
|---|---|
| Cashew enters ROI, gets one ID | **Not reliable yet.** In your run of 02:03-02:05, 10 of 32 log rows are LOST with only 1-4 frames; 7 of them are followed about 0.6 s later by a normally decided cashew in the same ROI. So most LOST are probably not missing cashews but a second short-lived ID (broken ID or a second box on the same cashew). I could not find the cause: the old log had no positions. A simulation of "cashew enters, frames missed" did **not** split the ID, so I did not change the tracker. The next run logs positions; then it can be decided. |
| YOLO finds it in every frame | Mostly: decided cashews had 17-24 frames of about 25 possible; gaps up to 114 ms were seen inside a track. |
| CNN every frame, voting, defect -> its ejector | Works in the log (UPW -> 2 / 22, TESTA SMALL -> 29, REJECTION -> 21). Note: 98 of 435 frames were "edge cut" (cashew touching the ROI edge) and do not vote; some cashews had only 2-8 usable frames of 22. |
| Undefined -> reject ejector 1 / 21 | Works (too_few_frames, LOST over 300 ms -> 1 / 21). A cashew that **vanishes** before the line (LOST) is not fired: there is nothing to time the pulse on. |
| Undefined -> auto capture | **Was not built. Built now**: `captures/undefined/<YYYYMMDD>/<HHMMSS>_<id>_<reason>_full.jpg` (whole frame), `..._v01.jpg ...` (every view, up to 30), `....json` (reason, votes, ejector). Vanished cashews go there too (`LOST_...`); DOUBLE goes to `captures/double/<date>/`. Own thread, nothing is deleted, pauses at 5,000 cashews per day or under 20 GB free (my assumption, same numbers as the inbox). GOOD cashews held back only by SIZE_NOT_READY are not captured. Home shows `CAPTURED n`. Tested with the simulated belt: 11 Undefined cashews -> 187 files, none missing. **Not tested on the machine.** |
| GOOD -> SW / WW -> size grade -> grade ejector | **Not working yet**: colour is not measured, the grade comes from the old length table, and all GOOD go to 1 / 21 while SIZE GRADES READY is off. This is PART 2. |

### Finding: the program can freeze when it is closed

In my test (fake PLC) closing the window (EXIT / `_on_close`) stalls inside Tk while destroying a TEACH button of the size
table on the Colour Size page. It closed normally when Colour Size was the page on screen, and stalled when another page was
on screen or several pages had been visited. Cause not found yet. If EXIT freezes on the machine: this is it; kill the
window, nothing is lost (settings are saved before that point). It came in with the Colour Size page (`66bd3fb`).

### Half done

- Nothing is half written in the code. One experiment for the freeze (TEACH buttons in a holder frame) was taken out again, untested.
- Not on a branch, not stashed: everything in the working tree is either committed or yours (config files, the
  models `cnn_Default_20261010_005648` and `..._022114`, the `golden_crops` deletions).

### Exact next steps

1. **Close freeze**: find the widget rule that stalls (try: TEACH buttons packed in a holder frame; then without
   `columnspan` / `weight` in that table), test with a script that closes inside a real Tk main loop after visiting 0, 1 and 4 pages.
2. **PART 1 rest**: read the new `shadow` notes of LOST rows after your next run and decide the tracker change; look at why
   so many frames are "edge cut" (ROI 2 reaches the image border at y 2047).
3. **PART 2 (c2)**: (a) colour value per frame inside the mask (proposal: LAB b* median; L and a are already measured per
   frame), median over the track; (b) SW / WW threshold + TEACH SW / TEACH WW with two histograms on Colour Size;
   (c) grade from the taught limits (`size_config.grades`, WW and SW sets), median length in mm; (d) SHADOW: log "would be
   W240 / SW210" until SIZE GRADES READY; (e) RECENT RESULTS columns class, SW / WW, colour, length, grade, ejector;
   (f) machine instructions for px per mm, colours, grades.
4. **PART 3**: fixed colour per class (14 colours) used in graphs, class list and confusion matrix; live per-class
   validation lines + overall white line; train / validation loss; clickable legend; red for FIRE classes under 98 %;
   PNG next to the model (per-epoch per-class accuracy must be added to the training history first).

---

## 2026-10-10 (3): PART C - remaining pages in the 5-track layout; summary of this session

### Pages

| Page | State | What changed |
|---|---|---|
| Vision ROI | **rebuilt** | Cards: ROIS AND DECISION LINE (coordinates, DRAW / RESET, SET LINE, shift), **CALIBRATION** (ROI 1 length in mm -> px per mm), IMAGE SENSOR (gain, exposure, FPS limit, colour gain ALL / R / G / B), CAMERA SOURCE, WHITE CALIBRATION, RECORD TEST FRAMES. One **SAVE VISION SETTINGS**. |
| Colour Size | **rebuilt** | Left: colour engines (OFF / SHADOW / ACTIVE), old OpenCV filter limits, LAST CASHEWS (measured L, W, area, solidity, decision). Right: size / shape engines, **SIZE GRADES READY** switch, size table, **TEACH FROM CAMERA** (3 histograms, mean -/+ k x std, k = 3, at least 30 cashews), shape limits, longest single cashew (double rule). One **SAVE COLOUR & SIZE**. |
| Dataset Capture | **rebuilt** | Same 20 folders, counts, OPEN, CAPTURE IMAGE TO FOLDER, AUTO CAPTURE (limit, CROP / FULL FRAME / BOTH). |
| Teaching YOLO | **rebuilt** | Steps 1 IMAGES, 2 CLASS TO MARK, 3 MARKS IN THIS IMAGE, 4 TRAIN. Same functions as before. |
| Network | **rebuilt** | PLC card (IP, port, unit, ejector 1 address, register type, status, SAVE & CONNECT - STOPPED only, PLC DIAGNOSTICS) and a CAMERA status card. |
| Programs (NEW / COPY / DELETE) | **not done - needs your OK** | See "Open" below. |
| Speed panel | **not done** | Belongs to step (c3): it needs the encoder / measured belt speed. |
| Test Console | unchanged | Old page (not on the 5-track machine). |

### Calibration (new, important)

- The program was using **10 px per mm, a placeholder**. All mm values (size, shape limits, double rule 45 mm, mm/s) were based on it.
- Vision ROI -> CALIBRATION: measure the real length of ROI 1 on the roller along the travel direction, type it, SAVE.
  px per mm = ROI 1 length in pixels / mm. The page shows NOT CALIBRATED in yellow until this is done. I did not enter a value.
- TEACH a size and SIZE GRADES READY are refused until the calibration is saved.

### Size grades: what is real and what is not

- Which grade a GOOD cashew gets is still the old table ("L below / W below", your earlier values). Unchanged.
- TEACH stores min / max of length, width and area per grade. **These taught limits are stored only; the decision does not use
  them yet** (step c2, with your grade rules). I did not invent a rule for them.
- SIZE GRADES READY off (as now) = every GOOD cashew goes to the reject ejector A01 / A21. Switching it on asks first.
- Teaching a size needs the machine RUNNING (cashews are only measured then), so the valves fire while teaching.
- Colour and SW / WW engines still do not measure (c2). The page says so.

### Tested (fake PLC, simulated belt, off-screen window, temporary settings file)

- `ast` check, **172 unit tests pass** (new: `tests/test_size_teach.py`).
- `tools/ui_selftest.py` with `tools/selftest_pages.py` and `tools/selftest_teaching.py`: calibration 300 mm -> 8.07 px per mm saved, a wrong
  entry keeps the saved value; teach and SIZE READY refused before calibration; teach with 44 simulated measurements + 1 double
  -> limits applied and saved, shape limit saved, old table kept; every page and window photographed (27 pictures, `design/pages/`).

### Not tested

- Real camera, real PLC, real cashews. Camera gain / exposure fields against the real camera after the rebuild (same code
  as before, new widgets). Buttons of Dataset Capture and Teaching YOLO were not pressed in the self-test (only drawn).

### Open - please answer

1. **Programs (NEW / COPY / DELETE).** This machine has one settings file (`pneumatic_config.json`) and one program folder
   (`programs/Default`). Real programs mean: settings, mapping, delays and dataset per program, and switching between them. A
   mistake there loads the wrong ejector mapping, so I want your OK on the plan first: each program = `programs/<name>/` with its
   own settings file; NEW = factory defaults, COPY = copy of the settings (not the images), DELETE = moved to a `deleted/` folder,
   never erased; switching only when STOPPED; the PLC coil table stays shared. OK, or different?
2. **KNIEF_MARK and SW classes** (you added them in Teaching CNN): they are not in the priority list, so a cashew decided as
   KNIEF_MARK is "Undefined" -> reject ejector. Which ejector should KNIEF_MARK get, and is SW again a CNN class (earlier you
   merged it into GOOD_CASHEW)?
3. **ROI 1 length in mm** for the calibration.

### Commits of this session

| Commit | Content |
|---|---|
| `cf0ed5b` | PART A: DRY-RUN and MASTER EJECTOR removed, START / STOP only, pulse buttons, real PLC when STOPPED |
| `e7b0dc5` | PART B: CHECK CONFUSION, CONFUSION folder, multi-step undo, large view, training graphs, 98 % warning |
| `66bd3fb` | PART C (1): Vision ROI and Colour Size |
| (this one) | PART C (2): Dataset Capture, Teaching YOLO, Network; report |

Earlier commits that had no report entry: `c021c1d` clean dataset + DOUBLE class, `7c903fa` AUTO LEARNING switch, `c557b61`
full ejector mapping (1/21 reject, 2 UPW, 3 PKW2, 4 PKW1, 5 OIL, 6 BROWN, 7 DW1+DW2, 8 WHITE_DW, 9 TESTA SMALL, 10 TESTA,
11-15 W180...W450, 16-20 SW180...SW450; ROI 2 = +20; BUDDS no fire) + branding, `da54f5d` AI Model Select + Info,
`99d9676` Pneumatic page + PLC mapping window. Training time on the RTX 3050: about 11.5 s per epoch on 2,000 images
(30 epochs about 6 minutes; your 3,200 images about 18 s per epoch).

### Assumptions I made (correct me)

1. A pulse = coil ON, re-written every 50 ms, then OFF - also for grading, not only for the buttons.
2. Pulse per ejector 5-2000 ms; FIRE ALL uses each row's own pulse.
3. LOW CONFIDENCE in CHECK CONFUSION = below 60 %.
4. "Delete" in Teaching CNN moves to `programs/Default/trash/` so UNDO can work; the trash is never emptied by the program.
5. UNDO history lasts until the program is closed (the log file stays).
6. FIRE class = every class that is not a PASS class on the Pneumatic page; 98 % is compared with recall on validation images.
7. CONFUSION is stored as `programs/Default/confusion/<class it came from>/`.
8. Calibration uses ROI 1 only; ROI 2 is assumed to have the same px per mm.
9. Size teach: k = 3, at least 30 cashews, doubles ignored.
10. Logo: the file from `~/Downloads/1001402664.png` (the name you gave did not exist).
11. I did not commit the 1,307 `cnn_final/golden_crops` files that git shows as deleted (you labelled them into the dataset),
    nor your new model files, nor `pneumatic_config.json` / `inovance_config.json` as changed by your own runs.

---

## 2026-10-10 (2): PART B - Teaching CNN: CHECK CONFUSION, CONFUSION folder, multi-step UNDO, large view, training graphs

| # | Asked | What is there now |
|---|---|---|
| 1 | CHECK CONFUSION before training | Button in step 2 LABEL (it replaces VERIFY LABELS). The **active** model runs on every labelled image; the list shows MISMATCH (model disagrees with the folder) and LOW CONFIDENCE (agrees, below 60 %). Per image: LABEL IS RIGHT - KEEP, MOVE TO THE MODEL'S CLASS, MOVE TO THIS CLASS (any class), MOVE TO CONFUSION, DELETE, UNDO. STOPPED only. |
| 2 | After training | The result file stores the images the new model still gets wrong (all training + validation images). Training panel: **STILL WRONG: n - REVIEW** opens the same list; **MOVE ALL LISTED TO CONFUSION** = one click (asks once). CONFUSION is `programs/Default/confusion/<class it came from>/`, beside the dataset, so no training reads it. |
| 3 | CONFUSION review | Third tab of the grid (CAPTURES / INBOX / CONFUSION), also the CONFUSION row under the class list. Thumbnails show "was: <class>". Press 1-9 or click a class = re-label (joins training again); DELETE SELECTED. |
| 4 | Multi-step UNDO + log | UNDO takes back one action per press, as many as were done since the program started: label, re-label, move, move to CONFUSION, delete, also those made in the review windows. Delete moves the image to `programs/Default/trash/<time>/` (that is what makes it undoable; the program never empties the trash). Every moved file and every undo is a line in `logs/dataset_changes.log`. |
| 5 | OPEN FOLDER, large view | Folder sign on every class row and on CONFUSION, OPEN FOLDER under the grid (captures / inbox / confusion): opens Ubuntu Files. Double click on any thumbnail (and on the picture in the review window) = large view. |
| 6 | Graphs | Per epoch: training loss, validation loss, training accuracy, validation accuracy. After training: precision / recall / accuracy bars per class and a confusion-matrix heatmap. Saved next to the model as `<model>_curves.png`, `_classes.png`, `_confusion.png`. Button GRAPHS on the Teaching CNN panel and on each CNN card in AI Model Select. |
| 7 | FIRE classes, 98 % | LAST RESULT marks every class PASS or FIRE; a FIRE class is green only at 98 % or more. Red line "WARNING: FIRE classes below 98 %: ..." on the Teaching CNN panel, on the model card in AI Model Select, and red bars in the class graph. |

Also new: **TEST SET REPORT** on each CNN card (AI Model Select): confusion matrix and per-class bars of that model on the
test set. It writes only `<model>_testset_*.png`; model and result file are not changed. This is how your model
`cnn_Default_20261010_005648` (trained before the graphs existed) can get its pictures. I did not run it on your model;
its files are untouched.

**Not carried over from the 5-track page:** double click on a capture used to open the auto-label tool with that image; it
is now the large view (the AUTO-LABEL TOOL button is still there).

### Tested

- 169 unit tests pass (new: metrics and the 98 % rule, CONFUSION folder outside the dataset, trash, 3-step undo, the log,
  per-epoch numbers, report pictures, low-confidence listing, test-set matrix without touching the model).
- `python3 tools/ui_selftest.py design/pages tools/selftest_teaching.py`: in the real window, on a **temporary copy** of
  240 dataset images with 4 labels made wrong on purpose and a 6-epoch model: CHECK CONFUSION found 4 of 4; move / confusion /
  delete / keep and 3 x UNDO restored the exact counts; STILL WRONG -> all to CONFUSION (training then saw fewer images);
  CONFUSION tab re-label and delete, UNDO back to the start, trash empty; folder buttons call `xdg-open` with the right
  folders; graphs and test-set report shown. Screenshots `design/pages/30...36`.

### Not tested

- On your full dataset (I did not run a check or a training on it), and Ubuntu Files really opening (the self-test only
  records the call). Graph numbers of the 6-epoch test model mean nothing; they only show the pictures work.

### How to test

1. Teaching CNN, machine STOPPED. CHECK CONFUSION uses the active model (now the old `cashew_color_cnn_new.pth`; classes it
   does not know are skipped and named). To check with your new model, activate it first in AI Model Select.
2. Train again (START TRAINING): after it, GRAPHS and STILL WRONG are filled.

---

## 2026-10-10 (1): After the power cut; PART A - DRY-RUN and MASTER EJECTOR removed, pulse buttons fire the real PLC

### Check after the power cut

- `git fsck` clean, history intact (last commit `99d9676`, step e2b). All `.py` parse, all `.json` load. Nothing was reset.
- Your work since then is on disk and untouched: 1,307 captures labelled into the dataset (they left `cnn_final/golden_crops`,
  which git shows as deleted files; I did not restore or commit that), new classes KNIEF_MARK and SW, and the model
  `cnn_Default_20261010_005648.pth` with its `.json` (100 epochs, best epoch 68, validation 85.7 %). Model and result are kept;
  it is not active (the grader still uses `cashew_color_cnn_new.pth`).
- Pages of your last prompt. **Done:** Home, Pneumatic, PLC / ejector mapping window, Sorting Logic, Teaching CNN,
  AI Model Select, Info. **Not rebuilt yet (still the older layout in the new colours):** Colour Size, Vision ROI,
  Dataset Capture, Teaching YOLO, Network; Programs and the Speed page do not exist yet.

### PART A - what changed

| Area | Change |
|---|---|
| DRY-RUN, MASTER EJECTOR | **Removed completely**: switches, top-bar pills, banners, questions, counters, log columns, tests. `relay_output.py` has no such state any more. |
| The only control | **START / STOP.** RUNNING: grading fires the valves. STOPPED: nothing fires from grading; pulses still waiting for their delay are cancelled and anything ON is switched OFF. The program opens STOPPED and nothing starts it by itself. |
| How a pulse is written | As the old diagnostics tool does: coil ON, written ON again every 50 ms while the pulse lasts, then OFF. **Grading uses the same engine** (it no longer goes through the PLC controller's own pulse queue), so a pulse that fires from a button fires the same way in production. |
| Pulse per ejector | New PULSE column in every row (5-2000 ms). "Pulse: set all ejectors" writes one value into every row. Grading and the buttons use the row's value. |
| Lightning button | Every ejector row on the **Pneumatic page** and in the **mapping window** has one: a click fires that ejector's coil at once with the row's pulse, without a question. Result next to it: `OK hh:mm:ss` or `PLC ERROR ...`. |
| FIRE ALL, SEQUENCE ALL, ALL RELAYS OFF NOW | In both places. FIRE ALL = ejectors 1-40 at once, SEQUENCE ALL = one after the other; both ask "Make sure nobody is at the nozzles." first. ALL RELAYS OFF NOW always works. |
| Only when STOPPED | Every manual button is refused while RUNNING ("refused: machine is RUNNING"), also inside `relay_output.py`. |
| Ejector check | Same rule: real PLC, STOPPED and conveyor OFF, asks first; the coil is held ON for the pulse; read-back as before. |
| M140 / M141 | Never driven: relays above 40 are refused, an ejector mapped to such an address is refused, the PLC's own all-off is never called. |
| Log | `logs/manual_fire_YYYYMMDD.csv`: time, action, ejector, M address, pulse, result, sent YES / NO, PLC connected, error. (A file with the older columns is kept as `..._old_HHMMSS.csv`.) |
| `tools/ui_selftest.py` (new) | The whole program with a fake PLC and a simulated belt, off-screen; photographs every page. This is how I test; it never touches the real PLC. |

### Read this before pressing START on the machine

With DRY-RUN gone, **START makes grading fire real valves at once.** As the settings are now:
1. Every GOOD cashew goes to the reject ejector A01 / A21 (SIZE_NOT_READY, your rule until sizes are taught).
2. The delay table is the one calculated for 300 mm/s; the real belt speed is not measured, so pulses may come early.
3. The active CNN is the old one; your new model is not activated.
So a first START is a valve and timing trial, not sorting. Use the lightning buttons and the ejector check first.

### Tested (fake PLC only - nothing was written to the real PLC by me)

- `ast` check; 167 unit tests pass. `tests/test_relay_output.py` is rewritten: no DRY-RUN / MASTER left; STOPPED nothing fires
  from grading; RUNNING fires after the delay and holds the coil ON; STOP cancels what waits; a manual pulse fires at once only
  when STOPPED; FIRE ALL is 40 single pulses on M100-M139; conveyor / feeder addresses refused; PLC errors reported; ALL RELAYS OFF.
- Self-test in the real window: lightning button -> coil ON then OFF, "OK"; FIRE ALL asked first and fired 1-40; SEQUENCE ALL
  stopped by ALL RELAYS OFF NOW; RUNNING on the simulated belt wrote coils for the decisions, a manual pulse was refused, STOP
  cancelled 3 waiting pulses and left no coil ON; mapping-window lightning button "OK". Screenshots in `design/pages/`.

### Not tested

- **The real PLC and valves.** Whether a pulse written this way fires your valves, and the real pulse length (each write is one
  Modbus round trip), can only be seen on the machine.

### How to test on the machine

1. Start the grader (it opens STOPPED). Pneumatic page.
2. Nobody at the nozzles. Click the lightning button of A01: the valve must blow once; the row shows `OK`.
3. If it shows `PLC ERROR`, read the text in the tools column (not connected / write failed) and tell me.
4. RUN EJECTOR CHECK 1-40 to go through all ejectors and mark AIR OK / NO AIR / WRONG VALVE.
5. `logs/manual_fire_<date>.csv` has one line per fire.

---

## 2026-10-09 (10): Combined ejectors - several classes per ejector, one ejector per class, summary before save

### What changed

| Area | Change |
|---|---|
| `relay_mapping.py` (new) | All rules in one place, no GUI: several classes / grades per ejector; a class on only one ejector per ROI; ejectors 1 / 21 take no class; a class or grade without an ejector goes to the reject ejector; an empty ejector is free; ROI 2 mirror (+20). |
| Grading | `_decide_track` asks only this mapping for the ejector. No class name is tied to an ejector number anywhere in the code (a test uses invented class names on arbitrary ejectors). |
| GOOD grades | A GOOD cashew looks for its grade name (`W180` for WW, `SW180` for SW, `BUDDS`, `SW_BUDDS`), then its colour group (`WW` / `SW`), then `GOOD_CASHEW`. With none of them assigned it goes to the reject ejector as "UNDEF no ejector (W240)". Before, a GOOD cashew without an ejector was simply not fired, and an SW grade silently used the W ejector; that hidden fallback is gone (tick both on one ejector instead). |
| Picker (gear button of a row) | Check boxes for all defect classes, the good groups, the WW and SW grades. Each shows where it is now ("on A07"). Ticking a class that is on another ejector asks **"Move it here?"**; NO leaves it where it was. "The same for ROI 2: ejector A26" is ticked by default. FREE THIS EJECTOR unticks everything. |
| Row text | `REJECT - FIXED` for 1 / 21, the classes for the others, `(free)` for an empty one. A free ejector is never fired by grading (manual tests still work). |
| Not assigned | Yellow list under the tables, per ROI: "ROI 1 -> reject ejector A01: UPW". |
| Duplicate | Cannot be made in the picker. If the settings file has one (edited by hand), a red "SAVE IS BLOCKED ... TESTA is on ejectors A03 and A08" appears and SAVE PNEUMATICS refuses until it is repaired. |
| SAVE PNEUMATICS | First shows the final table CLASS -> ROI 1 EJECTOR / ROI 2 EJECTOR (with "REJECT A01" and "A11 (via SW)" where that applies) and the free ejectors; CONFIRM AND SAVE writes, CANCEL writes nothing. |

### Tested

- `ast` check; **153 unit tests pass** (10 new in `tests/test_relay_mapping.py`): combined DW1 + DW2 and W180 + SW180; combining PKW1 + PKW2 frees the old ejector; moving a class between ejectors; duplicate found (names ignore case and underscores); unassigned -> reject ejector 1 / 21; reject ejectors take no class; ROI 2 mirror (+20) with a combined row, and ROI 1 only; the summary table; any invented combination; the stored config has no duplicate.
- Real window off-screen, fake PLC, temporary config: PKW2 ticked on A06 asked "PKW2 is on ejector A05 now. Move it here to A06?" -> A06 = PKW2, PKW1, A05 (free), the same on A26 / A25, grading tables point PKW2 to 6 / 26; answering NO moved nothing; W180 + SW180 combined on A12 for ROI 1 only; an injected duplicate blocked the save (file unchanged); the summary showed 22 rows and saved after CONFIRM; CANCEL left the file unchanged.
  Screenshots: `design/ejector_picker_combined.png`, `design/pneumatic_combined_mapping.png`, `design/mapping_summary_before_save.png`.

### Not tested

- On the machine with real cashews and valves.

### To know

- The mapping you set in the picker is used by grading at once (in memory). It is written to the settings file by SAVE PNEUMATICS; other save buttons of the program also write the whole settings file, so treat a picker change as live.
- With the config as it is: UPW is the only class without an ejector. `SW_BUDDS` and any unlisted SW grade use A11 / A31 through the group `SW`.
- A02 / A22 still list REJECTION, but REJECTION always goes to A01 / A21 (open question from before). You can now free A02 / A22 in the picker.
- The Pneumatic page is still the old layout; (e2b) rebuilds it and keeps this picker, list and summary.

### How to test

1. Pneumatic page, gear button of A06: tick PKW2, answer YES, OK. A05 shows (free), A06 "PKW2, PKW1", A25 / A26 the same.
2. The yellow list shows UPW. Give UPW an ejector (for example a free one): the list changes to "Every class and grade has an ejector."
3. SAVE PNEUMATICS: read the table, CONFIRM AND SAVE. Close and start the program: the mapping is still there.

---

## 2026-10-09 (9): Teaching CNN page rebuilt like the 5-track machine; clean dataset; production capture (INBOX)

Reference: `TCM_CASHEWXAI_2026-10-06/source` (the five files are identical in `CNN/CNN/app`) and manual B8 / B9.

### How it was done

The 5-track modules were **ported**, not rewritten, so layout, look and behaviour are the same:
`teach_ui.py` (the page), `dataset_ops.py` (confirm / test set / undo / warnings), `production_capture.py` (INBOX),
`training.py` (training, label check, AI suggest), `training_ui.py` (class chooser, live training window,
mismatch window). The sources on the drive are untouched. Adapted for this machine:

| Change | Why |
|---|---|
| `training.py` reads every image through **`preprocess.py`** (`prepare_crop`: tight crop around the cashew, padded to a square) and uses `preprocess.build_transform` and `preprocess.load_cnn_checkpoint` | the same crop, padding, size and normalisation as live grading (the 5-track file had its own copy of that code) |
| Thumbnails 152 x 112, compact class rows | 1920 px screen and 14 classes instead of 8 |
| A switch for the production capture on the page (bottom of CLASSES) | no programs yet to hold that setting |
| No password written in the code | CLEAR INBOX asks for the supervisor password; the first time you are asked to set one, only a salted hash is stored in `machine_config.json` |
| Teaching YOLO keeps its own epochs box and button | they used to overwrite the CNN page's (a bug the 5-track project had fixed too) |

### What the page has

- **1 CAPTURE**: START AUTO CAPTURE, SNAPSHOT, Limit, CROP / FULL / BOTH. **2 LABEL**: AI SUGGEST, AUTO-LABEL TOOL, VERIFY LABELS, number of unlabelled. **3 TRAIN**: CNN MobileNetV2, Epochs (default 30), START TRAINING.
- **Centre**: 5 x 5 image grid, click to select, arrows / PgUp / PgDn, PREV / NEXT, SELECT PAGE, DESELECT ALL, DELETE SELECTED; tabs CAPTURES and INBOX (INBOX adds CONFIRM ALL ON THIS PAGE AS SUGGESTED and CLEAR INBOX).
- **Right**: classes with count / 1000 and a bar; keys 1-9 or a click move the selected images into the class; UNDO, ADD, DELETE, RENAME, COLOR; DATASET WARNINGS (fewer than 100, unbalanced); TRAINING panel with loss and validation accuracy per epoch, LAST RESULT with one line per class; ACTIVATE A MODEL opens AI Model Select.
- **INBOX production capture while RUNNING** (5-track values): uncertain (confidence 0.40-0.85, or mixed classes, or an engine "would reject") 60 %, rejects 30 %, good 10 % (1 % of the sure good ones are candidates), 4 views per cashew, 5,000 images per day released hour by hour over 10 hours, 100,000 in the inbox, needs 20 GB free, **never deletes**, saved by its own thread (the frame loop only hands over a small copy of the crop). INBOX FULL / DISK SPACE LOW appear in the Home ALARMS box. Folder: `programs/Default/inbox/<date>/` with one JSON per cashew.
- **Test set**: every 10th image confirmed from the INBOX goes to `programs/Default/testset/<class>/` and is never trained on. (As on the 5-track machine, images labelled from the CAPTURES tab all go to the dataset.)
- **Training**: only when STOPPED; 20 % of every class held back for validation; best epoch kept; `<model>.json` with accuracy per class is written next to the model in `saved_models/`. **A new model is never activated**; the grader keeps `cashew_color_cnn_new.pth`.

### Clean dataset: `programs/Default/dataset/` (one folder per final class name)

Copied by `dataset_import.py` from `/home/phoenix/Documents/CNN_final/CNN/dataset` and from this project's
`dataset/`. **Nothing was moved or deleted**: both sources have the same 2,217 files as before.
GOOD -> GOOD_CASHEW, TESTA_SMALL -> TESTA SMALL, BACK_CUT_OIL and OIL_BACKCUT -> OIL. Every 10th image of a
class went to the test set.

| Class | Before | Imported | After (train) | Test set | Still needed to 1000 |
|---|---|---|---|---|---|
| GOOD_CASHEW | 0 | 619 | 558 | 61 | 442 |
| UPW | 0 | 240 | 216 | 24 | 784 |
| PKW1 | 0 | 199 | 180 | 19 | 820 |
| PKW2 | 0 | 223 | 201 | 22 | 799 |
| OIL | 0 | 69 | 63 | 6 | 937 |
| BROWN | 0 | 117 | 106 | 11 | 894 |
| DW1 | 0 | 0 | 0 | 0 | 1000 |
| DW2 | 0 | 41 | 37 | 4 | 963 |
| TESTA | 0 | 266 | 240 | 26 | 760 |
| TESTA SMALL | 0 | 248 | 224 | 24 | 776 |
| WHITE_DW | 0 | 21 | 19 | 2 | 981 |
| SW | 0 | 0 | 0 | 0 | 1000 |
| REJECTION | 0 | 93 | 84 | 9 | 916 |
| EDGECUT | 0 | 2 | 2 | 0 | 998 |
| **Total** | 0 | 2138 | 1930 | 208 | 12070 |

- Left out: `DOUBLE_KERNELS` (52, not a final class), `Class_1` (1 full frame), 23 exact duplicates. No unreadable file.
- **DW1 and SW have no image at all**; EDGECUT 2, WHITE_DW 19, DW2 37, OIL 63 are far too few. There are 15 SW images in `CNN_final/CNN/dataset_disabled/SW`; I did not import them because you named only the `dataset` folder. Say if they should go in.
- The program folder is not in git (images). Run `python3 dataset_import.py` again at any time; it never copies an image twice.

### Tested

- `ast` check; **142 unit tests pass** (22 new): the 5-track tests of `dataset_ops` and `production_capture` (shares, hourly release, limits, never deletes, saver thread), the import tool (merge, test set, duplicates, full frames, source untouched), the password hash, and a **real small training run** (20 images, 2 epochs): images go through `preprocess.py`, 20 % held back, result file written, the checkpoint loads with the grader's loader.
- The real window off-screen with a small dataset (110 images), fake PLC: 14 classes shown, warnings listed; 12 images labelled with a class and UNDO put them back; machine RUNNING on the simulated belt filled the inbox (about 50 images, 4 views per cashew, JSON per cashew, 0 skipped, 0 write errors); training refused while RUNNING; a **fake training run** of 30 epochs drew loss and validation accuracy and the result per class; the active CNN stayed the same.
  Screenshots: `design/teaching_cnn_simulated.png`, `design/teaching_cnn_inbox_simulated.png`, `design/teaching_cnn_trained_simulated.png`.

### Not tested

- A full training on the real dataset (1,930 images) and its accuracy. AI SUGGEST, VERIFY LABELS and the mismatch window on real data. The AUTO-LABEL TOOL button only starts the existing `ai_auto_labeler.py`.
- Production capture with the real camera at production speed (AI fps with capture ON: please compare).
- CLEAR INBOX and the password dialogs (they need typing; the hash logic is unit-tested).

### To know

- Production capture is **ON** by default and writes up to 5,000 images a day into `programs/Default/inbox`. The switch is at the bottom of the CLASSES column.
- AI SUGGEST and VERIFY LABELS use the active CNN, which knows 11 classes and has no REJECTION, DW1, EDGECUT.
- The old Teaching CNN page (YOLO-classifier options, text log) is replaced. Teaching YOLO, AI Model Select (test-set comparison, PASS / FIRE popup) and Dataset Capture are still the older pages: step (e4).

### How to test

```
cd /home/phoenix/Documents/GSN_V21/CNN
python3 cashew_cnn_grader.py        # Teaching CNN in the left menu
```

1. The CLASSES column shows the 14 classes with the counts of the table above.
2. Machine STOPPED: START TRAINING, choose the classes with enough images, CONFIRM & TRAIN. A window shows every epoch. Afterwards LAST RESULT shows the accuracy per class; the new model is in `saved_models/` and is not active.
3. Machine RUNNING with cashews for a few minutes, then INBOX tab: select images, press 1-9 or click a class.

---

## 2026-10-09 (8): Automatic EJECTOR CHECK (relays 1-40, 5 pulses each, read-back, operator marks)

### What was built

| Area | Detail |
|---|---|
| Where | Pneumatic page, button **RUN EJECTOR CHECK 1-40** (right column, under FIRE 1-40). It opens a tool window. The PLC diagnostics window of (e2b) will get the same button; the existing external diagnostics tool is not changed. |
| What it does | Relays in order, one at a time, 5 pulses each. Defaults, editable in the window: pulse 100 ms, gap between pulses 400 ms, gap between relays 1000 ms, countdown 3 s. Range: ALL 1-40, ROI 1 (1-20), ROI 2 (21-40) or SINGLE relay; Repeat n times (for one relay or a range). |
| Every pulse | Write coil ON -> read the coil back -> wait the pulse -> write OFF -> read back. Result per relay: **PLC OK** (all pulses confirmed) or **PLC FAIL** with the text ("write ON failed", "coil reads OFF after ON was written", "coil still reads ON after OFF was written", "no answer when reading the coil ..."). A PLC FAIL on one relay does not stop the check. |
| Live | Current relay, its M address, "pulse 3 / 5", table of all 40 relays (PLC address, pulses confirmed, PLC result, operator mark). |
| Operator marks | The relay being checked is selected; keys **1 / 2 / 3** or the buttons mark it AIR OK / NO AIR / WRONG VALVE. Click any row to mark another relay. Unmarked stays "not checked". |
| Top bar | New pill **EJECTOR CHECK**: NOT RUN, RUNNING R12, ALL OK (hh:mm), 20 OK, DRY-RUN, STOPPED, FAILED R7,R9, AIR FAULT R5, ERROR. |
| Log | `logs/ejector_check_YYYYMMDD.csv`: time, relay, M address, pulses sent, PLC OK / FAIL, operator mark, error. One line when a relay is finished and one more line each time you mark it (the last line of a relay counts). Each pulse is also in `logs/manual_fire_*.csv`. |

### Safety

- Starts only when the machine is **STOPPED** and the **conveyor is OFF**; both are checked again before every relay (if not, the check ends and all relays are switched OFF). The machine cannot be started while the check runs.
- Question before every start: "Make sure nobody is at the nozzles and air is ON." with the DRY-RUN state and the duration.
- **DRY-RUN** exactly as for every manual fire: with DRY-RUN ON nothing is written, the check only simulates, the window shows a yellow "DRY-RUN ON: nothing sent to the PLC" banner (red "DRY-RUN OFF: REAL VALVES" otherwise), and the result is DRY-RUN.
- Red **STOP / ALL RELAYS OFF NOW**: always works, also in DRY-RUN and when no check runs; it writes OFF to relays 1-40 one by one. (This OFF write is the one thing sent in DRY-RUN: switching OFF cannot fire a valve.)
- PLC disconnects during the check: the check stops, OFF is written to all relays (as far as possible), ERROR with the relay and pulse.
- Only through `relay_output.py`: relays 1-40 only; a relay whose address is M140 / M141 is refused; never `fire_all()` / `sequence_all()` / the PLC's `all_off()`. `innovance_plc.py` is unchanged (its single-coil write and Modbus read are used).

### Tested

- `ast` check; **120 unit tests pass** (15 new, fake PLC with coil memory): 40 relays x 5 pulses in order with the exact waits; ON / OFF / read-back per pulse; DRY-RUN writes nothing; ranges and limits; PLC FAIL per relay; disconnect stops the check; STOP switches 1-40 off and leaves M140 / M141 alone; not started / stopped when RUNNING or conveyor ON; operator marks; start in a thread and stop.
- The real window off-screen with a fake PLC: refused with conveyor ON and while RUNNING; DRY-RUN run of ROI 1 wrote nothing; a run with DRY-RUN off showed PLC FAIL for a relay whose coil does not switch and one whose write fails, marks by key, STOP at relay 30, pill "FAILED R7,R9". Screenshot: `design/ejector_check_window_fake_plc.png`.

### Not tested

- **The real PLC.** In particular the read-back: whether your PLC answers Modbus "read coils" for M100-M139 and whether the PLC program leaves those coils alone. If every relay shows PLC FAIL "no answer when reading the coil" although the valves fire, tell me; then the read function or address needs adjusting.
- Timing accuracy of the pulses on the real machine (each write and read is one Modbus round trip, so a 100 ms pulse is about 100 ms plus two round trips).

### How to run it on the machine

1. Start the grader. Machine STOPPED, CONVEYOR OFF, air ON.
2. Pneumatic page -> **RUN EJECTOR CHECK 1-40**.
3. First with **DRY-RUN ON** (yellow banner): START CHECK, answer YES. No valve may move; all rows end as DRY-RUN.
4. Then nobody at the nozzles: Pneumatic page, switch **DRY-RUN OFF** (the banner turns red). Try **SINGLE**, relay 1, START CHECK: valve 1 must blow 5 times and the row must say PLC OK 5 / 5. Press **1** (AIR OK), **2** (NO AIR) or **3** (WRONG VALVE).
5. Then ALL 1-40 (about 2 min 20 s). Watch each valve and press 1 / 2 / 3 while its relay is selected; it is easier with "Gap between relays" at 2000-3000 ms.
6. Red STOP at any time. Afterwards switch DRY-RUN ON again.
7. Result: the EJECTOR CHECK pill and `logs/ejector_check_<date>.csv`.

---

## 2026-10-09 (7): Sorting Logic page full width with SAVE; settings pages without the camera; new start values

### What changed

| Area | Change |
|---|---|
| Sorting Logic | No camera on this page; the whole width in three columns with bigger text: GRADING RULES + PRIORITY, VIEW LIMITS + LIVE VALUES, BELT COLOUR + TRACKER + OTHER. |
| Live preview | Kept, small: next to LIVE VALUES one picture of the latest measured cashew per ROI with its values. It is needed to see which cashew the numbers belong to while you set the view limits. |
| SAVE | A bar that is always on screen at the bottom: **SAVE SORTING LOGIC**. It shows "All changes saved", "UNSAVED CHANGES" after any edit, "Saved hh:mm:ss" after saving, or "NOT SAVED: wrong value ..." (nothing is stored then). |
| Start values | **NECK M = 0** (not-rolled rule on hold) and **"One defect frame blocks GOOD" = OFF**. With M = 0 a cashew whose NECK was never seen stays GOOD and is written to the log column `shadow` as `VIEW_NOT_ROLLED` ("would be not rolled"). |
| Old settings files | A file written by the earlier program (M = 1, blocks = ON) gets the two new start values once; after you save the page yourself, your own values are kept. |
| Pages without the camera now | **Sorting Logic, AI Model Select, Network** (new) besides Pneumatic, Colour Size, Teaching CNN (already so). The camera stays on Home, Vision ROI and Dataset Capture; Teaching YOLO and Test Console have their own picture. AI Model Select and Network only got the full width; their layout is rebuilt in (e4). |

The older blocks of the Sorting Logic page (combination rules, per-class confidence sliders of the old logic)
are still under the new part; the sliders are marked "not used by the grading rules".

### Tested

- `ast` check; 105 unit tests pass (3 new: start values, M = 0 still notes "not rolled", old files get the new start values once).
- Real window off-screen, fake PLC: page has no camera and is 1711 px wide; an edit shows UNSAVED CHANGES; a wrong
  value is refused; SAVE shows "Saved"; **after closing and starting the program again** M = 0, GOOD frames = 4
  and a NECK limit of 0.2 were still there (read from the screen and from the file).
  Screenshot: `design/sorting_logic_full_width_simulated.png`.

### Not tested

- On your screen with the real camera.

### Note

`assets/cashew_x_ai_logo_clean.png` (for the branding you asked for with (e2b)) is not in this project and I did
not find it on this PC or the external drive. Please copy it into `assets/`.

---

## 2026-10-09 (6): Safety fix - every manual fire button follows DRY-RUN; PLC diagnostics tool gated

Owner's report: with DRY-RUN ON the single relay test says "NOT sent", but FIRE 1-40 fired real valves.

### What I found

- In the code, FIRE 1-40 and SEQUENCE 1-40 already used the same DRY-RUN-checked call as the single test
  (`RelayOutput.test_fire`), in the version you were running too (`d0492ce`).
- `logs/grader.log` of 2026-10-09: one SEQUENCE 1-40 at 22:39:20-25 (40 lines "[DRY-RUN] would fire"), row
  tests at 22:38 and 22:41, **no FIRE 1-40 press and not one pulse sent to the PLC by the grader all day**
  (no "[FIRE]" line, no "DRY-RUN OFF" line).
- **The real bypass:** the buttons OPEN PLC DIAGNOSTICS (Pneumatic page) and "Open PLC Diagnostics Tool"
  (Network page) start `innovance_plc.py` as a separate program. It has its own PLC connection and its own
  fire buttons. It knows nothing of DRY-RUN or MASTER EJECTOR, fires at once, can also switch M140 / M141, and
  writes nothing into the grader's log. I cannot prove from the log that this is what fired your valves, but it
  is the only path in this project that ignores DRY-RUN. If you are sure you pressed FIRE 1-40 on the
  Pneumatic page itself with DRY-RUN ON, tell me: then something else is wrong and I look again.

### What changed

| Area | Change |
|---|---|
| One rule | Row test, single relay test, FIRE 1-40, SEQUENCE 1-40 all call `test_fire()`; there is no second way. One wording for all of them (`RESULT_TEXT`): "DRY-RUN: NOT sent to the PLC", "pulse sent to the PLC", "PLC OFFLINE: NOT sent". FIRE 1-40 shows e.g. "40 x DRY-RUN: NOT sent to the PLC". |
| Confirmation | FIRE 1-40 and SEQUENCE 1-40 always ask first: "Make sure nobody is at the nozzles." and say whether DRY-RUN is ON (nothing is sent) or OFF (real valves). |
| STOPPED only | All manual fires are refused while RUNNING (checked again after the question). |
| PLC diagnostics tool | Blocked while DRY-RUN is ON and while RUNNING. With DRY-RUN OFF it asks first and says plainly that it writes directly to the PLC. Opening (or the blocked attempt) is logged. |
| Manual fire log | `logs/manual_fire_YYYYMMDD.csv`: time, action, relay, M address, pulse ms, result, sent to PLC YES / NO, DRY-RUN ON / OFF, PLC connected. |

`innovance_plc.py` is unchanged.

### How FIRE 1-40 writes to the PLC compared with the single test

Identical path, 40 times: `test_fire(r)` -> `InovancePLCController.pulse_actuator(r, 0, pulse)` -> the controller's
dispatcher thread -> Modbus **FC05 (write single coil)**, ON at address 100 + (r - 1), and OFF after the pulse.
It never calls the PLC's `fire_all()` (48 coils). Differences only in timing: the dispatcher writes one coil per
Modbus round trip, so with 40 relays the ONs (and later the OFFs) go out one after the other, a few ms apart;
the pulse of each relay stays about the set length.

If the single test does not fire a valve with DRY-RUN OFF, read the result line:
- "PLC OFFLINE: NOT sent": no Modbus connection.
- "pulse sent to the PLC" but no valve: the pulse was handed to the controller's queue. A failed write is only
  printed in the terminal ("[PLC] Write error ..."), the grader is not told. Then check the M address of that
  relay, the wiring / valve / air, and the pulse length (default 100 ms; `inovance_config.json` has
  `pulse_duration` 25, which the diagnostics tool uses, not the grader).
- The diagnostics tool pulses differently: it keeps re-writing the coil ON during the pulse; the grader writes
  ON once and OFF once. If a relay fires from the tool but not from the grader, the PLC program may be
  resetting that coil; tell me and I look at it with you.

### Tested

- `ast` check; 102 unit tests pass (4 new: DRY-RUN blocks single / all / sequence alike; with DRY-RUN off all
  three send single pulses of relays 1-40 only; offline and refused are logged as not sent; one wording).
- The real window off-screen with a fake PLC: with DRY-RUN ON single, row, FIRE 1-40, SEQUENCE 1-40 gave
  0 pulses and the same message, the diagnostics tool did not open; answering NO sends nothing; with DRY-RUN
  OFF single gave 1 pulse and FIRE 1-40 gave 40 pulses of relays 1-40; while RUNNING everything is refused.

### Not tested

- The real PLC. Nothing was sent to it by me.

### How to test on the machine (close the running grader first, it still runs the old code)

1. Start the grader. Pneumatic page: DRY-RUN ON.
2. TEST FIRE THIS RELAY, a row test, FIRE 1-40, SEQUENCE 1-40: each must say "DRY-RUN: NOT sent to the PLC",
   FIRE and SEQUENCE must ask first, **no valve may move**.
3. OPEN PLC DIAGNOSTICS: must be refused.
4. Open `logs/manual_fire_<date>.csv`: one line per press, "sent_to_plc" = NO.
5. Only then, nobody at the nozzles: DRY-RUN OFF, single test on relay 1: valve 1 fires, the log says YES.
6. If a valve moves in step 2, stop and tell me exactly which button.

---

## 2026-10-09 (5): Step (e2a) done - delay table fixed, always-reject rules, shadow engines, alarms

Feature reference read: 5-track user manual v4.2 (`docs/CASHEW_X_AI_Manual_English.pdf`; it was not in `docs/`,
I copied it from `CNN/CNN/app/manual/` on the drive; not committed because it is 15 MB).
Step (e2) is split: **(e2a)** the logic behind the Pneumatic and Home pages (this entry), **(e2b)** the
Pneumatic page itself and the PLC diagnostics window (next).

### Delay bug

Formula now used everywhere (`fire_timing.py`, also by RECALCULATE FROM DISTANCES):
`delay = distance from the Decision line / speed - pulse / 2`, distances added up along each ROI.
Recalculated with the config values speed 300 mm/s and pulse 50 ms. Before: A16 and A36 were lower than the
relay before them. After: every delay rises along its ROI.

| Relay | Distance mm | Old ms | New ms | Change | Relay | Distance mm | Old ms | New ms | Change |
|---|---|---|---|---|---|---|---|---|---|
| A01 | 393.7 | 1312 | 1287 | -25 | A21 | 395.0 | 1317 | 1292 | -25 |
| A02 | 588.7 | 1962 | 1937 | -25 | A22 | 599.0 | 1992 | 1972 | -20 |
| A03 | 784.7 | 2815 | 2591 | -224 | A23 | 779.0 | 2700 | 2572 | -128 |
| A04 | 970.7 | 3435 | 3211 | -224 | A24 | 973.0 | 3437 | 3218 | -219 |
| A05 | 1165.7 | 4185 | 3861 | -324 | A25 | 1172.0 | 4200 | 3882 | -318 |
| A06 | 1359.7 | 4920 | 4507 | -413 | A26 | 1367.0 | 4950 | 4532 | -418 |
| A07 | 1557.7 | 5700 | 5167 | -533 | A27 | 1560.0 | 5793 | 5175 | -618 |
| A08 | 1753.7 | 6400 | 5821 | -579 | A28 | 1755.0 | 6543 | 5825 | -718 |
| A09 | 1946.7 | 7280 | 6464 | -816 | A29 | 1948.0 | 7486 | 6468 | -1018 |
| A10 | 2141.7 | 8038 | 7114 | -924 | A30 | 2141.0 | 8129 | 7112 | -1017 |
| A11 | 2336.7 | 8688 | 7764 | -924 | A31 | 2336.0 | 8950 | 7762 | -1188 |
| A12 | 2530.7 | 9435 | 8411 | -1024 | A32 | 2530.0 | 9682 | 8408 | -1274 |
| A13 | 2725.7 | 10385 | 9061 | -1324 | A33 | 2727.0 | 10383 | 9065 | -1318 |
| A14 | 2919.7 | 11100 | 9707 | -1393 | A34 | 2921.0 | 11730 | 9712 | -2018 |
| A15 | 3116.7 | 11850 | 10364 | -1486 | A35 | 3117.0 | 12020 | 10365 | -1655 |
| A16 | 3310.7 | 5500 | 11011 | +5511 | A36 | 3310.0 | 11026 | 11008 | -18 |
| A17 | 3504.7 | 11683 | 11657 | -26 | A37 | 3511.0 | 11696 | 11678 | -18 |
| A18 | 3698.7 | 12330 | 12304 | -26 | A38 | 3704.0 | 12339 | 12322 | -17 |
| A19 | 3893.7 | 12980 | 12954 | -26 | A39 | 3897.0 | 12982 | 12965 | -17 |
| A20 | 4088.7 | 13630 | 13604 | -26 | A40 | 4093.0 | 13635 | 13618 | -17 |

**Please read before using the new values on real valves.** The old delays were of two kinds:

- A01, A02, A17-A20, A21, A22, A36-A40: exactly distance / 300 mm/s (calculated, never tuned).
- A03-A15 and A23-A35: they correspond to a belt speed of **260-289 mm/s**, falling with distance. They look
  **hand-tuned on the machine**. A16 (5500 ms) was simply wrong.

So the real belt speed is probably nearer 265-280 mm/s than the 300 mm/s in the config, and the new table
(calculated at 300) may fire up to about 1.4 s early at the far relays. The old values are kept in the table
above and in git (`git show 1babfd8:pneumatic_config.json`). Tell me the real speed (or let the vision speed
measure it after px per mm is calibrated) and I recalculate; or say "restore the tuned delays".

Also: your grader was running while I wrote the config. When that program is closed it saves its own (old)
values again. Close it, then tell me and I apply the table again, or press RECALCULATE FROM DISTANCES.

### What changed

| Area | Change |
|---|---|
| Always-reject reasons (manual B14) | Checked before every class rule, all to the reject relay: `APPEARED_AFTER_LINE` (now fired, with the time since the line taken off the delay), `DOUBLE_OR_TOUCHING` (mask longer than 45 mm in at least 2 frames and half of its frames; switch in the config, ON), `STUCK` (slower than 30 % of the lane speed for 700 ms), `LOST_nnnMS` (not seen for more than 300 ms before the line). |
| LOST | A cashew that disappears before its line (`SEEN_ONLY_ONCE`, `LOST_BEFORE_LINE`, `LEFT_ROI_UNDECIDED`) is counted and logged as LOST and never fired. |
| Inspection engines | `engines.py`: OFF / SHADOW / ACTIVE per engine, all start in **SHADOW**. Working now: **View** (NECK seen) and **Shape** (solidity 0.75, area 100 mm2, length 16 mm, length / width 1.15). Listed but not measuring yet: Colour (e3), SW/WW and Size (c2). SHADOW writes "would reject" into the new CSV column `shadow`, the history detail and the Home line; it never fires. ACTIVE can only turn a GOOD into a reject. |
| NECK rule | It is the View engine now, in SHADOW: **a clean cashew is GOOD again**; "not rolled" is only logged until you switch View to ACTIVE. |
| Home | **ALARMS** box (only while something is wrong) and **SAFETY** pill: camera offline, PLC not connected, AI fps too low for the lane speed, disk below 20 GB, conveyor stopped with pending rejects (with ACKNOWLEDGE). kg/hr from the average cashew weight **1.8 g** (GOOD tile and THROUGHPUT tile). LOST count and "SHADOW would reject" counts in the line under the tiles. |
| Tracker | Measures the longest gap of each cashew, marks stuck cashews, reports ended tracks, knows when the conveyor stands; a stuck cashew no longer pulls the lane speed down. A missed cashew keeps its ID for 600 ms (was 250) so the 300 ms LOST rule can judge it. |
| Relay output | Remembers pulses that are still on their way (for the conveyor-stopped alarm). |
| Log | `cashews_*.csv` has a new last column `shadow`; a file of the same day with the old columns is kept as `..._old_HHMMSS.csv`. |

### Relay mapping against the grading rules (kept as it is)

| Mapping | Conflict | Today |
|---|---|---|
| A02 / A22 = REJECTION | The rules send REJECTION to relay 1 / 21 | A02 / A22 never fire. Say if REJECTION should use A02 / A22. |
| UPW has no relay | UPW is the top defect | goes to the reject relay |
| A11 / A31 = SW | CNN SW is a "no defect" vote; colour decides in (c2) | practically never fires |
| A12-A16 = "SW180, W180" ... | SW and WW of one size share a relay | no SW / WW split until (c2) |
| A10 / A30 = DW1 + DW2 | none | works |
| EDGECUT, DOUBLE | no relay, intended | reject relay |

### Tested

- `ast` check; **98 unit tests pass** (14 new: delay formula with the manual's example, always-reject order,
  view engine OFF / SHADOW / ACTIVE, shape limits, double rule, gap / stuck / stopped tracking, every alarm).
- Whole program off-screen with a fake PLC and the simulated belt: SAFETY OK at start; PLC offline gives the
  ALARMS box and SAFETY FAULT and clears again; kg/hr shown; 0 pulses to the PLC.
  Screenshot: `design/e2a_home_alarm_simulated.png`.

### Not tested / to know

- Real cashews and valves. The 45 mm double rule, the shape limits in mm and kg/hr all depend on px per mm,
  which is still the uncalibrated 10.0 from the config. **If that value is wrong, DOUBLE can reject good
  cashews.** It can be switched off (`grading.double.enabled`) until the calibration of (e3).
- AI-fps alarm limit here = lane speed in px/s / largest tracker step (160 px), not the 5-track pitch formula
  (I do not know this machine's cashew pitch).
- The conveyor-stopped alarm counts only real pulses (not DRY-RUN) and was tested only in unit tests.
- Password for SAVE MAPPING: the 5-track password is written in that project's source. I will not copy it
  into this code; in (e2b) the program asks you to set the supervisor password once and stores only a hash.

---

## 2026-10-09 (4): Step (e1) done - 5-track look: theme, top bar, left menu, Home

### What changed

| Area | Change |
|---|---|
| `ui_theme.py` (new) | Port of the 5-track `home_ui.py` (newest copy, with the pills that always fit): same colours, sizes, drawn START / STOP button, EXIT / SHUTDOWN, pills, LEDs, KPI tiles, flat buttons. Roboto instead of Segoe UI; three more menu icons; right column 640 px for the 1920 x 1080 screen. The source on the drive is untouched. |
| Top bar | Brand text, PROGRAM box, pills CAMERA, PLC, MASTER EJECTOR, MODE (DRY-RUN / LIVE), CONVEYOR, STATE, clock and date, EXIT, SHUTDOWN (both ask first). |
| Left menu | 176 px, drawn icons, active page marked in cyan; "GSN V21 CASHEW GRADER" and the running git commit at the bottom. |
| Home | Round MACHINE START / STOP with red / green ring, state text, CONVEYOR / FEEDER / Auto Capture buttons, tiles TOTAL, GOOD, UNDEFINED, REJECTION, THROUGHPUT (pcs/min), RESET COUNTERS, a line with NOT ROLLED / DOUBLE / other classes, RECENT RESULTS table. |
| Camera panel | CAM fps and AI fps (AI fps is new: frames graded per second), one LED per ROI (green / red for the last decision). |
| All other pages | Get the new colours and the new frame now; their layout is rebuilt in (e2)-(e4). |
| `ui_debug.py` (new) | The program photographs its own window placed outside the visible screen, so checking pages never shows a window on your display and never clicks. |
| Question 9a | BACK_CUT_OIL / OIL_BACKCUT are graded as OIL; EDGECUT stays its own class and goes to the reject relay. PLAN section 8 now says what dataset step (f) delivers. |

Design only: grading, tracking and relay logic are unchanged (apart from the OIL alias above).

### Shown as NOT AVAILABLE (honest list)

- Programs (NEW / COPY): moved to (e2).
- kg/hr: needs the average cashew weight from you.
- LATE counter: comes with centre fire in (c3).
- Customer logo / branding: text "NEWTON AUTOMATION" (the old header text) until you give the logo.

### Tested

- `ast` check on all files; 84 unit tests pass.
- Whole program with a fake PLC, a simulated belt as the test source and a temporary config, window placed off-screen: window is 1920 x 1080, all six pills fit, DRY-RUN ON and MASTER EJECTOR OFF at start, START / STOP changes button ring, pill and state text, tiles and history update, LEDs light, CONVEYOR button reaches only `set_conveyor` of the (fake) controller, 0 pulses to the PLC, the real config was not touched.
- Real screenshots taken by the program: `design/e1_home_stopped.png`, `design/e1_home_running_simulated.png`. Pneumatic, Sorting Logic and Vision ROI were also photographed and work in the new frame.

### Not tested

- SHUTDOWN (it would switch the PC off) and EXIT were not pressed; they call the existing close routine after a yes / no question.
- Real camera picture in the new frame, real cashews, AI FPS on the machine. Please compare the AI fps number with what you had before.

### How to test

```
cd /home/phoenix/Documents/GSN_V21/CNN
python3 cashew_cnn_grader.py
```

Look at Home: pills, clock, START / STOP, tiles. Open each page from the left menu. Tell me what should look
different before I do (e2): Relay Mapping, PLC test window, Sorting Logic, programs.

---

## 2026-10-09 (3): 5-track UI studied; PLAN step (e) rewritten and moved before (c2)

- Read (read only): `TCM_CASHEWXAI_2026-10-06/` `CLAUDE.md`, `PROJECT_HISTORY.md`, `VERSIONS.txt`, `TODO.md`,
  the theme `home_ui.py`, the page list, and the real screenshots / mockups in `design/`.
- `TCM_CASHEWXAI_2026-10-06/source` and `CNN/CNN/app` are the same project. `CNN/CNN/app` is 11 commits newer
  (2026-10-07 11:27 against 00:54). Same theme, same pages.
- `PLAN.md` section 7 now holds: which copy is which, the look to copy, the page-by-page table against this
  grader, the split of step (e) into (e1)-(e4), and questions E1-E7 with proposals.
- New order: (c1) done -> (e) -> (c2) -> (c3) -> (d) -> (f). No code changed in this entry.

---

## 2026-10-09 (2): Step (c1) done - tracking, views, voting, grading rules, log

### What changed

| Area | Change |
|---|---|
| `segmentation.py` (new) | Cashew mask by blue-background removal (belt hue band + dark roller gaps removed). TEACH BELT COLOUR reads the band from an empty belt. A box touching the ROI edge is a cut-off cashew: tracked, but it does not vote and is not measured. |
| `lane_tracker.py` (new) | One tracker and one ID series per ROI (`R1-000123`, `R2-000045`). Position = mask centroid, speed in px/ms along the belt, all tracks and detections paired at once (Hungarian), no backward match, sideways limit, a missed cashew keeps its ID for a set time. Replaces the old greedy tracker. |
| `view_features.py` (new) | Per frame: aspect ratio, solidity, deepest bay, hollow share, hollow offset, symmetry, texture; also L*, a*, b*, dark %, red % and spot count (logged only). View = TOP / BACK / NECK / OTHER from limits **you** set; no limit is built in. Values are drawn under every cashew. |
| `decision.py` (new) | The rules of PLAN 3.4 as one pure function: Double -> REJECTION (YOLO class, 0.50 in 2 frames) -> combination rules -> UPW (seen, or skin + another defect) -> priority list -> GOOD (3 frames at 0.80 and NECK in M frames) -> Undefined. NECK weight 2 for TESTA / TESTA SMALL. |
| `cashew_log.py` (new) | `logs/cashews_YYYYMMDD.csv` (one row per cashew) and `logs/frames_YYYYMMDD.jsonl` (frame-by-frame, deleted after 7 days). |
| `cashew_cnn_grader.py` | Frame loop rewritten around the above: every frame keeps the YOLO class **and** the CNN class; ONE decision when the centroid crosses the line; ONE fire. Relay 1 / 21 for REJECTION, Undefined, not rolled and classes without a relay. |
| Decision lines | One line per ROI. Vision ROI page: "One line for both ROIs" is ticked at start (as before); untick it and click or drag inside a ROI to set that ROI's line. |
| Sorting Logic page (now in the left menu) | Voting limits, priority order (UP / DOWN), view limits grid, live values and lane speed per ROI, result counters, belt colour values + TEACH, tracker values, old dark-spot filter switch (default OFF), APPLY GRADING SETTINGS. |
| History (Home) | Columns are now ID, RESULT, FRAMES, RELAY. Double click a row for the frame-by-frame votes of that cashew. |
| Config | New block `grading` in `pneumatic_config.json` (written the first time the program saves). Nothing old is removed. |

Not in this step, as planned: SW/WW colour and the new size tables (c2), relay distances / centre fire / LATE (c3), double kernel and auto capture (d). Until then a GOOD cashew uses the old size table and mapping, and fire timing uses the old delay table.

### Tested

- `ast` check on all files; **83 unit tests pass** (45 new: tracker, decision rules, mask, views, log, config).
- Tracker tests: one rolling cashew keeps one ID with a jittering centroid; two neighbours 150 px apart never swap or share an ID; missed frames keep the ID; a cashew behind is never taken for the one in front; separate IDs per ROI; both flow orientations.
- Whole grading loop on a **simulated belt** (your four sample crops pasted on a blue striped background, 5 cashews, both ROIs, real YOLO + CNN): 5 cashews -> 5 IDs, 5 decisions, 5 reject-relay requests, all DRY-RUN, 0 pulses to the fake PLC, CSV and frame log written.
- Whole program with the window hidden, fake PLC, temporary config: all pages build, DRY-RUN ON and MASTER EJECTOR OFF at start, settings applied and saved, a wrong value is refused, per-ROI line set / dragged / linked again, counters and live values update, belt teach works. The real config was not touched by the tests.

### Not tested (needs the machine or a recording)

- **Real cashews.** No recording exists yet, so: tracker values (step, sideways limit, hold time) are untuned; the ByteTrack / BoT-SORT comparison is not done; view limits are empty.
- Real valves: none fired, the PLC was never contacted.

### Things you must know

1. **With the start settings nothing can become GOOD yet.** The view limits are empty, so every frame is OTHER, NECK is never seen, and with M = 1 every clean cashew is "Undefined: not_rolled" -> reject relay. Either set NECK limits on the Sorting Logic page, or set M to 0 while tuning.
2. **Views, measured on your four frames:** outline values still cannot split NECK from BACK. "Red %" (testa colour inside the mask) was 0.7 and 0.5 on the two clean frames, 7.2 and 17.5 on the two frames with testa: promising for measuring skin, but it does not tell a clean NECK from BACK. I expect the view classifier (step f) to be needed for NECK.
3. **YOLO class is noisy.** On the simulated belt YOLO called the clean sample cashew DW1 (0.87) in some frames and REJECTION often enough to reject it. The simulation uses small upscaled crops, so this is a warning, not a result. Please check REJECTION counts on real good cashews first; the YOLO class of every frame is in the history detail and in the frame log.
4. **Spot count** from image processing is not usable yet (8 "spots" on a clean frame from surface texture). Logged only.
5. **"No defect in any frame":** one defect frame below N is not a defect, but it blocks GOOD -> Undefined. That follows the prompt literally and can be switched off on the page. Tell me if you want it off by default.
6. A cashew first seen after its line gets "Undefined: first_seen_after_line", is counted, and is **not fired** (no crossing time). Step (c3) can fire it by position.
7. The per-class confidence sliders and "defect allowances" of the old logic are no longer used by grading; the voting limits replace them.
8. UPW has no relay now, so UPW goes to the reject relay as "UNDEF no relay (UPW)" until you give the mapping.
9. Belt colour: the default hue band 95-135 fits the real belt (the live camera read 106-122 during a test).
10. `pneumatic_config.json` was re-saved by your own run of the grader at 22:10 (camera block and key order); I committed it as it is.

### How to test

```
cd /home/phoenix/Documents/GSN_V21/CNN
python3 -m unittest discover -s tests
python3 cashew_cnn_grader.py
```

1. Pneumatic page: DRY-RUN ON, MASTER EJECTOR OFF (leave them).
2. Sorting Logic page: with an empty belt press TEACH BELT COLOUR.
3. START RUNNING with cashews: every cashew should show one ID from entry to the line, with the view values under it. Watch for an ID that changes or two boxes on one cashew.
4. Home: history rows show ID, result, frames, relay; double click a row for the frame detail.
5. Make the recording (Vision ROI, RECORD) and tell me the cashew count; I need it for tuning and the tracker comparison.

### Still open

Question 9a (BACK_CUT_OIL = ?, EDGECUT = ?) was left blank in your answer; those classes are left out until you say.

---

## 2026-10-09: Recovery check after power cut; PLAN v3 (English) written, waiting for approval

- `git fsck` clean, working tree clean, step (b) is committed (`8ff570d`). All `.py` pass the syntax check,
  all `.json` parse, 38 unit tests pass. Nothing was reset or deleted.
- `samples/rolling_example/frame1..4.jpg` were already in place (identical to the zip) and are now committed.
- `FINAL_PROMPT.txt` was only in Downloads; copied into the project and committed.
- New `PLAN.md` (English) covers the whole final prompt. `PLAN_TAMIL.md` is unchanged. No code changed.
- Main findings (details in `PLAN.md` sections 1 and 2):
  - Too few images: OIL 0, DW1 0, WHITE_DW 21, DW2 41; low: REJECTION 93, BROWN 117.
  - The CNN in use has no REJECTION and no DW1 class.
  - On the four sample frames the outline (aspect ratio, solidity, convexity defect) cannot tell NECK from
    BACK; interior features are added to the view check.
  - The same cashew measures b* 13 to 19 depending on the view, so SW/WW uses the median over the track.
  - The config maps relay 1 / 21 to UPW, not reject; a new mapping is needed.
- Open questions: `PLAN.md` section 10 (16 questions). For (c1): 6-11 and a recording with cashews (15).

---

## 2026-10-08 (3): Step (b) done - camera backends, Bayer check, relay safety wrapper, single-relay test

Commit: see `git log` ("Step (b): Aravis camera backend ..."). The earlier cv2.VideoCapture version (`b910963`) is replaced.

### What changed

| Area | Change |
|---|---|
| `camera_backends.py` (new, replaces `usb_camera.py`) | One interface, two backends: **Aravis** (the HIFLY camera, default) and **file** (video / image / folder). The webcam backend is removed. |
| Reconnect (#17) | The camera thread no longer ends on an error. If no good frame arrives for 2 s, or Aravis reports "control lost", it closes the camera and retries every second. States: ONLINE / RECONNECTING / OFFLINE. |
| Pixel format | Read from the camera and looked up in a table (BayerGB8, BayerGR8, BayerRG8, BayerBG8, Mono8, RGB8, BGR8). Unknown format or wrong buffer size is an error, not a wrong picture. Incomplete buffers are skipped and counted. |
| Camera controls | Exposure (us), camera gain (1-16) and an optional FPS limit on the Vision ROI page; applied to the running camera without reopening. Limits are read from the camera. An empty field leaves the camera's own value. |
| Frame timestamps | Every frame carries its capture time (Aravis system timestamp). Needed for centre fire in step (c). |
| White calibration (FFC) | Restored (it was removed in `b910963`): TEACH WHITE button, on/off switch, map saved as `ffc_gain_map.png`. |
| Recorder | RECORD button saves real camera frames (JPEG, capture time in the file name) into `recordings/<date_time>/`. A recording replays as a test source with its original timing. |
| `relay_output.py` (new) | The only way the grader fires a relay. Relays 1-40 only; 41 (conveyor, M140), 42 (feeder, M141) and above are refused. Never calls the PLC's `fire_all()` / `sequence_all()`. DRY-RUN is on at every start, MASTER EJECTOR off. Counts per relay and PLC-offline drops. |
| Single relay test | Pneumatic page: choose relay 1-40, it shows "Relay 21 -> M120 (Modbus address 120), ROI 2, REJECT relay", TEST FIRE THIS RELAY, previous / next buttons, result line. Only when STOPPED. The per-row test buttons use the same path. FIRE 1-40 and SEQUENCE 1-40 pulse the 40 ejectors one by one. |
| `CLAUDE.md` (new) | English-only replies and the working / safety rules for future sessions. |

`innovance_plc.py` was not changed.

### Bayer conversion: verified, it was already correct

- Camera format: **BayerGB8** (row 0: G B, row 1: R G).
- Code in use: `cv2.COLOR_BayerGR2BGR`. OpenCV names its Bayer codes one position shifted from GenICam, so this is the right code for BayerGB8.
- Check 1 (synthetic, now a unit test): a known colour image turned into each of the four Bayer layouts comes back with error 0.0 only with the code in the table; any other code gives error 68-99.
- Check 2 (real frame): green positions - false-colour energy 5.9 with the code in use against 23.6 with a green-shifted code. Red / blue - the rollers come out blue (mean B 33.5, R 18.7); with the swapped code they come out orange.
- Saved for you to look at, in `captures/bayer_check/`:
  - `comparison_right_vs_wrong.jpg` (code in use | red/blue swapped | green shifted)
  - `frame_camera_colours_no_gain.jpg` (camera colours, no software gain)
  - `frame_as_grader_sees_it.jpg` (with the software colour gain from the config, B x 1.4)
- **Not checked:** there was no cashew under the camera, only rollers. Please look at a frame with white cashews; the software B gain of 1.4 will matter for SW/WW and should be reviewed with white calibration in step (c2).

### Tested

- `ast` syntax check on all files; 38 unit tests pass (Bayer formats, flat-field, file replay with recorded timing, reconnect with a fake camera that is unplugged and replugged, recorder, relay wrapper).
- Real camera through the new backend: 2448 x 2048, 68-75 fps free running, frame age about 19 ms. Exposure 801 -> 1201 us, gain 1.0 -> 1.5, FPS limit 30 were applied live (measured 30.0 fps) and then put back to the values found (801 us, gain 1.0, limit off, stored rate 20.0).
- Whole program with the window hidden, fake PLC, temporary copy of the config: all pages build; camera ONLINE in the app; exposure changed from the UI and restored; 1 s recording gave 19 frames, 0 dropped; single relay test, FIRE 1-40 and SEQUENCE 1-40 reached the fake PLC only as single pulses of relays 1-40; relay tests are refused while RUNNING.
- Grading on a file source in both line orientations: decisions at the line, 0 pulses to the PLC in DRY-RUN.
- The real config files were not modified by the tests.

### Not tested (needs you, on the machine)

- Pulling the camera cable and plugging it back (only simulated).
- Any real valve. No relay was fired; the PLC was never contacted by my tests.
- White calibration with a real white sheet.
- AI FPS under load with real cashews.

### How to test

```
cd /home/phoenix/Documents/GSN_V21/CNN
python3 -m unittest discover -s tests
python3 cashew_cnn_grader.py
```

1. Camera: the health label should read "Camera ONLINE"; Vision ROI shows model, pixel format, measured fps. Change the exposure, press APPLY CAMERA SETTINGS.
2. Reconnect: pull the USB3 cable, wait for RECONNECTING, plug it back, it should return to ONLINE.
3. Valve mapping check (machine STOPPED): Pneumatic page, DRY-RUN is ON at start, so TEST FIRE only logs. Switch DRY-RUN OFF (it asks for confirmation), choose relay 1, press TEST FIRE THIS RELAY, check which valve moves, press the next button and repeat to 40. MASTER EJECTOR can stay OFF for this.
4. Record: Vision ROI, RECORD 10 s with cashews running. Then choose "Test source", pick the folder under `recordings/`, APPLY. I need such a recording for the tracker comparison in step (c1).

### Things to know

- Relay 41 / 42: the CONVEYOR and FEEDER buttons on the main screen still drive them directly through the PLC controller. They are operator buttons, not grading logic; `relay_output.py` cannot address them.
- `innovance_plc.py` `stop()` calls `all_off()`, which switches all 48 coils off (conveyor and feeder included) when the program closes. Unchanged; tell me if you want that handled differently.
- The existing grading logic (old class -> actuator mapping, old delays) still runs; step (c) replaces it. With DRY-RUN on and MASTER EJECTOR off it cannot fire anything.
- Closing the program still prints `invalid command name ..._update_loop` lines; harmless, fixed in step (e).

### Open questions (PLAN section 10; 14 and 15 are answered)

| # | Question | Needed before |
|---|---|---|
| 1 | What does the "top side" of a cashew look like (the side with the split line, or the curved back)? A few sample images would help. | c1 |
| 2 | Starting values for voting: defect minimum frames (N), defect confidence, minimum frames per cashew, GOOD minimum frames, GOOD confidence, GOOD "top side" confidence and its minimum frames. Your 5-track machine notes had GOOD >= 3 views at confidence >= 0.80 and defect >= 0.50; use those as the start? | c1 |
| 3 | Defect priority order (which defect wins when several are seen). | c1 |
| 4 | Class -> relay mapping (ROI1: 2-20, ROI2: 22-40). For GOOD: since GOOD now goes to grade relays, which grade (if any) passes without firing? | c3 |
| 5 | CNN classes `SW` and `WHITE_DW`: are they defects with their own relay, or part of the GOOD family that then goes through colour and size? If the CNN says SW and the colour measurement says WW, which wins? | c1 |
| 6 | Size grade tables for SW and for WW: grade names, limits, relay numbers. Are the limits in length, width or area? | c2 |
| 7 | SW/WW threshold: you set it after measuring samples with the teach helper. OK? | c2 |
| 8 | The old "dark spot filter" (`opencv_filter`): keep it (default off) or remove it? | c2 |
| 9 | Belt speed (config says 300 mm/s), valve response delay (ms), pulse time (config says 50 ms). | c3 |
| 10 | Relay distances: how many mm from the decision line to the first relay (relay 1 and 21)? Is 220 mm spacing right for all the others? The current config has 393.7 mm then about 195 mm; is that from the old machine? | c3 |
| 11 | If the calculated fire time has already passed: fire at once, or skip (logged as LATE)? | c3 |
| 12 | Default daily capture limit per category, and how many days of per-frame logs to keep. | d / c1 |
| 13 | Double-kernel limits: you set them after watching the live measurements (default off). OK? | d |
| 16 | Branding name and logo. | e |
| 17 (new) | Encoder details when you have them: PPR, mm per pulse, which PLC input. Time mode until then. | later |

---

# பழைய அறிக்கைகள்

---

## 2026-10-08: திட்டம் பதிப்பு 2, camera விவரங்கள், relay → PLC அட்டவணை

**நிலை:** `PLAN_TAMIL.md` புதுப்பிக்கப்பட்டு commit ஆனது. Code எதுவும் மாற்றப்படவில்லை. உங்கள் ஒப்புதலுக்காக நிற்கிறேன்.

### 1. முதலில் தெரிந்துகொள்ள வேண்டியது

`git log`:

```
b910963 Step (b): USB camera, test source, decision line in either orientation
b980fb7 Step (a): critical bug fixes
12792a7 Baseline
```

படி (a) மீண்டும் செய்யப்படவில்லை. ஆனால் **படி (b) ஏற்கனவே ஒருமுறை செய்யப்பட்டுவிட்டது** (`b910963`), அது முந்தைய அறிவுறுத்தலின்படி camera-வை `cv2.VideoCapture`-க்கு மாற்றி Aravis code-ஐ நீக்கியது. உங்கள் camera USB3 Vision என்பதால்:

- **இப்போதைய HEAD-இல் camera படம் வராது.**
- இப்போதே இயந்திரத்தை ஓட்ட: `git checkout b980fb7` (Aravis + படி (a) திருத்தங்கள்). திரும்ப: `git checkout main`.
- புதிய படி (b)-இன் முதல் வேலை Aravis-ஐ மீட்பது. `b910963`-இன் மற்ற பகுதிகள் (சோதனை மூலம், reconnect அமைப்பு, horizontal/vertical line, `external_tools/`, test-கள்) பயன்படும்.

### 2. Camera (Aravis மூலம், அமைப்புகளை மாற்றாமல்)

| விவரம் | மதிப்பு |
|---|---|
| Brand / Model | HIFLY MV518UC-O2C |
| Protocol | USB3 Vision |
| Serial | U024700000658 |
| அதிகபட்ச resolution | 2448 × 2048 |
| Pixel format | BayerGB8 (மற்றொன்று: BayerGB12p) |
| அளந்த FPS | 70.2 fps (முழு frame, 800 µs exposure, 4 விநாடி, 0 பிழை) |
| Exposure | 801 µs இப்போது; 4.2 µs – 10 s |
| Camera gain | 1.0 இப்போது; 1.0 – 16.0 |
| Auto exposure / gain / white balance | மூன்றும் Off |

- Grader ஓடவில்லை என்பதால் camera திறந்தது.
- பழைய code-இன் `COLOR_BAYER_GR2BGR` என்பது BayerGB8-க்குச் சரியான மாற்றம் (OpenCV பெயரிடல் ஒரு இடம் நகர்ந்தது). நிறங்கள் சரி; ஆனால் code அதைச் சரிபார்க்கவில்லை. புதிய code சரிபார்க்கும்.
- Camera-விலேயே உயரத்தைக் குறைத்தால் FPS கூடலாம்; அதைச் சோதிக்கவில்லை.

### 3. Relay → PLC

Inovance EC320, Modbus TCP `192.168.1.88:502`, Unit 1, Coil (FC05). Relay *r* → முகவரி `100 + (r − 1)`.

| Relay | Coil முகவரி | PLC coil | பயன் |
|---|---|---|---|
| 1 | 100 | M100 | ROI1 reject |
| 2 – 20 | 101 – 119 | M101 – M119 | ROI1 class / grade |
| 21 | 120 | M120 | ROI2 reject |
| 22 – 40 | 121 – 139 | M121 – M139 | ROI2 class / grade |
| 41 | 140 | M140 | Conveyor |
| 42 | 141 | M141 | Feeder |
| 43 – 48 | 142 – 147 | M142 – M147 | config-இல் இல்லை, code-இல் மட்டும் |

**சந்தேகம்:** M coil-கள் எந்த உண்மையான output terminal-களுக்கு (Y…) போகின்றன என்பது PLC ladder program-இல் உள்ளது; Python code-இலிருந்து சரிபார்க்க முடியாது.

**`innovance_plc.py`-இல் கவனித்தவை (மாற்றவில்லை):**

1. `fire_all()`, `sequence_all()` 48 relay-களையும் pulse செய்கின்றன: conveyor (41), feeder (42), M142–M147 உட்பட. Grader-இன் FIRE ALL பொத்தான் இதை அழைக்கிறது. கோப்பை மாற்றாமல், என் உறை (`relay_output.py`) 1–40-ஐ மட்டும் pulse செய்யும்.
2. PLC இணைப்பு இல்லாதபோது fire கட்டளை எந்தப் பிழையும் இல்லாமல் விடுபடுகிறது. உறை அதை எண்ணிக் காட்டும்.
3. Fire நேரத்தில் சுமார் 5 ms அசைவு (300 mm/s-இல் 1.5 mm); ஏற்கத்தக்கது.
4. Center fire-க்கு இந்தக் கோப்பில் மாற்றம் தேவையில்லை.

### 4. திட்டத்தின் முக்கியப் பரிந்துரைகள்

- **Tracker:** சொந்த tracker-ஐ மேம்படுத்துவது (ROI வாரியாகத் தனி, contour மையம், நேர அடிப்படை வேகம், Hungarian பொருத்தம், "பின்னோக்கி இல்லை" விதி). ByteTrack / BoT-SORT-ஐ recorder பதிவில் ஒப்பிட்டு எண்களுடன் அறிக்கை தந்த பிறகே இறுதி முடிவு.
- **திரும்பாத GOOD:** இப்போது நீங்கள் சொன்ன confidence threshold. சிறந்த வழி: GOOD-ஐ தெரியும் பக்கத்தின்படி class-களாக label செய்து "இரண்டு பக்கமும் தெரிந்தால் மட்டும் GOOD" என்ற விதி; அல்லது retrain இல்லாமல் contour வடிவ மாற்றத்தால் சுழற்சியை அளப்பது (இது வேலை செய்கிறதா என்று அளந்து பார்க்க வேண்டும்).
- **SW/WW நிறம்:** LAB b\* (மஞ்சள் அளவு), சுருக்கிய contour mask-க்குள் median, frame-களின் median. பழைய `opencv_filter` நிறத்தை அளக்கவில்லை (அது கரும்புள்ளி எண்ணிக்கை), எனவே அதைப் பயன்படுத்த முடியாது.
- **Center fire:** கடக்கும் தருணத்தை frame timestamp-களால் இடைக்கணித்து, `fire தொடக்கம் = relay-ஐ அடையும் நேரம் − pulse/2 − valve delay`. Encoder-க்கு interface இடம் மட்டும்.
- **எச்சரிக்கை:** 20 relay × 220 mm = கடைசி relay சுமார் 4.4 m தொலைவில். Belt வேகத்தில் 1% பிழை அங்கே சுமார் 44 mm விலகல். தொலைவு relay-களுக்கு encoder இல்லாமல் center fire நம்பகமாக இருக்காது.
- **படி (c)-ஐ மூன்றாகப் பிரித்தல்:** (c1) tracking + voting + log, (c2) நிறம் + அளவு, (c3) relay + center fire + dry-run.

### 5. உங்களிடம் தேவை

16 கேள்விகள் `PLAN_TAMIL.md` பகுதி 10-இல் உள்ளன. படி (b)-க்கு இரண்டு மட்டும்:

- **14.** `cv2.VideoCapture` (webcam) backend-ஐ மூன்றாவது தேர்வாக வைக்கவா, நீக்கவா? (பரிந்துரை: வைத்தல்)
- **15.** FFC white calibration-ஐ மீட்கவா? (பரிந்துரை: வேண்டாம்; அது பயன்பாட்டில் இல்லை)

"சரி, படி (b)-ஐத் தொடங்கு" என்றால் தொடங்குகிறேன்.
