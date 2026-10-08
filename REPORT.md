# Reports (newest on top)

New entries are in English. Older entries below are in Tamil.

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
