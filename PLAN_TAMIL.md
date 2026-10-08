# திட்டம் (பதிப்பு 2): GSN_V21 முந்திரி Grader

தேதி: 2026-10-08
நிலை: **ஒப்புதலுக்காகக் காத்திருக்கிறது.** இந்தப் பதிப்பு உங்கள் புதிய தேவைகளை (Aravis camera, சுழலும் முந்திரி voting, SW/WW + size, center fire) உள்ளடக்கியது. பதிப்பு 1 git வரலாற்றில் உள்ளது.

---

## 0. தற்போதைய நிலை

| Commit | என்ன |
|---|---|
| `12792a7` | படி 0: தொடக்க நிலை |
| `b980fb7` | படி (a): critical bug fixes |
| `b910963` | படி (b), **முதல் முயற்சி**: camera-வை `cv2.VideoCapture`-க்கு மாற்றியது |

### முக்கியம்: `b910963` உங்கள் camera-வுடன் வேலை செய்யாது

முந்தைய அறிவுறுத்தலின்படி ("ஒரு USB camera, cv2.VideoCapture, GigE/Aravis தேவையில்லை") படி (b)-இல் Aravis code-ஐ நீக்கிவிட்டேன். உங்கள் camera USB3 Vision என்பதால் அது `/dev/video*`-இல் தெரியாது; அதனால் **இப்போதைய HEAD-இல் camera படம் வராது.**

- இயந்திரத்தை இப்போதே ஓட்ட வேண்டுமெனில்: `git checkout b980fb7` (Aravis உடன், படி (a) திருத்தங்களுடன்). திரும்ப: `git checkout main`.
- புதிய படி (b)-இன் முதல் வேலை: Aravis capture-ஐ மீட்டெடுப்பது (கீழே பகுதி 3).
- `b910963`-இல் எழுதியவற்றில் பயன்படுபவை அப்படியே இருக்கும்: சோதனை மூலம் (video / படக் கோப்புறை), தானாக மீண்டும் இணையும் thread அமைப்பு, decision line horizontal/vertical (`flow_geometry.py`), Vision ROI பக்கம், `external_tools/` நகல்கள், 22 unit test-கள்.

---

## 1. Camera விவரங்கள் (Aravis மூலம் படித்தவை)

Grader ஓடாத நிலையில், அமைப்புகள் எதையும் மாற்றாமல் படித்தேன். FPS-ஐ அளக்க 4 விநாடி படம் பிடித்தேன்.

| விவரம் | மதிப்பு |
|---|---|
| Brand / Model | **HIFLY MV518UC-O2C** |
| Serial | U024700000658 |
| Protocol | USB3 Vision (USB ID `04b4:d003`) |
| Firmware | BIN1:1.6.29.6355 |
| Sensor / அதிகபட்ச resolution | **2448 × 2048** (5 MP), நிறம் |
| Pixel format (இப்போது) | **BayerGB8** |
| கிடைக்கும் pixel format-கள் | BayerGB8, BayerGB12p |
| அளந்த FPS (முழு resolution, 800 µs exposure) | **70.2 fps**, 281 frame-இல் 0 பிழை |
| `AcquisitionFrameRate` | 20.0, ஆனால் `AcquisitionFrameRateEnable = False` (அதனால் free-run, 70 fps) |
| Exposure | இப்போது 801 µs; வரம்பு 4.2 µs – 10 s; auto: Off |
| Gain (camera hardware) | இப்போது 1.0; வரம்பு 1.0 – 16.0; auto: Off |
| White balance auto | Off |
| Trigger | Off (Continuous) |
| Binning | இல்லை |

**Pixel format சரிபார்ப்பு (#17-இன் ஒரு பகுதி):** பழைய code `COLOR_BAYER_GR2BGR`-ஐ நிலையாகப் பயன்படுத்தியது. OpenCV-இன் Bayer பெயரிடல் GenICam-இலிருந்து ஒரு இடம் நகர்ந்தது என்பதால் `BayerGB8` ↔ `COLOR_BayerGR2BGR` என்பது **சரியான** இணை. அதாவது இப்போதைய நிறங்கள் சரி; ஆனால் அது தற்செயல், code சரிபார்க்கவில்லை. புதிய code camera சொல்லும் format-ஐப் படித்து அதற்கான மாற்றத்தைத் தேர்ந்தெடுக்கும்; தெரியாத format என்றால் (எ.கா. BayerGB12p) தவறான நிறத்தில் ஓடாமல் பிழை காட்டும்.

**அதிகபட்ச FPS பற்றி:** camera சொல்லும் வரம்பு (1,000,000) பொருளற்றது. உண்மையில் அளந்தது முழு frame-இல் 70 fps. Camera-விலேயே உயரத்தைக் குறைத்தால் (sensor ROI) FPS கூடும்; அதை நான் சோதிக்கவில்லை.

---

## 2. Relay எண் → PLC output அட்டவணை

`innovance_plc.py` மற்றும் `inovance_config.json`-இலிருந்து (கோப்புகளை மாற்றவில்லை).

PLC: Inovance EC320, Modbus TCP, `192.168.1.88:502`, Unit ID 1, **Coil (FC05)**. விதி: Relay *r* → coil முகவரி `100 + (r − 1)`.

| Relay | Modbus coil முகவரி | PLC coil | பயன் |
|---|---|---|---|
| **1** | 100 | M100 | **ROI1 reject** (Double, Undefined, திரும்பாத GOOD) |
| 2 – 20 | 101 – 119 | M101 – M119 | ROI1 class / grade relay-கள் |
| **21** | 120 | M120 | **ROI2 reject** |
| 22 – 40 | 121 – 139 | M121 – M139 | ROI2 class / grade relay-கள் |
| 41 | 140 | M140 | Conveyor (`set_conveyor`) |
| 42 | 141 | M141 | Feeder (`set_feeder`) |
| 43 – 48 | 142 – 147 | M142 – M147 | config-இல் இல்லை; code-இல் மட்டும் (`TOTAL_RELAYS = 48`) |

**சந்தேகம்:** M100–M141 coil-கள் PLC-இன் எந்த உண்மையான output terminal-களுடன் (Y0, Y1 …) இணைக்கப்பட்டுள்ளன என்பது PLC ladder program-இல் உள்ளது; இந்த Python code-இலிருந்து அதைச் சரிபார்க்க முடியாது.

### `innovance_plc.py`-இல் கவனித்தவை (மாற்றவில்லை; உங்கள் முடிவு தேவை)

1. **`fire_all()`, `sequence_all()` 48 relay-களையும் pulse செய்கின்றன**, அதாவது Relay 41 (conveyor), 42 (feeder), M142–M147 உட்பட. Grader-இன் "FIRE ALL" பொத்தான் இதை அழைக்கிறது. **தீர்வு (கோப்பை மாற்றாமல்):** என் `relay_output.py` உறை 1–40-ஐ மட்டும் தனித்தனியாக pulse செய்யும்; PLC-இன் `fire_all`-ஐ அழைக்காது.
2. `all_off()`-உம் 48-ஐயும் OFF செய்கிறது (conveyor, feeder உட்பட). Emergency stop-க்கு அது சரி; ஆனால் program மூடும்போதும் (`stop()`) இது ஓடும். சொல்லி வைக்கிறேன்.
3. PLC இணைப்பு இல்லாதபோது `pulse_actuator` எந்தப் பிழையும் சொல்லாமல் கட்டளையை விட்டுவிடும். உறை இதை "PLC OFFLINE - dropped" என்று எண்ணிக் காட்டும்.
4. Dispatcher ஒரு நேரத்தில் ஒரு coil எழுதுகிறது, 5 ms இடைவெளியில் சோதிக்கிறது. அதனால் fire நேரத்தில் சுமார் 5 ms + network நேரம் அளவுக்கு அசைவு (jitter) இருக்கும். 300 mm/s-இல் 5 ms = 1.5 mm; ஏற்கத்தக்கது.
5. ஒரே relay-இல் அடுத்தடுத்த pulse-களுக்கு இடையே 25 ms OFF இடைவெளியை உறுதிசெய்ய முந்தைய pulse-ஐக் குறைக்கிறது. Center fire கணக்கில் இதைக் கணக்கில் எடுக்கிறேன் (log-இல் பதிவு).

Center fire-க்கு `innovance_plc.py`-இல் மாற்றம் **தேவையில்லை**: `pulse_actuator(relay, delay_ms, pulse_ms)`-க்கு "இப்போதிலிருந்து எவ்வளவு நேரம் கழித்து" என்ற delay-ஐக் கணக்கிட்டுக் கொடுத்தால் போதும்.

---

## 3. படி (b), புதியது: camera மேம்பாடு

`camera_backends.py`: ஒரே interface, இரண்டு backend-கள்.

```
CameraBackend:  open()  read() -> Frame(bgr, timestamp, width, height)  apply(settings)  info()  close()
  ├─ AravisBackend   உங்கள் HIFLY camera (default)
  └─ FileBackend     video / படம் / படக் கோப்புறை (camera இல்லாமல் சோதிக்க; ஏற்கனவே எழுதி சோதிக்கப்பட்டது)
```

`b910963`-இல் எழுதிய `cv2.VideoCapture` (webcam) code-ஐ மூன்றாவது தேர்வாக வைக்கிறேன், default அல்ல. வேண்டாம் என்றால் நீக்குகிறேன் (கேள்வி 14).

| # | வேலை | விவரம் |
|---|---|---|
| 1 | Aravis மீட்பு | `b980fb7`-இல் வேலை செய்த capture வரிசையை அப்படியே backend-க்குள் (stream, 10 buffer, `ExposureAuto Off`, exposure அமைத்தல், `start_acquisition`). நடத்தை மாறாது. |
| 2 | தானாக மீண்டும் இணைதல் (#17) | Frame 2 விநாடி வராவிட்டால் அல்லது device "control lost" என்றால்: மூடி, ஒவ்வொரு விநாடியும் `update_device_list` செய்து மீண்டும் திறக்கும். நிலை: ONLINE / RECONNECTING / OFFLINE. Thread இனி முடிந்துவிடாது. |
| 3 | Pixel format சரிபார்ப்பு | Camera-வின் format-ஐப் படித்து அட்டவணைப்படி மாற்றம்: BayerGB8, BayerRG8, BayerGR8, BayerBG8, Mono8, RGB8, BGR8. மற்றவை → தெளிவான பிழை. Buffer அளவும் அகலம் × உயரத்துடன் சரிபார்க்கப்படும். |
| 4 | UI கட்டுப்பாடு | Exposure (µs, slider + எண்), camera gain (1–16), FPS வரம்பு (ON/OFF + மதிப்பு; OFF = free-run). வரம்புகள் camera-விலிருந்தே படிக்கப்படும். மாற்றம் உடனே camera-வுக்குப் போகும்; camera ஏற்ற மதிப்பு திரையில் தெரியும். Software R/G/B gain தனியாகத் தொடரும். |
| 5 | Frame timestamp | ஒவ்வொரு frame-உடனும் Aravis buffer timestamp + கணினி நேரம். Center fire-க்கு (பகுதி 6) இது அடிப்படை. |
| 6 | சோதனைப் பதிவு (recorder) | "RECORD N விநாடி" பொத்தான்: உண்மையான camera frame-களை (colour gain-க்கு முன், timestamp உடன்) ஒரு கோப்புறையில் சேமிக்கும். பிறகு அதையே FileBackend மூலம் மீண்டும் ஓட்டி tracking, voting-ஐ camera, முந்திரி இல்லாமலே சோதிக்கலாம். Disk அளவு காட்டப்படும் (5 MB × 70 fps = அதிகம்; JPEG ஆகச் சேமிப்பு, அளவு வரம்புடன்). |

**நீக்கப்பட்டது, கவனிக்க:** FFC white calibration `b910963`-இல் நீக்கப்பட்டது (`ffc_gain_map.png` கோப்பு project-இல் இல்லை, அதாவது அது பயன்பாட்டில் இல்லை). வேண்டுமென்றால் மீட்கிறேன் (கேள்வி 15).

---

## 4. படி (c): ROI, tracking, voting, நிறம்/அளவு, relay, center fire

இது பெரிய படி. மூன்று commit-களாகப் பிரிக்கப் பரிந்துரைக்கிறேன்: **(c1)** tracking + voting + log, **(c2)** SW/WW நிறம் + size grade, **(c3)** relay mapping + center fire + dry-run. ஒவ்வொன்றுக்கும் பிறகு நிறுத்தி அறிக்கை.

### 4.1 ROI, decision line

- ROI1, ROI2: வரைதல், reset, config சேமிப்பு (உள்ளது).
- Decision line: horizontal / vertical + நகரும் திசை (`b910963`-இல் உள்ளது).
- **ஒவ்வொரு ROI-க்கும் தனி line நிலை**, "இரண்டையும் இணை" (link) தேர்வுடன். Default: இணைந்தது (இப்போது போல் ஒரே கோடு).

### 4.2 Tracking (3.1)

**இப்போதைய tracker-இன் ஆய்வு** (`_process_frame`):

| குறை | விளைவு |
|---|---|
| ஒவ்வொரு track-உம் வரிசையாகத் தனக்கு மிக அருகிலுள்ள box-ஐ எடுக்கிறது (greedy) | அருகருகே உள்ள இரண்டு முந்திரிகளில் முதல் track மற்றதன் box-ஐ எடுக்கலாம் → ID switch |
| Box-இன் நடுப்புள்ளியை centroid ஆகக் கொள்கிறது | சுழலும்போது box மாறுவதால் நடுப்புள்ளி குதிக்கும் |
| வேகம் "pixel per frame" (frame இடைவெளி மாறினால் தவறு) | AI FPS மாறும்போது prediction தவறும் |
| பின்னோக்கி நகரும் பொருத்தத்தையும் ஏற்கிறது | belt ஒரே திசையில் போகிறது; பின்னோக்கிய பொருத்தம் எப்போதும் தவறு |
| ROI1, ROI2-க்கு ஒரே பட்டியல், ஒரே ID வரிசை | தேவை: தனித்தனி |
| 3 frame காணாவிட்டால் track அழிகிறது; மீண்டும் தெரிந்தால் புதிய ID | ஒரே முந்திரிக்கு இரண்டு ID |

**பரிந்துரை: சொந்த tracker-ஐ மேம்படுத்துவது** (`lane_tracker.py`, ROI-க்கு ஒன்று):

- ID: `R1-000123`, `R2-000045` (ROI வாரியாகத் தனி எண்ணிக்கை).
- Centroid = முந்திரியின் contour (mask) மையம்; box நடு அல்ல.
- Prediction: நேர அடிப்படை (px/ms), belt திசையில் மட்டும். புதிய track-க்குத் தொடக்க வேகம் = அந்த lane-இன் சமீபத்திய சராசரி வேகம்.
- பொருத்தம்: எல்லா track-களுக்கும் box-களுக்கும் ஒரே நேரத்தில் சிறந்த இணை (Hungarian, `scipy` உள்ளது), செலவு = கணித்த இடத்திலிருந்து தூரம்.
- வாயில்கள் (gates): பின்னோக்கிய நகர்வு இல்லை; ஒரு frame-இல் அதிகபட்ச முன்னேற்றம் = வேகம் × நேரம் × வரம்பு; அதே roller channel.
- காணாமல் போன track: நேர அடிப்படையில் (ms) வைத்திருந்து, கணித்த இடத்தில் மீண்டும் தெரிந்தால் **அதே ID**.
- Fire ஆன track line-க்குப் பின் வரும் புதிய முந்திரியுடன் ஒருபோதும் பொருந்தாது (இப்போதும் உள்ளது).

**ByteTrack / BoT-SORT உடன் ஒப்பீடு:**

| | சொந்த tracker (மேம்படுத்தியது) | ultralytics ByteTrack / BoT-SORT |
|---|---|---|
| belt திசை, lane, "பின்னோக்கி இல்லை" விதிகள் | நேரடியாகச் சேர்க்கலாம் | சேர்க்க முடியாது (library உள்ளே) |
| சுழற்சியால் box வடிவ மாற்றம் | centroid மட்டும் பயன்படுத்துவதால் பாதிப்பு குறைவு | Kalman நிலையில் aspect ratio, உயரம் உள்ளன; IoU பொருத்தம்; வடிவம் மாறும்போது பலவீனம் |
| குறைந்த confidence box-களை மீட்பது | இல்லை (தேவையெனில் சேர்க்கலாம்) | ByteTrack-இன் பலம் |
| BoT-SORT-இன் camera அசைவு ஈடு, ReID | தேவையில்லை (camera நிலையானது) | தேவையற்ற செலவு |
| ROI வாரியாகத் தனி ID, நம் log வடிவம் | எளிது | கூடுதல் வேலை |
| சோதிக்கப்பட்ட நம்பகத்தன்மை | நாம் சோதிக்க வேண்டும் | பரவலாகப் பயன்பாட்டில் |

இங்கே இயக்கம் எளிமையானது (ஒரே திசை, தெரிந்த வேகம், நிலையான camera), எனவே விதிகளை நேரடியாக வைக்கக்கூடிய சொந்த tracker பொருத்தம். **ஆனால் இது என் மதிப்பீடு, அளவீடு அல்ல.** படி (c1)-இல் படி (b)-இன் recorder பதிவில் இரண்டையும் ஓட்டி இந்த எண்களை ஒப்பிட்டு அறிக்கை தருகிறேன்: மொத்த ID எண்ணிக்கை (உண்மையான முந்திரி எண்ணிக்கையுடன்), ROI நடுவில் பிறந்த / இறந்த track-கள் (ID உடைதலின் அறிகுறி), track நீளப் பரவல். அதன் பிறகு இறுதித் தேர்வு.

### 4.3 பல frame முடிவு (3.2)

ஒவ்வொரு ID-க்கும் ஒவ்வொரு frame-இலும் சேமிப்பது: நேரம், CNN class, confidence, நிற மதிப்பு, நீளம்/அகலம்/பரப்பு, box. Decision line-இல் **ஒருமுறை** முடிவு (`decision.py`, GUI சாராத தூய function, unit test உடன்).

முடிவு வரிசை:

1. **Double** (படி d; default OFF) → reject relay.
2. **Defect:** ஒரு defect class, confidence ≥ `defect_conf`, குறைந்தது `defect_min_frames` frame-களில் → அந்த defect. பல defect என்றால் **priority பட்டியல்** (UI-இல் வரிசை மாற்றலாம்; `priority_rules` (#22) இதற்கே பயன்படும்). → அந்த defect-இன் relay. Mapping இல்லாத defect → Undefined.
3. **Frame-கள் குறைவு:** மொத்த frame < `min_frames` → Undefined.
4. **GOOD:** GOOD frame-கள் (confidence ≥ `good_conf`) ≥ `good_min_frames`.
   - **திரும்பாத GOOD** (இடைக்கால விதி): GOOD confidence ≥ `good_top_conf` உள்ள frame-கள் < `good_top_min_frames` → `good_not_rotated` → reject relay.
   - இல்லையெனில் → நிறம் + அளவு (4.4).
5. மேலே எதுவும் பொருந்தாவிட்டால் (confidence குறைவு, கலவையான வாக்குகள்) → Undefined → reject relay.

எல்லா எண்களும் UI-இல் (பகுதி 8). **தொடக்க மதிப்புகளை நான் ஊகிக்கவில்லை** (கேள்வி 2).

**"திரும்பாத GOOD"-க்குச் சிறந்த வழிகள் (பரிந்துரை):**

"சுழலவில்லை" என்பது ஒரு படத்தின் பண்பு அல்ல; ஒரு படம் "எந்தப் பக்கம் தெரிகிறது" என்பதை மட்டுமே சொல்லும். அதனால் `GOOD_NOT_ROTATED` என்ற ஒற்றை class-ஐ விட இவை நம்பகமானவை:

- **(A) பக்க class-கள்:** GOOD-ஐ தெரியும் பக்கத்தின்படி இரண்டு (அல்லது மூன்று) class-களாக label செய்வது (எ.கா. `GOOD_TOP`, `GOOD_BOTTOM`). விதி: "இரண்டு பக்கங்களும் குறைந்தது K frame-களில் தெரிந்தால் மட்டும் GOOD". இது நேரடியான, அளக்கக்கூடிய விதி. Retrain தேவை.
- **(B) வடிவ மாற்றம் மூலம் சுழற்சியை அளப்பது (retrain இல்லாமல்):** track முழுவதும் contour-இன் அகலம் / aspect ratio / சாய்வுக் கோணம் எவ்வளவு மாறுகிறது என்று பார்ப்பது. சுழலும் முந்திரியில் இவை மாறும்; சுழலாததில் நிலையாக இருக்கும். இது உங்கள் முந்திரி, roller-இல் வேலை செய்கிறதா என்பதை recorder பதிவில் அளந்து பார்த்த பிறகே சொல்ல முடியும்.

பரிந்துரை: இப்போது நீங்கள் சொன்ன இடைக்கால விதி (confidence threshold); (B)-ஐ (c1)-இல் அளந்து பார்த்து அறிக்கை; பிறகு (A)-க்கு retrain. "Top side" எப்படித் தெரியும் என்று எனக்குத் தெரியாது (கேள்வி 1).

**Log / history:**

- `logs/cashews_YYYYMMDD.csv`: ஒரு முந்திரிக்கு ஒரு வரி: நேரம், ID, ROI, frame எண்ணிக்கை, class வாரியான வாக்குகள், இறுதி முடிவு, காரணம், நிற மதிப்பு, SW/WW, நீளம், அகலம், பரப்பு (mm), grade, relay, fire நேரம், dry-run / fired / late.
- `logs/frames_YYYYMMDD.jsonl`: ஒவ்வொரு ID-இன் frame வாரியான கணிப்புகள் (தவறான முடிவுகளை ஆய்வு செய்ய). ON/OFF, நாள் வாரியாகக் கோப்பு, வைத்திருக்கும் நாட்கள் UI-இல்.
- UI history-இல் ஒரு வரியை அழுத்தினால் அந்த ID-இன் frame வாரியான விவரம்.

### 4.4 GOOD-க்குப் பிறகு: நிறம் (SW / WW) + அளவு (3.3)

**இப்போதைய `opencv_filter`-இன் ஆய்வு:** `_calculate_defect_area` என்பது நிறத்தை அளக்கவில்லை. அது CNN "SW" அல்லது "WHITE_DW" என்று சொன்ன முந்திரியில், gray மதிப்பு `sw_color_thresh` / `ww_color_thresh`-க்குக் கீழ் உள்ள **கருமையான pixel-களின் எண்ணிக்கையை** எண்ணி, குறைவாக இருந்தால் class-ஐ `good_cashew` ஆக மாற்றுகிறது. அதாவது அது ஒரு "கரும்புள்ளி வடிகட்டி". மேலும் அதன் mask Otsu முறையில்; பின்னணி கலக்க வாய்ப்புண்டு. **SW/WW நிற அளவீட்டுக்கு இதை அப்படியே பயன்படுத்த முடியாது**; புதிய அளவீடு எழுத வேண்டும். பழைய வடிகட்டியை default OFF-இல் வைக்கலாம் அல்லது நீக்கலாம் (கேள்வி 8).

**நிற அளவீடு (`colour_size.py`):**

- Colour gain போட்ட frame-இல், முந்திரியின் contour mask-ஐ சில pixel உள்நோக்கிச் சுருக்கி (erode), விளிம்பு / பின்னணி / நிழல் கலக்காமல்.
- மிகப் பிரகாசமான (பளபளப்பு) மற்றும் மிகக் கருமையான pixel-களை விலக்கி.
- **பரிந்துரை: CIE LAB-இன் b\* (மஞ்சள் அளவு), mask-க்குள் median.** காரணம்: b\* மஞ்சள்–நீல அச்சை நேரடியாக அளக்கிறது, வெளிச்ச அளவிலிருந்து (L\*) ஓரளவு தனி. HSV-இன் saturation வெளிர் நிறங்களில் (வெள்ளைக்கு அருகில்) நிலையற்றது, hue அங்கே பொருளற்றது.
- ஒரு ID-க்கு: எல்லா frame-களின் **median** (சுழல்வதால்).
- UI: SW/WW எல்லை slider; live b\* மதிப்பு; கூடவே L\* மற்றும் HSV S-ஐயும் காட்டுகிறேன், ஒப்பிட்டுப் பார்க்க.
- **Teach உதவி:** தெரிந்த WW முந்திரிகளை ஓட்டி "WW மாதிரி", SW-ஐ ஓட்டி "SW மாதிரி" என்று பதிவு செய்தால் இரண்டின் பரவலையும் (histogram) காட்டி, நடுப்புள்ளியை threshold ஆகப் பரிந்துரைக்கும். இறுதி மதிப்பை நீங்களே அமைப்பீர்கள்.
- **எச்சரிக்கை:** இந்த அளவீடு exposure, gain, white balance, விளக்கைச் சார்ந்தது. Threshold அமைத்த பின் அவற்றை மாற்றினால் மீண்டும் அமைக்க வேண்டும். Camera-வில் auto exposure / gain / WB மூன்றும் Off-இல் உள்ளன (நல்லது).
- b\* உங்கள் SW/WW-ஐ உண்மையில் பிரிக்கிறதா என்பது **அளந்து பார்த்த பிறகே தெரியும்**; (c2)-இல் உங்கள் மாதிரிகளின் பரவலைக் காட்டுகிறேன்.

**அளவு:**

- Contour → `minAreaRect` நீளம், அகலம்; `contourArea` பரப்பு; `px_per_mm` மூலம் mm, mm².
- ஒரு ID-க்கு: நீளம் = median, அகலம் = max, பரப்பு = max என்பது என் முன்மொழிவு (roller-இல் நீள அச்சைச் சுற்றிச் சுழல்வதாகக் கொண்டால் நீளம் மாறாது, அகலம் மாறும்). எதை grade-க்குப் பயன்படுத்துவது என்பது உங்கள் முடிவு (கேள்வி 6).
- **px_per_mm calibration:** திரையில் தெரிந்த நீளமுள்ள பொருளின் (scale / அளவுகோல்) இரு முனைகளை அழுத்தி, mm-ஐத் தட்டச்சு செய்தால் கணக்கிடும். ROI வாரியாகத் தனி மதிப்பு.
- **Grade அட்டவணைகள்:** SW-க்கு ஒன்று, WW-க்கு ஒன்று. ஒவ்வொரு வரி: grade பெயர், min, max, ROI1 relay, ROI2 relay. UI-இல் வரி சேர்க்க / நீக்க / திருத்த. எல்லைகள் ஒன்றின் மேல் ஒன்று வந்தால் எச்சரிக்கை.
- எந்த grade-க்கும் பொருந்தாத அளவு → Undefined → reject relay.

**இறுதி ஓட்டம்:**

```
YOLO → CNN (ஒவ்வொரு frame) → tracker (ROI வாரியாக) → line கடக்கும்போது ஒருமுறை:
   Double?                         → reject relay (R1 / R21)
   Defect (N frame, priority)?     → அந்த defect-இன் relay
   Undefined / திரும்பாத GOOD?     → reject relay
   GOOD → b* median → SW / WW → அந்த நிறத்தின் size grade → grade relay
                                     (grade பொருந்தவில்லை → reject relay)
```

### 4.5 Relay mapping, dry-run

- Class → relay: ROI1-க்கு 2–20, ROI2-க்கு 22–40; "relay இல்லை (PASS)" தேர்வு. Relay 1, 21 நிலையான reject.
- `relay_output.py` (PLC கோப்பை மாற்றாத உறை): MASTER EJECTOR (default OFF), **DRY-RUN** (PLC-க்கு எதுவும் போகாது; fire ஆக வேண்டியவை log, UI-இல் மட்டும்), relay வாரியான எண்ணிக்கை, PLC offline எண்ணிக்கை, FIRE ALL = 1–40 மட்டும்.
- Dry-run default **ON**; உண்மையான fire-க்கு நீங்கள் DRY-RUN OFF + MASTER EJECTOR ON இரண்டையும் செய்ய வேண்டும்.

### 4.6 Center fire (5)

**கடக்கும் தருணம்:** line-க்கு முன் உள்ள கடைசி frame (நேரம் t₁, line-இலிருந்து தூரம் d₁) மற்றும் line-க்குப் பின் உள்ள முதல் frame (t₂, d₂) இடையே நேர்கோட்டு இடைக்கணிப்பு:

```
t_cross = t₁ + (t₂ − t₁) × d₁ / (d₁ + d₂)
```

t என்பது **frame பிடிக்கப்பட்ட நேரம்** (Aravis timestamp), processing முடிந்த நேரம் அல்ல. அதனால் YOLO/CNN எடுத்த நேரம் தானாக ஈடாகிறது. Camera-விலிருந்து கணினிக்கு வரும் தாமதத்துக்கு "camera latency (ms)" என்ற ஒரு அமைப்பு; exposure-இன் பாதியும் கழிக்கப்படும்.

**Fire நேரம்:**

```
centroid relay-ஐ அடையும் நேரம் = t_cross + (relay தூரம் mm + offset mm) ÷ belt வேகம்
fire தொடக்கம் = அந்த நேரம் − (pulse ÷ 2) − valve response delay + offset ms
```

- **Relay தூரம்:** ஒவ்வொரு relay-க்கும் line-இலிருந்து mm. "முதல் relay தூரம் + இடைவெளி (default 220 mm) கொண்டு நிரப்பு" பொத்தான்.
- **Pulse:** நிலையானது (ms), அல்லது கணக்கிட்டது = (முந்திரி நீளம் mm + விளிம்பு mm) ÷ belt வேகம், min/max வரம்புடன்.
- **Offset:** ஒவ்வொரு relay-க்கும் mm மற்றும் ms.
- **Belt வேகம் (time mode):** UI-இல் mm/s. Track-களிலிருந்து அளந்த வேகத்தையும் திரையில் காட்டுகிறேன்; அமைத்த மதிப்பிலிருந்து அதிகம் விலகினால் எச்சரிக்கை.
- **தாமதமான fire:** கணக்கிட்ட fire தொடக்கம் ஏற்கனவே கடந்திருந்தால் என்ன செய்வது என்பது உங்கள் முடிவு (கேள்வி 11); எப்படியும் log-இல் `LATE` என்று பதிவு.
- **Encoder-க்கு இடம்:** `ConveyorClock` என்ற interface. இப்போது `TimeClock` (வேகம் × நேரம்). பின்னர் `EncoderClock` (mm per pulse, pulse எண்ணிக்கை) அதே இடத்தில் செருகலாம். இப்போது implement செய்யவில்லை.

**எச்சரிக்கை (time mode-இன் வரம்பு):** 20 relay × 220 mm என்றால் கடைசி relay line-இலிருந்து சுமார் 4.4 m; 300 mm/s-இல் சுமார் 14.7 விநாடி. Belt வேகத்தில் 1% பிழை இருந்தால் கடைசி relay-இல் சுமார் 44 mm விலகல், அதாவது முந்திரியையே தவறவிடலாம். தொலைவிலுள்ள relay-களுக்கு encoder இல்லாமல் center fire நம்பகமாக இருக்காது.

---

## 5. படி (d): double kernel + auto capture

- **Double:** (i) contour நீளம் > வரம்பு, (ii) aspect ratio > வரம்பு, (iii) இரண்டு box overlap > வரம்பு. மூன்றும் தனித்தனியாக ON/OFF. **Default: எல்லாம் OFF.** திரையில் ஒவ்வொரு முந்திரியின் நீளம் (px, mm), ratio நேரடியாகத் தெரியும்; வரம்பை நீங்கள் அமைப்பீர்கள்.
- **Auto capture:** `captures/undefined/தேதி/`, `captures/double/தேதி/`, `captures/good_not_rotated/தேதி/`. ஒவ்வொரு முந்திரிக்கும் crop + முழு frame.
  கோப்புப் பெயர்: `143205_512_R1-000123_ROI1_TESTA_0.42_crop.jpg`, `..._full.jpg` (நேரம்_ms, ID, ROI, CNN கணிப்பு, confidence).
- Colour gain போட்ட frame-இலிருந்தே. தனி thread (AI FPS குறையாது). நாள் வரம்பு (வகை வாரியாக) UI-இல்; வரம்பு தாண்டினால் "LIMIT REACHED"; எதுவும் தானாக அழிக்கப்படாது. Disk காலி இடம் குறைந்தால் நிற்கும்.

---

## 6. படி (e): UI

Reference: `CNN/CNN/app/` (`home_ui.py`). நிறங்கள்: பின்னணி `#0E1420`, பலகை `#161E2E`, cyan `#22B8CF`, பச்சை `#2FBF71`, சிவப்பு `#E5484D`. Font: Segoe UI இங்கே இல்லை → Roboto. மேல் status pill-கள், இடது menu (176 px), நடுவில் camera, வலப்புறம் வட்ட START/STOP + KPI ஓடுகள் + history.

| பக்கம் | உள்ளடக்கம் |
|---|---|
| **Home** | camera; START/STOP; KPI: TOTAL, GOOD, UNDEFINED, DOUBLE, திரும்பாத GOOD, capture (இன்று/வரம்பு); **relay வாரியான live count**; **grade வாரியான count**; RECENT RESULTS (ID, ROI, முடிவு, நிறம், அளவு, grade, relay); வரியை அழுத்தினால் frame விவரம் |
| **Relay Mapping** | class → relay (ROI1, ROI2); Relay 1 / 21 "REJECT — FIXED"; relay தூரம், offset (mm, ms), pulse; TEST; **MASTER EJECTOR**; **DRY-RUN**; FIRE ALL / SEQUENCE (1–40 மட்டும்) |
| **Sorting Logic** (#16) | voting விதிகள் (N, threshold-கள், min frame); defect priority வரிசை; SW/WW threshold + live மதிப்பு + teach; SW grade அட்டவணை; WW grade அட்டவணை; px_per_mm calibration; double வரம்புகள்; capture வரம்பு |
| **Timing** | belt வேகம், valve response delay, camera latency, pulse முறை; அளந்த belt வேகம்; encoder: `NOT AVAILABLE` |
| **Vision ROI** | ROI1/ROI2; decision line (திசை, ROI வாரியாக / இணைந்து); camera (exposure, gain, FPS, source, recorder); colour gain |
| Teaching CNN, Dataset Capture, Teaching YOLO, AI Model Select, Network | இருக்கும் வசதிகள், புதிய தோற்றத்தில் |

மேல் pill-கள்: CAMERA, PLC, MASTER EJECTOR, MODE (LIVE / DRY-RUN), STATE.
வேலை செய்யாத வசதிகள் `NOT AVAILABLE` என்று மங்கலாகத் தெரியும். Branding: நீங்கள் தரும் வரை உரை மட்டும், logo இல்லை.

---

## 7. படி (f): training script

- `train_cnn.py` project-க்குள்; `dataset/`-ஐ நேரடியாகப் படிக்கும்; காலி கோப்புறைகளைத் தவிர்க்கும்.
- ஒரே முந்திரியின் frame-கள் (கோப்புப் பெயரிலுள்ள ID) ஒரே பக்கத்தில் இருக்கும்படி train / val / test split.
- `preprocess.py` (inference-உடன் ஒரே crop); class weight; சிறந்த val checkpoint; accuracy + confusion matrix கோப்பாக.
- புதிய model தானாகச் செயல்படுத்தப்படாது.
- பக்க class-கள் (4.3-A) முடிவானால் அவற்றுக்கான label பட்டியல் இங்கே.

---

## 8. புதிய / மாறும் கோப்புகள்

| கோப்பு | படி | பணி |
|---|---|---|
| `camera_backends.py` (`usb_camera.py`-ஐ மாற்றி) | b | Aravis, File (மற்றும் UVC) backend-கள், reconnect, pixel format, recorder |
| `lane_tracker.py` | c1 | ROI வாரியான tracker |
| `decision.py` | c1 | voting, priority, undefined, திரும்பாத GOOD |
| `cashew_log.py` | c1 | CSV + frame JSONL |
| `colour_size.py` | c2 | b\* அளவீடு, contour அளவு, grade அட்டவணை |
| `relay_output.py` | c3 | PLC உறை: master, dry-run, எண்ணிக்கை |
| `fire_timing.py` | c3 | கடக்கும் தருண இடைக்கணிப்பு, center fire, `ConveyorClock` |
| `auto_capture.py` | d | capture thread, நாள் வரம்பு |
| `ui_theme.py` + பக்கக் கோப்புகள் | e | பழைய theme, புதிய பக்கங்கள் |
| `train_cnn.py` | f | training |
| `tests/` | எல்லாம் | camera, PLC இல்லாமல் ஓடும் unit test-கள் |
| `cashew_cnn_grader.py`, `pneumatic_config.json` | எல்லாம் | இணைப்பு; config-இல் புதிய key-கள் (பழையவை அழியாது) |

**தொடாதவை:** `innovance_plc.py`, `inovance_config.json`, `saved_models/`, `dataset/`, `cnn_final/`, `v2_snapshot/`, `external_tools/` நகல்கள், பழைய drive, `RaniVision_Latest`.

---

## 9. சோதனை முறை

- ஒவ்வொரு படிக்கும்: `ast` syntax check + unit test-கள்.
- Recorder பதிவு + FileBackend + dry-run: முழு pipeline, PLC-ஐத் தொடாமல்.
- **நான் சோதிக்க முடியாதவை (நீங்கள் machine-இல்):** valve நேரம், center fire துல்லியம், உண்மையான முந்திரியில் voting / நிறம் / அளவு மதிப்புகள், camera cable-ஐப் பிடுங்கி reconnect, AI FPS சுமையில்.

---

## 10. உங்களிடம் கேள்விகள்

எதையும் ஊகிக்கவில்லை. படி எண் = அந்தப் படிக்கு முன் பதில் தேவை.

| # | கேள்வி | படி |
|---|---|---|
| 1 | "Top side" எப்படித் தெரியும் (பிளவுக் கோடு உள்ள பக்கமா, வளைந்த முதுகா)? ஒரு சில மாதிரிப் படங்கள் இருந்தால் நல்லது. | c1 |
| 2 | Voting தொடக்க மதிப்புகள்: `defect_min_frames` (N), `defect_conf`, `min_frames`, `good_min_frames`, `good_conf`, `good_top_conf`, `good_top_min_frames`. (உங்கள் 5 track machine-இன் குறிப்புகளில் இருந்தவை: GOOD-க்கு ≥ 3 view, confidence ≥ 0.80; defect ≥ 0.50. அவற்றையே தொடக்கமாக வைக்கவா?) | c1 |
| 3 | Defect priority வரிசை (பல defect தெரிந்தால் எது வெல்லும்)? | c1 |
| 4 | Class → relay mapping (ROI1: 2–20, ROI2: 22–40). GOOD: PASS-ஆ, relay-ஆ? (GOOD இனி grade relay-களுக்குப் போவதால், "PASS" என்பது எந்த grade-க்கு?) | c3 |
| 5 | CNN-இன் `SW`, `WHITE_DW` class-கள்: இவை defect-ஆ (சொந்த relay), அல்லது GOOD குடும்பமா (பிறகு நிறம்/அளவு)? CNN "SW" என்றும் நிற அளவீடு "WW" என்றும் சொன்னால் எது வெல்லும்? | c1 |
| 6 | Size grade அட்டவணைகள்: SW, WW ஒவ்வொன்றுக்கும் grade பெயர்கள், எல்லைகள், relay எண்கள். எல்லைகள் எந்த அளவில்: நீளம், அகலம், பரப்பு? | c2 |
| 7 | SW/WW threshold: teach உதவியில் மாதிரிகளை அளந்த பிறகு நீங்கள் அமைப்பீர்கள். சரியா? | c2 |
| 8 | பழைய "கரும்புள்ளி வடிகட்டி" (`opencv_filter`): வைக்கவா (default OFF), நீக்கவா? | c2 |
| 9 | Belt வேகம் (config-இல் 300 mm/s), valve response delay (ms), pulse நேரம் (config-இல் 50 ms)? | c3 |
| 10 | Relay தூரங்கள்: line-இலிருந்து முதல் relay (Relay 1, 21) எவ்வளவு mm? பிறகு 220 mm இடைவெளி எல்லாவற்றுக்கும் சரியா? (இப்போதைய config: முதல் 393.7 mm, பிறகு சுமார் 195 mm; அது பழைய machine-உடையதா?) | c3 |
| 11 | Fire நேரம் ஏற்கனவே கடந்திருந்தால்: உடனே fire செய்யவா, விட்டுவிடவா (log-இல் LATE)? | c3 |
| 12 | Capture நாள் வரம்பு default (ஒவ்வொரு வகைக்கும்)? Frame log-ஐ எத்தனை நாள் வைத்திருக்க? | d / c1 |
| 13 | Double kernel வரம்புகள்: திரையில் அளவீடு பார்த்த பிறகு நீங்கள் அமைப்பீர்கள் (default OFF). | d |
| 14 | `cv2.VideoCapture` (webcam) backend: மூன்றாவது தேர்வாக வைக்கவா, நீக்கவா? | b |
| 15 | FFC white calibration: மீட்கவா? | b |
| 16 | Branding பெயர், logo. | e |

---

## 11. ஒப்புதல்

"சரி, படி (b)-ஐத் தொடங்கு" என்றால் camera மேம்பாட்டைச் செய்து, commit செய்து, அறிக்கையுடன் நிறுத்துவேன். படி (b)-க்கு கேள்விகள் 14, 15 மட்டும் தேவை; பதில் இல்லையெனில் என் பரிந்துரைப்படி (webcam backend-ஐ வைத்தல், FFC-ஐ மீட்காமல் விடுதல்) செய்வேன்.
