# திட்டம்: GSN_V21 முந்திரி Grader — USB camera, பல relay, புதிய UI

தேதி: 2026-10-08
நிலை: **ஒப்புதலுக்காகக் காத்திருக்கிறது.** இந்தக் கோப்பைத் தவிர project-இல் இதுவரை எதுவும் மாற்றப்படவில்லை.

---

## 0. தொடங்கும் முன் தடை: `git` நிறுவப்படவில்லை

இந்த PC-இல் `git` கட்டளை இல்லை (`git: command not found`). நிறுவ `sudo` தேவை, அதை நான் ஓட்ட முடியாது.

நீங்கள் ஓட்ட வேண்டியது (Claude Code prompt-இல்):

```
! sudo apt install -y git
```

நிறுவிய பின் நான் `git init` செய்து, தற்போதைய நிலையை முதல் commit ஆக்குவேன். `__pycache__/` மட்டும் `.gitignore`-இல் போகும்; model கோப்புகள், dataset, golden crops (மொத்தம் சுமார் 108 MB) எல்லாம் commit ஆகும்.

---

## 1. பழைய project-இல் நான் படித்தவை

Drive: `/media/phoenix/00A46655A4664D6A/CNN` (படிக்க மட்டும்; எதுவும் மாற்றப்படவில்லை).

அங்கே இரண்டு பதிப்புகள் உள்ளன:

| கோப்புறை | கடைசி மாற்றம் | UI |
|---|---|---|
| `CNN/app/` | 05-10-2026 | பழைய தோற்றம், ஒரே கோப்பு (7501 வரிகள்) |
| `CNN/CNN/app/` | 07-10-2026 | புதிய தோற்றம், தனி theme கோப்பு `home_ui.py`, `design/` கோப்புறையில் உண்மையான screenshot-கள் |

"நேற்று செய்தது, UI மிகச் சிறப்பு" என்பதற்கு **`CNN/CNN/app/`** பொருந்துகிறது; அதன் `CLAUDE.md`-உம் "இதுதான் தற்போதைய பதிப்பு, மற்றவை பழையவை" என்கிறது. அதையே reference ஆக எடுத்துள்ளேன். **இது தவறு என்றால் சொல்லுங்கள் (கேள்வி 1).**

படித்தவை: `home_ui.py` (முழுமையாக, 853 வரிகள்), `cashew_cnn_grader.py`-இன் `_build_gui` மற்றும் USB camera பகுதி, `CLAUDE.md`, `TODO.md`, `CODEX_PROGRESS.md`, `run_5track10.bat`, மற்றும் screenshot-கள் (`home_real_screenshot.png`, `pneumatic_mapping_real.png`, `real_vision_roi.png`, `ejector_mapping_real.png`).

### 1.1 பழைய UI-இன் விவரம்

**நிறங்கள் (colours)**

| பயன் | நிறம் |
|---|---|
| பின்னணி (BG) | `#0E1420` |
| பலகை (PANEL) | `#161E2E` |
| எல்லைக்கோடு (BORDER) | `#263044` |
| எழுத்து (TEXT) / மங்கிய எழுத்து (MUTED) | `#E6EAF2` / `#8A94A8` |
| முக்கிய நிறம் (ACCENT, cyan) | `#22B8CF` |
| நல்லது (GOOD, பச்சை) / எச்சரிக்கை (WARN) / தவறு (BAD, சிவப்பு) | `#2FBF71` / `#F2A93B` / `#E5484D` |
| உள்ளீட்டுப் பெட்டி (FIELD) | `#0B111B` |

**எழுத்துரு (font):** Segoe UI; நான்கு அளவுகள் மட்டும்: 13, 16, 22, 34 px.
இந்த Linux PC-இல் Segoe UI இல்லை; **Roboto** நிறுவப்பட்டுள்ளது, அதைப் பயன்படுத்துவேன் (தோற்றம் மிக நெருக்கம்).

**அமைப்பு (layout)**

- **மேல் பட்டை** (64 px): logo, PROGRAM தேர்வு, நிலை pill-கள் (புள்ளி + பெயர் + மதிப்பு: CAMERA, controller, SAFETY, STATE, MODE), கடிகாரம் + தேதி, வட்ட EXIT / SHUTDOWN பொத்தான்கள்.
- **இடது menu** (176 px): கோட்டு icon + பெயர்; தேர்ந்தெடுத்த பக்கத்தில் இடப்புறம் cyan பட்டை. பக்கங்கள்: Home, Pneumatic, Colour Size, Vision ROI, Teaching CNN, Dataset Capture, Teaching YOLO, AI Model Select, Network, Info. கீழே product logo + version.
- **நடுப்பகுதி:** camera படம் (எல்லைக்கோட்டுடன்), அதன் கீழ்-இடதில் `CAM xx fps • AI xx fps`, படத்தின் கீழே lane LED வரிசை.
- **வலது பலகை** (696 px): பெரிய வட்ட MACHINE START/STOP பொத்தான் (நிற்கும்போது சிவப்பு வளையம், ஓடும்போது பச்சை), அதன் கீழ் நிலை வரி, பிறகு பக்கத்தின் உள்ளடக்கம்.

**Home பக்கம்:** KPI ஓடுகள் (TOTAL, GOOD, BAD, REJECT %, THROUGHPUT, RESET COUNTERS), RECENT RESULTS அட்டவணை (TIME, ID, LANE, RESULT/CLASS, EJ).

**Pneumatic பக்கம்:** EJECTOR LANES அட்டவணை (lane, relay, delay, pulse, TEST, நிலை, FIRES TODAY); REJECT CLASSES chip-கள் (PASS பச்சை / FIRE சிவப்பு, அழுத்தினால் மாறும்); RECENT EJECTOR EVENTS; வலப்புறம் MASTER EJECTOR சுவிட்ச், FIRE ALL, SEQUENCE ALL, SAVE.

**Vision ROI பக்கம்:** ஒவ்வொரு lane-க்கும் DRAW / RESET; Decision line SET LINE / RESET / Shift / திசை (horizontal-vertical); px-per-mm calibration; camera rotation; Gain, Exposure slider-கள், R/G/B gain; SAVE.

**Widget-கள் (எல்லாம் `home_ui.py`-இல் Pillow மூலம் வரையப்படுகின்றன, வெளிப் படங்கள் இல்லை):** `Pill`, `NavItem`, `KpiTile`, `EngineButton`, `IconButton`, `LedLabel`, `FlatButton` (outline / danger / primary), `ToggleSwitch`, `Chip`, `Segmented`, `Slider`, `ProgressBar`, `LineChart`.

### 1.2 பழைய project-இலிருந்து எடுக்காதவை

- ESP32 serial ejector (`ejector.py`): புதிய machine `innovance_plc.py`-ஐப் பயன்படுத்தும்.
- MindVision `mvsdk`, hardware lock (motherboard serial சோதனை), welcome video, `.exe` build.
- 5 lane, ஒரே valve (accept/reject) தர்க்கம்: புதியது 2 ROI, பல relay.
- Gulf logo, "CASHEW X AI" பெயர்: வேறு customer-இன் branding என்பதால் தானாக நகலெடுக்க மாட்டேன் (கேள்வி 9).

---

## 2. Feature ஒப்பீட்டு அட்டவணை

| Feature | பழைய project (5 track) | புதிய project இப்போது | புதிய project திட்டப்படி |
|---|---|---|---|
| Camera | MindVision GigE + OpenCV fallback | Aravis GigE மட்டும்; reconnect இல்லை | **USB (`cv2.VideoCapture`) மட்டும்**; index, resolution, FPS, exposure UI-இல்; தானாக reconnect |
| ROI | 5 lane | ROI1, ROI2 | ROI1, ROI2 (மாற்றமில்லை), புதிய தோற்றத்தில் |
| Decision line | horizontal / vertical | vertical (`line_x`) மட்டும் | கேள்வி 7-இன் பதிலைப் பொறுத்து |
| Detection + class | YOLO + CNN | YOLO + CNN | அதே; class பெயர்களுக்கு ஒரே மூலம் |
| முடிவு எப்போது | line கடக்கும்போது ஒருமுறை | line கடக்கும்போது ஒருமுறை | அதே |
| Output | lane-க்கு 1 valve | 40 actuator, class → actuator | **class → relay எண்** (UI-இல் mapping) |
| Undefined | UNKNOWN → reject | fire ஆகாது (mapping இல்லையெனில் ஒன்றும் நடக்காது) | **Relay 1** |
| Double kernel | "நீளமான box → reject" நிலையான விதி | இல்லை | **Relay 1**; threshold-கள் UI-இல் |
| Auto capture | production capture (inbox) | golden capture (கைமுறையாகத் தொடங்குவது) | undefined / double ஒவ்வொன்றுக்கும் crop + முழு frame, நாள் வரம்புடன் |
| Relay controller | ESP32 serial | Inovance PLC Modbus TCP | அதே `innovance_plc.py`, **மாற்றாமல்** |
| MASTER EJECTOR | default ON | default ON | **default OFF** |
| Dry-run | இல்லை | இல்லை | **உண்டு**: fire ஆக வேண்டியதை log/UI-இல் மட்டும் காட்டும் |
| Live count | TOTAL / GOOD / BAD | detected எண்ணிக்கை மட்டும் | TOTAL, ஒவ்வொரு relay-க்கும், UNDEFINED, DOUBLE, capture எண்ணிக்கை |
| History | RECENT RESULTS | Production History | RECENT RESULTS (புதிய தோற்றம்) |
| Training | program-க்குள்ளேயே, val split உடன் | வெளி script, val split இல்லை | project-க்குள் `train_cnn.py`, train/val/test split |
| UI theme | `home_ui.py` | Slate நிறங்கள், ttk | பழைய theme அப்படியே |

---

## 3. மாற்றப்படும் / புதிதாக வரும் கோப்புகள்

**மாற்றப்படுபவை**

| கோப்பு | என்ன மாறும் |
|---|---|
| `cashew_cnn_grader.py` | camera thread, முடிவு தர்க்கம், UI முழுவதும், bug fixes |
| `ai_auto_labeler.py` | பொதுவான preprocess function, class பெயர் ஒற்றுமை, paths |
| `pneumatic_config.json` | புதிய key-கள் சேரும் (camera, class→relay, double, capture). பழைய key-கள் அழியாது; முதலில் backup நகல் |

**புதியவை**

| கோப்பு | பணி |
|---|---|
| `app_paths.py` | எல்லா paths-உம் `Path(__file__).parent` அடிப்படையில் (#12, #13) |
| `class_registry.py` | class பெயர்களுக்கு ஒரே மூலம்: YOLO → `model.names`, CNN → checkpoint `class_names` (#1) |
| `preprocess.py` | train, auto-labeler, inference மூன்றுக்கும் ஒரே crop + pad + normalize (#6) |
| `usb_camera.py` | USB camera thread: அமைப்புகள், reconnect, சோதனைக்கு video/படக் கோப்புறை மூலம் |
| `decision.py` | class → relay, undefined, double kernel விதிகள் (GUI சாராத தூய function-கள், சோதிக்க எளிது) |
| `auto_capture.py` | undefined / double சேமிப்பு, நாள் வரம்பு, தனி thread |
| `relay_output.py` | `innovance_plc.py`-ஐச் சுற்றிய மெல்லிய உறை: MASTER EJECTOR, dry-run, relay வாரியான எண்ணிக்கை |
| `ui_theme.py` | பழைய `home_ui.py`-இன் நிறங்கள், fonts, widget-கள் (Linux font உடன்) |
| `train_cnn.py` | CNN training, train/val/test split (#2, #3) |
| `tests/` | camera, PLC இல்லாமல் ஓடும் unit test-கள் |
| `.gitignore` | `__pycache__/`, `captures/`, `logs/` |

**தொடாதவை:** `innovance_plc.py`, `inovance_config.json`, `saved_models/`, `dataset/`, `cnn_final/`, `v2_snapshot/`, பழைய drive முழுவதும்.

`relay_output.py` பற்றி: `innovance_plc.py`-இல் dry-run வசதி இல்லை. அதை மாற்றாமல், அதே interface (`pulse_actuator`, `is_connected`, `start`, `stop`) கொண்ட உறை ஒன்றை வெளியே வைக்கிறேன். Dry-run-இல் PLC-க்கு எந்தக் கட்டளையும் போகாது.

---

## 4. படிப்படியான வரிசை

ஒவ்வொரு படிக்கும்: தனி commit, `ast` syntax check, என்ன மாறியது + எப்படிச் சோதிப்பது என்ற தமிழ்ச் சுருக்கம். சோதனைகள் எல்லாம் dry-run-இல்; உண்மையான relay fire ஆகாது.

### படி 0 — git
`git init`, `.gitignore`, தற்போதைய நிலை commit.

### படி (a) — critical bug fixes
| # | சரிசெய்தல் |
|---|---|
| 1 | `class_registry.py`; `cashew_class_mapping.json` இனி YOLO ID-களை மேலெழுதாது |
| 8 | `opencv_filter`, YOLO confidence ஆகியவை processing state-க்குள் |
| 9 | `_start_yolo_train` `TypeError` |
| 11 | வேறு கோப்புறை config-களை எழுதுவது நீக்கம் |
| 12, 13 | எல்லா paths-உம் `app_paths.py` வழியாக |
| 14 | `except: pass` → `logging` (`logs/grader.log`) |
| 15 | class பட்டியலுக்கு lock + நகல் |
| 19 | "Clear All" `.pt` மட்டும் நீக்கும்; `.pth` அழியாது |
| 20–25 | `GOOD_CLASSES`, `new_ip_var`, பயன்படாத constants, Keras/TF குறியீடு, shebang, தவறான log செய்தி |

#6 (பொது preprocess) இந்தப் படியிலேயே inference-க்கும் auto-labeler-க்கும்; training-க்கு படி (f)-இல்.

### படி (b) — USB camera
- Aravis, `mvsdk`, Bayer மாற்றம், FFC நீக்கம்.
- `usb_camera.py`: index, அகலம் × உயரம், FPS, exposure (auto / manual). Camera உண்மையில் ஏற்றுக்கொண்ட மதிப்புகளை UI-இல் காட்டும் (கேட்டதும் கிடைத்ததும் வேறுபடலாம்).
- Frame வராவிட்டால் (2 விநாடி) அல்லது திறக்க முடியாவிட்டால் தானாக மீண்டும் முயற்சி; நிலை pill: ONLINE / RECONNECTING / OFFLINE.
- **சோதனை மூலம்:** camera-க்குப் பதில் video கோப்பு அல்லது படக் கோப்புறை. இப்போது இந்த PC-இல் USB camera இணைக்கப்படவில்லை (`/dev/video*` இல்லை), எனவே இது தேவை.

### படி (c) — ROI / decision line / relay தர்க்கம்
- ROI1, ROI2, decision line: வரைதல், reset, நகர்த்தல், config சேமிப்பு.
- `decision.py`: class → relay mapping. "relay இல்லை (PASS)" என்ற தேர்வும் உண்டு.
- ஒரு முந்திரிக்கு ஒரே ஒரு முடிவு (tracker-இன் `fired` கொடி; unit test உடன்).
- `relay_output.py`: MASTER EJECTOR default OFF, dry-run, relay வாரியான எண்ணிக்கை.

### படி (d) — undefined + double kernel + auto capture
- **Undefined:** CNN confidence < threshold, அல்லது class mapping-இல் இல்லை → Relay 1.
- **Double:** (i) box நீளம் > வரம்பு, (ii) aspect ratio > வரம்பு, (iii) இரண்டு box-கள் overlap > வரம்பு. மூன்றும் தனித்தனியாக ON/OFF, மதிப்புகள் UI-இல். நேரடி அளவீடு (தற்போதைய box நீளம், ratio) திரையில் தெரியும், அதைப் பார்த்து வரம்பை அமைக்கலாம். → Relay 1.
- **Auto capture:** `captures/undefined/2026-10-08/` மற்றும் `captures/double/2026-10-08/`.
  கோப்புப் பெயர்: `143205_512_ROI1_TESTA_0.42_crop.jpg` மற்றும் `..._full.jpg` (நேரம்_ms, ROI, CNN கணிப்பு, confidence).
  Colour gain போட்ட frame-இலிருந்தே சேமிப்பு. தனி thread (AI FPS குறையாது). நாள் வரம்பு தாண்டினால் சேமிப்பு நிற்கும், UI-இல் "LIMIT REACHED" என்று காட்டும்; எதுவும் தானாக அழிக்கப்படாது.

### படி (e) — UI மாற்றம்
`ui_theme.py` உடன் எல்லாப் பக்கங்களும் பழைய தோற்றத்தில்:

| பக்கம் | உள்ளடக்கம் |
|---|---|
| **Home** | camera படம்; START/STOP; KPI: TOTAL, UNDEFINED, DOUBLE, capture எண்ணிக்கை (இன்று / வரம்பு); **ஒவ்வொரு relay-க்கும் live count**; RECENT RESULTS |
| **Relay Mapping** (புதிது) | class chip-கள் → relay எண் தேர்வு; Relay 1 "UNDEFINED / DOUBLE — FIXED RULE"; delay, pulse; TEST (dry-run-இல் log மட்டும்); **MASTER EJECTOR** சுவிட்ச்; DRY-RUN சுவிட்ச் |
| **Vision ROI** | ROI1 / ROI2 DRAW, RESET; decision line; camera அமைப்புகள் (index, resolution, FPS, exposure); colour gain |
| **Detection** (புதிது) | CNN confidence threshold; double kernel வரம்புகள்; capture நாள் வரம்பு; YOLO confidence |
| Teaching CNN, Dataset Capture, Teaching YOLO, AI Model Select, Network | இருக்கும் வசதிகள், புதிய தோற்றத்தில் |

மேல் பட்டை pill-கள்: CAMERA, PLC, MASTER EJECTOR, MODE (LIVE / DRY-RUN), STATE.
வேலை செய்யாத வசதி எதையும் வேலை செய்வது போல் காட்ட மாட்டேன்; `NOT AVAILABLE` என்று மங்கலாகக் காட்டுவேன்.

### படி (f) — training script (கடைசி, தனிப் படி)
- `train_cnn.py` project-க்குள்; `dataset/`-ஐ நேரடியாகப் படிக்கும்; காலி கோப்புறைகளைத் தவிர்க்கும்.
- ஒரே முந்திரியின் frame-கள் (கோப்புப் பெயரிலுள்ள object ID) ஒரே பக்கத்தில் இருக்கும்படி train / val / test split.
- `preprocess.py`-ஐப் பயன்படுத்தும்; class weight; சிறந்த val checkpoint மட்டும் சேமிப்பு; accuracy + confusion matrix கோப்பாக.
- புதிய model **தானாகச் செயல்படுத்தப்படாது**; AI Model Select-இல் நீங்கள் தேர்ந்தெடுக்க வேண்டும்.
- GUI-இன் "START TRAINING" இந்த script-ஐ அழைக்கும். Training-ஐ நான் ஓட்ட மாட்டேன்; 1 epoch சிறு சோதனை மட்டும், உங்கள் அனுமதியுடன்.

---

## 5. சோதனை முறை

- `tests/` unit test-கள்: class → relay, undefined, double விதிகள், "ஒருமுறை மட்டும் fire", capture நாள் வரம்பு, கோப்புப் பெயர்கள். Camera, PLC தேவையில்லை.
- Dry-run + video/படக் கோப்புறை மூலம்: முழு pipeline ஓடும்; fire ஆக வேண்டியவை log-இலும் UI-இலும் மட்டும்.
- **நான் சோதிக்க முடியாதவை (நீங்கள் machine-இல் பார்க்க வேண்டும்):** உண்மையான USB camera-வின் exposure கட்டுப்பாடு, PLC இணைப்பு, valve நேரம் (delay), உண்மையான முந்திரியில் double kernel வரம்புகள், AI FPS.

---

## 6. உங்களிடம் கேள்விகள்

ஊகிக்காமல் கேட்கிறேன். அடைப்புக்குறிக்குள் என் பரிந்துரை.

1. **UI reference:** `CNN/CNN/app/` (07-10, `home_ui.py`) தானா, அல்லது `CNN/app/` (05-10) ஆ? *(பரிந்துரை: `CNN/CNN/app/`)*
2. **Relay எண்கள்:** மொத்தம் எத்தனை relay பயன்படும்? ROI1-க்கும் ROI2-க்கும் **தனித்தனி relay வரிசையா** (இப்போது: ROI1 → 1–20, ROI2 → 21–40), அல்லது இரண்டுக்கும் ஒரே mapping-ஆ?
3. **Relay 1:** ROI1, ROI2 இரண்டிலும் undefined/double-க்கு அதே Relay 1 தானா? அல்லது ROI2-க்கு வேறு relay (எ.கா. 21) ஆ?
4. **நல்ல முந்திரி (GOOD):** ஒரு relay-க்குப் போக வேண்டுமா, அல்லது fire இல்லாமல் கடந்து போக வேண்டுமா (PASS)?
5. **Camera:** index (0 / 1?), resolution, FPS என்ன? இப்போது இந்த PC-இல் USB camera இணைக்கப்படவில்லை; இணைத்த பின் `v4l2-ctl --list-formats-ext` மூலம் சரியான மதிப்புகளைப் பார்க்கலாம்.
6. **Double kernel:** ஒரு சாதாரண முந்திரியின் அதிகபட்ச நீளம் (mm அல்லது px) தெரியுமா? தெரியாவிட்டால் double கண்டறிதலை முதலில் OFF-இல் வைத்து, திரையில் நேரடி அளவீடு பார்த்து நீங்கள் அமைக்கலாம். *(பரிந்துரை: இரண்டாவது)*
7. **Decision line திசை:** USB camera-வில் முந்திரி எந்தத் திசையில் நகரும்? இப்போதைய code: செங்குத்துக் கோடு, வலமிருந்து இடம். இரண்டு திசைகளையும் (horizontal / vertical) UI-இல் தேர்வாக வைக்கவா? *(பரிந்துரை: ஆம்)*
8. **Delay / pulse:** இருக்கும் relay வாரியான delay (தூரம் ÷ conveyor வேகம்) முறையை அப்படியே வைக்கவா? *(பரிந்துரை: ஆம்)*
9. **Branding:** மேல்-இடது logo, கீழ்-இடது பெயர் என்ன? இப்போது "NEWTON AUTOMATION | INSTANT FIRE GRADER". பழையதில் Gulf logo + CASHEW X AI. *(பரிந்துரை: நீங்கள் சொல்லும் வரை உரை மட்டும், logo இல்லை)*
10. **அளவு தரம் பிரித்தல் (W180, W210 …) மற்றும் SW/WHITE_DW OpenCV வடிகட்டி:** புதிய machine-இல் வேண்டுமா? புதிய விவரக்குறிப்பில் class → relay மட்டுமே உள்ளது. *(பரிந்துரை: code-ஐ வைத்து, default OFF)*
11. **Capture நாள் வரம்பு default:** undefined-க்கு 1000, double-க்கு 1000 (ஒவ்வொன்றும் crop + முழு frame = 2 கோப்புகள்) சரியா?
12. **பல frame முடிவு:** இப்போது ஒரு frame-இல் defect தெரிந்தாலும் defect என்று முடிவு. அப்படியே வைக்கவா, அல்லது பெரும்பான்மை (majority) ஆ? *(பரிந்துரை: அப்படியே)*
13. **CNN model:** இப்போதைய checkpoint-இல் 11 class-கள் (DW1, REJECTION இல்லை), `dataset/`-இல் வேறு பெயர்கள் (GOOD, TESTA_SMALL, EDGECUT, OIL_BACKCUT). இறுதி class பட்டியலை படி (f)-க்கு முன் நீங்கள் முடிவு செய்ய வேண்டும்; அதுவரை இப்போதைய model-ஐயே பயன்படுத்துகிறேன். சரியா?

---

## 7. ஒப்புதல்

"சரி, தொடங்கு" என்று சொன்னால் (git நிறுவிய பின்) படி 0, பிறகு (a) முதல் வரிசையாகச் செய்வேன். கேள்விகள் 2, 3, 4 படி (c)-க்கு முன்பும், 5, 7 படி (b)-க்கு முன்பும், 6, 11 படி (d)-க்கு முன்பும் தேவை; படி (a)-ஐ அவை இல்லாமலே தொடங்கலாம்.
