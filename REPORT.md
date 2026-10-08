# அறிக்கைகள் (புதியது மேலே)

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
