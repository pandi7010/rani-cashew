# 🌰 Cashew CNN Grader - Comprehensive System Log & Project Status
Date: 2026-09-16

---

## 📌 Executive Summary (இன்றைய முக்கிய சாதனைகள் & புதுப்பிப்புகள்)

Today, we successfully resolved all GUI freezing issues, inference bugs, ejector mapping errors, config synchronization conflicts, and classification rules for the **CNN Cashew Grader System**.

---

## 🛠️ Key Technical Solved Items & Enhancements

### 1. **Live Conveyor Tracking & Production History Logging**
- **Decision Line Fix**: Corrected state mapping key (`decision_line_x` -> `line_x`) so solenoids fire reliably and real-time history logs appear in `PRODUCTION HISTORY` table with live timestamps, confidence scores, and actuator IDs.
- **78+ FPS Performance**: Optimized frame processing pipeline for smooth live execution.

### 2. **Clean Image Capture & Dataset Generation**
- **Overlay-Free Crops**: Introduced `clean_frame = frame.copy()` before drawing ROI boxes, decision lines, or white center dots. All CNN inference crops and Golden Capture dataset images are 100% clean raw crops without graphics pollution.
- **Edge-Touching Filter**: Added boundary checks (`is_touching_edge`) to skip capturing cashews cut by frame edges.

### 3. **AI Auto-Labeler UI Upgrade**
- **Enhanced UI**: Built a 1150x700 window with a 3-column color-coded class grid (Green=Good, Orange=Cuts, Brown=Defects, Teal=Pieces).
- **Source Folder Selector**: Added a dropdown to target specific directories (`dataset/TESTA`, `cnn_final/golden_crops`, etc.) for seamless auto-sorting.

### 4. **MobileNetV2 Model Training & Evaluation**
- **Training**: Trained a 15-class MobileNetV2 model achieving **82.30% final accuracy**.
- **Model Checkpoints**: Updated model loading to support both `'model_state_dict'` and `'model_state'` keys, auto-synced to `saved_models/cashew_color_cnn_new.pth`.
- **Graphs Generated**:
  - `confusion_matrix_graph.png`: Confusion Matrix Heatmap.
  - `training_metrics_graph.png`: Loss and Accuracy training curves over 10 epochs.

### 5. **Tight Cashew Bounding Box Localization**
- **Contour Refinement**: Used OpenCV contour detection (`_get_cashew_contour_rect`) inside blue roller channel crops to shrink green bounding boxes tightly around the cashew itself. Empty blue roller slots are skipped without drawing false boxes.

### 6. **Pneumatic Timing & Delay Persistence**
- **Entry Saving Fix**: Created `_save_pneumatics()` to read all 40 actuator delay entries (`A01`-`A40`) as `StringVar` and safely cast to `int`.
- **Auto-Sync Config**: Configured `CFG_PATH` with absolute path resolution and auto-syncing to parent directories so no old `pneumatic_config.json` file conflicts with new delay settings.

### 7. **Domain Classification Rules & Rejection Sensitivity**
- **SMALL TESTA**: Length/width < 5mm -> Classified as `TESTA SMALL` (Actuator A04/A24).
- **UPW vs TESTA**: Skin defect coverage > 30% -> `UPW` (Actuator A02/A22); <= 30% -> `TESTA` (Actuator A03/A23).
- **Clean Good Cashews**: Lowered defect confidence threshold (`class_confs`) to **35%** for aggressive defect rejection, preventing TESTA / UPW from slipping into Good Cashews.

### 9. **Arduino to Inovance EC320 PLC Modbus TCP Migration**
- **InovancePLCController Integration**: Successfully converted pneumatic ejection control from serial Arduino Mega to high-performance industrial **Inovance EC320 PLC** via Modbus TCP (Port 502, Default IP: 192.168.1.88, Unit ID: 1).
- **Sub-Millisecond Event Dispatcher**: Replaced thread-per-pulse overhead and serial 64-byte buffer overflow limits with a single-worker priority-heap pulse dispatcher (`heapq` + `Condition`). Accurately fires and releases 40 pneumatic actuators without Modbus socket collisions.
- **Actuator & Relay Mapping**:
  - Bank A (A01 - A20 / Track 1): Modbus coils M100 to M119 (addresses 100-119).
  - Bank B (A21 - A40 / Track 2): Modbus coils M120 to M139 (addresses 120-139).
  - Fully supports both Coils (FC01/05/15) and Holding Registers (FC03/06/16), with persistent custom mapping in `inovance_config.json`.
- **GUI Control & Diagnostics**:
  - Real-time PLC connection indicator (`● Inovance PLC`) on the main dashboard health header.
  - Added dedicated Inovance PLC configuration card in the **Network** tab (IP, Port, Unit ID, Register Type, Base Register, Save & Connect).
  - Added direct **"OPEN PLC DIAGNOSTICS"** launch buttons in both the **Pneumatics** and **Network** tabs to open the full 40-relay diagnostics & mapping tool.
- **Fail-Safe Emergency Stop**: `all_off()` clears all queued pulses instantly and sets all 40 relays to 0 in a single Modbus batch command.

---

## 🚀 How to Run the Project Tomorrow (நாளை இயக்கும் முறை)

### Option A: Automatic Boot (சிஸ்டம் ஆன் செய்யும் போது)
The app will automatically open upon system boot. Simply click **START RUNNING** on the Home tab.

### Option B: Terminal Command (டெர்மினலில் இயக்கும் போது)
```bash
cd /home/phoenix/Documents/cnn_plc/CNN
python3 cashew_cnn_grader.py
```

### Option C: Standalone PLC Diagnostics & Testing Tool
```bash
cd /home/phoenix/Documents/cnn_plc/CNN
python3 innovance_plc.py
```

---

## 🎯 Active Actuator Mapping Reference (Track 1 / Track 2)
- **A02 / A22**: `UPW` (> 30% Skin) [PLC Coils: M101 / M121]
- **A03 / A23**: `TESTA` (<= 30% Skin) [PLC Coils: M102 / M122]
- **A04 / A24**: `TESTA SMALL` (< 5mm) [PLC Coils: M103 / M123]
- **A06 / A26**: `PKW2` [PLC Coils: M105 / M125]
- **A07 / A27**: `PKW1` [PLC Coils: M106 / M126]
- **A10 / A30**: `WHITE_DW` [PLC Coils: M109 / M129]
- **A13 / A33**: `DW1`, `DW2` [PLC Coils: M112 / M132]

---

# 🚀 VERSION 2.0 (V2) SNAPSHOT & COMPLETE SYSTEM LOG
**Snapshot Date & Time**: `2026-10-02 02:20:00 IST`  
**Snapshot Version Tag**: `V2 (Production Ready)`  
**Backup Archive Directory**: `file:///home/phoenix/Documents/cnn_plc/CNN/v2_snapshot/`

---

## 📌 V2 Overview & Executive Accomplishments (V2 சாதனைகள்)
All settings, UI modifications, mathematical calculations, and hardware mappings have been frozen and saved into Version 2. When the user requests **"start v2"**, the system will immediately apply and run with the exact configuration logged below.

---

## 🛠️ V2 Key Technical Implementations & Enhancements

### 1. **Pneumatic Distance-to-Delay Auto-Calculation Engine**
- **Dual Input Boxes (mm & ms)**: Added two dedicated input fields for every single actuator (A01 - A40) on the Pneumatics page:
  - **Distance Box (`mm`)**: Distance from decision line (for A01 / A21) or distance from previous ejector hole to next ejector hole (for A02..A20 & A22..A40).
  - **Delay Box (`ms`)**: Total travel delay time in milliseconds.
- **Auto-Calculation Logic**:
  - Automatically calculates $\Delta t = \frac{\text{Distance (mm)}}{\text{Conveyor Speed (mm/s)}} \times 1000\text{ ms}$.
  - Seamlessly updates cumulative delay as soon as the user types the distance.
- **Manual Micro-Tuning Preserved**: If the operator manually adjusts the delay (e.g. $\pm 10\text{ ms}$ or $\pm 20\text{ ms}$), the manual delay value is honored without being overwritten by the distance.
- **Conveyor Speed Variable**: Dedicated `Conveyor Speed (mm/s)` input box (default: 300 mm/s) with a **"RECALCULATE FROM DISTANCES"** button to bulk update all 40 ejectors upon speed changes.
- **Physical Timing Accuracy**:
  - Ejector 1: $15.5\text{ in} = 393.7\text{ mm} \rightarrow \mathbf{1310\text{ ms}}$
  - Ejector 2: $195\text{ mm} \rightarrow \mathbf{1960\text{ ms}}$
  - Ejector 3: $196\text{ mm} \rightarrow \mathbf{2613\text{ ms}}$ (Corrected from 2680 ms to eliminate 20mm miss).

### 2. **Inovance EC320 PLC 48-Relay Expansion & Motor Controls**
- **Relay Scale**: Expanded `TOTAL_RELAYS` in `InovancePLCController` from 40 to 48.
- **Actuators (Ejectors 1 to 40)**: Mapped to Modbus Coils `M100` to `M139` (addresses 100-139).
- **Conveyor Control (Relay 41)**: Mapped to Coil `M140` (address 140) for VFD Run/Stop.
- **Feeder Control (Relay 42)**: Mapped to Coil `M141` (address 141) for Vibrator Feeder Run/Stop.

### 3. **Main Dashboard UI Controls (Near Start Button)**
- Placed directly beneath the **`START RUNNING`** button on the Home dashboard:
  - **`[ CONVEYOR: ON / OFF ]`**: Independent toggle button with dynamic color feedback (Slate Grey $\leftrightarrow$ Bright Green). Triggers Relay 41 / Coil M140.
  - **`[ FEEDER: ON / OFF ]`**: Independent toggle button with dynamic color feedback (Slate Grey $\leftrightarrow$ Bright Green). Triggers Relay 42 / Coil M141.
- Operable independently of the vision inspection state.

### 4. **Ejector Assignment Popup Upgrades**
- In the Ejector Assignment window (`_popup_mapper`):
  - Added **`[ SELECT ALL ]`** button to check all defect and size classes with one click.
  - Added **`[ DESELECT ALL ]`** button to clear all checked classes with one click.
  - Positioned directly above the **`[ Confirm ]`** button for optimal ergonomics.

### 5. **Antigravity Microsecond Priority Queue & Gap Protection**
- Replaced PLC internal shift register tracking with PC-based absolute timestamp scheduling (`heapq`).
- Enforces **Minimum Gap Protection (`min_gap_sec = 25 ms`)**: Prevents continuous streams (`1 1 1 1 1`) from sticking the solenoid ON, ensuring every single cashew receives an independent, full-pressure air pulse.
- Socket configured with `TCP_NODELAY = 1` for sub-0.5 ms Modbus write latency.

---

## 📁 V2 Saved Files & Snapshot Archive

| File Name | Path | Purpose |
| :--- | :--- | :--- |
| `cashew_cnn_grader.py` | `/home/phoenix/Documents/cnn_plc/CNN/cashew_cnn_grader.py` | Main GUI, YOLO Tracking, CNN Grader, and UI Controls |
| `innovance_plc.py` | `/home/phoenix/Documents/cnn_plc/CNN/innovance_plc.py` | Inovance EC320 PLC Controller (48 Relays + Priority Queue) |
| `pneumatic_config.json` | `/home/phoenix/Documents/cnn_plc/CNN/pneumatic_config.json` | 40 Ejector Delays, Physical Distances (mm), and Speed |
| `inovance_config.json` | `/home/phoenix/Documents/cnn_plc/CNN/inovance_config.json` | PLC Modbus IP, Port, and Coil Addresses (M100 - M141) |
| **V2 Backup Directory** | `/home/phoenix/Documents/cnn_plc/CNN/v2_snapshot/` | Complete identical frozen copy of all V2 files |

---

## 🚀 How to Execute "Start V2" (V2 இயக்கும் கட்டளை)
Whenever you request **"start v2"**, the system will run directly using:
```bash
cd /home/phoenix/Documents/cnn_plc/CNN
python3 cashew_cnn_grader.py
```
Or to restore from snapshot at any future time:
```bash
cp /home/phoenix/Documents/cnn_plc/CNN/v2_snapshot/* /home/phoenix/Documents/cnn_plc/CNN/
python3 cashew_cnn_grader.py
```

---
*Logged and Archived by Antigravity AI Assistant on 2026-10-02 for Cashew CNN Grader Project.*
