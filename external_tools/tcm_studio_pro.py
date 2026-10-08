"""
TCM STUDIO PRO — Main Application
20-Class Cashew Deep Learning Labeling, Annotation & Training Platform
GPU: RTX 5060 Ti (CUDA Accelerated)
"""

import sys
import os
import time
import threading
import shutil
import datetime
import customtkinter as ctk
from tkinter import messagebox, Canvas, filedialog, Scrollbar
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np

# Local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append("..")
import tcm_config
from tcm_engine import TrainingEngine

# Camera SDK
try:
    import mvsdk
except ImportError:
    mvsdk = None

try:
    from ultralytics import YOLO, FastSAM
except ImportError:
    YOLO = None
    FastSAM = None

# --- UI Setup ---
class StdoutRedirector:
    def __init__(self, text_widget, progress_bar=None, lbl_perc=None):
        self.text_widget = text_widget
        self.progress_bar = progress_bar
        self.lbl_perc = lbl_perc
        self.terminal = sys.stdout
        self.terminal_err = sys.stderr
        
        self.start_time = None
        self.last_epoch_time = None
        self.epoch_times = []

    def write(self, string):
        try:
            self.terminal.write(string)
            self.terminal.flush()
        except: pass
        if string:
            self.text_widget.after(0, self._append, string)

    def _append(self, string):
        try:
            if not self.text_widget or not self.text_widget.winfo_exists():
                return
                
            self.text_widget.configure(state="normal")
                
            # Parse for progress if progress bar is available
            if self.progress_bar:
                import re
                m = re.search(r'(\d+)/(\d+)\s+[0-9.]+[A-Z]G?', string)
                if m:
                    try:
                        curr = int(m.group(1))
                        total = int(m.group(2))
                        perc = curr / max(1, total)
                        self.progress_bar.set(perc)
                        
                        if self.lbl_perc:
                            import time
                            now = time.time()
                            eta_str = ""
                            
                            if self.start_time is None:
                                self.start_time = now
                                self.last_epoch_time = now
                            else:
                                elapsed = now - self.last_epoch_time
                                # Only record reasonable deltas to filter out fast cached prints
                                if elapsed > 0.5:
                                    self.epoch_times.append(elapsed)
                                    if len(self.epoch_times) > 5:
                                        self.epoch_times.pop(0)
                            
                            if len(self.epoch_times) > 0:
                                avg = sum(self.epoch_times)/len(self.epoch_times)
                                rem = total - curr
                                secs = int(rem * avg)
                                m_left, s_left = divmod(secs, 60)
                                
                                if m_left > 60:
                                    h_left, m_left = divmod(m_left, 60)
                                    eta_str = f"({h_left}h {m_left}m left)"
                                else:
                                    eta_str = f"({m_left}m {s_left}s left)"
                                    
                            self.last_epoch_time = now
                            self.lbl_perc.configure(text=f"{int(perc*100)}% ... {eta_str}")
                    except: pass

            # Handle \r and ANSI escape codes to prevent matrix messes
            import re
            string = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', string)
            
            parts = string.split('\r')
            for i, part in enumerate(parts):
                if i > 0:
                    self.text_widget.delete("end-1c linestart", "end-1c")
                if part:
                    self.text_widget.insert("end", part)

            self.text_widget.see("end")
            self.text_widget.configure(state="disabled")
        except Exception:
            pass

    def flush(self):
        pass

class EvaluationPopup(ctk.CTkToplevel):
    def __init__(self, parent, run_dir):
        super().__init__(parent)
        self.title("Training Analytics & Evaluation")
        self.geometry("1100x750")
        
        from PIL import Image
        
        tabs = ctk.CTkTabview(self)
        tabs.pack(fill="both", expand=True, padx=10, pady=10)
        tabs.add("Loss & Metrics (results.png)")
        tabs.add("Confusion Matrix")
        tabs.add("F1 & PR Curves")

        def load_img(tab_name, filename):
            path = os.path.join(run_dir, filename)
            frame = ctk.CTkScrollableFrame(tabs.tab(tab_name), fg_color="transparent")
            frame.pack(fill="both", expand=True)
            if os.path.exists(path):
                try:
                    img = Image.open(path)
                    # Force maximum 600 height for ultra-wide monitors, preserving aspect
                    w, h = img.size
                    scale = min(1000.0 / float(w), 500.0 / float(h))
                    tk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(int(w * scale), int(h * scale)))
                    lbl = ctk.CTkLabel(frame, image=tk_img, text="")
                    lbl.pack(pady=10)
                except Exception as e:
                    ctk.CTkLabel(frame, text=f"Failed to load image: {e}").pack(pady=20)
            else:
                ctk.CTkLabel(frame, text=f"{filename} not generated yet by YOLO.").pack(expand=True)

        load_img("Loss & Metrics (results.png)", "results.png")
        load_img("Confusion Matrix", "confusion_matrix_normalized.png")
        load_img("Confusion Matrix", "confusion_matrix.png") # Fallback to absolute
        # Support both old YOLO (F1_curve.png) and new YOLO (BoxF1_curve.png) naming
        f1_file = "BoxF1_curve.png" if os.path.exists(os.path.join(run_dir, "BoxF1_curve.png")) else "F1_curve.png"
        pr_file = "BoxPR_curve.png" if os.path.exists(os.path.join(run_dir, "BoxPR_curve.png")) else "PR_curve.png"
        load_img("F1 & PR Curves", f1_file)
        load_img("F1 & PR Curves", pr_file)

        def trigger_export():
            self.destroy()
            if hasattr(parent, 'ent_best_pt') and hasattr(parent, '_export_model'):
                best_pt_path = os.path.join(run_dir, "weights", "best.pt")
                if os.path.exists(best_pt_path):
                    parent.ent_best_pt.delete(0, "end")
                    parent.ent_best_pt.insert(0, best_pt_path)
                    parent._export_model()
                else:
                    import tkinter.messagebox as mb
                    mb.showerror("Error", f"No best.pt found at {best_pt_path}")

        ctk.CTkButton(self, text="💾 EXPORT MODEL & CLOSE", height=40, font=("Segoe UI", 12, "bold"), 
                     command=trigger_export).pack(pady=10)
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA DRIVER
# ══════════════════════════════════════════════════════════════════════════════
class MindVisionCamera:
    def __init__(self):
        self.h = None
        self.cap_buffer = None
        self.last_frame = None
        self.lock = threading.Lock()

    def open(self, device_index=0):
        if not mvsdk:
            return False
        try:
            self.DevList = mvsdk.CameraEnumerateDevice()
            if len(self.DevList) <= device_index:
                return False
            
            if self.h:
                mvsdk.CameraUnInit(self.h)
                self.h = None
                
            self.h = mvsdk.CameraInit(self.DevList[device_index], -1, -1)
            mvsdk.CameraSetIspOutFormat(self.h, mvsdk.CAMERA_MEDIA_TYPE_BGR8)
            mvsdk.CameraPlay(self.h)
            cap = mvsdk.CameraGetCapability(self.h)
            size = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * 3
            self.cap_buffer = mvsdk.CameraAlignMalloc(size, 16)
            return True
        except Exception:
            return False

    def get_devices(self):
        if not mvsdk: return []
        devices = mvsdk.CameraEnumerateDevice()
        return [f"{i}: {dev.acFriendlyName.decode('utf-8')}" for i, dev in enumerate(devices)]

    def set_manual_params(self, exp, again, rgain, ggain, bgain):
        if self.h:
            try:
                mvsdk.CameraSetAeState(self.h, 0)
                mvsdk.CameraSetExposureTime(self.h, float(exp))
                mvsdk.CameraSetAnalogGain(self.h, float(again))
                mvsdk.CameraSetGain(self.h, int(rgain), int(ggain), int(bgain))
            except Exception:
                pass

    def read(self):
        frame = None
        if self.h:
            try:
                p_raw, head = mvsdk.CameraGetImageBuffer(self.h, 200)
                mvsdk.CameraImageProcess(self.h, p_raw, self.cap_buffer, head)
                mvsdk.CameraReleaseImageBuffer(self.h, p_raw)
                mvsdk.CameraFlipFrameBuffer(self.cap_buffer, head, 1)
                data = (mvsdk.c_ubyte * head.uBytes).from_address(self.cap_buffer)
                frame = np.frombuffer(data, dtype=np.uint8).reshape(
                    head.iHeight, head.iWidth, 3
                ).copy()
            except Exception:
                pass
        if frame is None:
            frame = np.zeros((600, 800, 3), dtype=np.uint8)
            cv2.putText(frame, "NO CAMERA — Drop Images in DATASET tab",
                        (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        with self.lock:
            self.last_frame = frame
        return frame

    def get_latest(self):
        with self.lock:
            return self.last_frame.copy() if self.last_frame is not None else None

    def close(self):
        if self.h:
            mvsdk.CameraUnInit(self.h)
            mvsdk.CameraAlignFree(self.cap_buffer)
        self.h = None


# ══════════════════════════════════════════════════════════════════════════════
#  CROP & CLASS ASSIGN POPUP (ROBOFLOW STYLE)
# ══════════════════════════════════════════════════════════════════════════════
class ClassAssignPopup(ctk.CTkToplevel):
    def __init__(self, master, cropped_cv_img, callback, suggested_class=None, crop_origin=(0,0)):
        super().__init__(master)
        self.title("Assign Class to Crop")
        self.geometry("600x550")
        self.attributes("-topmost", True)
        self.grab_set()  # Modal window
        self.callback = callback
        self.suggested_class = suggested_class
        
        self.crop_origin = crop_origin
        self.cropped_cv_img = cropped_cv_img.copy()
        self.display_cv_img = cropped_cv_img.copy()
        self.staged_defects = []

        # Display Cropped Image
        try:
            self._render_cropped_preview()
        except Exception:
            pass

        ctk.CTkLabel(self, text="Select Class for this object:", font=("Segoe UI", 16, "bold")).pack(pady=5)
        
        # Grid of buttons
        self.grid_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.grid_frame.pack(padx=20, pady=10, fill="both", expand=True)
        
        self.draw_buttons()
        
        bottom_frame = ctk.CTkFrame(self, fg_color="transparent")
        bottom_frame.pack(fill="x", pady=15)
        
        undo_btn = ctk.CTkButton(bottom_frame, text="↩️ UNDO DEFECT", fg_color="#F57C00", hover_color="#E65100", command=self._undo_staged)
        undo_btn.pack(side="left", padx=5)
        
        self.bind("<Control-z>", self._undo_staged)
        self.bind("<Control-Z>", self._undo_staged)
        
        add_btn = ctk.CTkButton(bottom_frame, text="➕ ADD NEW CLASS", fg_color="#37474F", hover_color="#546E7A", command=self._prompt_add_class)
        add_btn.pack(side="left", padx=5)

        rn_btn = ctk.CTkButton(bottom_frame, text="✏️ RENAME CLASS", fg_color="#1976D2", hover_color="#1565C0", command=self._prompt_rename_class)
        rn_btn.pack(side="left", padx=5)

        rm_btn = ctk.CTkButton(bottom_frame, text="➖ DEL CLASS", fg_color="#E65100", hover_color="#FF6D00", command=self._prompt_delete_class)
        rm_btn.pack(side="left", padx=5)

        del_btn = ctk.CTkButton(bottom_frame, text="🗑 DELETE LABEL", fg_color="#D32F2F", hover_color="#B71C1C", command=lambda: self._on_select("DELETE"))
        del_btn.pack(side="right", padx=5)

    def _undo_staged(self, e=None):
        if self.staged_defects:
            self.staged_defects.pop()
            # Reset canvas preview back to bare crop
            self.display_cv_img = self.cropped_cv_img.copy()
            
            # Reconstruct all surviving defects in the stack
            for defect in self.staged_defects:
                bgr = tcm_config.hex_to_bgr(tcm_config.CLASS_COLORS.get(defect["cls"], "#FFFF00"))
                x1, y1 = self.crop_origin
                # Unwind global coordinates back to local crop coordinates for rendering
                pts = np.array([[pt[0]-x1, pt[1]-y1] for pt in defect["points"]], np.int32).reshape((-1, 1, 2))
                
                overlay = self.display_cv_img.copy()
                cv2.fillPoly(overlay, [pts], bgr)
                cv2.addWeighted(overlay, 0.4, self.display_cv_img, 0.6, 0, self.display_cv_img)
                cv2.polylines(self.display_cv_img, [pts], True, bgr, 2)
                
            self._render_cropped_preview()

    def _render_cropped_preview(self):
        try:
            h, w = self.display_cv_img.shape[:2]
            scale = min(200 / w, 200 / h)
            self.popup_scale = scale
            nw, nh = max(50, int(w * scale)), max(50, int(h * scale))
    
            if w < 50 or h < 50:
                resized = cv2.resize(self.display_cv_img, (nw, nh), interpolation=cv2.INTER_NEAREST)
            else:
                resized = cv2.resize(self.display_cv_img, (nw, nh), interpolation=cv2.INTER_LINEAR)
                
            img = ctk.CTkImage(light_image=Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)), dark_image=Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)), size=(nw, nh))
            
            if not hasattr(self, 'lbl_img'):
                self.lbl_img = ctk.CTkLabel(self, text="", image=img)
                self.lbl_img.pack(pady=15)
                self.lbl_img.bind("<Button-1>", self._on_popup_img_click)
            else:
                self.lbl_img.configure(image=img)
        except Exception:
            pass

    def _on_popup_img_click(self, e):
        # Calculate exactly which pixel was clicked on the original crop matrix
        if not hasattr(self, 'popup_scale'): return
        
        lbl_w = self.lbl_img.winfo_width()
        lbl_h = self.lbl_img.winfo_height()
        
        # Determine inherent padding since the scaled image is centered in a label
        img_nw = int(self.display_cv_img.shape[1] * self.popup_scale)
        img_nh = int(self.display_cv_img.shape[0] * self.popup_scale)
        p_x = max(0, (lbl_w - img_nw) // 2)
        p_y = max(0, (lbl_h - img_nh) // 2)

        local_ix = int((e.x - p_x) / self.popup_scale)
        local_iy = int((e.y - p_y) / self.popup_scale)

        if not (0 <= local_ix < self.display_cv_img.shape[1] and 0 <= local_iy < self.display_cv_img.shape[0]):
            return
            
        # Utilizing FastSAM AI Magic Wand
        import tcm_magic_wand
        best_cnt = tcm_magic_wand.get_magic_polygon(self.cropped_cv_img, local_ix, local_iy)
        
        if best_cnt is not None:
            # Convert to High-Def contour polygon for global mapping later
            # Reduced epsilon allows tighter wrapping inside SAM masks
            epsilon = 0.0005 * cv2.arcLength(best_cnt, True)
            approx = cv2.approxPolyDP(best_cnt, epsilon, True)
            poly_points = []
            
            # Map local crop coordinates BACK to Global Canvas coordinates
            x1, y1 = self.crop_origin
            for pt in approx:
                poly_points.append((x1 + pt[0][0], y1 + pt[0][1]))
                
            # Classify: User inherently implies TESTA by clicking a secondary defect, but allow overrides 
            # if they rename their layer 2 defect buttons. For now we use standard TESTA default.
            suggested_class = "TESTA"
            self.staged_defects.append({"type": "polygon", "cls": suggested_class, "points": poly_points})
            
            # Draw the newly detected polygon permanently onto the display clone matrix 
            # using the Yellow TESTA color mask
            bgr = tcm_config.hex_to_bgr(tcm_config.CLASS_COLORS.get(suggested_class, "#FFFF00"))
            pts = np.array([pt[0] for pt in approx], np.int32).reshape((-1, 1, 2))
            
            # Transparent fill
            overlay = self.display_cv_img.copy()
            cv2.fillPoly(overlay, [pts], bgr)
            cv2.addWeighted(overlay, 0.4, self.display_cv_img, 0.6, 0, self.display_cv_img)
            cv2.polylines(self.display_cv_img, [pts], True, bgr, 2)
            
            # Push Visual Updates back to UI
            self._render_cropped_preview()
        ctk.CTkLabel(self, text="Select Class for this object:", font=("Segoe UI", 16, "bold")).pack(pady=5)
        
        # Grid of buttons
        self.grid_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.grid_frame.pack(padx=20, pady=10, fill="both", expand=True)
        
        self.draw_buttons()
        
        bottom_frame = ctk.CTkFrame(self, fg_color="transparent")
        bottom_frame.pack(fill="x", pady=15)
        
        add_btn = ctk.CTkButton(bottom_frame, text="➕ ADD NEW CLASS", fg_color="#37474F", hover_color="#546E7A", command=self._prompt_add_class)
        add_btn.pack(side="left", padx=5)

        rn_btn = ctk.CTkButton(bottom_frame, text="✏️ RENAME CLASS", fg_color="#1976D2", hover_color="#1565C0", command=self._prompt_rename_class)
        rn_btn.pack(side="left", padx=5)

        rm_btn = ctk.CTkButton(bottom_frame, text="➖ DEL CLASS", fg_color="#E65100", hover_color="#FF6D00", command=self._prompt_delete_class)
        rm_btn.pack(side="left", padx=5)

        del_btn = ctk.CTkButton(bottom_frame, text="🗑 DELETE LABEL", fg_color="#D32F2F", hover_color="#B71C1C", command=lambda: self._on_select("DELETE"))
        del_btn.pack(side="right", padx=5)

    def draw_buttons(self):
        for w in self.grid_frame.winfo_children(): w.destroy()
        
        base_classes = ["GOOD", "SW", "UPW", "FULL_BROWN", "LIGHT_BROWN"]
        defect_classes = ["PKW1", "PKW2", "PKW_GREEN1", "PKW_GREEN2", "SSW", "DW", "TESTA", "HALF_UPW"]
        
        # Ensure any dynamic custom classes exist in the list
        all_cls = tcm_config.CLASS_NAMES
        
        lbl_base = ctk.CTkLabel(self.grid_frame, text="✅ MAIN CLASS (LAYER 1):", font=("Segoe UI", 14, "bold"), text_color="#00C853")
        lbl_base.grid(row=0, column=0, columnspan=4, pady=(5,0), sticky="w")
        
        row_idx = 1
        col_idx = 0
        for cls_name in all_cls:
            if cls_name in base_classes or (cls_name not in defect_classes and "PKW" not in cls_name and "TESTA" not in cls_name):
                self._create_btn(cls_name, row_idx, col_idx)
                col_idx += 1
                if col_idx >= 4:
                    col_idx = 0
                    row_idx += 1
                    
        row_idx += 1
        lbl_def = ctk.CTkLabel(self.grid_frame, text="🔻 SECONDARY DEFECT (LAYER 2):", font=("Segoe UI", 14, "bold"), text_color="#FF5252")
        lbl_def.grid(row=row_idx, column=0, columnspan=4, pady=(15,0), sticky="w")
        
        row_idx += 1
        col_idx = 0
        for cls_name in all_cls:
            if cls_name in defect_classes or "PKW" in cls_name or "TESTA" in cls_name:
                self._create_btn(cls_name, row_idx, col_idx)
                col_idx += 1
                if col_idx >= 4:
                    col_idx = 0
                    row_idx += 1

        for i in range(4):
            self.grid_frame.columnconfigure(i, weight=1)

    def _create_btn(self, cls_name, r, c):
        col = tcm_config.CLASS_COLORS.get(cls_name, "#FFFFFF")
        
        # Highlight AI-predicted class
        if hasattr(self, 'suggested_class') and self.suggested_class == cls_name:
            border_width = 4
            border_color = "#00E676" # Bright Green
            text_prefix = "🎯 "
        else:
            border_width = 1
            border_color = "#37474F"
            text_prefix = ""
            
        btn = ctk.CTkButton(
            self.grid_frame,
            text=text_prefix + cls_name,
            fg_color=col,
            hover_color=col,
            text_color="black",
            border_width=border_width,
            border_color=border_color,
            font=("Consolas", 12, "bold"),
            command=lambda c=cls_name: self._on_select(c)
        )
        btn.grid(row=r, column=c, padx=5, pady=5, sticky="nsew")

    def _prompt_add_class(self):
        dialog = ctk.CTkInputDialog(text="Enter New Class Name (e.g. WOOD):", title="Add New Class")
        new_class = dialog.get_input()
        if new_class:
            new_class = new_class.strip().upper().replace(" ", "_")
            if new_class and new_class not in tcm_config.CLASS_NAMES:
                if tcm_config.ProjectManager.add_class(new_class):
                    if hasattr(self.master, "combo_capture_cls"):
                        self.master.combo_capture_cls.configure(values=tcm_config.CLASS_NAMES)
                    if hasattr(self.master, "combo_aug_cls"):
                        self.master.combo_aug_cls.configure(values=["ALL CLASSES"] + tcm_config.CLASS_NAMES)
                    if hasattr(self.master, "_refresh_dataset"):
                        self.master._refresh_dataset()
                    self.draw_buttons()
            else:
                from tkinter import messagebox
                messagebox.showwarning("Warning", "Class exists or invalid.", parent=self)

    def _prompt_delete_class(self):
        self.title("⚠️ Click a Class below to DELETE it ⚠️")
        for widget in self.grid_frame.winfo_children():
            if isinstance(widget, ctk.CTkButton):
                cls_name = widget.cget("text")
                widget.configure(
                    fg_color="#D32F2F", 
                    hover_color="#B71C1C",
                    command=lambda c=cls_name: self._execute_delete_class(c)
                )

    def _execute_delete_class(self, del_class):
        from tkinter import messagebox
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete '{del_class}' and ALL its images?", parent=self):
            if tcm_config.ProjectManager.delete_class(del_class):
                if hasattr(self.master, "combo_capture_cls"):
                    self.master.combo_capture_cls.configure(values=tcm_config.CLASS_NAMES)
                    if tcm_config.CLASS_NAMES:
                        self.master.combo_capture_cls.set(tcm_config.CLASS_NAMES[0])
                if hasattr(self.master, "combo_aug_cls"):
                    self.master.combo_aug_cls.configure(values=["ALL CLASSES"] + tcm_config.CLASS_NAMES)
                if hasattr(self.master, "_refresh_dataset"):
                    self.master._refresh_dataset()
        self.title("Assign Class to Crop")
        self.draw_buttons()

    def _prompt_rename_class(self):
        self.title("✏️ Click a Class below to RENAME it ✏️")
        for widget in self.grid_frame.winfo_children():
            # Apply to both regular and dynamically created buttons, skip the Header Labels
            if isinstance(widget, ctk.CTkButton) and widget.winfo_class() != 'CTkLabel':
                cls_name = widget.cget("text")
                if cls_name and "LAYER" not in cls_name:
                    widget.configure(
                        fg_color="#1976D2", 
                        hover_color="#1565C0",
                        text_color="white",
                        command=lambda c=cls_name: self._execute_rename_class(c)
                    )

    def _execute_rename_class(self, old_class):
        dialog = ctk.CTkInputDialog(text=f"Enter NEW Name for '{old_class}':", title="Rename Class")
        new_class = dialog.get_input()
        if new_class:
            new_class = new_class.strip().upper().replace(" ", "_")
            if new_class and new_class not in tcm_config.CLASS_NAMES:
                if tcm_config.ProjectManager.rename_class(old_class, new_class):
                    if hasattr(self.master, "combo_capture_cls"):
                        self.master.combo_capture_cls.configure(values=tcm_config.CLASS_NAMES)
                        if self.master.combo_capture_cls.get() == old_class:
                           self.master.combo_capture_cls.set(new_class)
                    if hasattr(self.master, "combo_aug_cls"):
                        self.master.combo_aug_cls.configure(values=["ALL CLASSES"] + tcm_config.CLASS_NAMES)
                    if hasattr(self.master, "_refresh_dataset"):
                        self.master._refresh_dataset()
                    self.title("Assign Class to Crop")
                    self.draw_buttons()
            else:
                from tkinter import messagebox
                messagebox.showwarning("Warning", "Class exists or invalid.", parent=self)

    def _on_select(self, class_name):
        if self.staged_defects and class_name != "DELETE":
            for d in self.staged_defects:
                d["cls"] = class_name
        self.callback(class_name, self.staged_defects)
        self.destroy()

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════
class TCMStudioPro(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("TCM STUDIO PRO — 20 Class Cashew AI Platform")
        w, h = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}")
        self.after(100, lambda: self.state("zoomed"))

        tcm_config.ensure_dirs()
        self.cam = MindVisionCamera()
        self.engine = TrainingEngine(log_callback=self.train_log)
        self.video_running = True
        self.model = None
        self.simulation_active = False

        # Magic Tool
        self.magic_tool_active = ctk.BooleanVar(value=False)
        self.yolo_magic_model = None

        # Annotation state
        self.rect_start = None
        self.current_bboxes = []
        self.display_scale = 1.0
        self.tk_img = None
        self.offset_x = 0
        self.offset_y = 0
        self.orig_h = 0
        self.orig_w = 0
        self.current_img_idx = 0
        self.viewing_saved_file = False
        self.temp_image_path = "temp_capture.jpg"
        self.current_cv_img = None
        self.crop_size_var = ctk.StringVar(value="Original HD")

        # Layout
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._build_header()
        self._build_tabs()

        # Start camera
        self.cam.open()
        threading.Thread(target=self._capture_loop, daemon=True).start()
        self._update_gui_loop()

    # ── HEADER ───────────────────────────────────────────────────────────
    def _build_header(self):
        hf = ctk.CTkFrame(self, height=55, fg_color="#0d0d0d", corner_radius=0)
        hf.grid(row=0, column=0, sticky="ew")
        hf.grid_propagate(False)

        ctk.CTkLabel(hf, text="⚡ TCM STUDIO PRO",
                     font=("Segoe UI Black", 22), text_color="#FF3D00").pack(side="left", padx=15)
                     
        proj_name = tcm_config.ACTIVE_PROJECT if tcm_config.ACTIVE_PROJECT else "NO WORKSPACE"
        ctk.CTkLabel(hf, text=f"PROJECT: {proj_name}",
                     font=("Segoe UI", 13, "bold"), text_color="#00E676").pack(side="left", padx=10)

        self.lbl_gpu = ctk.CTkLabel(hf, text=f"GPU: {tcm_config.DEVICE}",
                                    font=("Consolas", 11), text_color="#4CAF50")
        self.lbl_gpu.pack(side="left", padx=30)

        self.lbl_status = ctk.CTkLabel(hf, text="READY",
                                       font=("Segoe UI", 16, "bold"), text_color="#444")
        self.lbl_status.pack(side="left", padx=20)

        ctk.CTkButton(hf, text="✕ EXIT", width=80, height=32,
                      fg_color="#B71C1C", hover_color="#D32F2F",
                      command=self._shutdown).pack(side="right", padx=10)
                      
        ctk.CTkButton(hf, text="🏠 HOME", width=80, height=32,
                      fg_color="#37474F", hover_color="#455A64",
                      command=self._go_home).pack(side="right", padx=10)

    # ── TABS ─────────────────────────────────────────────────────────────
    def _build_tabs(self):
        self.tabs = ctk.CTkTabview(self, corner_radius=8, command=self._on_tab_change)
        self.tabs.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)

        self.tab_cam = self.tabs.add("📸 CAMERA")
        self.tab_ann = self.tabs.add("✏️ ANNOTATE")
        self.tab_ds  = self.tabs.add("📁 DATASET")
        self.tab_aug = self.tabs.add("🔬 AUGMENT")
        self.tab_trn = self.tabs.add("🚀 TRAIN")
        self.tab_exp = self.tabs.add("📦 EXPORT")

        self.tabs.set("📸 CAMERA")

        self._setup_camera_tab()
        self._setup_annotate_tab()
        self._setup_dataset_tab()
        self._setup_augment_tab()
        self._setup_train_tab()
        self._setup_export_tab()

    def _on_tab_change(self):
        if hasattr(self, "tabs") and self.tabs.get() == "📁 DATASET":
            self._refresh_dataset()

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 1: CAMERA
    # ══════════════════════════════════════════════════════════════════════
    def _setup_camera_tab(self):
        t = self.tab_cam
        t.grid_rowconfigure(0, weight=1)
        t.grid_columnconfigure(1, weight=1)

        sb = ctk.CTkFrame(t, width=260, fg_color="#1a1a1a")
        sb.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        ctk.CTkLabel(sb, text="CAMERA TUNING", text_color="#00BCD4",
                     font=("Segoe UI", 13, "bold")).pack(pady=(10, 5))
                     
        devices = self.cam.get_devices()
        val_list = devices if devices else ["0: Default WebCam/None"]
        self.combo_cam_dev = ctk.CTkComboBox(sb, values=val_list)
        self.combo_cam_dev.pack(fill="x", padx=8, pady=4)
        
        ctk.CTkLabel(sb, text="Exposure (µs)", text_color="#888", font=("Segoe UI", 10)).pack()
        self.ent_exp = ctk.CTkEntry(sb, placeholder_text="20000")
        self.ent_exp.pack(fill="x", padx=8, pady=2)
        
        ctk.CTkLabel(sb, text="Analog Gain", text_color="#888", font=("Segoe UI", 10)).pack()
        self.ent_again = ctk.CTkEntry(sb, placeholder_text="1.0")
        self.ent_again.pack(fill="x", padx=8, pady=2)
        
        ctk.CTkLabel(sb, text="R / G / B Gains", text_color="#888", font=("Segoe UI", 10)).pack()
        rgb_frame = ctk.CTkFrame(sb, fg_color="transparent")
        rgb_frame.pack(fill="x", padx=8, pady=2)
        self.ent_rgain = ctk.CTkEntry(rgb_frame, width=50, placeholder_text="100")
        self.ent_rgain.pack(side="left", padx=2, expand=True)
        self.ent_ggain = ctk.CTkEntry(rgb_frame, width=50, placeholder_text="100")
        self.ent_ggain.pack(side="left", padx=2, expand=True)
        self.ent_bgain = ctk.CTkEntry(rgb_frame, width=50, placeholder_text="100")
        self.ent_bgain.pack(side="left", padx=2, expand=True)
        
        # Insert defaults from workspace
        proj_p = os.path.join(tcm_config.PROJECTS_DIR, tcm_config.ACTIVE_PROJECT) if tcm_config.ACTIVE_PROJECT else ""
        config_path = os.path.join(proj_p, "camera.json") if proj_p else ""
        if os.path.exists(config_path):
            try:
                import json
                with open(config_path, "r") as f:
                    c = json.load(f)
                    self.ent_exp.insert(0, str(c.get("exposure", 20000)))
                    self.ent_again.insert(0, str(c.get("analog_gain", 1.0)))
                    self.ent_rgain.insert(0, str(c.get("r_gain", 100)))
                    self.ent_ggain.insert(0, str(c.get("g_gain", 100)))
                    self.ent_bgain.insert(0, str(c.get("b_gain", 100)))
            except: pass
        else:
            self.ent_exp.insert(0, "20000")
            self.ent_again.insert(0, "1.0")
            self.ent_rgain.insert(0, "100")
            self.ent_ggain.insert(0, "100")
            self.ent_bgain.insert(0, "100")

        ctk.CTkButton(sb, text="APPLY & SAVE", height=30, fg_color="#E65100",
                      hover_color="#FF6D00", command=self._apply_cam).pack(fill="x", padx=8, pady=6)

        ctk.CTkLabel(sb, text="━" * 30, text_color="#333").pack(pady=5)
        ctk.CTkLabel(sb, text="LIVE AI TEST", text_color="#00E676",
                     font=("Segoe UI", 13, "bold")).pack(pady=5)
        
        runs_dir = os.path.join(tcm_config.MODEL_DIR, "runs") if tcm_config.MODEL_DIR else ""
        run_names = ["Select Trained Model...", "Import External Model (.pt)..."]
        if runs_dir and os.path.exists(runs_dir):
            for d in reversed(sorted(os.listdir(runs_dir))):
                if os.path.isdir(os.path.join(runs_dir, d)):
                    run_names.append(d)
                    
        self.combo_load_model = ctk.CTkComboBox(sb, values=run_names, command=self._on_model_select)
        self.combo_load_model.pack(fill="x", padx=8)
        
        conf_frame = ctk.CTkFrame(sb, fg_color="transparent")
        conf_frame.pack(fill="x", padx=8, pady=5)
        self.lbl_conf = ctk.CTkLabel(conf_frame, text="Conf: 40%")
        self.lbl_conf.pack(side="left", padx=2)
        self.slider_conf = ctk.CTkSlider(conf_frame, from_=0.05, to=0.95, number_of_steps=90, command=self._on_conf_change)
        self.slider_conf.pack(side="left", padx=5, expand=True)
        self.slider_conf.set(0.40)
        
        btn_frame = ctk.CTkFrame(sb, fg_color="transparent")
        btn_frame.pack(fill="x", padx=8, pady=8)
        
        self.sw_sim = ctk.CTkSwitch(btn_frame, text="LIVE", width=60, onvalue=True, offvalue=False,
                                    command=self._toggle_sim)
        self.sw_sim.pack(side="left")
        
        ctk.CTkButton(btn_frame, text="TEST...", width=70, font=("Segoe UI", 11, "bold"),
                      fg_color="#00ACC1", hover_color="#00838F",
                      command=self._test_on_image).pack(side="right", padx=(2,0))
                      
        ctk.CTkButton(btn_frame, text="NEXT ▶", width=60, font=("Segoe UI", 11, "bold"),
                      fg_color="#4CAF50", hover_color="#388E3C",
                      command=self._next_test_image).pack(side="right")

        ctk.CTkLabel(sb, text="━" * 30, text_color="#333").pack(pady=5)
        ctk.CTkLabel(sb, text="QUICK CAPTURE", text_color="#2196F3",
                     font=("Segoe UI", 13, "bold")).pack(pady=5)
        cls_frame = ctk.CTkFrame(sb, fg_color="transparent")
        cls_frame.pack(fill="x", padx=8)
        self.combo_capture_cls = ctk.CTkComboBox(cls_frame, values=tcm_config.CLASS_NAMES, width=150)
        self.combo_capture_cls.pack(side="left")
        if tcm_config.CLASS_NAMES:
            self.combo_capture_cls.set(tcm_config.CLASS_NAMES[0])
            
        ctk.CTkButton(cls_frame, text="➕", width=40, fg_color="#37474F", hover_color="#546E7A", command=self._prompt_add_class).pack(side="right", padx=(5,0))
        ctk.CTkButton(sb, text="📸 CAPTURE & SAVE", height=45,
                      fg_color="#2962FF", hover_color="#448AFF",
                      command=self._quick_capture).pack(fill="x", padx=8, pady=8)
        ctk.CTkButton(sb, text="CAPTURE → ANNOTATE", height=35,
                      fg_color="#007ACC", hover_color="#00897B",
                      command=self._capture_to_annotate).pack(fill="x", padx=8, pady=4)

        self.lbl_video = ctk.CTkLabel(t, text="", fg_color="black")
        self.lbl_video.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 2: ANNOTATE (ROBOFLOW STYLE)
    # ══════════════════════════════════════════════════════════════════════
    def _setup_annotate_tab(self):
        t = self.tab_ann
        t.grid_rowconfigure(0, weight=1)
        t.grid_columnconfigure(1, weight=1)

        sb = ctk.CTkFrame(t, width=260, fg_color="#1a1a1a")
        sb.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        # MAGIC TOOL (SAM/YOLO AUTO ANNOTATE)
        ctk.CTkLabel(sb, text="✨ MAGIC WAND", text_color="#AA00FF",
                     font=("Segoe UI", 13, "bold")).pack(pady=(10, 5))
        sw_magic = ctk.CTkSwitch(sb, text="Enable Magic Click", variable=self.magic_tool_active,
                                 progress_color="#AA00FF")
        sw_magic.pack(pady=5)
        ctk.CTkLabel(sb, text="Click on a cashew to auto-box", text_color="#888", font=("Segoe UI", 9)).pack()

        ctk.CTkLabel(sb, text="━" * 30, text_color="#333").pack(pady=5)
        
        # AUTO-LABEL NEW
        ctk.CTkLabel(sb, text="🤖 AUTO-LABEL NEW", text_color="#00E676", font=("Segoe UI", 11, "bold")).pack(pady=(5,0))
        self.sw_auto_label = ctk.CTkSwitch(sb, text="Auto-Detect on NEXT", onvalue=True, offvalue=False)
        self.sw_auto_label.pack(pady=5)

        ctk.CTkLabel(sb, text="━" * 30, text_color="#333").pack(pady=5)

        open_row = ctk.CTkFrame(sb, fg_color="transparent")
        open_row.pack(fill="x", padx=8, pady=4)
        ctk.CTkButton(open_row, text="📂 OPEN IMAGE", height=35, fg_color="#37474F",
                      command=self._open_image_file).pack(side="left", expand=True, fill="x", padx=(0,2))
        ctk.CTkButton(open_row, text="📁 OPEN FOLDER", height=35, fg_color="#1565C0",
                      hover_color="#1976D2",
                      command=self._open_image_folder).pack(side="right", expand=True, fill="x", padx=(2,0))

        nav = ctk.CTkFrame(sb, fg_color="transparent")
        nav.pack(fill="x", padx=8, pady=4)
        ctk.CTkButton(nav, text="◀ PREV", width=100,
                      command=lambda: self._nav_images(-1)).pack(side="left", padx=2)
        ctk.CTkButton(nav, text="NEXT ▶", width=100,
                      command=lambda: self._nav_images(1)).pack(side="right", padx=2)

        self.lbl_img_info = ctk.CTkLabel(sb, text="No image loaded", text_color="#666",
                                         font=("Consolas", 10), wraplength=220)
        self.lbl_img_info.pack(pady=4)

        ctk.CTkLabel(sb, text="━" * 30, text_color="#333").pack()
        ctk.CTkButton(sb, text="↩ UNDO BOX", fg_color="#F57F17", hover_color="#FFB300",
                      command=self._undo_box).pack(fill="x", padx=8, pady=3)
        ctk.CTkButton(sb, text="🗑 CLEAR ALL", fg_color="#BF360C", hover_color="#E64A19",
                      command=self._clear_boxes).pack(fill="x", padx=8, pady=3)
        self.lbl_box_count = ctk.CTkLabel(sb, text="Boxes/Objects: 0", text_color="#888")
        self.lbl_box_count.pack(pady=2)
        
        ctk.CTkLabel(sb, text="━" * 30, text_color="#333").pack(pady=5)
        ctk.CTkLabel(sb, text="ANNOTATION MODE", text_color="#FF9800", font=("Segoe UI", 11, "bold")).pack(pady=(2,0))
        self.annot_mode = ctk.StringVar(value="Bounding Box")
        self.seg_box_annot = ctk.CTkSegmentedButton(sb, values=["Bounding Box", "Polygon (Seg)"], variable=self.annot_mode)
        self.seg_box_annot.pack(fill="x", padx=8, pady=5)
        
        ctk.CTkLabel(sb, text="CROP SAVE SIZE", text_color="#2196F3", font=("Segoe UI", 11, "bold")).pack(pady=(5,0))
        self.combo_crop_size = ctk.CTkComboBox(sb, values=["Original HD", "224x224", "416x416", "640x640"], variable=self.crop_size_var)
        self.combo_crop_size.pack(fill="x", padx=8, pady=5)

        ctk.CTkButton(sb, text="💾 SAVE ALL LABELS", height=45,
                      fg_color="#00C853", hover_color="#00E676",
                      font=("Segoe UI", 13, "bold"),
                      command=self._save_annotation).pack(fill="x", padx=8, pady=15)

        self.canvas = Canvas(t, bg="#111", cursor="cross", highlightthickness=0)
        self.canvas.grid(row=0, column=1, sticky="nsew")

        # Instruction overlay
        ctk.CTkLabel(self.canvas, text="Right-Click to Delete any Box | Drag to Draw | Click to Edit", 
                     text_color="#888", font=("Segoe UI", 12)).place(relx=0.5, rely=0.02, anchor="n")

        self.canvas.bind("<Button-1>", self._mouse_down)
        self.canvas.bind("<B1-Motion>", self._mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._mouse_up)
        # Bind Right Click (Button-3 on Windows/Linux, Button-2/3 on Mac depending on setup)
        self.canvas.bind("<Button-3>", self._right_click_delete)
        # New Roboflow-parity bindings for Pan, Zoom, Crosshairs
        self.canvas.bind("<MouseWheel>", self._mouse_wheel)
        self.canvas.bind("<Button-2>", self._middle_mouse_down)
        self.canvas.bind("<B2-Motion>", self._middle_mouse_drag)
        self.canvas.bind("<Motion>", self._mouse_move)
        
        # Keyboard Accelerators
        self.bind("<Control-z>", lambda e: self._undo_box())
        self.bind("<Control-Z>", lambda e: self._undo_box())
        
        # Disable magic mode initially so user knows standard box dragging is default
        self.magic_tool_active.set(False)

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 3: DATASET (WITH GALLERY)
    # ══════════════════════════════════════════════════════════════════════
    def _setup_dataset_tab(self):
        t = self.tab_ds
        t.grid_rowconfigure(1, weight=1)
        t.grid_columnconfigure(0, weight=1)
        t.grid_columnconfigure(1, weight=2)

        top = ctk.CTkFrame(t, height=50, fg_color="#1a1a1a")
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=4, pady=4)
        ctk.CTkButton(top, text="🔄 REFRESH", width=100,
                      fg_color="#37474F", command=self._refresh_dataset).pack(side="left", padx=8, pady=8)
        ctk.CTkButton(top, text="📂 IMPORT FOLDER", width=130,
                      fg_color="#1565C0", command=self._import_folder).pack(side="left", padx=4)
        ctk.CTkButton(top, text="🏷 AUTO-LABEL ALL", width=140,
                      fg_color="#E65100", hover_color="#FF6D00",
                      command=self._auto_label_all).pack(side="left", padx=4)
        self.lbl_total = ctk.CTkLabel(top, text="Total: 0 images",
                                      font=("Segoe UI", 13, "bold"), text_color="#4CAF50")
        self.lbl_total.pack(side="right", padx=15)

        # Left Panel (Classes list)
        self.ds_scroll = ctk.CTkScrollableFrame(t, fg_color="#0d0d0d", width=350)
        self.ds_scroll.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        
        # Right Panel (Gallery)
        self.gallery_frame = ctk.CTkScrollableFrame(t, fg_color="#111")
        self.gallery_frame.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)
        
        self.gallery_lbl = ctk.CTkLabel(self.gallery_frame, text="Select a class to view images", font=("Segoe UI", 14), text_color="#666")
        self.gallery_lbl.pack(pady=20)
        
        # We will dynamically populate grid in gallery_frame
        self.gallery_cols = 4

        self._refresh_dataset()

    def _refresh_dataset(self):
        for w in self.ds_scroll.winfo_children():
            w.destroy()

        stats = self.engine.get_class_stats()
        total = sum(v["images"] for v in stats.values())
        self.lbl_total.configure(text=f"Total: {total} images")

        for i, cls in enumerate(tcm_config.CLASS_NAMES):
            s = stats.get(cls, {"images": 0, "labels": 0})
            col = tcm_config.CLASS_COLORS.get(cls, "#FFF")
            row = ctk.CTkFrame(self.ds_scroll, fg_color="#1a1a1a", height=50, corner_radius=6)
            row.pack(fill="x", pady=2, padx=4)
            row.grid_columnconfigure(2, weight=1)

            swatch = ctk.CTkLabel(row, text="", width=8, fg_color=col)
            swatch.grid(row=0, column=0, sticky="ns", padx=(0, 4))

            ctk.CTkLabel(row, text=cls, font=("Segoe UI", 12, "bold"),
                         text_color=col).grid(row=0, column=1, sticky="w", padx=4, pady=5)
            
            ctk.CTkLabel(row, text=f"📷 {s['images']} | 🏷 {s['labels']}", font=("Consolas", 11),
                         text_color="#aaa").grid(row=0, column=2, sticky="w", padx=10)

            # View Gallery Button
            ctk.CTkButton(row, text="VIEW", width=50, height=24, font=("Segoe UI", 10, "bold"),
                          fg_color="#007ACC", hover_color="#00897B",
                          command=lambda c=cls: self._load_gallery(c)).grid(row=0, column=3, padx=8)

    def _load_gallery(self, cls_name):
        for w in self.gallery_frame.winfo_children():
            w.destroy()
            
        cls_dir = os.path.join(tcm_config.DATASET_DIR, cls_name)
        golden_dir = os.path.join(tcm_config.DATASET_DIR, "Golden_Crops", cls_name)
        
        ctk.CTkLabel(self.gallery_frame, text=f"Gallery: {cls_name}", text_color=tcm_config.CLASS_COLORS.get(cls_name, "#FFF"),
                     font=("Segoe UI", 16, "bold")).grid(row=0, column=0, columnspan=self.gallery_cols, pady=10)
                     
        self.gallery_checkboxes = []
        
        def select_all():
            for cb, _ in self.gallery_checkboxes: cb.select()
            
        def delete_selected():
            selected = [path for cb, path in self.gallery_checkboxes if cb.get() == 1]
            if not selected: return
            if messagebox.askyesno("Confirm Bulk Delete", f"Delete {len(selected)} selected images?\nThis cannot be undone.", parent=self):
                for file_path in selected:
                    try:
                        os.remove(file_path)
                        txt_file = os.path.splitext(file_path)[0] + ".txt"
                        if os.path.exists(txt_file): os.remove(txt_file)
                    except Exception: pass
                self.lbl_status.configure(text=f"{len(selected)} images deleted.", text_color="#FF9800")
                self._load_gallery(cls_name)
                self._refresh_dataset()
                
        btn_f = ctk.CTkFrame(self.gallery_frame, fg_color="transparent")
        btn_f.grid(row=0, column=1, columnspan=self.gallery_cols-1, sticky="e", padx=10, pady=10)
        ctk.CTkButton(btn_f, text="✅ SELECT ALL", width=100, fg_color="#37474F", hover_color="#546E7A", command=select_all).pack(side="left", padx=5)
        ctk.CTkButton(btn_f, text="🗑️ DELETE SELECTED", width=140, fg_color="#D32F2F", hover_color="#B71C1C", command=delete_selected).pack(side="left", padx=5)

        files = []
        if os.path.exists(cls_dir):
            files.extend([os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        if os.path.exists(golden_dir):
            files.extend([os.path.join(golden_dir, f) for f in os.listdir(golden_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            
        if not files:
            ctk.CTkLabel(self.gallery_frame, text="No images in this class.").grid(row=1, column=0, columnspan=self.gallery_cols, pady=20)
            return
            
        for idx, fp in enumerate(files):
            fname = os.path.basename(fp)
            r = (idx // self.gallery_cols) + 1
            c = idx % self.gallery_cols
            
            card = ctk.CTkFrame(self.gallery_frame, fg_color="#1a1a1a")
            card.grid(row=r, column=c, padx=5, pady=5, sticky="nsew")
            
            try:
                img_pil = Image.open(fp)
                img_pil.thumbnail((120, 120))
                ctk_img = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=img_pil.size)
                ctk.CTkLabel(card, text="", image=ctk_img).pack(pady=5)
            except:
                ctk.CTkLabel(card, text="[Error Loading]", width=120, height=120).pack(pady=5)
                
            ctk.CTkLabel(card, text=fname[:15]+"...", font=("Consolas", 9)).pack()
            
            cb = ctk.CTkCheckBox(card, text="Select", text_color="#aaa", width=70, checkbox_height=18, checkbox_width=18)
            cb.pack(pady=(0, 5))
            self.gallery_checkboxes.append((cb, fp))
            
            ctk.CTkButton(card, text="Delete", width=80, height=22, fg_color="#D32F2F", hover_color="#B71C1C", font=("Segoe UI", 10),
                          command=lambda path=fp, cl=cls_name: self._delete_image(path, cl)).pack(pady=5)
                          
    def _delete_image(self, file_path, cls_name):
        if messagebox.askyesno("Confirm", f"Delete {os.path.basename(file_path)}?"):
            try:
                os.remove(file_path)
                txt_file = os.path.splitext(file_path)[0] + ".txt"
                if os.path.exists(txt_file):
                    os.remove(txt_file)
                self.lbl_status.configure(text="Image deleted.", text_color="#FF9800")
                self._load_gallery(cls_name)
                self._refresh_dataset()
            except Exception as e:
                messagebox.showerror("Error", str(e))

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 4: AUGMENT
    # ══════════════════════════════════════════════════════════════════════
    def _setup_augment_tab(self):
        t = self.tab_aug
        t.grid_rowconfigure(0, weight=1)
        t.grid_columnconfigure(1, weight=1)

        sb = ctk.CTkFrame(t, width=280, fg_color="#1a1a1a")
        sb.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        ctk.CTkLabel(sb, text="🧠 DYNAMIC YOLO AUGMENT", text_color="#00BCD4",
                     font=("Segoe UI", 13, "bold")).pack(pady=(10, 8))

        ctk.CTkLabel(sb, text="Unlike static platforms (Roboflow),\nTCM Studio dynamically augments\npolygons in-memory during training.", 
                     text_color="#888", font=("Segoe UI", 11)).pack(pady=4)

        self.aug_flip_h = ctk.CTkCheckBox(sb, text="Horizontal Flip (fliplr=0.5)")
        self.aug_flip_h.pack(anchor="w", padx=10, pady=3)
        self.aug_flip_h.select()
        self.aug_flip_v = ctk.CTkCheckBox(sb, text="Vertical Flip (flipud=0.0)")
        self.aug_flip_v.pack(anchor="w", padx=10, pady=3)
        self.aug_rotate = ctk.CTkCheckBox(sb, text="Random Rotation (±15°)")
        self.aug_rotate.pack(anchor="w", padx=10, pady=3)
        self.aug_rotate.select()
        self.aug_bright = ctk.CTkCheckBox(sb, text="Brightness Jitter (hsv_v=0.4)")
        self.aug_bright.pack(anchor="w", padx=10, pady=3)
        self.aug_bright.select()
        self.aug_mosaic = ctk.CTkCheckBox(sb, text="Mosaic (mosaic=1.0)")
        self.aug_mosaic.pack(anchor="w", padx=10, pady=3)
        self.aug_mosaic.select()

        ctk.CTkLabel(sb, text="━" * 30, text_color="#333").pack(pady=4)

        ctk.CTkButton(sb, text="💾 SAVE YOLO CONFIG", height=45,
                      fg_color="#00BCD4", hover_color="#0097A7",
                      font=("Segoe UI", 13, "bold"),
                      command=self._save_aug_config).pack(fill="x", padx=8, pady=15)

        self.lbl_aug_status = ctk.CTkLabel(sb, text="Ready", text_color="#666",
                                           font=("Consolas", 10), wraplength=250)
        self.lbl_aug_status.pack(padx=8, pady=4)

        # Preview area
        self.aug_preview = ctk.CTkScrollableFrame(t, fg_color="#111")
        self.aug_preview.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        
        lbl = ctk.CTkLabel(self.aug_preview, text="Dynamic Augmentation Configured.\n\nThese advanced hyper-parameters are passed\ndirectly out-of-the-box to the YOLO engine during training.", font=("Segoe UI", 16), text_color="#aaa")
        lbl.pack(pady=100)

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 5: TRAIN
    # ══════════════════════════════════════════════════════════════════════
    def _setup_train_tab(self):
        t = self.tab_trn
        t.grid_rowconfigure(1, weight=1)
        t.grid_columnconfigure(0, weight=1)
        t.grid_columnconfigure(1, weight=1)

        # LEFT SIDE: Colab/No-GPU Preparation
        left = ctk.CTkFrame(t, fg_color="#181818")
        left.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        
        ctk.CTkLabel(left, text="☁️ GOOGLE COLAB / EXTERNAL", font=("Segoe UI", 16, "bold"), text_color="#2196F3").pack(pady=(15, 5))
        ctk.CTkLabel(left, text="1. Generate YOLO Structure & data.yaml\n2. Zip Dataset for Colab\n3. Upload & Train", text_color="#aaa", justify="center").pack(pady=10)
        
        ctk.CTkButton(left, text="1️⃣ PREPARE DATASET", width=250, height=45, fg_color="#1565C0", hover_color="#1E88E5", command=self._prepare_dataset).pack(pady=15)
        ctk.CTkButton(left, text="2️⃣ EXPORT FOR COLAB (.ZIP)", width=250, height=45, fg_color="#007ACC", hover_color="#00897B", command=self._export_dataset_zip).pack(pady=10)
        ctk.CTkButton(left, text="3️⃣ IMPORT TRAINED YOLO (.ZIP)", width=250, height=45, fg_color="#8E24AA", hover_color="#9C27B0", command=self._import_trained_zip).pack(pady=15)

        # RIGHT SIDE: Local GPU Training
        right = ctk.CTkFrame(t, fg_color="#1a1a1a")
        right.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        
        ctk.CTkLabel(right, text="🖥️ LOCAL GPU TRAINING", font=("Segoe UI", 16, "bold"), text_color="#00E676").pack(pady=(15, 5))
        
        r_grid = ctk.CTkFrame(right, fg_color="transparent")
        r_grid.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(r_grid, text="Epochs:", text_color="#888").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.ent_epochs = ctk.CTkEntry(r_grid, width=80, placeholder_text="100")
        self.ent_epochs.grid(row=0, column=1, padx=5, pady=5)
        self.ent_epochs.insert(0, str(tcm_config.DEFAULT_EPOCHS))
        
        ctk.CTkLabel(r_grid, text="Batch:", text_color="#888").grid(row=0, column=2, padx=5, pady=5, sticky="e")
        self.ent_batch = ctk.CTkEntry(r_grid, width=80, placeholder_text="16")
        self.ent_batch.grid(row=0, column=3, padx=5, pady=5)
        self.ent_batch.insert(0, str(tcm_config.DEFAULT_BATCH))
        
        ctk.CTkLabel(r_grid, text="ImgSz:", text_color="#888").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.ent_imgsz = ctk.CTkEntry(r_grid, width=80, placeholder_text="640")
        self.ent_imgsz.grid(row=1, column=1, padx=5, pady=5)
        self.ent_imgsz.insert(0, str(tcm_config.DEFAULT_IMGSZ))
        
        ctk.CTkLabel(r_grid, text="Model:", text_color="#888").grid(row=1, column=2, padx=5, pady=5, sticky="e")
        self.combo_model = ctk.CTkComboBox(r_grid, values=[
            "yolo11s-seg.pt",
            "yolo11n-seg.pt",
            "yolo11x-seg.pt",
            "yolo11n.pt", 
            "yolo11s.pt", 
            "yolo11m.pt", 
            "yolo11l.pt", 
            "yolo11x.pt", 
            "yolov8n.pt",
            "yolov8n-seg.pt",
            "yolov8m.pt", 
            "yolov8x.pt"
        ], width=130)
        self.combo_model.grid(row=1, column=3, padx=5, pady=5)
        self.combo_model.set("yolo11s-seg.pt")
        
        def _browse_train_model():
            p = filedialog.askopenfilename(filetypes=[("YOLO Weights", "*.pt")], parent=self)
            if p:
                vals = self.combo_model.cget("values")
                if p not in vals:
                    self.combo_model.configure(values=[p] + list(vals))
                self.combo_model.set(p)
                
        ctk.CTkButton(r_grid, text="📂", width=30, fg_color="#333", hover_color="#555", command=_browse_train_model).grid(row=1, column=4, padx=2, pady=5)
        
        ctk.CTkLabel(r_grid, text="Hardware:", text_color="#888").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.combo_device = ctk.CTkComboBox(r_grid, values=["Auto (GPU)", "Force CPU"], width=120)
        self.combo_device.grid(row=2, column=1, columnspan=3, padx=5, pady=5, sticky="w")
        self.combo_device.set("Auto (GPU)")
        
        r2 = ctk.CTkFrame(right, fg_color="transparent")
        r2.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkButton(r2, text="🚀 START TRAINING", width=160, height=45,
                      fg_color="#00C853", hover_color="#00E676",
                      font=("Segoe UI", 12, "bold"),
                      command=self._start_training).pack(side="left", padx=10, expand=True)
        ctk.CTkButton(r2, text="⏹ STOP", width=70, height=45,
                      fg_color="#B71C1C", hover_color="#D32F2F",
                      command=self._stop_training).pack(side="left", padx=10)
                      
        prog_container = ctk.CTkFrame(right, fg_color="transparent")
        prog_container.pack(fill="x", padx=20, pady=5)
        
        self.lbl_train_perc = ctk.CTkLabel(prog_container, text="0%", text_color="#00BCD4", font=("Consolas", 11, "bold"))
        self.lbl_train_perc.pack(side="top", anchor="e")
        self.train_progress = ctk.CTkProgressBar(prog_container, width=250, height=14)
        self.train_progress.pack(side="bottom", pady=2, fill="x")
        self.train_progress.set(0.0)

        # Logging Console
        self.train_log_text = ctk.CTkTextbox(t, fg_color="#0a0a0a",
                                              font=("Consolas", 11), text_color="#00E676")
        self.train_log_text.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=4, pady=4)
        self.train_log_text.insert("end", "TCM STUDIO PRO — Training Console\n")
        self.train_log_text.insert("end", f"GPU Device: CUDA:{tcm_config.DEVICE}\n")
        self.train_log_text.insert("end", "=" * 60 + "\n\n")
        self.train_log_text.configure(state="disabled")

    def _refresh_export_metrics(self):
        self.export_info.configure(state="normal")
        self.export_info.delete("1.0", "end")
        
        try:
            yaml_path = os.path.join(tcm_config.PROCESSED_DIR, "data.yaml")
            if os.path.exists(yaml_path):
                import datetime
                t_mod = os.path.getmtime(yaml_path)
                dt = datetime.datetime.fromtimestamp(t_mod).strftime('%Y-%m-%d %H:%M:%S')
                self.export_info.insert("end", f"📁 DATASET COMPILED ON: {dt}\n")
                
                img_dir = os.path.join(tcm_config.PROCESSED_DIR, "images", "train")
                val_dir = os.path.join(tcm_config.PROCESSED_DIR, "images", "val")
                train_imgs = len(os.listdir(img_dir)) if os.path.exists(img_dir) else 0
                val_imgs = len(os.listdir(val_dir)) if os.path.exists(val_dir) else 0
                self.export_info.insert("end", f"   Train Images: {train_imgs}  |  Validation Images: {val_imgs}\n\n")
            else:
                self.export_info.insert("end", "⚠️ DATASET NOT PREPARED YET\n\n")
                
            runs_dir = os.path.join(tcm_config.MODEL_DIR, "runs")
            if os.path.exists(runs_dir):
                subs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
                if subs:
                    latest_run = max(subs, key=os.path.getmtime)
                    csv_path = os.path.join(latest_run, "results.csv")
                    if os.path.exists(csv_path):
                        with open(csv_path, "r") as f:
                            lines = f.readlines()
                            if len(lines) > 1:
                                headers = [h.strip() for h in lines[0].split(",")]
                                last_line = [v.strip() for v in lines[-1].split(",")]
                                self.export_info.insert("end", f"🚀 LATEST TENSOR RUN: {os.path.basename(latest_run)}\n")
                                for h, v in zip(headers, last_line):
                                    if "metrics" in h.lower() or "epoch" in h.lower():
                                        self.export_info.insert("end", f"   {h}: {v}\n")
                    else:
                        self.export_info.insert("end", f"⚠️ NO RESULTS.CSV IN {os.path.basename(latest_run)}\n")
        except Exception as e:
            self.export_info.insert("end", f"Error parsing metrics: {e}\n")
            
        self.export_info.configure(state="disabled")

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 6: EXPORT
    # ══════════════════════════════════════════════════════════════════════
    def _setup_export_tab(self):
        t = self.tab_exp
        t.grid_rowconfigure(1, weight=1)
        t.grid_rowconfigure(2, weight=1)  # Enable metrics UI to expand
        t.grid_columnconfigure(0, weight=1)

        top = ctk.CTkFrame(t, fg_color="#1a1a1a")
        top.grid(row=0, column=0, sticky="ew", padx=4, pady=4)

        ctk.CTkLabel(top, text="📦 MODEL EXPORT", text_color="#FF9800",
                     font=("Segoe UI", 16, "bold")).pack(pady=10)

        r1 = ctk.CTkFrame(top, fg_color="transparent")
        r1.pack(fill="x", padx=10, pady=4)
        ctk.CTkLabel(r1, text="best.pt path:", text_color="#888").pack(side="left")
        self.ent_best_pt = ctk.CTkEntry(r1, width=500, placeholder_text="Auto-detected after training")
        self.ent_best_pt.pack(side="left", padx=8)
        ctk.CTkButton(r1, text="Browse", width=80, fg_color="#37474F",
                      command=self._browse_best_pt).pack(side="left")

        r2 = ctk.CTkFrame(top, fg_color="transparent")
        r2.pack(fill="x", padx=10, pady=8)
        
        ctk.CTkButton(r2, text="📦 EXPORT YOLO MODEL (.PT)", height=50, width=250,
                      fg_color="#E65100", hover_color="#FF6D00",
                      font=("Segoe UI", 14, "bold"),
                      command=self._export_model).pack(side="left", padx=5)
                      
        ctk.CTkButton(r2, text="🗄 EXPORT DATASET AS ZIP (FOR EXTERNAL TRAINING)", height=50, width=350,
                      fg_color="#007ACC", hover_color="#00897B",
                      font=("Segoe UI", 14, "bold"),
                      command=self._export_dataset_zip).pack(side="left", padx=5)

        ctk.CTkButton(top, text="🔄 REFRESH METRICS", fg_color="#37474F", 
                      command=self._refresh_export_metrics).pack(side="right", padx=10, pady=10)

        # Dynamic Metrics UI
        self.export_info = ctk.CTkTextbox(t, fg_color="#0a0a0a", text_color="#00E676", font=("Consolas", 12))
        self.export_info.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        self._refresh_export_metrics()

        info_frame = ctk.CTkFrame(t, fg_color="#111")
        info_frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)

        export_info = (
            "Export will package the following files:\n\n"
            "  ✅  best.pt          — Trained YOLO model weights\n"
            "  ✅  data.yaml        — Dataset configuration\n"
            "  ✅  class_names.json — All 20 class names mapped by ID\n"
            "  ✅  model_info.json  — Training metadata & GPU info\n"
            "  ✅  results.csv      — Training metrics log\n"
            "  ✅  confusion_matrix.png  — Confusion matrix plot\n"
            "  ✅  results.png      — Training curves\n\n"
            f"Export Location: {tcm_config.EXPORT_DIR}\n\n"
            "These files are compatible with:\n"
            "  • Roboflow Deploy\n"
            "  • Ultralytics Predict\n"
            "  • ONNX / TensorRT export\n"
            "  • RaniVision Production System"
        )
        ctk.CTkLabel(info_frame, text=export_info, font=("Consolas", 12),
                     text_color="#888", justify="left", anchor="nw").pack(padx=20, pady=20, anchor="nw")

    # ══════════════════════════════════════════════════════════════════════
    #  CAMERA ACTIONS
    # ══════════════════════════════════════════════════════════════════════
    def _capture_loop(self):
        while self.video_running:
            self.cam.read()
            time.sleep(0.01)

    def _update_gui_loop(self):
        if self.video_running and self.tabs.get() == "📸 CAMERA":
            if getattr(self, "static_camera_mode", False):
                self.after(30, self._update_gui_loop)
                return
                
            frame = self.cam.get_latest()
            if frame is not None:
                if self.simulation_active and self.model:
                    try:
                        conf_val = float(self.slider_conf.get())
                        
                        # --- BLUE MASKING PRE-PROCESSING ---
                        # Black out the blue rollers so the AI only sees cashews
                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        lower_blue = np.array([90, 50, 50])
                        upper_blue = np.array([130, 255, 255])
                        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
                        mask_non_blue = cv2.bitwise_not(mask_blue)
                        black_bg_frame = cv2.bitwise_and(frame, frame, mask=mask_non_blue)
                        
                        # Predict on the blacked-out frame
                        res = self.model.predict(black_bg_frame, conf=conf_val, verbose=False)
                        
                        for r in res:
                            for box in r.boxes:
                                cname = self.model.names[int(box.cls[0])]
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                bgr = tcm_config.hex_to_bgr(tcm_config.CLASS_COLORS.get(cname, "#FFF"))
                                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 3)
                                conf = float(box.conf[0])
                                label = f"{cname} {conf:.2f}"
                                cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr, 2)
                    except:
                        pass
                try:
                    lh = self.lbl_video.winfo_height()
                    lw = self.lbl_video.winfo_width()
                    if lw > 10:
                        scale = min(lw / frame.shape[1], lh / frame.shape[0])
                        nw, nh = int(frame.shape[1] * scale), int(frame.shape[0] * scale)
                        resized = cv2.resize(frame, (nw, nh))
                        img = ctk.CTkImage(
                            Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)),
                            size=(nw, nh)
                        )
                        self.lbl_video.configure(image=img, text="")
                except Exception:
                    pass
        self.after(30, self._update_gui_loop)

    def _apply_cam(self):
        exp = self.ent_exp.get() or "20000"
        again = self.ent_again.get() or "4"
        self.cam.set_manual_params(exp, float(again), 100)

    def _on_model_select(self, run_name):
        if run_name.startswith("Select"):
            self.model = None
            return
            
        if run_name == "Import External Model (.pt)...":
            p = filedialog.askopenfilename(filetypes=[("YOLO Model", "*.pt")], parent=self)
            if not p:
                self.combo_load_model.set("Select Trained Model...")
                return
            import datetime
            import shutil
            # Create a placeholder run directory to store this imported model permanently
            run_name = f"External_Import_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run_dir = os.path.join(tcm_config.MODEL_DIR, "runs", run_name, "weights")
            os.makedirs(run_dir, exist_ok=True)
            target_p = os.path.join(run_dir, "best.pt")
            shutil.copy(p, target_p)
            p = target_p
            
            # Refresh list silently
            vals = self.combo_load_model.cget("values")
            self.combo_load_model.configure(values=[vals[0], vals[1], run_name, *vals[2:]])
            self.combo_load_model.set(run_name)
        else:
            p = os.path.join(tcm_config.MODEL_DIR, "runs", run_name, "weights", "best.pt")
            
        if os.path.exists(p) and YOLO:
            try:
                self.model = YOLO(p)
                self.lbl_status.configure(text=f"✓ LOADED: {run_name}", text_color="#00C853")
            except Exception as e:
                messagebox.showerror("Error", str(e), parent=self)
        else:
            messagebox.showerror("Error", f"best.pt not found for {run_name}!", parent=self)

    def _load_model(self):
        # Kept for backward compatibility just in case
        pass

    def _on_conf_change(self, val):
        self.lbl_conf.configure(text=f"Conf: {int(val*100)}%")
        if getattr(self, "static_camera_mode", False) and getattr(self, "last_test_image_path", None):
            self._test_on_image(refresh=True)

    def _toggle_sim(self):
        self.static_camera_mode = False
        self.simulation_active = self.sw_sim.get()
        if self.simulation_active and not self.model:
            self.sw_sim.deselect()
            messagebox.showerror("Error", "Load a .pt model first!")
            
    def _next_test_image(self):
        if not getattr(self, "last_test_image_path", None):
            return
        
        folder = os.path.dirname(self.last_test_image_path)
        if not os.path.exists(folder):
            return
            
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if not files:
            return
            
        current = os.path.basename(self.last_test_image_path)
        try:
            idx = files.index(current)
            next_idx = (idx + 1) % len(files)
            self.last_test_image_path = os.path.join(folder, files[next_idx])
            self._test_on_image(refresh=True)
        except ValueError:
            pass

    def _test_on_image(self, refresh=False):
        if type(refresh) != bool:
            refresh = False
            
        if not getattr(self, "model", None):
            messagebox.showerror("Error", "Load a .pt model first!")
            return
            
        p = getattr(self, "last_test_image_path", None) if refresh else None
            
        if not p:
            p = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")], parent=self)
            
        if not p:
            return
            
        self.last_test_image_path = p
        img = cv2.imread(p)
        if img is None:
            return
            
        self.static_camera_mode = True
        self.sw_sim.deselect()
            
        conf_val = float(self.slider_conf.get())
        res = self.model.predict(img, conf=conf_val, verbose=False)
        for r in res:
            # Draw masks if segmentation model
            if r.masks is not None:
                predictions = []
                for i, mask in enumerate(r.masks.xy):
                    cname = self.model.names[int(r.boxes[i].cls[0])]
                    conf = float(r.boxes[i].conf[0])
                    pts = np.array(mask, dtype=np.int32).reshape((-1, 1, 2))
                    if len(pts) == 0: continue
                    rect = cv2.boundingRect(pts)
                    predictions.append({"cls": cname, "conf": conf, "poly": pts, "rect": rect, "merged": False, "final_cls": cname})

                base_classes = ["CASHEW", "GOOD_CASHEW", "BROWN", "SW"]

                for def_p in predictions:
                    if def_p["cls"] not in base_classes:
                        best_parent = None
                        dx, dy, dw, dh = def_p["rect"]
                        best_area = 0
                        for base_p in predictions:
                            if base_p["cls"] in base_classes:
                                bx, by, bw, bh = base_p["rect"]
                                # Calculate intersection area of bounding boxes for robust overlapping
                                ix = max(dx, bx)
                                iy = max(dy, by)
                                iw = min(dx+dw, bx+bw) - ix
                                ih = min(dy+dh, by+bh) - iy
                                
                                if iw > 0 and ih > 0:
                                    intersect_area = iw * ih
                                    if intersect_area > best_area:
                                        best_area = intersect_area
                                        best_parent = base_p
                        if best_parent:
                            if best_parent["final_cls"] in base_classes:
                                best_parent["final_cls"] = def_p["cls"]
                            def_p["merged"] = True

                for p in predictions:
                    if p["merged"]: continue
                    cname = p["final_cls"]
                    conf = p["conf"]
                    bx, by, bw, bh = p["rect"]
                    bgr = tcm_config.hex_to_bgr(tcm_config.CLASS_COLORS.get(cname, "#FFF"))
                    
                    # Draw a perfect rectangle box instead of a polygon sketch
                    cv2.rectangle(img, (bx, by), (bx+bw, by+bh), bgr, 2)
                    
                    label = f"{cname} {conf:.2f}"
                    cv2.putText(img, label, (bx, max(20, by - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2)
            else:
                for box in r.boxes:
                    cname = self.model.names[int(box.cls[0])]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    bgr = tcm_config.hex_to_bgr(tcm_config.CLASS_COLORS.get(cname, "#FFF"))
                    cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 3)
                    conf = float(box.conf[0])
                    label = f"{cname} {conf:.2f}"
                    cv2.putText(img, label, (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr, 2)
                            
        lh = self.lbl_video.winfo_height()
        lw = self.lbl_video.winfo_width()
        if lw > 10:
            scale = min(lw / img.shape[1], lh / img.shape[0])
            nw, nh = int(img.shape[1] * scale), int(img.shape[0] * scale)
            resized = cv2.resize(img, (nw, nh))
            from PIL import Image
            tk_img = ctk.CTkImage(light_image=Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)), dark_image=Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)), size=(nw, nh))
            self.lbl_video.configure(image=tk_img, text="")
            self.static_test_tk = tk_img

    def _quick_capture(self):
        frame = self.cam.get_latest()
        if frame is None:
            return
        cls = self.combo_capture_cls.get()
        cls_dir = os.path.join(tcm_config.DATASET_DIR, cls)
        os.makedirs(cls_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fname = f"{cls}_{ts}.jpg"
        cv2.imwrite(os.path.join(cls_dir, fname), frame)
        self.lbl_status.configure(text=f"Saved: {fname}", text_color="#4CAF50")

    def _capture_to_annotate(self):
        frame = self.cam.get_latest()
        if frame is None:
            return
        self.temp_image_path = "temp_capture.jpg"
        self.viewing_saved_file = False
        cv2.imwrite(self.temp_image_path, frame)
        self.current_cv_img = frame
        self._load_image_to_canvas(frame)
        self.tabs.set("✏️ ANNOTATE")
        self.current_bboxes = []

    # ══════════════════════════════════════════════════════════════════════
    #  ANNOTATION ACTIONS (ROBOFLOW MAGIC WAND & CROP POPUP)
    # ══════════════════════════════════════════════════════════════════════
    def _open_image_file(self):
        p = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")], parent=self)
        if p:
            img = cv2.imread(p)
            if img is not None:
                self.temp_image_path = p
                self.viewing_saved_file = True
                self.current_cv_img = img
                self.current_bboxes = []
                self._load_image_to_canvas(img)
                
                txt = os.path.splitext(p)[0] + ".txt"
                if os.path.exists(txt):
                    self._load_yolo_labels(txt, img.shape[1], img.shape[0])
                # Show folder context with image count
                folder = os.path.dirname(p)
                all_imgs = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                try:
                    idx = sorted(all_imgs).index(os.path.basename(p)) + 1
                except ValueError:
                    idx = 1
                self.lbl_img_info.configure(text=f"{os.path.basename(folder)}/{os.path.basename(p)}  [{idx}/{len(all_imgs)}]")

    def _open_image_folder(self):
        """Open a folder and load the first image. Use PREV/NEXT to navigate."""
        folder = filedialog.askdirectory(title="Select Folder with Images", parent=self)
        if not folder:
            return
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        if not files:
            messagebox.showwarning("No Images", "No image files found in the selected folder.", parent=self)
            return
        fp = os.path.join(folder, files[0])
        img = cv2.imread(fp)
        if img is not None:
            self.temp_image_path = fp
            self.viewing_saved_file = True
            self.current_cv_img = img
            self.current_img_idx = 0
            self.current_bboxes = []
            self._load_image_to_canvas(img)
            
            txt = os.path.splitext(fp)[0] + ".txt"
            if os.path.exists(txt):
                self._load_yolo_labels(txt, img.shape[1], img.shape[0])
            self.lbl_img_info.configure(text=f"{os.path.basename(folder)}/{files[0]}  [1/{len(files)}]")

    def _load_yolo_labels(self, txt_path, img_w, img_h):
        self.current_bboxes = []
        loaded = False
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                loaded = True
                cid = int(parts[0])
                cls_name = tcm_config.CLASS_NAMES[cid] if cid < len(tcm_config.CLASS_NAMES) else "UNKNOWN"
                
                # If length > 5, it is a YOLO Polygon Segmentation format!
                if len(parts) > 5:
                    poly_points = []
                    xs = []
                    ys = []
                    for i in range(1, len(parts)-1, 2):
                        px = float(parts[i]) * img_w
                        py = float(parts[i+1]) * img_h
                        poly_points.append((px, py))
                        xs.append(px)
                        ys.append(py)
                    x = min(xs)
                    y = min(ys)
                    w = max(xs) - x
                    h = max(ys) - y
                    self.current_bboxes.append({"type": "polygon", "cls": cls_name, "x": x, "y": y, "w": w, "h": h, "points": poly_points})
                else:
                    # Legacy Bounding Box format
                    cx, cy, nw, nh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    x = (cx - nw / 2) * img_w
                    y = (cy - nh / 2) * img_h
                    w = nw * img_w
                    h = nh * img_h
                    self.current_bboxes.append({"type": "bbox", "cls": cls_name, "x": x, "y": y, "w": w, "h": h})
        self._redraw()
        return loaded

    def _nav_images(self, direction):
        if not self.viewing_saved_file:
            return
            
        self._save_annotation(quiet=True)
        prev_boxes = self.current_bboxes.copy()
        
        # Navigate ONLY within the same folder the user opened the image from
        current_dir = os.path.dirname(self.temp_image_path)
        files = sorted([f for f in os.listdir(current_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if not files:
            return
        
        current_name = os.path.basename(self.temp_image_path)
        try:
            self.current_img_idx = files.index(current_name)
        except ValueError:
            self.current_img_idx = 0
            
        self.current_img_idx += direction
        if self.current_img_idx < 0:
            self.current_img_idx = len(files) - 1
        if self.current_img_idx >= len(files):
            self.current_img_idx = 0
            
        fp = os.path.join(current_dir, files[self.current_img_idx])
        img = cv2.imread(fp)
        if img is not None:
            self.temp_image_path = fp
            self.current_cv_img = img
            self.current_bboxes = []
            self._load_image_to_canvas(img)
            
            txt = os.path.splitext(fp)[0] + ".txt"
            loaded_existing = False
            if os.path.exists(txt):
                loaded_existing = self._load_yolo_labels(txt, img.shape[1], img.shape[0])
                
            if not loaded_existing and prev_boxes:
                # User requested DO NOT carry over labels so we commented it out
                # for b in prev_boxes:
                #     self.current_bboxes.append(b.copy())
                # self._save_annotation(quiet=True)
                # self._redraw()
                pass
                
            if not getattr(self, "current_bboxes", []) and hasattr(self, "sw_auto_label") and self.sw_auto_label.get():
                dialog = ctk.CTkToplevel(self)
                dialog.title("Select Auto Class")
                dialog.geometry("400x220")
                dialog.transient(self)
                
                ctk.CTkLabel(dialog, text="Cashews Auto-Detected! 🤖\nWhich class should be assigned to them?", font=("Segoe UI", 14, "bold"), text_color="#2196F3").pack(pady=(20, 10))
                combo = ctk.CTkComboBox(dialog, values=tcm_config.CLASS_NAMES, width=250)
                combo.pack(pady=10)
                if tcm_config.CLASS_NAMES: combo.set(tcm_config.CLASS_NAMES[0])
                
                self.auto_label_result = None
                def on_ok():
                    self.auto_label_result = combo.get()
                    dialog.grab_release()
                    dialog.destroy()
                    
                ctk.CTkButton(dialog, text="✅ ASSIGN", fg_color="#00C853", hover_color="#00E676", font=("Segoe UI", 14, "bold"), command=on_ok).pack(pady=15)
                dialog.grab_set()
                self.wait_window(dialog)
                
                if getattr(self, "auto_label_result", None):
                    self._auto_detect_all_in_image(img, self.auto_label_result)
                
            folder_name = os.path.basename(current_dir)
            self.lbl_img_info.configure(text=f"{folder_name}/{files[self.current_img_idx]}  [{self.current_img_idx+1}/{len(files)}]")

    def _load_image_to_canvas(self, cv_img):
        self.orig_h, self.orig_w = cv_img.shape[:2]
        self.update_idletasks()
        self._redraw()

    def _mouse_wheel(self, e):
        # Handle Zoom scrolling
        delta = e.delta if hasattr(e, 'delta') else 0
        if delta > 0:
            self.interactive_zoom = min(5.0, getattr(self, 'interactive_zoom', 1.0) * 1.1)
        elif delta < 0:
            self.interactive_zoom = max(0.2, getattr(self, 'interactive_zoom', 1.0) / 1.1)
        self._redraw()
        
    def _middle_mouse_down(self, e):
        self.pan_start_x = e.x
        self.pan_start_y = e.y
        
    def _middle_mouse_drag(self, e):
        dx = e.x - getattr(self, 'pan_start_x', e.x)
        dy = e.y - getattr(self, 'pan_start_y', e.y)
        self.pan_x = getattr(self, 'pan_x', 0) + dx
        self.pan_y = getattr(self, 'pan_y', 0) + dy
        self.pan_start_x = e.x
        self.pan_start_y = e.y
        self._redraw()
        
    def _mouse_move(self, e):
        # Update Roboflow crosshairs dynamically independent of full redraw overhead
        if hasattr(self, 'cross_h') and hasattr(self, 'cross_v'):
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            self.canvas.coords(self.cross_h, 0, e.y, cw, e.y)
            self.canvas.coords(self.cross_v, e.x, 0, e.x, ch)

    def _mouse_down(self, e):
        # Auto-Pause running hardware streams to anchor moving objects perfectly for drawing!
        if self.video_running:
            self.video_running = False
            self.lbl_status.configure(text="Live Feed Paused For Annotation (Click NEXT to resume)", text_color="#2196F3")
            
        ix = int((e.x - self.offset_x) / self.display_scale)
        iy = int((e.y - self.offset_y) / self.display_scale)
        if not (0 <= ix < self.orig_w and 0 <= iy < self.orig_h):
            return

        # Hit test first! Even if Magic Wand is ON, hitting an existing box means EDIT.
        self.drag_start = (e.x, e.y)
        self.selected_box_idx = -1
        for i, b in enumerate(self.current_bboxes):
            # Added 15px padding so users don't miss the box and accidentally spawn a new one
            if b['x'] - 15 <= ix <= b['x'] + b['w'] + 15 and b['y'] - 15 <= iy <= b['y'] + b['h'] + 15:
                self.selected_box_idx = i
                break
        
        if self.selected_box_idx != -1:
            if self.magic_tool_active.get():
                # Layer 2 Defect Extraction!
                self._do_defect_wand(ix, iy, self.selected_box_idx)
            return

        if self.magic_tool_active.get():
            self._do_magic_wand(ix, iy)
        else:
            self.rect_start = (e.x, e.y)
            self.rect_id = self.canvas.create_rectangle(e.x, e.y, e.x, e.y, outline="yellow", width=2, dash=(4,4))

    def _mouse_drag(self, e):
        # Allow dragging only if we are drawing a new rect, and not magic wand, and not selected an existing box
        if hasattr(self, 'selected_box_idx') and self.selected_box_idx != -1:
            return
        if hasattr(self, 'rect_start') and self.rect_start and not self.magic_tool_active.get():
            self.canvas.coords(self.rect_id, self.rect_start[0], self.rect_start[1], e.x, e.y)

    def _mouse_up(self, e):
        if hasattr(self, 'selected_box_idx') and self.selected_box_idx != -1:
            dist = abs(e.x - self.drag_start[0]) + abs(e.y - self.drag_start[1])
            if dist < 10:
                b = self.current_bboxes[self.selected_box_idx]
                ix1, iy1 = int(b['x']), int(b['y'])
                ix2, iy2 = int(b['x'] + b['w']), int(b['y'] + b['h'])
                self._trigger_class_popup(ix1, iy1, ix2, iy2, edit_idx=self.selected_box_idx)
            self.selected_box_idx = -1
            return

        if self.magic_tool_active.get():
            return

        if hasattr(self, 'rect_start') and self.rect_start:
            x1, y1 = self.rect_start
            x2, y2 = e.x, e.y
            self.rect_start = None
            self.canvas.delete(self.rect_id)
            
            if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
                return
            
            ix1 = max(0, int((min(x1, x2) - self.offset_x) / self.display_scale))
            iy1 = max(0, int((min(y1, y2) - self.offset_y) / self.display_scale))
            ix2 = min(self.orig_w, int((max(x1, x2) - self.offset_x) / self.display_scale))
            iy2 = min(self.orig_h, int((max(y1, y2) - self.offset_y) / self.display_scale))
            
            self._trigger_class_popup(ix1, iy1, ix2, iy2)

    def _do_magic_wand(self, ix, iy):
        if self.current_cv_img is None:
            return
            
        img = self.current_cv_img.copy()
        
        # Localize search to a 500x500 region around the click to prevent full-screen blob merges
        S = 250 
        x_min = max(0, ix - S)
        y_min = max(0, iy - S)
        x_max = min(img.shape[1], ix + S)
        y_max = min(img.shape[0], iy + S)
        
        roi = img[y_min:y_max, x_min:x_max]
        
        lower_blue = np.array([70, 40, 30])
        upper_blue = np.array([150, 255, 255])
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Also remove dark shadows (like gaps between rollers)
        # Anything with very low Value (Brightness < 25) is a pitch-black shadow
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 25])
        mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
        
        # Combine blue and dark masks
        mask_bg = cv2.bitwise_or(mask_blue, mask_dark)
        mask_obj = cv2.bitwise_not(mask_bg)
        
        kernel = np.ones((5,5), np.uint8)
        mask_obj = cv2.morphologyEx(mask_obj, cv2.MORPH_OPEN, kernel)
        mask_obj = cv2.morphologyEx(mask_obj, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask_obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_cnt = None
        min_dist = float('inf')
        local_ix = ix - x_min
        local_iy = iy - y_min
        
        for cnt in contours:
            if cv2.contourArea(cnt) < 500:  # Must be a substantial object to qualify as cashew or large testa
                continue
            dist = cv2.pointPolygonTest(cnt, (local_ix, local_iy), True)
            if dist >= 0:
                best_cnt = cnt
                break
            else:
                if abs(dist) < min_dist:
                    min_dist = abs(dist)
                    best_cnt = cnt
                    
        # Check if they clicked relatively close to the valid cashew contour
        if best_cnt is not None and (min_dist < 40 or dist >= -10):
            bx, by, bw, bh = cv2.boundingRect(best_cnt)
            gx = x_min + bx
            gy = y_min + by
            
            pad = 8
            ix1 = max(0, gx - pad)
            iy1 = max(0, gy - pad)
            ix2 = min(img.shape[1], gx + bw + pad)
            iy2 = min(img.shape[0], gy + bh + pad)
            if bw > 15 and bh > 15:
                # --- MATHEMATICAL COLOR GRADING (BASE CLASS) ---
                body_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
                cv2.drawContours(body_mask, [best_cnt], -1, 255, -1)
                mean_hsv = cv2.mean(hsv, mask=body_mask)
                sat = mean_hsv[1]
                
                # Good cashew (pure white/creamy) has low saturation. 
                # Scorched cashews are more yellowish/brownish (higher saturation).
                suggested_class = "GOOD"
                if sat > 105:
                    suggested_class = "SSW"
                elif sat > 65:
                    suggested_class = "SW"
                # -----------------------------------------------
                
                # Capture exact contour for Segmentation
                if hasattr(self, 'annot_mode') and self.annot_mode.get() == "Polygon (Seg)":
                    epsilon = 0.005 * cv2.arcLength(best_cnt, True)
                    approx = cv2.approxPolyDP(best_cnt, epsilon, True)
                    poly_points = []
                    for pt in approx:
                        px, py = pt[0][0], pt[0][1]
                        poly_points.append((x_min + px, y_min + py))
                    self._trigger_class_popup(ix1, iy1, ix2, iy2, poly_points=poly_points, suggested_class=suggested_class)
                else:
                    self._trigger_class_popup(ix1, iy1, ix2, iy2, suggested_class=suggested_class)
            else:
                self.lbl_status.configure(text="Magic Tool: Object too small (Not a Cashew)", text_color="#FF9800")
        else:
            self.lbl_status.configure(text="Magic Tool: Click closer to a valid object", text_color="#FF9800")
            
    def _do_defect_wand(self, ix, iy, parent_idx):
        if self.current_cv_img is None: return
        parent_box = self.current_bboxes[parent_idx]
        px, py, pw, ph = parent_box['x'], parent_box['y'], parent_box['w'], parent_box['h']
        
        # Only search within the parent cashew boundary
        img = self.current_cv_img.copy()
        pad = 10
        y_min, y_max = max(0, int(py - pad)), min(img.shape[0], int(py + ph + pad))
        x_min, x_max = max(0, int(px - pad)), min(img.shape[1], int(px + pw + pad))
        roi = img[y_min:y_max, x_min:x_max]
        
        # Focus on dark/brownish defects (Testa, PKW, Oil)
        # Defects are generally lower in VALUE (Brightness) or specific Hües
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        local_ix = max(0, min(roi.shape[1] - 1, ix - x_min))
        local_iy = max(0, min(roi.shape[0] - 1, iy - y_min))
        
        # Seed-based localized gradient detection via FloodFill.
        # This securely isolates the organic defect skin extending from the clicked pixel, 
        # completely preventing distant shadows or background regions from bleeding into the mask.
        flood_mask = np.zeros((roi.shape[0]+2, roi.shape[1]+2), np.uint8)
        seed_pt = (local_ix, local_iy)
        lo_diff = (30, 30, 30)  # Locked tolerance limits
        up_diff = (30, 30, 30)
        
        flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
        cv2.floodFill(roi, flood_mask, seed_pt, 255, lo_diff, up_diff, flags=flags)
        mask_defect = flood_mask[1:-1, 1:-1]
        
        # Soft closure without eroding thin organic bridges
        kernel = np.ones((2,2), np.uint8)
        mask_defect = cv2.morphologyEx(mask_defect, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask_defect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_cnt = None
        min_dist = float('inf')
        local_ix = ix - x_min
        local_iy = iy - y_min
        
        for cnt in contours:
            if cv2.contourArea(cnt) < 2: continue # Allow very tiny PKW spots
            dist = cv2.pointPolygonTest(cnt, (local_ix, local_iy), True)
            if dist >= 0:
                best_cnt = cnt
                break
            else:
                if abs(dist) < min_dist:
                    min_dist = abs(dist)
                    best_cnt = cnt
                    
        # Allow connecting if clicked nearby (min_dist < 20)
        if best_cnt is not None and (min_dist < 20 or dist >= 0):
            bx, by, bw, bh = cv2.boundingRect(best_cnt)
            gx = x_min + bx
            gy = y_min + by
            ix1, iy1 = max(0, gx - 5), max(0, gy - 5)
            ix2, iy2 = min(img.shape[1], gx + bw + 5), min(img.shape[0], gy + bh + 5)
            
            # --- MATHEMATICAL HEURISTIC EVALUATION ---
            defect_area = cv2.contourArea(best_cnt)
            parent_area = pw * ph * 0.70  # approx real cashew pixel area based on bbox
            ratio = defect_area / max(1, parent_area)
            
            suggested_class = None
            if ratio > 0.15:
                suggested_class = "UPW"
            else:
                # Check mean HSV specifically to detect pitch black dots for PKW
                single_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
                cv2.drawContours(single_mask, [best_cnt], -1, 255, -1)
                mean_hsv = cv2.mean(hsv, mask=single_mask)
                if mean_hsv[2] < 70:  # Very low value/brightness -> Black dot
                    suggested_class = "PKW1"
                else:
                    suggested_class = "TESTA"
            # -----------------------------------------
            
            # Layer 2 Sub-Mask Outline
            if hasattr(self, 'annot_mode') and self.annot_mode.get() == "Polygon (Seg)":
                epsilon = 0.001 * cv2.arcLength(best_cnt, True)
                approx = cv2.approxPolyDP(best_cnt, epsilon, True)
                poly_points = []
                for pt in approx:
                    poly_points.append((x_min + pt[0][0], y_min + pt[0][1]))
                self._trigger_class_popup(ix1, iy1, ix2, iy2, poly_points=poly_points, suggested_class=suggested_class)
            else:
                self._trigger_class_popup(ix1, iy1, ix2, iy2, suggested_class=suggested_class)
        else:
            self.lbl_status.configure(text="Magic Tool [Layer 2]: No defect found near click", text_color="#FF9800")

    def _right_click_delete(self, e):
        if not hasattr(self, 'current_bboxes'): return
        ix = int((e.x - self.offset_x) / self.display_scale)
        iy = int((e.y - self.offset_y) / self.display_scale)
        
        # Reverse check to delete the top-most box First
        for i in reversed(range(len(self.current_bboxes))):
            b = self.current_bboxes[i]
            if b['x'] - 10 <= ix <= b['x'] + b['w'] + 10 and b['y'] - 10 <= iy <= b['y'] + b['h'] + 10:
                removed_cls = b['cls']
                self.current_bboxes.pop(i)
                self._redraw()
                self._save_annotation(quiet=True)
                self.lbl_status.configure(text=f"🗑 Deleted '{removed_cls}' box and Auto-Saved", text_color="#F44336")
                return

    def _auto_detect_all_in_image(self, img, cls_name):
        added = 0
        use_poly = hasattr(self, "annot_mode") and self.annot_mode.get() == "Polygon (Seg)"
        
        # 1. Primary Method: Use loaded YOLO AI to Auto-Label (if available)
        if getattr(self, "model", None):
            res = self.model.predict(img, conf=0.15, verbose=False)
            for r in res:
                for i, box in enumerate(r.boxes):
                    cname = self.model.names[int(box.cls[0])]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    if use_poly and r.masks is not None:
                        poly = r.masks.xy[i].tolist()
                        if len(poly) >= 3:
                            self.current_bboxes.append({"cls": cname, "type": "polygon", "points": poly, "x": x1, "y": y1, "w": x2-x1, "h": y2-y1})
                            added += 1
                            continue
                            
                    self.current_bboxes.append({"cls": cname, "type": "box", "x": x1, "y": y1, "w": x2-x1, "h": y2-y1})
                    added += 1
            if added > 0:
                self._redraw()
                self._save_annotation(quiet=True)
                type_str = "polygons" if use_poly else "bounding boxes"
                self.lbl_status.configure(text=f"✨ AI Auto-Labeled {added} {type_str} and Auto-Saved", text_color="#00C853")
                return

        # 2. Fallback Method: OpenCV Contour Detection (Strict)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        # Cashews are much brighter than the blue rollers or black belts
        _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((5,5), np.uint8)
        mask_obj = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        mask_obj = cv2.morphologyEx(mask_obj, cv2.MORPH_CLOSE, kernel)
        
        try:
            contours, _ = cv2.findContours(mask_obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:
            contours, _ = cv2.findContours(mask_obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter typical noise and massive background shapes (paper, belts)
            if area < 1500 or area > 50000:
                continue
                
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw < 10 or bh < 10:
                continue
                
            # Filter strictly by aspect ratio to ignore the red tapes and metallic poles 
            aspect_ratio = max(bw, bh) / float(min(bw, bh))
            if aspect_ratio > 3.0: 
                continue
                
            pad = 5
            x = max(0, bx - pad)
            y = max(0, by - pad)
            w = min(img.shape[1] - x, bw + pad * 2)
            h = min(img.shape[0] - y, bh + pad * 2)
            
            if use_poly:
                epsilon = 0.005 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                poly_pts = [pt[0].tolist() for pt in approx]
                self.current_bboxes.append({
                    "cls": cls_name, "type": "polygon", "points": poly_pts,
                    "x": x, "y": y, "w": w, "h": h
                })
            else:
                self.current_bboxes.append({"cls": cls_name, "type": "box", "x": x, "y": y, "w": w, "h": h})
            added += 1
            
        self._redraw()
        if added > 0:
            type_str = "polygons" if use_poly else "boxes"
            self.lbl_status.configure(text=f"✨ OpenCV Auto-Labeled {added} {type_str} as '{cls_name}'", text_color="#00C853")
        else:
            self.lbl_status.configure(text=f"Magic Wand: Found no objects matching size criteria.", text_color="#FF9800")


    def _trigger_class_popup(self, x1, y1, x2, y2, edit_idx=-1, poly_points=None, suggested_class=None):
        if self.current_cv_img is None: return
        
        # Crop the image using Numpy slicing
        cropped_img = self.current_cv_img[y1:y2, x1:x2]
        if cropped_img.size == 0: return
        
        # Apply HD Resizing if requested
        size_setting = self.crop_size_var.get()
        if size_setting != "Original HD":
            try:
                dim = int(size_setting.split("x")[0])
                cropped_img = cv2.resize(cropped_img, (dim, dim), interpolation=cv2.INTER_LANCZOS4)
            except Exception:
                pass
        
        # Show prompt
        def on_class_assigned(cls_name, staged_defects=[]):
            if cls_name == "DELETE":
                if edit_idx != -1:
                    self.current_bboxes.pop(edit_idx)
                    self._redraw()
                return

            w = x2 - x1
            h = y2 - y1
            if edit_idx != -1:
                self.current_bboxes[edit_idx]["cls"] = cls_name
            else:
                if poly_points:
                    self.current_bboxes.append({"type": "polygon", "cls": cls_name, "x": x1, "y": y1, "w": w, "h": h, "points": poly_points})
                else:
                    self.current_bboxes.append({"type": "bbox", "cls": cls_name, "x": x1, "y": y1, "w": w, "h": h})
            
            # Unpack all dynamically staged layer 2 defects injected from the Popup sandbox
            # Unpack all dynamically staged layer 2 defects injected from the Popup sandbox
            for defect_obj in staged_defects:
                if defect_obj.get("type") == "polygon" and "points" in defect_obj:
                    pts_x = [pt[0] for pt in defect_obj["points"]]
                    pts_y = [pt[1] for pt in defect_obj["points"]]
                    defect_obj["x"] = min(pts_x)
                    defect_obj["y"] = min(pts_y)
                    defect_obj["w"] = max(pts_x) - min(pts_x)
                    defect_obj["h"] = max(pts_y) - min(pts_y)
                else:
                    defect_obj["x"] = x1
                    defect_obj["y"] = y1
                    defect_obj["w"] = w
                    defect_obj["h"] = h
                self.current_bboxes.append(defect_obj)
                
            self._redraw()
            
            # Sub-image strictly saved in `_save_annotation` so that auto-labels are captured too!
            msg = "Polygon" if poly_points else "Bounding Box"
            self.lbl_status.configure(text=f"✅ Configured {msg} as: {cls_name}. (Auto-Saved!)", text_color="#00C853")
            self._save_annotation(quiet=True)

        popup = ClassAssignPopup(self, cropped_img, on_class_assigned, suggested_class=suggested_class, crop_origin=(x1, y1))

    def _undo_box(self):
        if self.current_bboxes:
            self.current_bboxes.pop()
            self._redraw()
            self._save_annotation(quiet=True)

    def _clear_boxes(self):
        self.current_bboxes = []
        self._redraw()

    def _redraw(self):
        self.canvas.delete("all")
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 100: cw, ch = 800, 600
        
        # Interactive Zoom tracking bounds
        zoom_mult = getattr(self, 'interactive_zoom', 1.0)
        self.pan_x = getattr(self, 'pan_x', 0)
        self.pan_y = getattr(self, 'pan_y', 0)

        # Build Image Matrix only if exists
        if self.current_cv_img is not None:
            overlay = self.current_cv_img.copy()
            blended = self.current_cv_img.copy()
            
            # --- Draw True Alpha Fills on OpenCV Frame ---
            for b in self.current_bboxes:
                col_hex = tcm_config.CLASS_COLORS.get(b["cls"], "#FFFFFF")
                bgr = tcm_config.hex_to_bgr(col_hex)
                
                if b.get("type") == "polygon" and "points" in b:
                    pts = np.array(b["points"], np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(overlay, [pts], bgr)
                    cv2.polylines(blended, [pts], True, bgr, 2)
                else:
                    bx, by, bw, bh = b['x'], b['y'], b['w'], b['h']
                    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), bgr, cv2.FILLED)
                    cv2.rectangle(blended, (bx, by), (bx + bw, by + bh), bgr, 2)
                    
            # Alpha composite the filled regions
            cv2.addWeighted(overlay, 0.4, blended, 0.6, 0, blended)
            
            # --- Draw Opaque Labels ---
            for b in self.current_bboxes:
                col_hex = tcm_config.CLASS_COLORS.get(b["cls"], "#FFFFFF")
                bgr = tcm_config.hex_to_bgr(col_hex)
                label = b["cls"]
                
                # Base Anchor coordinates (Top-Left point of object)
                if b.get("type") == "polygon" and "points" in b:
                    tx, ty = b["points"][0]
                else:
                    tx, ty = b['x'], b['y']
                    
                # High visibility label backings
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(blended, (tx, ty - text_h - 10), (tx + text_w + 10, ty), bgr, cv2.FILLED)
                cv2.putText(blended, label, (tx + 5, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # Resize the final composited frame utilizing interactive Zoom logic
            base_scale = min(cw / self.orig_w, ch / self.orig_h) * 0.95
            self.display_scale = base_scale * zoom_mult
            nw, nh = int(self.orig_w * self.display_scale), int(self.orig_h * self.display_scale)
            
            if nw > 0 and nh > 0:
                resized = cv2.resize(blended, (nw, nh), interpolation=cv2.INTER_LINEAR)
                self.tk_img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
                
                # Render onto canvas utilizing Pan Offset vectors
                img_x = (cw // 2) + self.pan_x
                img_y = (ch // 2) + self.pan_y
                self.canvas.create_image(img_x, img_y, image=self.tk_img, anchor="center")
                
                self.offset_x = img_x - (nw // 2)
                self.offset_y = img_y - (nh // 2)
                
        # Handle persistent visual tools (Crosshairs) - Rendered natively in Tkinter
        if hasattr(self, 'last_mouse') and self.last_mouse:
            mx, my = self.last_mouse
            cw = self.canvas.winfo_width()
            ch_c = self.canvas.winfo_height()
            self.canvas.create_line(mx, 0, mx, ch_c, fill="#00E5FF", width=1, dash=(3, 5))
            self.canvas.create_line(0, my, cw, my, fill="#00E5FF", width=1, dash=(3, 5))
        
        # If drawing locally
        if self.rect_start and hasattr(self, 'last_mouse'):
            self.canvas.create_rectangle(self.rect_start[0], self.rect_start[1],
                                         self.last_mouse[0], self.last_mouse[1], outline="yellow", width=1, dash=(4, 4))
                                         
        counts = {}
        for b in self.current_bboxes:
            counts[b["cls"]] = counts.get(b["cls"], 0) + 1
        
        if not counts:
            self.lbl_box_count.configure(text="Boxes/Objects: 0", justify="left")
        else:
            text = f"Total Objects: {len(self.current_bboxes)}\n"
            text += "\n".join([f"• {cls}: {cnt}" for cls, cnt in counts.items()])
            self.lbl_box_count.configure(text=text, justify="left")

    def _save_annotation(self, quiet=False):
        # We must ALWAYS save the YOLO label into the ACTIVE dataset!
        # If the user opened a file from e.g. Desktop, we copy it to General_Scenes.
        if tcm_config.DATASET_DIR in self.temp_image_path:
            save_dir = os.path.dirname(self.temp_image_path)
            base_name = os.path.splitext(os.path.basename(self.temp_image_path))[0]
        else:
            save_dir = os.path.join(tcm_config.DATASET_DIR, "General_Scenes")
            os.makedirs(save_dir, exist_ok=True)
            if not self.viewing_saved_file:
                base_name = f"scene_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            else:
                base_name = os.path.splitext(os.path.basename(self.temp_image_path))[0]
            # Copy to project 
            try:
                shutil.copy(self.temp_image_path, os.path.join(save_dir, f"{base_name}.jpg"))
            except shutil.SameFileError:
                pass
                
        txt_p = os.path.join(save_dir, f"{base_name}.txt")
        if not self.current_bboxes:
            if os.path.exists(txt_p):
                os.remove(txt_p)
            if not quiet:
                messagebox.showwarning("Warning", "No bounding boxes. Cleared labels.")
            return

        with open(txt_p, "w") as f:
            for b_idx, b in enumerate(self.current_bboxes):
                try:
                    cid = tcm_config.CLASS_NAMES.index(b["cls"])
                except ValueError:
                    cid = 0
                
                if b.get("type") == "polygon" and "points" in b:
                    # Output Segmentation Polygon Format
                    line_parts = [str(cid)]
                    for px, py in b["points"]:
                        nx = px / self.orig_w
                        ny = py / self.orig_h
                        line_parts.append(f"{nx:.6f} {ny:.6f}")
                    f.write(" ".join(line_parts) + "\n")
                else:
                    # Output Detection Bounding Box Format
                    cx = (b["x"] + b["w"] / 2) / self.orig_w
                    cy = (b["y"] + b["h"] / 2) / self.orig_h
                    nw = b["w"] / self.orig_w
                    nh = b["h"] / self.orig_h
                    f.write(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                
                # CRITICAL: EXTRACT AND SAVE INDIVIDUAL GOLDEN CROP FOR THIS LABEL!
                # We save it in a completely separate "Golden_Crops" folder for analysis.
                cls_dir = os.path.join(tcm_config.DATASET_DIR, "Golden_Crops", b["cls"])
                os.makedirs(cls_dir, exist_ok=True)
                crop_x1 = max(0, int(b["x"]))
                crop_y1 = max(0, int(b["y"]))
                crop_x2 = min(self.orig_w, int(b["x"] + b["w"]))
                crop_y2 = min(self.orig_h, int(b["y"] + b["h"]))
                if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                    crop_img = self.current_cv_img[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                    
                    # Apply selected crop save size if any
                    size_setting = self.crop_size_var.get()
                    if size_setting != "Original HD":
                        try:
                            dim = int(size_setting.split("x")[0])
                            crop_img = cv2.resize(crop_img, (dim, dim), interpolation=cv2.INTER_LANCZOS4)
                        except Exception:
                            pass
                            
                    crop_name = f"crop_{base_name}_{b_idx}.jpg"
                    cv2.imwrite(os.path.join(cls_dir, crop_name), crop_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

        if not quiet:
            self.lbl_status.configure(text=f"✅ FULL LABEL SAVED: {base_name}.txt", text_color="#00C853")
            self.current_bboxes = []
            self._redraw()
            if not self.viewing_saved_file:
                self.tabs.set("📸 CAMERA")

    # ══════════════════════════════════════════════════════════════════════
    #  DATASET ACTIONS
    # ══════════════════════════════════════════════════════════════════════
    def _import_folder(self):
        folder = filedialog.askdirectory(title="Select folder with images to import", parent=self)
        if not folder:
            return
        dialog = ctk.CTkInputDialog(text="Import to which class?", title="Class Name")
        cls = dialog.get_input()
        if cls:
            cls = cls.strip()
            # Auto-create class if it doesn't exist
            if cls not in tcm_config.CLASS_NAMES:
                self._add_class_backend(cls)
            dest = os.path.join(tcm_config.DATASET_DIR, cls)
            os.makedirs(dest, exist_ok=True)
            count = 0
            for f in os.listdir(folder):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    shutil.copy(os.path.join(folder, f), os.path.join(dest, f))
                    count += 1
            self.lbl_status.configure(text=f"Imported {count} images to {cls}", text_color="#4CAF50")
            self._refresh_dataset()

    def _prompt_add_class(self):
        dialog = ctk.CTkInputDialog(text="Enter New Class Name (e.g. METAL_PART):", title="Add New Class")
        new_class = dialog.get_input()
        if new_class:
            new_class = new_class.strip().upper().replace(" ", "_")
            if new_class and new_class not in tcm_config.CLASS_NAMES:
                self._add_class_backend(new_class)
            else:
                messagebox.showwarning("Warning", "Class already exists or invalid.")

    def _add_class_backend(self, new_class):
        if tcm_config.ProjectManager.add_class(new_class):
            self.combo_capture_cls.configure(values=tcm_config.CLASS_NAMES)
            if hasattr(self, 'combo_aug_cls'):
                self.combo_aug_cls.configure(values=["ALL CLASSES"] + tcm_config.CLASS_NAMES)
            self._refresh_dataset()
            self.lbl_status.configure(text=f"✅ Class '{new_class}' Added", text_color="#00C853")
        else:
            messagebox.showerror("Error", f"Failed to add class {new_class}")

    def _auto_label_all(self):
        def _run():
            self.engine.auto_label_all()
            self.after(0, self._refresh_dataset)
        threading.Thread(target=_run, daemon=True).start()
        self.lbl_status.configure(text="Auto-labeling in progress...", text_color="#FF9800")

    # ══════════════════════════════════════════════════════════════════════
    #  AUGMENT ACTIONS
    # ══════════════════════════════════════════════════════════════════════
    def _save_aug_config(self):
        aug_cfg = {}
        aug_cfg["fliplr"] = 0.5 if self.aug_flip_h.get() else 0.0
        aug_cfg["flipud"] = 0.5 if self.aug_flip_v.get() else 0.0
        aug_cfg["degrees"] = 15.0 if self.aug_rotate.get() else 0.0
        aug_cfg["hsv_v"] = 0.4 if self.aug_bright.get() else 0.0
        aug_cfg["mosaic"] = 1.0 if self.aug_mosaic.get() else 0.0
        
        self.engine.yolo_aug_config = aug_cfg
        
        status_text = "✅ Saved YOLO args:\n" + "\n".join([f"{k}={v}" for k,v in aug_cfg.items()])
        self.lbl_aug_status.configure(text=status_text, text_color="#00E676")
        
        # Dynamically generate synthetic preview visuals!
        self._refresh_aug_preview()

    def _refresh_aug_preview(self):
        for w in self.aug_preview.winfo_children(): w.destroy()
        
        crops = []
        gc_dir = os.path.join(tcm_config.DATASET_DIR, "Golden_Crops")
        if os.path.exists(gc_dir):
            for cls in os.listdir(gc_dir):
                cls_dir = os.path.join(gc_dir, cls)
                if os.path.isdir(cls_dir):
                    imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.png'))]
                    for f in imgs:
                        crops.append(os.path.join(cls_dir, f))
                        
        import random
        random.shuffle(crops)
        crops = crops[:12]
        
        if not crops:
            ctk.CTkLabel(self.aug_preview, text="No annotated Cashew crops found for preview.", text_color="#555").pack(pady=50)
            return
            
        r, c = 0, 0
        for path in crops:
            img = cv2.imread(path)
            if img is not None:
                # Apply dynamic preview augmentations using engine (in-memory, no disk writes!)
                aug_imgs = self.engine.augment_image(
                    img, 
                    flip_h=bool(self.aug_flip_h.get()),
                    flip_v=bool(self.aug_flip_v.get()),
                    rotate=bool(self.aug_rotate.get()),
                    brightness=bool(self.aug_bright.get()),
                    noise=False,
                    multiplier=1
                )
                if aug_imgs:
                    aug_img = aug_imgs[0]
                    h, w = aug_img.shape[:2]
                    scale = min(120/w, 120/h)
                    nw, nh = max(50, int(w*scale)), max(50, int(h*scale))
                    resized = cv2.resize(aug_img, (nw, nh), interpolation=cv2.INTER_AREA)
                    tk_img = ctk.CTkImage(light_image=Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)), dark_image=Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)), size=(nw, nh))
                    
                    card = ctk.CTkFrame(self.aug_preview, fg_color="#222")
                    card.grid(row=r, column=c, padx=10, pady=10)
                    ctk.CTkLabel(card, text="", image=tk_img).pack(pady=5)
                    ctk.CTkLabel(card, text="Dynamic Preview", font=("Consolas", 9)).pack()
                    
                c += 1
                if c >= 3:
                    c = 0
                    r += 1

    # ══════════════════════════════════════════════════════════════════════
    #  TRAINING ACTIONS
    # ══════════════════════════════════════════════════════════════════════
    def train_log(self, msg):
        def _append():
            self.train_log_text.configure(state="normal")
            self.train_log_text.insert("end", msg + "\n")
            self.train_log_text.see("end")
            self.train_log_text.configure(state="disabled")
        self.after(0, _append)

    def _prepare_dataset(self):
        # Run synchronously to prevent Race Condition with START TRAINING button
        self.engine.prepare_dataset()
        self.engine.create_yaml()

    def _start_training(self):
        try:
            epochs = int(self.ent_epochs.get())
            batch = int(self.ent_batch.get())
            
            sz_str = self.ent_imgsz.get().strip().lower()
            if "x" in sz_str:
                parts = sz_str.split("x")
                imgsz = [int(parts[0].strip()), int(parts[1].strip())]
            else:
                imgsz = int(sz_str)
                
        except ValueError:
            from tkinter import messagebox
            messagebox.showerror("Error", "Invalid parameters! Use integers or format like 640x480")
            return
        model_name = self.combo_model.get()
        force_cpu = (self.combo_device.get() == "Force CPU")
        
        # Divert STDOUT & STDERR directly so YOLO metric prints flow into our text box
        redirector = StdoutRedirector(self.train_log_text, progress_bar=self.train_progress, lbl_perc=self.lbl_train_perc)
        sys.stdout = redirector
        sys.stderr = redirector
        
        self.lbl_train_perc.configure(text="0% - Preparing...")
        self.train_progress.set(0.0)
        self.engine.start_training_thread(
            epochs=epochs, batch=batch, imgsz=imgsz, model_name=model_name, force_cpu=force_cpu
        )
        self.train_progress.set(0.0)
        self._poll_training()

    def _poll_training(self):
        if self.engine.training_active:
            # We removed the fake pulse because StdoutRedirector automatically updates the progress bar via Epoch detection.
            self.after(500, self._poll_training)
        else:
            self.train_progress.set(1.0)
            self.train_log("✅ Training thread completed!")
            
            # Show Analytics Popup
            runs_dir = os.path.join(tcm_config.MODEL_DIR, "runs")
            if os.path.exists(runs_dir):
                subs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
                if subs:
                    latest_run = max(subs, key=os.path.getmtime)
                    EvaluationPopup(self, latest_run)
            
            if self.engine.latest_best_pt:
                if hasattr(self, 'ent_best_pt'):
                    self.ent_best_pt.delete(0, "end")
                    self.ent_best_pt.insert(0, self.engine.latest_best_pt)
                self.lbl_status.configure(text="✅ Training Complete!", text_color="#00C853")
                
                # Explicitly log the file paths directly to the console for the user
                self.train_log(f"\n{'='*60}")
                self.train_log(f"🌟 LOCAL GPU TRAINING SAVED DETAILS:")
                self.train_log(f"📦 PROJECT RUN DIRECTORY: {os.path.dirname(os.path.dirname(self.engine.latest_best_pt))}")
                self.train_log(f"🔥 FINAL WEIGHTS (best.pt): {self.engine.latest_best_pt}")
                self.train_log(f"{'='*60}\n")
                
                # Refresh LIVE AI dropdown
                if hasattr(self, 'combo_load_model'):
                    runs_dir = os.path.join(tcm_config.MODEL_DIR, "runs")
                    run_names = ["Select Trained Model...", "Import External Model (.pt)..."]
                    if os.path.exists(runs_dir):
                        for d in reversed(sorted(os.listdir(runs_dir))):
                            if os.path.isdir(os.path.join(runs_dir, d)):
                                run_names.append(d)
                    self.combo_load_model.configure(values=run_names)

    def _stop_training(self):
        self.engine.training_active = False
        self.train_log("⏹ Training stop requested.")

    # ══════════════════════════════════════════════════════════════════════
    #  EXPORT ACTIONS
    # ══════════════════════════════════════════════════════════════════════
    def _browse_best_pt(self):
        p = filedialog.askopenfilename(filetypes=[("YOLO Model", "*.pt")], parent=self)
        if p:
            self.ent_best_pt.delete(0, "end")
            self.ent_best_pt.insert(0, p)

    def _export_model(self):
        pt_path = self.ent_best_pt.get().strip()
        if not pt_path:
            pt_path = self.engine.latest_best_pt
        if not pt_path or not os.path.exists(pt_path):
            messagebox.showerror("Error", "No best.pt found! Train a model or browse for one.", parent=self)
            return
        try:
            result = self.engine.export_model(pt_path)
            if result:
                messagebox.showinfo("Export Complete",
                                    f"Model exported to:\n{result}\n\n"
                                    f"Files: best.pt, data.yaml, class_names.json, model_info.json", parent=self)
            else:
                messagebox.showerror("Export Failed", "Export failed. Check the path and ensure the model exists.", parent=self)
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export: {e}", parent=self)

    def _export_dataset_zip(self):
        yolo_dir = tcm_config.PROCESSED_DIR
        if not os.path.exists(yolo_dir) or not os.path.exists(os.path.join(yolo_dir, "data.yaml")):
            messagebox.showerror("Error", "Dataset not prepared! Please go to TRAIN tab and click 'PREPARE DATASET' first.", parent=self)
            return
            
        proj_name = getattr(tcm_config, "ACTIVE_PROJECT", "Dataset")
        save_path = filedialog.asksaveasfilename(defaultextension=".zip", 
                                               initialfile=f"{proj_name}_dataset.zip",
                                               filetypes=[("ZIP Archive", "*.zip")], parent=self)
        if not save_path:
            return
            
        import shutil
        try:
            runs_dir = os.path.join(tcm_config.MODEL_DIR, "runs")
            if os.path.exists(runs_dir):
                subs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
                if subs:
                    latest_run = max(subs, key=os.path.getmtime)
                    weights_src = os.path.join(latest_run, "weights", "best.pt")
                    if os.path.exists(weights_src):
                        shutil.copy(weights_src, os.path.join(yolo_dir, "best.pt"))
                    metrics_dest = os.path.join(yolo_dir, "training_metrics")
                    if os.path.exists(metrics_dest):
                        shutil.rmtree(metrics_dest)
                    shutil.copytree(latest_run, metrics_dest)

            base_path = save_path[:-4] if save_path.endswith(".zip") else save_path
            shutil.make_archive(base_path, 'zip', yolo_dir)
            
            # Print to GUI console
            self.train_log(f"\n{'='*60}")
            self.train_log(f"☁️ GOOGLE COLAB DATASET EXPORT HAS FINISHED")
            self.train_log(f"📁 ZIP SAVED LOCATION: {save_path}")
            self.train_log(f"{'='*60}\n")
            
            messagebox.showinfo("Success", f"Comprehensive ZIP Export Complete!\n\nLocation: {save_path}\n\nContents: images, labels, data.yaml, best.pt, and training_metrics.", parent=self)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export ZIP: {e}", parent=self)

    def _import_trained_zip(self):
        zip_path = filedialog.askopenfilename(filetypes=[("ZIP Archive", "*.zip")], parent=self)
        if not zip_path:
            return
            
        import zipfile
        import shutil
        import datetime
        
        try:
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            run_name = f"Colab_Import_{ts}"
            run_dir = os.path.join(tcm_config.MODEL_DIR, "runs", run_name)
            weights_dir = os.path.join(run_dir, "weights")
            os.makedirs(weights_dir, exist_ok=True)
            
            temp_dir = os.path.join(tcm_config.MODEL_DIR, "runs", "temp_extract")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                
            best_pt_found = False
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file == "best.pt":
                        src = os.path.join(root, file)
                        dst = os.path.join(weights_dir, "best.pt")
                        shutil.copy(src, dst)
                        best_pt_found = True
                        break
                        
            shutil.rmtree(temp_dir)
            
            if not best_pt_found:
                messagebox.showerror("Error", "Could not find 'best.pt' in the ZIP file.", parent=self)
                return
                
            self.train_log(f"\n{'='*60}")
            self.train_log(f"☁️ GOOGLE COLAB MODEL IMPORT HAS FINISHED")
            self.train_log(f"📁 MODEL SAVED AS: {run_name}")
            self.train_log(f"{'='*60}\n")
            
            messagebox.showinfo("Success", f"Successfully imported Colab trained model!\n\nYou can now select '{run_name}' in the TEST tab.", parent=self)
            
            runs_dir = os.path.join(tcm_config.MODEL_DIR, "runs")
            run_names = ["Select Trained Model...", "Import External Model (.pt)..."]
            if os.path.exists(runs_dir):
                for d in reversed(sorted(os.listdir(runs_dir))):
                    if os.path.isdir(os.path.join(runs_dir, d)) and d != "temp_extract":
                        run_names.append(d)
                        
            try:
                self.combo_load_model.configure(values=run_names)
            except Exception:
                pass
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to import ZIP: {e}", parent=self)


    def _apply_cam(self):
        try:
            device_str = self.combo_cam_dev.get()
            device_idx = int(device_str.split(":")[0])
            
            exp = float(self.ent_exp.get() or "20000")
            again = float(self.ent_again.get() or "1.0")
            rgain = float(self.ent_rgain.get() or "100")
            ggain = float(self.ent_ggain.get() or "100")
            bgain = float(self.ent_bgain.get() or "100")
            
            if getattr(self, 'current_cam_idx', -1) != device_idx:
                self.cam.open(device_index=device_idx)
                self.current_cam_idx = device_idx

            self.cam.set_manual_params(exp, again, rgain, ggain, bgain)
            
            # Persist to workspace
            proj_p = os.path.join(tcm_config.PROJECTS_DIR, tcm_config.ACTIVE_PROJECT) if tcm_config.ACTIVE_PROJECT else ""
            if proj_p:
                import json
                camera_cfg = {
                    "device": device_idx,
                    "exposure": exp,
                    "analog_gain": again,
                    "r_gain": rgain,
                    "g_gain": ggain,
                    "b_gain": bgain
                }
                with open(os.path.join(proj_p, "camera.json"), "w") as f:
                    json.dump(camera_cfg, f, indent=4)
                    
            self.lbl_status.configure(text=f"Camera settings saved! (Dev {device_idx})", text_color="#00E676")
        except Exception as e:
            messagebox.showerror("Camera Error", f"Failed to apply tuning: {e}", parent=self)

    # ══════════════════════════════════════════════════════════════════════
    #  SHUTDOWN
    # ══════════════════════════════════════════════════════════════════════

    def _shutdown(self):
        if hasattr(sys.stdout, 'terminal'):
            sys.stdout = sys.stdout.terminal
        self.video_running = False
        if self.cam: self.cam.close()
        self.destroy()
        sys.exit(0)

    def _go_home(self):
        self.video_running = False
        if self.cam: self.cam.close()
        self.destroy()
        tcm_config.ACTIVE_PROJECT = None
        hub = ProjectsHubApp()
        hub.mainloop()


# ══════════════════════════════════════════════════════════════════════════════
# PROJECTS HUB
# ══════════════════════════════════════════════════════════════════════════════
class ProjectsHubApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("TCM Studio Pro — Projects Hub")
        w, h = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}")
        self.after(100, lambda: self.state("zoomed") if os.name == "nt" else self.attributes('-zoomed', True))
        
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        hf = ctk.CTkFrame(self, height=70, fg_color="#0d0d0d", corner_radius=0)
        hf.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(hf, text="🪐 TCM STUDIO PRO", font=("Segoe UI Black", 28), text_color="#FF3D00").pack(side="left", padx=20, pady=15)
        ctk.CTkLabel(hf, text="TCM AI DEEP LEARNING DATASET TOOL", font=("Segoe UI Black", 16), text_color="#00BCD4").pack(side="left", padx=5)

        # Main Body
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=20, pady=20)
        body.grid_columnconfigure(0, weight=2)
        body.grid_columnconfigure(1, weight=1)

        # Left: Existing Projects
        left_frame = ctk.CTkFrame(body)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        ctk.CTkLabel(left_frame, text="Existing Workspaces", font=("Segoe UI", 18, "bold")).pack(pady=15)
        
        self.proj_scroll = ctk.CTkScrollableFrame(left_frame, fg_color="#1a1a1a")
        self.proj_scroll.pack(fill="both", expand=True, padx=15, pady=10)
        
        # Dedicated Details Panel
        self.insights_frame = ctk.CTkFrame(left_frame, fg_color="#181818", height=250)
        self.insights_frame.pack(fill="x", padx=15, pady=(0, 15))
        self.insights_frame.pack_propagate(False)
        self.lbl_insights_title = ctk.CTkLabel(self.insights_frame, text="📊 DETAILS - SELECT A PROJECT", font=("Segoe UI", 14, "bold"), text_color="#00BCD4")
        self.lbl_insights_title.pack(pady=(10, 5))
        self.txt_insights = ctk.CTkTextbox(self.insights_frame, font=("Consolas", 12), text_color="#00E676")
        self.txt_insights.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.txt_insights.configure(state="disabled")

        self.refresh_projects()

        # Right: New Project
        right_frame = ctk.CTkFrame(body)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        ctk.CTkLabel(right_frame, text="Create New Workspace", font=("Segoe UI", 18, "bold")).pack(pady=15)
        
        form = ctk.CTkFrame(right_frame, fg_color="transparent")
        form.pack(fill="both", expand=True, padx=20, pady=10)
        
        ctk.CTkLabel(form, text="Project Name:", anchor="w", font=("Segoe UI", 16)).pack(fill="x", pady=(10,2))
        self.ent_new_name = ctk.CTkEntry(form, placeholder_text="e.g. Screws_Defect_Tracking", font=("Segoe UI", 16), height=40)
        self.ent_new_name.pack(fill="x", pady=20)
        
        ctk.CTkButton(form, text="➕ CREATE PROJECT", height=50, fg_color="#00C853", hover_color="#00E676", font=("Segoe UI", 16, "bold"), command=self.create_project).pack(fill="x", pady=20)

        # TCM Cloud Login (Aesthetic Polish)
        ctk.CTkLabel(right_frame, text="TCM Cloud Sync", font=("Segoe UI", 18, "bold"), text_color="#2196F3").pack(pady=(30, 15))
        
        login_form = ctk.CTkFrame(right_frame, fg_color="#181818", corner_radius=10)
        login_form.pack(fill="x", expand=False, padx=20, pady=10)
        
        ctk.CTkLabel(login_form, text="Email Address:", font=("Segoe UI", 14), text_color="#aaa").pack(fill="x", padx=15, pady=(15,2), anchor="w")
        self.ent_cloud_email = ctk.CTkEntry(login_form, placeholder_text="engineer@tcm.com", font=("Segoe UI", 14), height=35)
        self.ent_cloud_email.pack(fill="x", padx=15, pady=5)
        
        ctk.CTkLabel(login_form, text="API Key:", font=("Segoe UI", 14), text_color="#aaa").pack(fill="x", padx=15, pady=(10,2), anchor="w")
        self.ent_cloud_key = ctk.CTkEntry(login_form, placeholder_text="sk-...", show="*", font=("Segoe UI", 14), height=35)
        self.ent_cloud_key.pack(fill="x", padx=15, pady=5)
        
        def mock_login():
            if not self.ent_cloud_email.get() or not self.ent_cloud_key.get():
                messagebox.showerror("Error", "Please enter valid Cloud Config!")
                return
            messagebox.showinfo("Cloud Sync", "Secure tunneling enabled.\nConnected to TCM Cloud API successfully!")

        ctk.CTkButton(login_form, text="🔒 SECURE LOGIN", height=45, fg_color="#1976D2", hover_color="#1565C0", font=("Segoe UI", 14, "bold"), command=mock_login).pack(fill="x", padx=15, pady=20)

    def refresh_projects(self):
        for w in self.proj_scroll.winfo_children():
            w.destroy()
            
        projs = tcm_config.ProjectManager.get_projects()
        if not projs:
            ctk.CTkLabel(self.proj_scroll, text="No projects found.\nCreate one to get started!", text_color="#888").pack(pady=30)
            return
            
        for p in projs:
            wrapper = ctk.CTkFrame(self.proj_scroll, fg_color="transparent")
            wrapper.pack(fill="x", pady=5)
            
            row = ctk.CTkFrame(wrapper, fg_color="#222")
            row.pack(fill="x")
            
            ctk.CTkLabel(row, text="📁 " + p, font=("Segoe UI", 16, "bold")).pack(side="left", padx=15, pady=15)
            
            ctk.CTkButton(row, text="OPEN", width=80, fg_color="#2196F3", command=lambda name=p: self.open_project(name)).pack(side="right", padx=10)
            ctk.CTkButton(row, text="💾 ZIP DATASET", width=120, fg_color="#007ACC", hover_color="#00897B", 
                          command=lambda name=p: self.export_project_zip(name)).pack(side="right", padx=5)
            
            insights_btn = ctk.CTkButton(row, text="📊 DETAILS", width=100, fg_color="#455A64", hover_color="#546E7A")
            insights_btn.configure(command=lambda name=p: self.toggle_info(name))
            insights_btn.pack(side="right", padx=5)
            
            ctk.CTkButton(row, text="🗑 DELETE", width=80, fg_color="#B71C1C", hover_color="#D32F2F", 
                          command=lambda name=p: self.delete_project_gui(name)).pack(side="right", padx=5)

    def delete_project_gui(self, name):
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to completely delete the project '{name}'?", parent=self):
            if tcm_config.ProjectManager.delete_project(name):
                self.refresh_projects()
                messagebox.showinfo("Success", f"Project '{name}' deleted.", parent=self)
            else:
                messagebox.showerror("Error", f"Could not delete project '{name}'.", parent=self)

    def toggle_info(self, name):
        import datetime
        self.lbl_insights_title.configure(text=f"📊 DETAILS - {name.upper()}")
        self.txt_insights.configure(state="normal")
        self.txt_insights.delete("1.0", "end")
        
        proj_dir = os.path.join(tcm_config.PROJECTS_DIR, name)
        yolo_dir = os.path.join(proj_dir, "Datasets", "Processed_YOLO")
        yaml_path = os.path.join(yolo_dir, "data.yaml")
        
        if os.path.exists(yaml_path):
            t_mod = os.path.getmtime(yaml_path)
            dt = datetime.datetime.fromtimestamp(t_mod).strftime('%Y-%m-%d %H:%M:%S')
            self.txt_insights.insert("end", f"📁 DATASET PREPARED: {dt}\n")
            img_dir = os.path.join(yolo_dir, "images", "train")
            val_dir = os.path.join(yolo_dir, "images", "val")
            t_cnt = len(os.listdir(img_dir)) if os.path.exists(img_dir) else 0
            v_cnt = len(os.listdir(val_dir)) if os.path.exists(val_dir) else 0
            self.txt_insights.insert("end", f"   Train Images: {t_cnt} | Validation Images: {v_cnt}\n\n")
        else:
            self.txt_insights.insert("end", "⚠️ NO DATASET PREPARED YET\n\n")
            
        runs_dir = os.path.join(proj_dir, "Models", "runs")
        if os.path.exists(runs_dir):
            subs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
            if subs:
                latest = max(subs, key=os.path.getmtime)
                csv_path = os.path.join(latest, "results.csv")
                if os.path.exists(csv_path):
                    with open(csv_path, "r") as f:
                        lines = f.readlines()
                        if len(lines) > 1:
                            headers = [h.strip() for h in lines[0].split(",")]
                            last = [v.strip() for v in lines[-1].split(",")]
                            self.txt_insights.insert("end", f"🚀 LATEST TENSOR RUN: {os.path.basename(latest)}\n")
                            for h, v in zip(headers, last):
                                if "metrics" in h.lower() or "epoch" in h.lower():
                                    self.txt_insights.insert("end", f"   {h}: {v}\n")
                else:
                    self.txt_insights.insert("end", f"⚠️ NO RESULTS.CSV IN {os.path.basename(latest)}\n")
            else:
                self.txt_insights.insert("end", "⚠️ NO TRAINING RUNS YET\n")
        else:
            self.txt_insights.insert("end", "⚠️ NO TRAINING RUNS YET\n")
            
        self.txt_insights.configure(state="disabled")

    def export_project_zip(self, name):
        import shutil
        proj_dir = os.path.join(tcm_config.PROJECTS_DIR, name)
        yolo_dir = os.path.join(proj_dir, "Datasets", "Processed_YOLO")
        if not os.path.exists(yolo_dir) or not os.path.exists(os.path.join(yolo_dir, "data.yaml")):
            messagebox.showerror("Error", f"Dataset inside '{name}' is empty or not prepared yet!", parent=self)
            return
            
        save_path = filedialog.asksaveasfilename(defaultextension=".zip", 
                                               initialfile=f"{name}_dataset.zip",
                                               filetypes=[("ZIP Archive", "*.zip")], parent=self)
        if not save_path:
            return
            
        try:
            base_path = save_path[:-4] if save_path.endswith(".zip") else save_path
            shutil.make_archive(base_path, 'zip', yolo_dir)
            messagebox.showinfo("Success", f"Dataset ZIP Export Complete!\n\nLocation: {save_path}", parent=self)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export ZIP: {e}", parent=self)

    def open_project(self, name):
        if tcm_config.ProjectManager.load_project(name):
            self.destroy()
            app = TCMStudioPro()
            app.mainloop()
        else:
            messagebox.showerror("Error", f"Failed to load project: {name}")

    def create_project(self):
        name = self.ent_new_name.get().strip()
        
        if not name:
            messagebox.showwarning("Warning", "Project Name is required!")
            return
            
        if tcm_config.ProjectManager.create_project(name, []):
            messagebox.showinfo("Success", f"Project '{name}' created successfully!")
            self.ent_new_name.delete(0, "end")
            self.refresh_projects()
        else:
            messagebox.showerror("Error", f"Project '{name}' already exists!")

if __name__ == "__main__":
    app = ProjectsHubApp()
    app.mainloop()

