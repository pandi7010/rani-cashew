import os
import shutil
import time
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import torch
import torch.nn.functional as F
from torchvision import models, transforms

class CashewColorCNN:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        candidates = [
            model_path,
            "saved_models/cashew_color_cnn_new.pth",
            "saved_models/cashew_color_cnn.pth",
            "../CNN_Experiment/cashew_color_cnn.pth"
        ]
        
        selected_path = None
        for path in candidates:
            if path and os.path.exists(path):
                selected_path = path
                break
                
        if not selected_path:
            raise FileNotFoundError("No trained CNN model found in saved_models/")
            
        print(f"[AUTO-LABELER] Loading CNN Model from: {selected_path}")
        checkpoint = torch.load(selected_path, map_location=self.device)
        self.class_names = checkpoint['class_names']
        
        # Initialize model architecture
        self.model = models.mobilenet_v2(pretrained=False)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = torch.nn.Linear(num_ftrs, len(self.class_names))
        
        # Load weights
        state_dict = checkpoint.get('model_state_dict') or checkpoint.get('model_state')
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Transforms for inference
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        print(f"CNN Model loaded with classes: {self.class_names}")

    def predict_with_confidence(self, bgr_image_crop):
        rgb_image = cv2.cvtColor(bgr_image_crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_image)
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, 1)
            
        raw_pred = self.class_names[preds[0]]
        norm_pred = raw_pred.upper().replace("GOOD_CASHEW", "GOOD").replace(" ", "_")
        return norm_pred, conf.item()

class AIAutoLabeler(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Auto-Labeler (Fast Review)")
        self.geometry("1150x700")
        self.configure(bg="#2b2b2b")
        
        # Style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("TButton", font=("Roboto", 12), padding=10)
        self.style.configure("Agree.TButton", font=("Roboto", 14, "bold"), background="#4CAF50", foreground="white")
        self.style.map("Agree.TButton", background=[("active", "#45a049")])
        self.style.configure("Class.TButton", font=("Roboto", 10))
        
        self.colors = {"bg": "#2b2b2b", "fg": "#ffffff", "accent": "#2196f3"}
        
        # Paths
        self.golden_dir = "cnn_final/golden_crops"
        self.dataset_crops_dir = "dataset"
        os.makedirs(self.dataset_crops_dir, exist_ok=True)
        
        # Model (Optional for manual sorting)
        try:
            self.cnn = CashewColorCNN()
        except Exception as e:
            self.cnn = None
            print(f"[WARN] Failed to load CNN model: {e}. Running without AI model.")
            
        # Discover existing classes strictly from dataset directory (no auto-creation)
        if os.path.exists(self.dataset_crops_dir):
            self.available_classes = sorted([d for d in os.listdir(self.dataset_crops_dir) if os.path.isdir(os.path.join(self.dataset_crops_dir, d))])
        else:
            self.available_classes = ["GOOD", "SW", "BROWN", "REJECTION"]
            for c in self.available_classes:
                os.makedirs(os.path.join(self.dataset_crops_dir, c), exist_ok=True)
            
        # Load image list
        self.image_files = []
        self.load_image_list()
        
        self.current_img_path = None
        self.current_pred = None
        self.current_conf = 0.0
        self._class_check_vars = {}  # {class_name: tk.BooleanVar} for multi-select delete
        self._class_buttons = {}     # {class_name: tk.Button} for live count updates
        
        self.build_ui()
        self.load_next_image()
        
        # Bindings
        self.bind("<Return>", lambda e: self.agree_prediction())

    def get_class_counts(self):
        counts = {}
        total_annotated = 0
        if os.path.exists(self.dataset_crops_dir):
            for d in sorted(os.listdir(self.dataset_crops_dir)):
                cp = os.path.join(self.dataset_crops_dir, d)
                if os.path.isdir(cp):
                    cnt = len([f for f in os.listdir(cp) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    counts[d] = cnt
                    total_annotated += cnt
        return counts, total_annotated
        
    def load_image_list(self):
        if not os.path.exists(self.golden_dir):
            os.makedirs(self.golden_dir)
        self.image_files = [os.path.join(self.golden_dir, f) for f in os.listdir(self.golden_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_files.sort()
        
    def refresh_source_options(self):
        src_options = ["cnn_final/golden_crops"]
        if os.path.exists(self.dataset_crops_dir):
            for d in sorted(os.listdir(self.dataset_crops_dir)):
                full_d = os.path.join(self.dataset_crops_dir, d)
                if os.path.isdir(full_d):
                    src_options.append(full_d)
        if hasattr(self, 'combo_src'):
            self.combo_src['values'] = src_options
        return src_options

    def change_source_folder(self):
        new_dir = self.src_folder_var.get()
        if os.path.exists(new_dir):
            self.golden_dir = new_dir
            self.load_image_list()
            self.update_count()
            self.load_next_image()
            print(f"[AUTO-LABELER] Switched Source Folder to: {new_dir} ({len(self.image_files)} images)")
        
    def build_ui(self):
        # Top Bar
        top_bar = tk.Frame(self, bg="#212121", padx=10, pady=8)
        top_bar.pack(fill="x")
        
        counts, total_annotated = self.get_class_counts()
        
        count_frame = tk.Frame(top_bar, bg="#212121")
        count_frame.pack(side="left")
        
        self.lbl_remaining = tk.Label(count_frame, text=f"📥 Inbox (Unannotated): {len(self.image_files)}", bg="#212121", fg="#FFD54F", font=("Roboto", 11, "bold"))
        self.lbl_remaining.pack(side="left", padx=(0, 12))
        self.lbl_count = self.lbl_remaining  # backwards-compatible alias
        
        self.lbl_annotated = tk.Label(count_frame, text=f"✅ Annotated in Dataset: {total_annotated}", bg="#212121", fg="#81C784", font=("Roboto", 11, "bold"))
        self.lbl_annotated.pack(side="left", padx=(0, 15))
        
        # Source Folder Selector
        src_frame = tk.Frame(top_bar, bg="#212121")
        src_frame.pack(side="left", padx=(5, 0))
        
        tk.Label(src_frame, text="Source:", bg="#212121", fg="#E0E0E0", font=("Roboto", 10, "bold")).pack(side="left", padx=2)
        
        self.src_folder_var = tk.StringVar(value=self.golden_dir)
        src_options = self.refresh_source_options()
                    
        self.combo_src = ttk.Combobox(src_frame, textvariable=self.src_folder_var, values=src_options, width=24, state="readonly")
        self.combo_src.pack(side="left", padx=2)
        self.combo_src.bind("<<ComboboxSelected>>", lambda e: self.change_source_folder())
        
        # Open Dataset Folder in OS File Manager Button
        def open_dataset_folder():
            import subprocess
            os.makedirs(self.dataset_crops_dir, exist_ok=True)
            subprocess.Popen(['xdg-open', self.dataset_crops_dir])

        tk.Button(
            top_bar, text="📁 Open Dataset Folder",
            bg="#37474f", fg="white", font=("Roboto", 9, "bold"),
            relief="flat", cursor="hand2", padx=8, pady=4,
            command=open_dataset_folder
        ).pack(side="left", padx=(12, 0))
        
        # Auto Sort
        auto_frame = tk.Frame(top_bar, bg="#212121")
        auto_frame.pack(side="right")
        
        tk.Label(auto_frame, text="Threshold:", bg="#212121", fg="white", font=("Roboto", 10)).pack(side="left", padx=5)
        self.thresh_var = tk.StringVar(value="95")
        ttk.Combobox(auto_frame, textvariable=self.thresh_var, values=["90", "95", "98", "99"], width=4, state="readonly").pack(side="left", padx=5)
        ttk.Label(auto_frame, text="%").pack(side="left")
        
        ttk.Button(auto_frame, text="⚡ Bulk Auto-Sort (High Confidence)", command=self.bulk_auto_sort).pack(side="left", padx=10)
        
        # Main Content
        main_frame = tk.Frame(self, bg=self.colors["bg"])
        main_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Left: Image Viewer & Navigation
        left_frame = tk.Frame(main_frame, bg=self.colors["bg"])
        left_frame.pack(side="left", fill="both", expand=True)
        
        self.lbl_prediction = tk.Label(left_frame, text="AI Suggestion: N/A", font=("Roboto", 22, "bold"), bg=self.colors["bg"], fg="yellow")
        self.lbl_prediction.pack(pady=(0, 2))
        
        self.lbl_status = tk.Label(left_frame, text="Status: ⏳ Raw Crop (Unannotated)", font=("Roboto", 10, "italic"), bg=self.colors["bg"], fg="#FFB74D")
        self.lbl_status.pack(pady=(0, 5))
        
        self.lbl_image = tk.Label(left_frame, bg="black")
        self.lbl_image.pack(fill="both", expand=True)
        
        nav_frame = tk.Frame(left_frame, bg=self.colors["bg"])
        nav_frame.pack(pady=10)
        
        ttk.Button(nav_frame, text="⏪ Previous", command=self.prev_image).pack(side="left", padx=5)
        ttk.Button(nav_frame, text="🗑️ Delete Image", command=self.delete_image).pack(side="left", padx=5)
        ttk.Button(nav_frame, text="Next ⏩", command=self.skip_image).pack(side="left", padx=5)
        
        # Agree Button
        self.btn_agree = ttk.Button(left_frame, text="AGREE [Press Enter]", style="Agree.TButton", command=self.agree_prediction)
        self.btn_agree.pack(pady=(0, 10), ipadx=20, ipady=5)
        
        # Right: Class Buttons & Class Management (3-Column Layout with Distinct Colors)
        right_frame = tk.Frame(main_frame, bg=self.colors["bg"], width=460)
        right_frame.pack(side="right", fill="both", expand=False, padx=(15, 0))
        
        tk.Label(right_frame, text="Manual Correction (Click to Sort):", font=("Roboto", 13, "bold"), bg=self.colors["bg"], fg="#FFD700").pack(pady=(0, 5), anchor="w")
        
        # Scrollable canvas for class buttons in 3 columns
        canvas = tk.Canvas(right_frame, bg=self.colors["bg"], highlightthickness=0, width=440)
        scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas, bg=self.colors["bg"])
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw", width=420)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="top", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        class_mgmt_frame = tk.Frame(right_frame, bg=self.colors["bg"])
        class_mgmt_frame.pack(side="bottom", fill="x", pady=8)
        
        tk.Button(class_mgmt_frame, text="➕ Add Class", bg="#00897b", fg="white", font=("Roboto", 10, "bold"), relief="flat", cursor="hand2", padx=5, pady=6, command=self.add_class).pack(side="left", expand=True, fill="x", padx=2)
        tk.Button(class_mgmt_frame, text="✏️ Rename", bg="#f57c00", fg="white", font=("Roboto", 10, "bold"), relief="flat", cursor="hand2", padx=5, pady=6, command=self.rename_class).pack(side="left", expand=True, fill="x", padx=2)
        tk.Button(class_mgmt_frame, text="❌ Delete Selected", bg="#c62828", fg="white", font=("Roboto", 10, "bold"), relief="flat", cursor="hand2", padx=5, pady=6, command=self.delete_class).pack(side="left", expand=True, fill="x", padx=2)

        # Select All / Deselect All toggle
        sel_frame = tk.Frame(right_frame, bg=self.colors["bg"])
        sel_frame.pack(side="bottom", fill="x", pady=(0, 2))
        self._select_all_var = tk.BooleanVar(value=False)
        tk.Checkbutton(sel_frame, text="Select All", variable=self._select_all_var,
                       command=self._toggle_select_all, bg=self.colors["bg"], fg="#FFD700",
                       selectcolor="#333", activebackground=self.colors["bg"],
                       activeforeground="#FFD700", font=("Roboto", 10, "bold"),
                       cursor="hand2").pack(side="left", padx=5)

        self.refresh_class_buttons()
        
    def _get_class_color(self, cls_name):
        c = cls_name.upper()
        if c in ["GOOD", "GOOD_CASHEW", "WHITE_DW", "UPW"]:
            return "#2e7d32", "#ffffff"  # Emerald Green (Good / Quality)
        elif "CUT" in c or "EDGE" in c or "DOUBLE" in c or "DW" in c:
            return "#e65100", "#ffffff"  # Amber Orange (Cuts / Splits)
        elif "PKW" in c or "SW" in c or "BB" in c:
            return "#00838f", "#ffffff"  # Dark Cyan / Teal (Pieces / Broken)
        elif "BROWN" in c or "OIL" in c or "TESTA" in c or "REJECT" in c:
            return "#6d4c41", "#ffffff"  # Warm Brown / Defects
        else:
            return "#455a64", "#ffffff"  # Slate Gray (Other)

    def refresh_class_buttons(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        self._class_check_vars = {}
        self._class_buttons = {}
        self._select_all_var.set(False)

        if os.path.exists(self.dataset_crops_dir):
            self.available_classes = sorted([d for d in os.listdir(self.dataset_crops_dir) if os.path.isdir(os.path.join(self.dataset_crops_dir, d))])
        else:
            self.available_classes = self.cnn.class_names if self.cnn else ["good", "sw", "bb", "bbw"]

        counts, total_annotated = self.get_class_counts()

        # Layout: [☐ ClassBtn] [☐ ClassBtn] [☐ ClassBtn]
        cols = 3  # number of class items per row
        for col_i in range(cols * 2):  # checkbox + button per column
            self.scrollable_frame.columnconfigure(col_i, weight=0 if col_i % 2 == 0 else 1)

        for i, cls_name in enumerate(self.available_classes):
            bg_col, fg_col = self._get_class_color(cls_name)
            row = i // cols
            col_base = (i % cols) * 2

            # Checkbox for selection
            var = tk.BooleanVar(value=False)
            self._class_check_vars[cls_name] = var
            cb = tk.Checkbutton(
                self.scrollable_frame, variable=var,
                bg=self.colors["bg"], selectcolor="#333",
                activebackground=self.colors["bg"],
                bd=0, highlightthickness=0
            )
            cb.grid(row=row, column=col_base, padx=(3, 0), pady=3, sticky="e")

            # Class button showing image count
            cnt = counts.get(cls_name, 0)
            btn = tk.Button(
                self.scrollable_frame,
                text=f"{cls_name}\n({cnt})",
                bg=bg_col,
                fg=fg_col,
                activebackground="#1565c0",
                activeforeground="white",
                font=("Roboto", 9, "bold"),
                relief="flat",
                bd=0,
                padx=4,
                pady=6,
                cursor="hand2",
                command=lambda c=cls_name: self.move_image(c)
            )
            btn.grid(row=row, column=col_base + 1, padx=(0, 3), pady=3, sticky="nsew")
            self._class_buttons[cls_name] = btn

    def _toggle_select_all(self):
        val = self._select_all_var.get()
        for var in self._class_check_vars.values():
            var.set(val)
            
    def add_class(self):
        import tkinter.simpledialog as sd
        new_class = sd.askstring("Add Class", "Enter new class name:", parent=self)
        if new_class and new_class.strip():
            new_class = new_class.strip().replace(" ", "_")
            os.makedirs(os.path.join(self.dataset_crops_dir, new_class), exist_ok=True)
            self.refresh_class_buttons()
            
    def rename_class(self):
        import tkinter.simpledialog as sd
        old_class = sd.askstring("Rename Class", "Enter the exact name of the class to rename:", parent=self)
        if old_class and old_class in self.available_classes:
            new_class = sd.askstring("Rename Class", f"Rename '{old_class}' to:", parent=self)
            if new_class and new_class.strip():
                new_class = new_class.strip().replace(" ", "_")
                src = os.path.join(self.dataset_crops_dir, old_class)
                dst = os.path.join(self.dataset_crops_dir, new_class)
                try:
                    os.rename(src, dst)
                    self.refresh_class_buttons()
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to rename: {e}")
        elif old_class:
            messagebox.showerror("Error", "Class not found.")

    def delete_class(self):
        to_delete = [cn for cn, var in self._class_check_vars.items() if var.get()]
        if not to_delete:
            messagebox.showwarning("No Selection", "Check the boxes next to the classes you want to delete, then click Delete.")
            return
        names_str = ", ".join(to_delete)
        if not messagebox.askyesno("Confirm Delete", f"Delete {len(to_delete)} class(es) and ALL their images?\n\n{names_str}"):
            return
        errors = []
        for cls_name in to_delete:
            try:
                shutil.rmtree(os.path.join(self.dataset_crops_dir, cls_name))
            except Exception as e:
                errors.append(f"{cls_name}: {e}")
        self.refresh_class_buttons()
        if errors:
            messagebox.showerror("Errors", "Some classes could not be deleted:\n" + "\n".join(errors))
        else:
            messagebox.showinfo("Success", f"Deleted {len(to_delete)} class(es): {names_str}")

    def prev_image(self):
        # Move current image to the end of the list and pop the last one to front
        if len(self.image_files) > 1:
            self.image_files.insert(0, self.image_files.pop())
            self.load_next_image()
            
    def skip_image(self):
        # Move current image to the end of the list
        if len(self.image_files) > 1:
            self.image_files.append(self.image_files.pop(0))
            self.load_next_image()
            
    def delete_image(self):
        if not self.current_img_path:
            return
        if messagebox.askyesno("Delete", "Are you sure you want to delete this image?"):
            try:
                os.remove(self.current_img_path)
                self.image_files.pop(0)
                self.update_count()
                self.load_next_image()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete: {e}")
                
    def load_next_image(self):
        is_reviewing = "dataset" in self.golden_dir
        folder_base = os.path.basename(self.golden_dir)

        if not self.image_files:
            self.lbl_image.config(image='')
            if is_reviewing:
                self.lbl_prediction.config(text=f"Done reviewing [{folder_base}]!", fg="#4CAF50")
                if hasattr(self, 'lbl_status'):
                    self.lbl_status.config(text=f"No more images in [{folder_base}] to review.", fg="#4CAF50")
            else:
                self.lbl_prediction.config(text="Done! No more images in inbox.", fg="#4CAF50")
                if hasattr(self, 'lbl_status'):
                    self.lbl_status.config(text="🎉 All images from inbox have been sorted and annotated into dataset!", fg="#4CAF50")
            self.btn_agree.config(text="AGREE [Press Enter]", state="disabled")
            self.current_img_path = None
            return
            
        self.btn_agree.config(state="normal")
        self.current_img_path = self.image_files[0]
        
        # Load and predict
        bgr = cv2.imread(self.current_img_path)
        if bgr is None:
            # Corrupted image, delete and move to next
            try:
                os.remove(self.current_img_path)
            except:
                pass
            self.image_files.pop(0)
            self.update_count()
            self.load_next_image()
            return
            
        if self.cnn:
            self.current_pred, self.current_conf = self.cnn.predict_with_confidence(bgr)
        else:
            self.current_pred, self.current_conf = "load ai model file", 0.0
            
        # Display Prediction & Status based on mode
        conf_pct = int(self.current_conf * 100)
        clean_pred = self.current_pred.upper().replace("GOOD_CASHEW", "GOOD").replace(" ", "_") if self.current_pred else "UNKNOWN"
        
        if is_reviewing:
            self.lbl_prediction.config(text=f"Folder: [{folder_base}]  (AI: {clean_pred})", fg="#81C784")
            if hasattr(self, 'lbl_status'):
                self.lbl_status.config(text=f"🔍 REVIEWING [{folder_base}] ➔ Wrong class? Click correct class on right to RE-SORT | Correct? Hit Enter to keep", fg="#64B5F6")
            self.btn_agree.config(text=f"KEEP IN [{folder_base}]  [Press Enter]")
        else:
            self.lbl_prediction.config(text=f"AI Suggestion: {clean_pred} ({conf_pct}%)")
            if hasattr(self, 'lbl_status'):
                self.lbl_status.config(text=f"⏳ UNANNOTATED RAW CROP  ➔  Click AGREE to label as [{clean_pred}] or pick class on right", fg="#FFB74D")
            self.btn_agree.config(text=f"AGREE ➔ Move to [{clean_pred}]  [Press Enter]")
            
            # Color coding: Green if confident, Orange if unsure
            if conf_pct >= 95:
                self.lbl_prediction.config(fg="#4CAF50")
            elif conf_pct >= 80:
                self.lbl_prediction.config(fg="#FF9800")
            else:
                self.lbl_prediction.config(fg="#F44336")
        
        # Display Image (Resize to fit)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        # Resize preserving aspect ratio (Shrink slightly to leave room for buttons)
        w, h = pil_img.size
        target_size = 350
        scale = min(target_size/w, target_size/h)
        pil_img = pil_img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.lbl_image.config(image=self.tk_img)
        
    def move_image(self, target_class):
        if not self.current_img_path:
            return
            
        if not os.path.exists(self.current_img_path):
            print(f"[AUTO-LABELER WARNING] File no longer exists, skipping: {self.current_img_path}")
            if self.image_files:
                self.image_files.pop(0)
            self.update_count()
            self.load_next_image()
            return

        # Ensure destination exists
        target_class_clean = target_class.upper().replace("GOOD_CASHEW", "GOOD").replace(" ", "_")
        dest_dir = os.path.join(self.dataset_crops_dir, target_class_clean)
        os.makedirs(dest_dir, exist_ok=True)
        
        filename = os.path.basename(self.current_img_path)
        dest_path = os.path.join(dest_dir, filename)
        
        try:
            shutil.move(self.current_img_path, dest_path)
            print(f"[MANUAL-LABEL] Moved: {filename} -> {dest_dir}")
            self.image_files.pop(0)
            self.update_count()
            self.load_next_image()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to move file: {e}")
            
    def agree_prediction(self):
        if "dataset" in self.golden_dir:
            # In review mode, pressing Agree/Enter keeps it in current class and advances
            self.skip_image()
        else:
            if self.current_pred and self.current_pred != "load ai model file":
                self.move_image(self.current_pred)
            
    def update_count(self):
        counts, total_annotated = self.get_class_counts()
        is_reviewing = "dataset" in self.golden_dir
        folder_base = os.path.basename(self.golden_dir)

        if is_reviewing:
            inbox_text = f"🔍 Reviewing [{folder_base}]: {len(self.image_files)} imgs"
        else:
            inbox_text = f"📥 Inbox (Unannotated): {len(self.image_files)}"

        annotated_text = f"✅ Annotated in Dataset: {total_annotated}"
        if hasattr(self, 'lbl_remaining'):
            self.lbl_remaining.config(text=inbox_text)
        if hasattr(self, 'lbl_annotated'):
            self.lbl_annotated.config(text=annotated_text)
        if hasattr(self, 'lbl_count'):
            self.lbl_count.config(text=inbox_text)
        if hasattr(self, '_class_buttons'):
            for cls_name, btn in self._class_buttons.items():
                cnt = counts.get(cls_name, 0)
                btn.config(text=f"{cls_name}\n({cnt})")
        self.refresh_source_options()
        
    def bulk_auto_sort(self):
        try:
            thresh_pct = int(self.thresh_var.get())
            thresh = thresh_pct / 100.0
        except:
            thresh = 0.95
            
        if not self.image_files:
            messagebox.showinfo("Empty", "No images to sort.")
            return
            
        if not messagebox.askyesno("Confirm", f"This will automatically move all images with >{thresh_pct}% confidence to their predicted folders.\n\nContinue?"):
            return
            
        # Show progress window
        prog_win = tk.Toplevel(self)
        prog_win.title("Auto-Sorting...")
        prog_win.geometry("300x100")
        prog_win.transient(self)
        prog_win.grab_set()
        
        tk.Label(prog_win, text="Processing images...").pack(pady=10)
        prog_bar = ttk.Progressbar(prog_win, length=250, mode='determinate', maximum=len(self.image_files))
        prog_bar.pack()
        
        moved_count = 0
        remaining_files = []
        
        for i, img_path in enumerate(self.image_files):
            bgr = cv2.imread(img_path)
            if bgr is not None:
                if self.cnn:
                    pred, conf = self.cnn.predict_with_confidence(bgr)
                    if conf >= thresh:
                        pred_clean = pred.upper().replace("GOOD_CASHEW", "GOOD").replace(" ", "_")
                        dest_dir = os.path.join(self.dataset_crops_dir, pred_clean)
                        os.makedirs(dest_dir, exist_ok=True)
                        dest_path = os.path.join(dest_dir, os.path.basename(img_path))
                        try:
                            shutil.move(img_path, dest_path)
                            print(f"[AUTO-LABELER] Moved: {os.path.basename(img_path)} -> {dest_dir}")
                            moved_count += 1
                        except Exception as e:
                            print(f"[AUTO-LABELER ERROR] Failed to move {img_path}: {e}")
                        continue
                remaining_files.append(img_path)
            else:
                os.remove(img_path)
                
            prog_bar['value'] = i + 1
            if i % 10 == 0:
                prog_win.update()
                
        prog_win.destroy()
        
        self.image_files = remaining_files
        self.update_count()
        self.load_next_image()
        
        messagebox.showinfo("Complete", f"Successfully auto-sorted {moved_count} images.\nRemaining: {len(self.image_files)} for manual review.")

if __name__ == "__main__":
    app = AIAutoLabeler()
    app.mainloop()
