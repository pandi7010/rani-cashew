"""
TCM STUDIO PRO — Training Engine
GPU-Accelerated YOLO Training Pipeline (RTX 5060 Ti)
"""
import os
import shutil
import random
import datetime
import threading
import yaml
import cv2
import numpy as np
import json

import tcm_config

LOWER_BLUE_HSV = (90, 50, 50)
UPPER_BLUE_HSV = (130, 255, 255)


class TrainingEngine:
    """Handles dataset preparation, augmentation, training, and export."""

    def __init__(self, log_callback=None):
        self.log_cb = log_callback or print
        self.training_active = False
        self.train_thread = None
        self.latest_best_pt = None

    def log(self, msg):
        self.log_cb(msg)

    # ── Dataset Statistics ───────────────────────────────────────────────
    def get_class_stats(self):
        """Return dict: {class_name: {'images': N, 'labels': N}}"""
        stats = {}
        for cls in tcm_config.CLASS_NAMES:
            cls_dir = os.path.join(tcm_config.DATASET_DIR, cls)
            golden_dir = os.path.join(tcm_config.DATASET_DIR, "Golden_Crops", cls)
            
            imgs = 0
            lbls = 0
            
            if os.path.isdir(cls_dir):
                imgs += len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                lbls += len([f for f in os.listdir(cls_dir) if f.endswith('.txt')])
                
            if os.path.isdir(golden_dir):
                imgs += len([f for f in os.listdir(golden_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                # Golden crops don't typically have their own YOLO text files as they are pure cropped items.
                lbls += imgs  # Assume 1 object per golden crop 
                
            stats[cls] = {"images": imgs, "labels": lbls}
        return stats

    def get_total_images(self):
        return sum(v["images"] for v in self.get_class_stats().values())

    # ── Auto-Label (Blue Belt Removal) ───────────────────────────────────
    def auto_label_class(self, class_name, class_id):
        """Auto-generate YOLO labels for one class using blue-belt background removal on full frames."""
        cls_dir = os.path.join(tcm_config.DATASET_DIR, class_name)
        if not os.path.isdir(cls_dir):
            return 0
        lower = np.array(LOWER_BLUE_HSV)
        upper = np.array(UPPER_BLUE_HSV)
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        def process_image(fname):
            txt_name = os.path.splitext(fname)[0] + ".txt"
            txt_path = os.path.join(cls_dir, txt_name)
            if os.path.exists(txt_path):
                return 0  # skip already labeled
            img = cv2.imread(os.path.join(cls_dir, fname))
            if img is None:
                return 0
            h, w = img.shape[:2]
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask_bg = cv2.inRange(hsv, lower, upper)
            mask_obj = cv2.bitwise_not(mask_bg)
            contours, _ = cv2.findContours(mask_obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            lines = []
            for cnt in contours:
                if cv2.contourArea(cnt) > 500:
                    epsilon = 0.005 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    if len(approx) > 2:
                        pts = []
                        for pt in approx:
                            # normalize
                            pts.append(f"{pt[0][0]/w:.6f} {pt[0][1]/h:.6f}")
                        lines.append(f"{class_id} " + " ".join(pts))
            if lines:
                with open(txt_path, "w") as f:
                    f.write("\n".join(lines))
                return 1
            return 0

        import concurrent.futures
        count = 0
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(process_image, files)
            count = sum(results)
            
        return count

    def auto_label_all(self):
        """Auto-label all classes."""
        self.log("🔍 AUTO-LABELING: Starting blue-belt background removal...")
        total = 0
        for i, cls in enumerate(tcm_config.CLASS_NAMES):
            n = self.auto_label_class(cls, i)
            total += n
            self.log(f"   [{i+1:02d}/20] {cls}: {n} new labels")
        self.log(f"✅ AUTO-LABELING COMPLETE: {total} images labeled")
        return total

    # ── Augmentation ─────────────────────────────────────────────────────
    def augment_image(self, img, flip_h=True, flip_v=False, rotate=True, brightness=True, noise=True, multiplier=3):
        """Apply random augmentations to a single image. Returns list of augmented copies."""
        results = []
        import random
        for _ in range(multiplier):
            res = img.copy()
            if flip_h and random.random() > 0.5:
                res = cv2.flip(res, 1)
            if flip_v and random.random() > 0.5:
                res = cv2.flip(res, 0)
            if rotate and random.random() > 0.5:
                angle = random.uniform(-15, 15)
                h, w = res.shape[:2]
                M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                res = cv2.warpAffine(res, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            if brightness and random.random() > 0.5:
                factor = random.uniform(0.7, 1.3)
                res = np.clip(res.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            if noise and random.random() > 0.5:
                gauss = np.random.normal(0, 10, res.shape).astype(np.uint8)
                res = cv2.add(res, gauss)
            results.append(res)
        return results

    def augment_class(self, class_name, **aug_opts):
        """Augment all images in a class folder, copying labels for each augmented image."""
        cls_dir = os.path.join(tcm_config.DATASET_DIR, class_name)
        if not os.path.isdir(cls_dir):
            return 0
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        count = 0
        for fname in files:
            if "_aug" in fname or fname.lower().startswith("crop_"):
                continue
            img = cv2.imread(os.path.join(cls_dir, fname))
            if img is None:
                continue
            base = os.path.splitext(fname)[0]
            txt_src = os.path.join(cls_dir, base + ".txt")
            aug_imgs = self.augment_image(img, **aug_opts)
            for j, aug in enumerate(aug_imgs):
                aug_name = f"{base}_aug_{random.randint(1000, 9999)}"
                cv2.imwrite(os.path.join(cls_dir, aug_name + ".jpg"), aug)
                if os.path.exists(txt_src):
                    shutil.copy(txt_src, os.path.join(cls_dir, aug_name + ".txt"))
                count += 1
        return count

    # ── Prepare YOLO Dataset ─────────────────────────────────────────────
    def prepare_dataset(self, val_ratio=0.2):
        """Split labeled data into train/val in YOLO format."""
        self.log("📦 PREPARING YOLO DATASET...")
        if os.path.exists(tcm_config.PROCESSED_DIR):
            shutil.rmtree(tcm_config.PROCESSED_DIR)

        dirs = {}
        for split in ["train", "val"]:
            for kind in ["images", "labels"]:
                p = os.path.join(tcm_config.PROCESSED_DIR, kind, split)
                os.makedirs(p, exist_ok=True)
                dirs[f"{kind}_{split}"] = p

        total = 0
        labeled_files = []

        self.log(f"🔎 Scanning for dataset files in: {tcm_config.DATASET_DIR}")

        # Traverse dataset directory
        for root, dirs_list, files in os.walk(tcm_config.DATASET_DIR):
            # Skip output directories so we don't duplicate
            if "Processed_YOLO" in root or "runs" in root:
                continue
                
            folder_name = os.path.basename(root)
            cls_id = None
            if folder_name in tcm_config.CLASS_NAMES:
                cls_id = tcm_config.CLASS_NAMES.index(folder_name)

            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    txt = os.path.splitext(f)[0] + ".txt"
                    txt_path = os.path.join(root, txt)
                    
                    if os.path.exists(txt_path):
                        labeled_files.append({"root": root, "img": f, "txt_path": txt_path, "cls_id": None})
                    elif folder_name.lower() in ["background", "general_scenes", "bg"]:
                        # NEGATIVE TRAINING: Include image with artificial empty label
                        labeled_files.append({"root": root, "img": f, "txt_path": "EMPTY_BG", "cls_id": -1})
                    elif cls_id is not None:
                        labeled_files.append({"root": root, "img": f, "txt_path": None, "cls_id": cls_id})
                    else:
                        print(f"Skipping {f} (no {txt_path})")

        self.log(f"🔎 Found {len(labeled_files)} valid labeled images (including auto-synthesized crops).")

        random.shuffle(labeled_files)
        split_idx = int(len(labeled_files) * (1 - val_ratio))

        for i, item in enumerate(labeled_files):
            split = "train" if i < split_idx else "val"
            rt = item["root"]
            f = item["img"]
            shutil.copy(os.path.join(rt, f), os.path.join(dirs[f"images_{split}"], f))
            
            
            txt_name = os.path.splitext(f)[0] + ".txt"
            dest_txt = os.path.join(dirs[f"labels_{split}"], txt_name)
            
            if item["txt_path"] and item["txt_path"] != "EMPTY_BG":
                shutil.copy(item["txt_path"], dest_txt)
            elif item["txt_path"] == "EMPTY_BG":
                # Create an empty txt file for YOLO background/negative training
                open(dest_txt, 'w').close()
            else:
                with open(dest_txt, "w") as out_f:
                    # Synthetic full-image polygon for segmentation model
                    out_f.write(f"{item['cls_id']} 0.05 0.05 0.95 0.05 0.95 0.95 0.05 0.95\n")
            total += 1
            
        self.log(f"✅ DATASET READY: {total} images total ({split_idx} train / {total - split_idx} val)")
        return total

    def create_yaml(self):
        """Generate data.yaml for YOLO training."""
        yaml_path = os.path.join(tcm_config.PROCESSED_DIR, "data.yaml")
        data = {
            "path": os.path.abspath(tcm_config.PROCESSED_DIR).replace("\\", "/"),
            "train": "images/train",
            "val": "images/val",
            "nc": len(tcm_config.CLASS_NAMES),
            "names": {i: name for i, name in enumerate(tcm_config.CLASS_NAMES)}
        }
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        self.log(f"📄 data.yaml created: {yaml_path}")
        return yaml_path

    # ── Training (GPU) ───────────────────────────────────────────────────
    def train(self, epochs=None, batch=None, imgsz=None, model_name=None, force_cpu=False):
        """Run YOLO training on GPU. Meant to be called in a thread."""
        try:
            from ultralytics import YOLO
        except ImportError:
            self.log("❌ ultralytics not installed! Run: pip install ultralytics")
            return None

        epochs = epochs or tcm_config.DEFAULT_EPOCHS
        batch = batch or tcm_config.DEFAULT_BATCH
        imgsz = imgsz or tcm_config.DEFAULT_IMGSZ
        model_name = model_name or tcm_config.DEFAULT_MODEL
        
        # Auto-detect Instance Segmentation dataset based on txt files having >5 columns
        is_segmentation = False
        train_labels = os.path.join(tcm_config.PROCESSED_DIR, "labels", "train")
        if os.path.exists(train_labels):
            for f in os.listdir(train_labels):
                if f.endswith(".txt"):
                    with open(os.path.join(train_labels, f), "r") as txtf:
                        for line in txtf:
                            if len(line.strip().split()) > 5:
                                is_segmentation = True
                                break
                    if is_segmentation:
                        break
        
        if is_segmentation and "seg" not in model_name.lower():
            base_model = os.path.splitext(model_name)[0]
            model_name = f"{base_model}-seg.pt"
            self.log(f"✨ Segmentation annotations detected. Switching to Segment model ({model_name}).")
        elif "seg" in model_name.lower():
            self.log("✨ Segment model requested. Make sure your dataset annotations are polygons.")

        self.training_active = True
        self.log(f"\n{'='*60}")
        if force_cpu:
            self.log(f"🚀 TRAINING STARTED — FORCED CPU")
        else:
            self.log(f"🚀 TRAINING STARTED — AUTO GPU")
        self.log(f"   Model: {model_name} | Epochs: {epochs} | Batch: {batch} | ImgSz: {imgsz}")
        self.log(f"{'='*60}\n")

        yaml_path = self.create_yaml()
        model = YOLO(model_name)

        try:
            import torch
            if force_cpu:
                device_str = "cpu"
            else:
                device_str = 0 if torch.cuda.is_available() else "cpu"
            
            project_name_safe = tcm_config.PROJECT_NAME if hasattr(tcm_config, "PROJECT_NAME") and tcm_config.PROJECT_NAME else "TCM_Model"
            
            kwargs = {
                "data": yaml_path,
                "epochs": epochs,
                "imgsz": imgsz,
                "batch": batch,
                "device": device_str,
                "name": f"{project_name_safe}_Run",
                "project": os.path.join(tcm_config.MODEL_DIR, "runs"),
                "exist_ok": False,
                "verbose": True,
                # Advanced Augmentations (to match/beat Roboflow)
                "mosaic": 1.0,      # Force 4-image mosaic
                "mixup": 0.1,       # Image mixup probability
                "copy_paste": 0.1,  # Segment copy-paste probability 
                "hsv_h": 0.015,     # HSV-Hue augmentation
                "hsv_s": 0.7,       # HSV-Saturation augmentation
                "hsv_v": 0.4,       # HSV-Value augmentation
                "degrees": 15.0,    # Rotation augmentation
                "translate": 0.1,   # Translation augmentation
                "scale": 0.5,       # Scale augmentation
                "shear": 0.0,
                "perspective": 0.0,
                "flipud": 0.0,      # Flip up-down
                "fliplr": 0.5,      # Flip left-right
                "bgr": 0.0          # BGR to RGB flip
            }
            if getattr(self, "yolo_aug_config", None):
                kwargs.update(self.yolo_aug_config)
                self.log(f"🧠 DYNAMIC AUGMENTATION ACTIVATED: {self.yolo_aug_config}")
                
            results = model.train(**kwargs)
            best_pt = os.path.join(results.save_dir, "weights", "best.pt")
            if os.path.exists(best_pt):
                self.latest_best_pt = best_pt
                self.log(f"\n✅ TRAINING COMPLETE!")
                self.log(f"📁 best.pt: {best_pt}")
            else:
                self.log("⚠️ Training finished but best.pt not found.")
        except Exception as e:
            self.log(f"❌ TRAINING ERROR: {e}")
        finally:
            self.training_active = False

    def start_training_thread(self, **kwargs):
        if self.training_active:
            self.log("⚠️ Training already running!")
            return
        self.train_thread = threading.Thread(target=self.train, kwargs=kwargs, daemon=True)
        self.train_thread.start()

    # ── Export ────────────────────────────────────────────────────────────
    def export_model(self, best_pt_path=None):
        """Package best.pt + data.yaml + class_names.json into export folder."""
        best_pt = best_pt_path or self.latest_best_pt
        if not best_pt or not os.path.exists(best_pt):
            self.log("❌ No best.pt found. Train a model first.")
            return None

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        export_folder = os.path.join(tcm_config.EXPORT_DIR, f"cashew_20class_{ts}")
        os.makedirs(export_folder, exist_ok=True)

        # Copy best.pt
        shutil.copy(best_pt, os.path.join(export_folder, "best.pt"))

        # Create class_names.json
        with open(os.path.join(export_folder, "class_names.json"), "w") as f:
            json.dump({i: n for i, n in enumerate(tcm_config.CLASS_NAMES)}, f, indent=2)

        # Copy data.yaml
        yaml_src = os.path.join(tcm_config.PROCESSED_DIR, "data.yaml")
        if os.path.exists(yaml_src):
            shutil.copy(yaml_src, os.path.join(export_folder, "data.yaml"))

        # Model info
        info = {
            "exported_at": ts,
            "num_classes": len(tcm_config.CLASS_NAMES),
            "classes": tcm_config.CLASS_NAMES,
            "gpu_device": tcm_config.DEVICE,
            "source_best_pt": best_pt,
        }
        with open(os.path.join(export_folder, "model_info.json"), "w") as f:
            json.dump(info, f, indent=2)

        # Copy training results if available
        results_dir = os.path.dirname(os.path.dirname(best_pt))  # go up from weights/
        for fname in ["results.csv", "confusion_matrix.png", "results.png"]:
            src = os.path.join(results_dir, fname)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(export_folder, fname))

        self.log(f"\n📦 EXPORT COMPLETE: {export_folder}")
        self.log(f"   ✓ best.pt")
        self.log(f"   ✓ data.yaml")
        self.log(f"   ✓ class_names.json")
        self.log(f"   ✓ model_info.json")
        return export_folder
