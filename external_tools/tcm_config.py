"""
TCM STUDIO PRO — Configuration
20-Class Cashew Classification Platform
"""
import os

import json

# --- PATHS ---
BASE_DIR = "C:/TCM_VISION"
PROJECTS_DIR = os.path.join(BASE_DIR, "Projects")

# Global Active Project State
ACTIVE_PROJECT = None
DATASET_DIR = None
PROCESSED_DIR = None
MODEL_DIR = None
EXPORT_DIR = None
CLASS_NAMES = []
CLASS_COLORS = {}

# --- GPU CONFIG (RTX 5060 Ti) ---
DEVICE = "0"  # CUDA GPU index — use "cpu" if no GPU

# Default training params
DEFAULT_EPOCHS = 100
DEFAULT_BATCH = 16
DEFAULT_IMGSZ = 640
DEFAULT_MODEL = "yolo11n.pt"

class ProjectManager:
    @staticmethod
    def get_projects():
        os.makedirs(PROJECTS_DIR, exist_ok=True)
        return [d for d in os.listdir(PROJECTS_DIR) if os.path.isdir(os.path.join(PROJECTS_DIR, d))]
        
    @staticmethod
    def create_project(name, classes):
        """Classes should be a list of strings."""
        os.makedirs(PROJECTS_DIR, exist_ok=True)
        proj_dir = os.path.join(PROJECTS_DIR, name)
        
        if os.path.exists(proj_dir):
            return False # Already exists
            
        # Basic dynamic color generator
        colors = ["#4CAF50", "#FF9800", "#2196F3", "#9C27B0", "#F44336", "#00BCD4", "#E91E63", "#CDDC39", "#3F51B5", "#8BC34A"]
        class_colors = {cls: colors[i % len(colors)] for i, cls in enumerate(classes)}
        
        config = {
            "name": name,
            "classes": classes,
            "colors": class_colors,
            "created_at": "Unknown"
        }
        
        os.makedirs(os.path.join(proj_dir, "Datasets", "Raw_Images"), exist_ok=True)
        os.makedirs(os.path.join(proj_dir, "Datasets", "Processed_YOLO"), exist_ok=True)
        os.makedirs(os.path.join(proj_dir, "Models"), exist_ok=True)
        os.makedirs(os.path.join(proj_dir, "Exports"), exist_ok=True)
        
        for cls in classes:
            os.makedirs(os.path.join(proj_dir, "Datasets", "Raw_Images", cls), exist_ok=True)
            
        with open(os.path.join(proj_dir, "project.json"), "w") as f:
            json.dump(config, f, indent=4)
            
        return True

    @staticmethod
    def load_project(name):
        global ACTIVE_PROJECT, DATASET_DIR, PROCESSED_DIR, MODEL_DIR, EXPORT_DIR, CLASS_NAMES, CLASS_COLORS
        
        proj_dir = os.path.join(PROJECTS_DIR, name)
        json_path = os.path.join(proj_dir, "project.json")
        
        if not os.path.exists(json_path):
            # Try to populate default cashew if missing
            return False
            
        with open(json_path, "r") as f:
            config = json.load(f)
            
        ACTIVE_PROJECT = name
        CLASS_NAMES = config.get("classes", [])
        CLASS_COLORS = config.get("colors", {})
        
        DATASET_DIR = os.path.join(proj_dir, "Datasets", "Raw_Images")
        PROCESSED_DIR = os.path.join(proj_dir, "Datasets", "Processed_YOLO")
        MODEL_DIR = os.path.join(proj_dir, "Models")
        EXPORT_DIR = os.path.join(proj_dir, "Exports")
        return True

    @staticmethod
    def delete_project(name):
        import shutil
        proj_dir = os.path.join(PROJECTS_DIR, name)
        if os.path.exists(proj_dir):
            try:
                shutil.rmtree(proj_dir)
                return True
            except Exception:
                return False
        return False


    @staticmethod
    def add_class(new_class):
        global CLASS_NAMES, CLASS_COLORS, DATASET_DIR
        if not ACTIVE_PROJECT or new_class in CLASS_NAMES:
            return False
            
        proj_dir = os.path.join(PROJECTS_DIR, ACTIVE_PROJECT)
        json_path = os.path.join(proj_dir, "project.json")
        
        with open(json_path, "r") as f:
            config = json.load(f)
            
        config["classes"].append(new_class)
        colors = ["#4CAF50", "#FF9800", "#2196F3", "#9C27B0", "#F44336", "#00BCD4", "#E91E63", "#CDDC39", "#3F51B5", "#8BC34A"]
        new_color = colors[len(config["classes"]) % len(colors)]
        config["colors"][new_class] = new_color
        
        with open(json_path, "w") as f:
            json.dump(config, f, indent=4)
            
        CLASS_NAMES.append(new_class)
        CLASS_COLORS[new_class] = new_color
        os.makedirs(os.path.join(DATASET_DIR, new_class), exist_ok=True)
        return True

    @staticmethod
    def rename_class(old_name, new_name):
        global CLASS_NAMES, CLASS_COLORS, DATASET_DIR
        if not ACTIVE_PROJECT or old_name not in CLASS_NAMES or new_name in CLASS_NAMES:
            return False
            
        proj_dir = os.path.join(PROJECTS_DIR, ACTIVE_PROJECT)
        json_path = os.path.join(proj_dir, "project.json")
        
        with open(json_path, "r") as f:
            config = json.load(f)
            
        idx = config["classes"].index(old_name)
        config["classes"][idx] = new_name
        config["colors"][new_name] = config["colors"].pop(old_name)
        
        with open(json_path, "w") as f:
            json.dump(config, f, indent=4)
            
        old_dir = os.path.join(DATASET_DIR, old_name)
        new_dir = os.path.join(DATASET_DIR, new_name)
        if os.path.exists(old_dir):
            os.rename(old_dir, new_dir)
            
        # Also rename all labels in new_dir (replace class ID)
        idx_old = CLASS_NAMES.index(old_name)
        idx_new = config["classes"].index(new_name)
        # Note: changing class ID order might mess up YOLO data, but keeping it simple for now. 
        # Wait, if we keep the same index, ID doesn't change, which is PERFECT.
        
        CLASS_NAMES[idx] = new_name
        CLASS_COLORS[new_name] = CLASS_COLORS.pop(old_name)
        return True

    @staticmethod
    def delete_class(del_name):
        import shutil
        global CLASS_NAMES, CLASS_COLORS, DATASET_DIR
        if not ACTIVE_PROJECT or del_name not in CLASS_NAMES:
            return False
            
        proj_dir = os.path.join(PROJECTS_DIR, ACTIVE_PROJECT)
        json_path = os.path.join(proj_dir, "project.json")
        
        with open(json_path, "r") as f:
            config = json.load(f)
            
        config["classes"].remove(del_name)
        config["colors"].pop(del_name, None)
        
        with open(json_path, "w") as f:
            json.dump(config, f, indent=4)
            
        cls_dir = os.path.join(DATASET_DIR, del_name)
        if os.path.exists(cls_dir):
            shutil.rmtree(cls_dir)
            
        CLASS_NAMES.remove(del_name)
        CLASS_COLORS.pop(del_name, None)
        return True

def ensure_dirs():
    """Create all required directories for ACTIVE project."""
    if not ACTIVE_PROJECT:
        return
    for d in [DATASET_DIR, PROCESSED_DIR, MODEL_DIR, EXPORT_DIR]:
        os.makedirs(d, exist_ok=True)
    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(DATASET_DIR, cls), exist_ok=True)

def hex_to_bgr(hex_color):
    """Convert hex color to BGR tuple for OpenCV."""
    h = hex_color.lstrip('#')
    if len(h) != 6:
        return (255, 255, 255) # default white safely
    try:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return (b, g, r)
    except:
        return (255,255,255)

def hex_to_rgb(hex_color):
    h = hex_color.lstrip('#')
    if len(h) != 6:
        return (255, 255, 255)
    try:
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except:
        return (255, 255, 255)
