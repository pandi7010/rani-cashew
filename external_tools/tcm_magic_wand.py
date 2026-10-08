import cv2
import numpy as np

# Lazy load FastSAM to avoid blocking the main UI thread during import
_fast_sam_model = None

def get_sam_model():
    global _fast_sam_model
    if _fast_sam_model is None:
        try:
            from ultralytics import FastSAM
            _fast_sam_model = FastSAM('FastSAM-s.pt')
            print("FastSAM initialized successfully.", flush=True)
        except Exception as e:
            print(f"Error loading FastSAM: {e}")
    return _fast_sam_model

def get_magic_polygon(image_matrix, click_x, click_y):
    """
    Given a BGR image matrix and a single click coordinate (x, y),
    Return a numpy array of polygon coordinates (shape: N, 1, 2) utilizing FastSAM.
    Fallback to OpenCV FloodFill if FastSAM fails or isn't available.
    """
    model = get_sam_model()
    if model is not None:
        try:
            # Predict using point prompt
            results = model.predict(
                source=image_matrix, 
                points=[[click_x, click_y]], 
                labels=[1], 
                conf=0.4, 
                verbose=False
            )
            
            # Extract highest confidence mask polygon
            if results and len(results) > 0 and results[0].masks is not None:
                # results[0].masks.xy is a list of segments
                best_segment = results[0].masks.xy[0] 
                if len(best_segment) > 3:
                    # Convert float coordinates to integers for cv2 contour compat
                    poly = np.array(best_segment, dtype=np.int32).reshape((-1, 1, 2))
                    return poly
        except Exception as e:
            print(f"FastSAM prediction failed: {e}")
            pass
            
    # --- FALLBACK: Standard OpenCV FloodFill (Current TCM Style) ---
    flood_mask = np.zeros((image_matrix.shape[0]+2, image_matrix.shape[1]+2), np.uint8)
    lo_diff = (30, 30, 30)
    up_diff = (30, 30, 30)
    flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
    
    cv2.floodFill(image_matrix.copy(), flood_mask, (click_x, click_y), 255, lo_diff, up_diff, flags=flags)
    
    mask_defect = flood_mask[1:-1, 1:-1]
    kernel = np.ones((2,2), np.uint8)
    mask_defect = cv2.morphologyEx(mask_defect, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask_defect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_cnt = None
    min_dist = float('inf')
    for cnt in contours:
        if cv2.contourArea(cnt) < 2: continue
        dist = cv2.pointPolygonTest(cnt, (click_x, click_y), True)
        if dist >= 0:
            best_cnt = cnt
            break
        elif abs(dist) < min_dist:
            min_dist = abs(dist)
            best_cnt = cnt
            
    return best_cnt
