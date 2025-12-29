import cv2
import numpy as np
import time

# --- CONSTANTS ---
# COCO Class Mapping (Standard YOLOv11)
# 0:Person, 1:Bicycle, 2:Car, 3:Motorcycle, 5:Bus, 7:Truck, 15:Cat, 16:Dog
# Note: TFLite model usually keeps original COCO indices unless retrained.
# If using 'best.pt' custom model, indices might differ. Assuming standard COCO for now or mapping.
# User asked for 'yolo11n.pt' base export, so standard COCO.

HAZARD_CLASSES = [0, 1, 3, 15, 16] # Person, Bike, Moto, Cat, Dog
VEHICLE_CLASSES = [2, 3, 5, 7] # Car, Moto, Bus, Truck
BIKER_CLASSES = [1, 3]
TRAFFIC_CLASSES = [9, 11] # Traffic Light, Stop Sign

class BoxShim:
    """Numpy-based Shim to mimic Ultralytics Box object for HazardAnalyzer"""
    def __init__(self, data):
        # data: {'box': [x1, y1, x2, y2], 'class_id': int, 'conf': float, 'id': int/None}
        self.cls = np.array([float(data['class_id'])])
        self.id = np.array([float(data['id'])]) if data.get('id') is not None else None
        self.xyxy = np.array([data['box']])
        self.conf = np.array([data['conf']])

class BoxesShim:
    """Numpy-based Shim to mimic Ultralytics Boxes object"""
    def __init__(self, obj_list):
        self.obj_list = obj_list
        if obj_list:
            ids = [o.id[0] for o in obj_list if o.id is not None]
            self.id = np.array(ids) if ids else None
        else:
            self.id = None
            
    def __iter__(self):
        return iter(self.obj_list)
    def __len__(self):
        return len(self.obj_list)

class DistanceMonitor:
    def __init__(self, frame_width=1920, focal_length_px=1000):
        self.focal_length = focal_length_px
        self.frame_width = frame_width
        self.real_widths = {
            2: 1.8, 5: 2.5, 7: 2.5, 3: 0.8, 1: 0.6, 0: 0.5
        }
        self.critical_distance = 15.0 
        self.tailgating_distance = 8.0 

    def estimate_distance(self, box, class_id):
        x1, y1, x2, y2 = box
        img_w = x2 - x1
        if img_w <= 0: return 999.0
        real_w = self.real_widths.get(class_id, 1.5)
        distance = (real_w * self.focal_length) / img_w
        return distance

    def check_safe_distance(self, detections):
        center_x = self.frame_width // 2
        lane_center_threshold = self.frame_width * 0.3
        closest_dist = 999.0
        closest_obj = None
        
        for det in detections:
            if det['class_id'] not in [2, 3, 5, 7]: continue
            x1, y1, x2, y2 = det['box']
            obj_center = (x1 + x2) / 2
            
            if abs(obj_center - center_x) < lane_center_threshold:
                dist = self.estimate_distance(det['box'], det['class_id'])
                if dist < closest_dist:
                    closest_dist = dist
                    closest_obj = det
                    
        if closest_obj:
            if closest_dist < self.tailgating_distance:
                return "CRITICAL: BRAKE!", (255, 0, 0), closest_dist, closest_obj['box'] # RGB Red
            elif closest_dist < self.critical_distance:
                return "WARNING: Too Close", (255, 165, 0), closest_dist, closest_obj['box'] # RGB Orange
            else:
                return f"Dist: {closest_dist:.1f}m", (0, 255, 0), closest_dist, closest_obj['box']
        return None, (0, 255, 0), None, None

class HazardStabilizer:
    def __init__(self, frame_width=1920, frame_height=1080, buffer_frames=3):
        self.width = frame_width
        self.height = frame_height
        self.buffer_frames = buffer_frames
        self.hazard_counter = {} 
        
        # User defined Red Zone (V3 from Desktop)
        # Normalized: Bottom(0,1)-(1,1), Top(0.365, 0.70)-(0.635, 0.70)
        p1 = (0.0, 1.0)
        p2 = (0.365, 0.70)
        p3 = (0.635, 0.70)
        p4 = (1.0, 1.0)

        self.roi_poly = np.array([
            [int(p1[0]*self.width), int(p1[1]*self.height)],
            [int(p2[0]*self.width), int(p2[1]*self.height)],
            [int(p3[0]*self.width), int(p3[1]*self.height)],
            [int(p4[0]*self.width), int(p4[1]*self.height)]
        ], dtype=np.int32)

    def is_in_danger_zone(self, box):
        x1, y1, x2, y2 = box
        center_x = int((x1 + x2) / 2)
        center_y = int(y2)
        result = cv2.pointPolygonTest(self.roi_poly, (center_x, center_y), False)
        return result >= 0

    def update(self, detections, current_frame_detected_ids):
        valid_hazards = []
        track_ids_to_remove = [tid for tid in self.hazard_counter if tid not in current_frame_detected_ids]
        for tid in track_ids_to_remove: del self.hazard_counter[tid]

        for det in detections:
            track_id = det.get('id', -1)
            
            # Spatial Filter
            if not self.is_in_danger_zone(det['box']):
                continue 
            
            # Temporal Filter
            if track_id != -1:
                self.hazard_counter[track_id] = self.hazard_counter.get(track_id, 0) + 1
                if self.hazard_counter[track_id] >= self.buffer_frames:
                    valid_hazards.append(det)
            else:
                valid_hazards.append(det)
        return valid_hazards

    def draw_debug_zone(self, frame):
        # Draw Red Zone (BGR for OpenCV usually, but Kivy loads texture RGB? 
        # We'll assume OpenCV usage usually BGR. 
        # BUT Kivy camera texture might be RGB. 
        # Let's stick to BGR (0,0,255) red for consistency if converting.
        cv2.polylines(frame, [self.roi_poly], isClosed=True, color=(0, 0, 255), thickness=3)
        return frame

class HazardAnalyzer:
    def __init__(self):
        self.track_history = {}
        
    def analyze(self, boxes, names, frame_shape=None, lane_bounds=None):
        alerts = []
        current_ids = []
        
        # Extract track IDs manually from Shim
        track_ids = []
        if boxes.id is not None:
             track_ids = boxes.id.astype(int).tolist()
        else:
             track_ids = [-1] * len(boxes)

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            track_id = track_ids[i]
            x1, y1, x2, y2 = box.xyxy[0]
            w = x2 - x1
            h = y2 - y1
            area = w * h
            
            # Safe access to name
            name = names.get(cls_id, 'Unknown') if isinstance(names, dict) else names[cls_id]

            # Logic ported from Desktop (Simplified)
            # Since inputs are FILTERED by Stabilizer -> Everything is "In Zone"
            
            # 1. Approach Logic (retained basic approach)
            is_approaching = False
            
            # History
            if track_id != -1:
                current_ids.append(track_id)
                if track_id not in self.track_history:
                    self.track_history[track_id] = {'area': [], 'center': []}
                
                history = self.track_history[track_id]
                center_x = (x1 + x2) / 2
                history['area'].append(area)
                history['center'].append((center_x, (y1+y2)/2))
                
                if len(history['area']) > 10: history['area'].pop(0)
                if len(history['center']) > 10: history['center'].pop(0)
                
                # Fast Approach check
                if len(history['area']) >= 4:
                    prev_area = history['area'][-4]
                    if prev_area > 0:
                        growth = (area - prev_area) / prev_area
                        if growth > 0.30 and area > 1500:
                            alerts.append(f"CRASH IMMINENT: {name}")

            # 2. Specific Alerts
            if cls_id in TRAFFIC_CLASSES:
                if cls_id == 11: alerts.append(f"STOP SIGN")
                elif cls_id == 9: alerts.append(f"TRAFFIC LIGHT")
            elif cls_id in BIKER_CLASSES:
                alerts.append(f"BIKER: {name}")
            elif cls_id in HAZARD_CLASSES:
                 alerts.append(f"HAZARD: {name}")
            elif cls_id in VEHICLE_CLASSES:
                 # Only alert if merging or crash imminent (already handled above)
                 pass
            else:
                 if area > 3000: alerts.append("OBSTACLE")

        return list(set(alerts))
