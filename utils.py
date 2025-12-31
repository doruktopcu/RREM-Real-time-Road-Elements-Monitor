import cv2

# COCO Class Mapping (Standard YOLOv8/11)
# We can use the model's names dict, but having some constants here helps.

# Vulnerable Road Users and Animal Hazards
# Vulnerable Road Users and Animal Hazards
# Updated for UNIFIED DATASET (Custom Model)
# Must match data.yaml: 0:Person, 1:Bicycle, 3:Motorcycle, 6:Cat, 7:Dog, 10:Accident
HAZARD_CLASSES = [
    0,  # Person
    1,  # Bicycle
    3,  # Motorcycle
    6,  # Cat (Mapped from COCO #15)
    7,  # Dog (Mapped from COCO #16)
    10, # Accident
]

# Vehicle Classes
# 2: Car, 3: Motorcycle, 4: Bus, 5: Truck
VEHICLE_CLASSES = [
    2,  # Car
    3,  # Motorcycle
    4,  # Bus
    5,  # Truck
]

BIKER_CLASSES = [
    1, # Bicycle
    3 # Motorcycle
]

# Traffic Control
TRAFFIC_CLASSES = [
    9,  # Traffic Light
    11, # Stop Sign
]

import numpy as np

# ... (Existing constants unchanged) ...

class LaneTracker:
    def __init__(self):
        # Smoothing factors
        self.avg_x1 = 0
        self.avg_x2 = 0
        self.initialized = False
        self.alpha = 0.2 # Exponential Moving Average factor (0.0 - 1.0)
        self.frame_count = 0
        self.skip_interval = 5 # Run detection every 5th frame
        
    def update(self, frame):
        """
        Detects lane lines and returns dynamic (x1, x2) bounds.
        Fallback to None if detection fails.
        """
        if frame is None: return None, None
        
        self.frame_count += 1
        
        # Optimization: Return cached values if not on interval
        if self.initialized and self.frame_count % self.skip_interval != 0:
             return int(self.avg_x1), int(self.avg_x2)
             
        h, w = frame.shape[:2]
        
        # Region of Interest: Bottom Half only
        roi = frame[h//2:, :]
        
        # 1. Edge Detection / Color Filtering
        # White properties
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        # 2. Hough Lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=50)
        
        left_lines = []
        right_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Avoid vertical/horizontal noise
                if x2 - x1 == 0: continue
                slope = (y2 - y1) / (x2 - x1)
                
                # Check for reasonable slopes for road lanes
                # Left lane: Negative slope (usually -0.5 to -2.0)
                # Right lane: Positive slope (usually 0.5 to 2.0)
                if -2.5 < slope < -0.3:
                    left_lines.append(line[0])
                elif 0.3 < slope < 2.5:
                    right_lines.append(line[0])
                    
        # 3. Fit Average Lines (Using x-intercept at bottom of frame)
        
        current_x1 = None
        current_x2 = None
        
        if left_lines:
            try:
                # Collect all points
                lx = []
                ly = []
                for l in left_lines:
                    lx.extend([l[0], l[2]])
                    ly.extend([l[1], l[3]])
                poly_left = np.poly1d(np.polyfit(ly, lx, 1)) # Fit x as function of y
                current_x1 = int(poly_left(h//2)) # x at bottom of ROI 
            except: pass

        if right_lines:
            try:
                rx = []
                ry = []
                for l in right_lines:
                    rx.extend([l[0], l[2]])
                    ry.extend([l[1], l[3]])
                poly_right = np.poly1d(np.polyfit(ry, rx, 1))
                current_x2 = int(poly_right(h//2))
            except: pass
            
        # Defaults if detection fails (Centered 50%)
        default_x1 = int(w * 0.25)
        default_x2 = int(w * 0.75)
        
        # 4. Smoothing (EMA)
        if not self.initialized:
            self.avg_x1 = current_x1 if current_x1 else default_x1
            self.avg_x2 = current_x2 if current_x2 else default_x2
            self.initialized = True
        else:
            if current_x1:
                self.avg_x1 = self.alpha * current_x1 + (1 - self.alpha) * self.avg_x1
            else:
                # Decay towards default if lost
                self.avg_x1 = self.alpha * default_x1 + (1 - self.alpha) * self.avg_x1
                
            if current_x2:
                self.avg_x2 = self.alpha * current_x2 + (1 - self.alpha) * self.avg_x2
            else:
                 self.avg_x2 = self.alpha * default_x2 + (1 - self.alpha) * self.avg_x2
                 
        # Safety Clamps
        self.avg_x1 = max(0, min(self.avg_x1, w//2 - 50))
        self.avg_x2 = max(w//2 + 50, min(self.avg_x2, w))
        
        return int(self.avg_x1), int(self.avg_x2)

class SimpleTracker:
    def __init__(self, max_missed=5, iou_threshold=0.3):
        self.next_id = 1
        self.tracks = {} # {id: {'box': [x1,y1,x2,y2], 'missed': 0}}
        self.max_missed = max_missed
        self.iou_threshold = iou_threshold

    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def update(self, detected_boxes):
        """
        detected_boxes: list of [x1, y1, x2, y2]
        Returns: list of {'id': int, 'box': [x1,y1,x2,y2]}
        """
        updated_tracks = []
        
        # Greedy Matching
        # Create a matrix of IoUs? Or just iterate.
        # For simplicity/speed without scipy:
        
        # 1. Prediction (assume static for simple version, or Kalman but that's complex)
        # We will match existing tracks to new boxes.
        
        matched_track_ids = set()
        matched_detection_indices = set()
        
        # Sort existing tracks? No.
        
        matches = [] # (track_id, det_idx, iou)
        
        for t_id, t_data in self.tracks.items():
            t_box = t_data['box']
            for d_idx, d_box in enumerate(detected_boxes):
                 score = self.iou(t_box, d_box)
                 if score > self.iou_threshold:
                     matches.append((t_id, d_idx, score))
                     
        # Sort matches by score descending
        matches.sort(key=lambda x: x[2], reverse=True)
        
        for t_id, d_idx, score in matches:
            if t_id in matched_track_ids or d_idx in matched_detection_indices:
                continue
            
            matched_track_ids.add(t_id)
            matched_detection_indices.add(d_idx)
            
            # Update track
            self.tracks[t_id]['box'] = detected_boxes[d_idx]
            self.tracks[t_id]['missed'] = 0
            updated_tracks.append({'id': t_id, 'box': detected_boxes[d_idx]})
            
        # Handle New Detections
        for d_idx, d_box in enumerate(detected_boxes):
            if d_idx not in matched_detection_indices:
                # New Track
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = {'box': d_box, 'missed': 0}
                updated_tracks.append({'id': new_id, 'box': d_box})
                
        # Handle Lost Tracks
        to_remove = []
        for t_id in self.tracks:
            if t_id not in matched_track_ids:
                self.tracks[t_id]['missed'] += 1
                if self.tracks[t_id]['missed'] > self.max_missed:
                    to_remove.append(t_id)
        
        for t_id in to_remove:
            del self.tracks[t_id]
            
        return updated_tracks

class HazardStabilizer:
    def __init__(self, frame_width=1920, frame_height=1080, buffer_frames=5):
        """
        Args:
            frame_width (int): Video width (e.g., 1920 for 1080p)
            frame_height (int): Video height (e.g., 1080 for 1080p)
            buffer_frames (int): Number of consecutive frames a hazard must be seen to trigger alert.
        """
        self.width = frame_width
        self.height = frame_height
        self.buffer_frames = buffer_frames
        
        # Track potential hazards: {track_id: consecutive_count}
        self.hazard_counter = {} 
        
        # --- DEFINE THE DANGER ZONES (3 Levels) ---
        # Normalized coordinates (0.0 to 1.0)
        
        # 1. GREEN ZONE (Far) - Early Warning
        # Top: y=0.55, Bottom: y=0.65
        g_p1 = (0.25, 0.55)  # Top Left
        g_p2 = (0.75, 0.55)  # Top Right
        g_p3 = (0.60, 0.65)  # Bottom Right (Matches Yellow Top)
        g_p4 = (0.40, 0.65)  # Bottom Left (Matches Yellow Top)

        # 2. YELLOW ZONE (Medium) - Standard Alert
        # Top: y=0.65, Bottom: y=0.75
        y_p1 = (0.40, 0.65)
        y_p2 = (0.60, 0.65)
        y_p3 = (0.70, 0.75)  # Bottom Right
        y_p4 = (0.30, 0.75)  # Bottom Left

        # 3. RED ZONE (Close) - Critical Brake
        # Top: y=0.75, Bottom: y=1.0
        r_p1 = (0.30, 0.75)
        r_p2 = (0.70, 0.75)
        r_p3 = (1.0, 1.0)
        r_p4 = (0.0, 1.0)
        
        # 4. SIDE ZONES (Green/Info)
        # Left Side: Fill area between x=0 and Yellow/Red left edge
        sl_p1 = (0.0, 0.65)   # Top Left Corner of side zone
        sl_p2 = y_p1          # Connect to Yellow Top Left
        sl_p3 = r_p1          # Connect to Red Top Left
        sl_p4 = r_p4          # Connect to Red Bottom Left
        sl_p5 = (0.0, 1.0)    # Bottom Left Corner
        
        # Right Side: Fill area between Yellow/Red right edge and x=1
        sr_p1 = y_p2          # Connect to Yellow Top Right
        sr_p2 = (1.0, 0.65)   # Top Right Corner
        sr_p3 = (1.0, 1.0)    # Bottom Right Corner
        sr_p4 = r_p3          # Connect to Red Bottom Right
        sr_p5 = r_p2          # Connect to Red Top Right

        def make_poly(pts):
            return np.array([
                [int(p[0]*self.width), int(p[1]*self.height)] for p in pts
            ], dtype=np.int32)

        self.poly_green = make_poly([g_p4, g_p1, g_p2, g_p3]) 
        self.poly_yellow = make_poly([y_p4, y_p1, y_p2, y_p3])
        self.poly_red = make_poly([r_p4, r_p1, r_p2, r_p3])
        
        self.poly_side_left = make_poly([sl_p5, sl_p1, sl_p2, sl_p3, sl_p4])
        self.poly_side_right = make_poly([sr_p4, sr_p5, sr_p1, sr_p2, sr_p3])
        
        # Keep roi_poly as union for backward compat or broad checks?
        # Let's just keep it as the total area for "is generally inside"
        # Total area is Green + Yellow + Red
        self.roi_poly = make_poly([r_p4, g_p1, g_p2, r_p3]) # Approx outer shell

    def get_zone_level(self, box):
        """
        Checks which zone the box center is in.
        Returns: 0 (None), 1 (Green), 2 (Yellow), 3 (Red)
        """
        x1, y1, x2, y2 = box
        center_x = int((x1 + x2) / 2)
        center_y = int(y2) # Bottom center
        
        # Check Red first (Most critical)
        if cv2.pointPolygonTest(self.poly_red, (center_x, center_y), False) >= 0:
            return 3
        # Check Yellow
        if cv2.pointPolygonTest(self.poly_yellow, (center_x, center_y), False) >= 0:
            return 2
        # Check Green
        if cv2.pointPolygonTest(self.poly_green, (center_x, center_y), False) >= 0:
            return 1
            
        # Check Side Zones (Treat as Level 1 / Info)
        if cv2.pointPolygonTest(self.poly_side_left, (center_x, center_y), False) >= 0:
            return 1
        if cv2.pointPolygonTest(self.poly_side_right, (center_x, center_y), False) >= 0:
            return 1
            
        return 0

    def is_in_danger_zone(self, box):
        # Backward compatibility wrapper
        return self.get_zone_level(box) > 0

    def update(self, detections, current_frame_detected_ids):
        """
        Filter detections and update persistence counters.
        """
        valid_hazards = []
        
        # 1. Decay missing hazards
        track_ids_to_remove = []
        for tid in self.hazard_counter:
            if tid not in current_frame_detected_ids:
                self.hazard_counter[tid] -= 1 # Decay
                if self.hazard_counter[tid] <= 0:
                    track_ids_to_remove.append(tid)
        
        for tid in track_ids_to_remove:
            del self.hazard_counter[tid]

        # 2. Process new detections
        for det in detections:
            track_id = det.get('id', -1)
            
            # --- FILTER 1: SPATIAL ---
            zone_level = self.get_zone_level(det['box'])
            if zone_level == 0:
                continue 
            
            # Inject zone level into detection dict for Analyzer/Monitor
            det['zone'] = zone_level 

            # --- FILTER 2: LOGICAL CORRECTIONS ---
            
            # REMOVED: Geometric/Size filters that were suppressing accidents
            # We trust the custom model + new confidence thresholds
            
            # B) OVERLAP SUPPRESSION (Person > Accident)
            # If we detect an "Accident" (10) (and it wasn't fixed above) but it overlaps 
            # significantly with a "Person" (0), ignore the accident.
            if det['class_id'] == 10: 
                frame_has_person = False
                accident_box = det['box']
                
                # Check against other detections in this frame
                for other in detections:
                    if other['class_id'] == 0: # Person
                         # Simple Intersection check
                         p_box = other['box']
                         xA = max(accident_box[0], p_box[0])
                         yA = max(accident_box[1], p_box[1])
                         xB = min(accident_box[2], p_box[2])
                         yB = min(accident_box[3], p_box[3])
                         
                         if xB > xA and yB > yA:
                             frame_has_person = True
                             break
                
                if frame_has_person:
                    continue

            # --- FILTER 3: TEMPORAL ---
            if track_id != -1:
                self.hazard_counter[track_id] = self.hazard_counter.get(track_id, 0) + 1
                
                # Only return as a valid hazard if it has been seen for N frames
                if self.hazard_counter[track_id] >= self.buffer_frames:
                    valid_hazards.append(det)
            else:
                valid_hazards.append(det)

        return valid_hazards

    def draw_debug_zone(self, frame):
        """Draws the 3 danger zones on the frame."""
        # Green (Far) - BGR: (0, 255, 0)
        cv2.polylines(frame, [self.poly_green], isClosed=True, color=(0, 255, 0), thickness=2)
        # Side Zones (Green)
        cv2.polylines(frame, [self.poly_side_left], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(frame, [self.poly_side_right], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Yellow (Medium) - BGR: (0, 255, 255)
        cv2.polylines(frame, [self.poly_yellow], isClosed=True, color=(0, 255, 255), thickness=2)
        # Red (Close) - BGR: (0, 0, 255)
        cv2.polylines(frame, [self.poly_red], isClosed=True, color=(0, 0, 255), thickness=2)
        return frame

class HazardAnalyzer:
    def __init__(self):
        # Store history: {track_id: {'area': [float], 'center': [(x,y)], 'last_seen': timestamp}}
        self.track_history = {}
        # Thresholds
        self.fast_approach_threshold = 0.05 # 5% growth per frame approx
        self.side_aspect_ratio_threshold = 1.6 # Width / Height
        
    def analyze(self, boxes, names, frame_shape=None, lane_bounds=None):
        alerts = []
        
        current_ids = []
        
        if boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
        else:
            track_ids = [-1] * len(boxes) # No tracking available
            
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            track_id = track_ids[i]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w = x2 - x1
            h = y2 - y1
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            name = names[cls_id]
            zone_level = getattr(box, 'zone', 0) # 3=Red, 2=Yellow, 1=Green

            # Determine direction if frame_shape is provided
            direction_str = ""
            center_x = (x1 + x2) / 2
            if frame_shape:
                w_frame = frame_shape[1] # shape is (H, W, C) usually, or (H,W)
                if center_x < w_frame * 0.33:
                    direction_str = " (Left)"
                elif center_x > w_frame * 0.66:
                    direction_str = " (Right)"
                else:
                    direction_str = " (Ahead)"

            # Lane Logic
            # Use dynamic bounds if provided, else fallback to 25%-75%
            lane_x1 = 0
            lane_x2 = 0
            
            if lane_bounds:
                lane_x1, lane_x2 = lane_bounds
            elif frame_shape:
                w_frame = frame_shape[1]
                lane_x1 = w_frame * 0.25
                lane_x2 = w_frame * 0.75
                
            in_lane = False
            # Check if center is within lane bounds
            if center_x > lane_x1 and center_x < lane_x2:
                 in_lane = True
            
            # Tracking for Approach/Merge
            is_approaching = False
            dx = 0
            
            if track_id != -1:
                # History update moved to main loop below for efficiency
                pass 

            # Refactored Loop Logic:
            # 1. Update History
            # Track known classes OR large unknown objects
            is_known = cls_id in VEHICLE_CLASSES + BIKER_CLASSES + HAZARD_CLASSES
            should_track = is_known or (area > 3000)

            if track_id != -1 and should_track:
                current_ids.append(track_id)
                if track_id not in self.track_history:
                    self.track_history[track_id] = {'area': [], 'center': []}
                
                history = self.track_history[track_id]
                history['area'].append(area)
                history['center'].append((center_x, (y1+y2)/2))
                
                if len(history['area']) > 10: history['area'].pop(0)
                if len(history['center']) > 10: history['center'].pop(0)
                
                # Calculate DX
                if len(history['center']) >= 4:
                    # avg movement over last 3 frames
                    prev_x = history['center'][-4][0]
                    dx = center_x - prev_x # + means moving Right, - means moving Left
                    
                    # Check Approach
                    if frame_shape:
                        # Left of lane and moving Right (substantial move)
                        if center_x < lane_x1 and dx > 5: 
                            is_approaching = True
                        # Right of lane and moving Left
                        elif center_x > lane_x2 and dx < -5:
                            is_approaching = True

            # 2. Generate Alerts
            
            # Determine Prefix based on Zone (and Type)
            # Red (3) -> CRITICAL BRAKE
            # Yellow (2) -> CAUTION
            # Green (1) -> INFO / AHEAD
            
            prefix = ""
            if zone_level == 3:
                prefix = "CRITICAL BRAKE: "
            elif zone_level == 2:
                prefix = "CAUTION: "
            elif zone_level == 1:
                prefix = "AHEAD: "
            
            # --- GENERATE ALERT STRING ---
            alert_str = ""
            
            # BIKER
            if cls_id in BIKER_CLASSES:
                if zone_level == 3:
                     alert_str = f"CRITICAL BRAKE: BIKER!"
                else:
                     alert_str = f"{prefix}BIKER {direction_str}"

            # HAZARD (Animals/Peds)
            elif cls_id in HAZARD_CLASSES:
                if zone_level == 3:
                     alert_str = f"CRITICAL BRAKE: {name}!"
                else:
                     alert_str = f"{prefix}{name}{direction_str}"

            # VEHICLES (Standard)
            elif cls_id in VEHICLE_CLASSES:
                 if zone_level == 3:
                      # Red zone vehicle -> BRAKE
                      alert_str = f"CRITICAL BRAKE: {name}!"
                 elif is_approaching:
                      alert_str = f"MERGING: {name}"
                 elif zone_level >= 2:
                      # Yellow zone vehicle -> Caution
                      alert_str = f"{prefix}{name}"
                 # Green zone vehicles are ignored unless approaching?
                 elif zone_level == 1 and is_approaching:
                       alert_str = f"AHEAD: {name}"

            # TRAFFIC SIGNS
            elif cls_id in TRAFFIC_CLASSES:
                if cls_id == 11 and zone_level >= 2: # Stop Sign in Yellow/Red
                     alert_str = f"STOP SIGN"
                elif cls_id == 9 and zone_level >= 2:
                     alert_str = f"TRAFFIC LIGHT"
            
            # GENERIC OBSTACLE
            elif (cls_id not in VEHICLE_CLASSES and 
                  cls_id not in BIKER_CLASSES and 
                  cls_id not in HAZARD_CLASSES and
                  cls_id not in TRAFFIC_CLASSES):
                  if area > 3000:
                      if zone_level == 3:
                          alert_str = "CRITICAL BRAKE: OBSTACLE!"
                      elif zone_level >= 1:
                          alert_str = f"{prefix}OBSTACLE"

            if alert_str:
                alerts.append(alert_str)
            
            # 3. Fast Approach (Only if in lane)
            # Apply to ALL tracked objects (Vehicles + Generic Obstacles)
            is_valid_target = (cls_id in VEHICLE_CLASSES) or (area > 3000 and cls_id not in BIKER_CLASSES + HAZARD_CLASSES)
            
            if in_lane and track_id != -1 and is_valid_target:
                 # Logic ... (use existing history)
                 history = self.track_history.get(track_id)
                 if history and len(history['area']) >= 4:
                    prev_area = history['area'][-4]
                    if prev_area > 0:
                        growth = (area - prev_area) / prev_area
                        disp_name = name if (cls_id in VEHICLE_CLASSES) else "OBSTACLE"
                        
                        # Only issue CRASH/BRAKE alerts if in Red/Yellow zones
                        # Logic: Red = Brake, Yellow = Caution
                        if zone_level == 3:
                            if growth > 0.30 and area > 1500:
                                 alerts.append(f"CRASH IMMINENT (<1s): {disp_name}")
                            elif growth > 0.10 and area > 1500:
                                 alerts.append(f"CRASH WARNING (<2s): {disp_name}")
                        elif zone_level == 2:
                            if growth > 0.10 and area > 1500:
                                 alerts.append(f"CAUTION: Fast Approach {disp_name}")

            # 4. Side Traffic (Refined)
            # Only if NOT in lane (obviously) and is very wide/close
            if not in_lane and aspect_ratio > self.side_aspect_ratio_threshold and area > 5000:
                 alerts.append(f"SIDE TRAFFIC: {name}")
                     
        # 5. Crash / Overlap Detection
        # Simple O(N^2) check for significant overlaps between vehicles
        # Using boxes list as it contains current frame boxes
        vehicle_boxes = []
        for i, box in enumerate(boxes):
            if int(box.cls[0]) in VEHICLE_CLASSES:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                vehicle_boxes.append({'box': (x1,y1,x2,y2), 'id': track_ids[i]})
        
        for i in range(len(vehicle_boxes)):
            for j in range(i + 1, len(vehicle_boxes)):
                b1 = vehicle_boxes[i]['box']
                b2 = vehicle_boxes[j]['box']
                vid1 = vehicle_boxes[i]['id']
                vid2 = vehicle_boxes[j]['id']
                
                # Determine intersection rectangle
                x_left = max(b1[0], b2[0])
                y_top = max(b1[1], b2[1])
                x_right = min(b1[2], b2[2])
                y_bottom = min(b1[3], b2[3])

                if x_right > x_left and y_bottom > y_top:
                    intersection_area = (x_right - x_left) * (y_bottom - y_top)
                    area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
                    area2 = (b2[2]-b2[0]) * (b2[3]-b1[1])
                    iou = intersection_area / float(area1 + area2 - intersection_area)
                    
                    if iou > 0.30: # Significant overlap
                        alerts.append("POSSIBLE CRASH")
                        
        # 6. Lane Deviation (Flow Heuristic)
        # Disabled
        
        return list(set(alerts)) # Deduplicate

def draw_alert(frame, message, color=(0, 0, 255)):
    """Draws a prominent alert message on the frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.2
    thickness = 3
    text_size = cv2.getTextSize(message, font, scale, thickness)[0]
    
    # Top center position
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = 50
    
    # Draw background rectangle for visibility
    cv2.rectangle(frame, (text_x - 10, text_y - 35), (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
    cv2.putText(frame, message, (text_x, text_y), font, scale, color, thickness)

def get_hazard_level(class_id):
    """Returns a hazard level string/color based on class ID."""
    if class_id in HAZARD_CLASSES:
        return "HAZARD", (0, 0, 255) # Red
    return "NORMAL", (0, 255, 0) # Green

class DistanceMonitor:
    def __init__(self, frame_width=1920, focal_length_px=1000):
        """
        Estimates distance to vehicles and warns if too close.
        Assumes standard dashcam FOV.
        """
        self.focal_length = focal_length_px
        self.frame_width = frame_width
        
        # Real world widths (approx meters)
        self.real_widths = {
            2: 1.8,  # Car
            4: 2.5,  # Bus
            5: 2.5,  # Truck
            3: 0.8,  # Motorcycle
            1: 0.6,  # Bike
            0: 0.5   # Person
        }
        
        self.critical_distance = 15.0 # meters (Warning threshold)
        self.tailgating_distance = 8.0 # meters (Danger threshold - typical stopping dist at city speed)

    def estimate_distance(self, box, class_id):
        """
        Distance = (Real Width * Focal Length) / Image Width
        """
        x1, y1, x2, y2 = box
        img_w = x2 - x1
        if img_w <= 0: return 999.0
        
        real_w = self.real_widths.get(class_id, 1.5) # Default 1.5m
        
        distance = (real_w * self.focal_length) / img_w
        return distance

    def check_safe_distance(self, detections):
        """
        Analyzes detections to find the 'Leading Vehicle' (center lane, closest).
        Returns: (status_message, color_code, distance, box)
        """
        # Find car in center lane
        center_x = self.frame_width // 2
        lane_center_threshold = self.frame_width * 0.3 # Must be within center 30%
        
        closest_dist = 999.0
        closest_obj = None
        
        for det in detections:
            # Check only vehicles
            if det['class_id'] not in [2, 3, 4, 5]: # Car, Moto, Bus, Truck
                continue
                
            x1, y1, x2, y2 = det['box']
            obj_center = (x1 + x2) / 2
            
            # Is it in front of us? (Center of screen)
            if abs(obj_center - center_x) < lane_center_threshold:
                dist = self.estimate_distance(det['box'], det['class_id'])
                if dist < closest_dist:
                    closest_dist = dist
                    closest_obj = det
                    
        if closest_obj:
            # Check zone of closest object for Critical Alert
            # Logic: Yellow = Caution, Red = Brake
            obj_zone = closest_obj.get('zone', 0)
            
            if closest_dist < self.tailgating_distance:
                if obj_zone == 3: # Red
                    return "CRITICAL: BRAKE!", (0, 0, 255), closest_dist, closest_obj['box']
                elif obj_zone == 2: # Yellow
                     return "CAUTION: Too Close", (0, 165, 255), closest_dist, closest_obj['box']
                else: 
                     return "WARNING: Too Close", (0, 165, 255), closest_dist, closest_obj['box']
            elif closest_dist < self.critical_distance:
                return "WARNING: Too Close", (0, 165, 255), closest_dist, closest_obj['box']
            else:
                return f"Distance: {closest_dist:.1f}m", (0, 255, 0), closest_dist, closest_obj['box']
        
        return None, (0, 255, 0), None, None
