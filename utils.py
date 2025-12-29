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
        
        # --- DEFINE THE DANGER ZONE (Trapezoid) ---
        # Normalized coordinates (0.0 to 1.0)
        # User specified "Red Line" - refined V3
        # Even lower top edge (0.70), narrower top width (0.27)
        p1 = (0.0, 1.0)    # Bottom Left
        p2 = (0.365, 0.70) # Top Left
        p3 = (0.635, 0.70) # Top Right
        p4 = (1.0, 1.0)    # Bottom Right

        # Convert to pixels
        roi_points = np.array([
            [int(p1[0]*self.width), int(p1[1]*self.height)],
            [int(p2[0]*self.width), int(p2[1]*self.height)],
            [int(p3[0]*self.width), int(p3[1]*self.height)],
            [int(p4[0]*self.width), int(p4[1]*self.height)]
        ], dtype=np.int32)
        
        self.roi_poly = roi_points

    def is_in_danger_zone(self, box, mask=None):
        """
        Checks if the center of the bounding box is inside the driving lane (mask).
        If mask is None (e.g. not yet generated), returns False (or could default to True/Old Logic).
        box format: [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = box
        center_x = int((x1 + x2) / 2)
        center_y = int(y2) # Use the bottom-center point
        
        if mask is None:
            # Fallback: strict safety or maintain old trapezoid if really needed. 
            # For now, safer to assume NOT in danger if we don't know the road.
            return False
            
        # Check image bounds
        h, w = mask.shape
        if 0 <= center_x < w and 0 <= center_y < h:
            # Mask value should be 1 (True) for road
            return mask[center_y, center_x] > 0
            
        return False

    def update(self, detections, current_frame_detected_ids, mask=None):
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
            if not self.is_in_danger_zone(det['box']):
                continue 

            # --- FILTER 2: LOGICAL CORRECTIONS ---
            
            # A) GEOMETRIC CORRECTION (Accident -> Person)
            # If the model detects "Accident" (10) but the box is taller than it is wide (like a person),
            # it is almost certainly a misclassified person.
            # Aspect Ratio = Height / Width
            w = det['box'][2] - det['box'][0]
            h = det['box'][3] - det['box'][1]
            aspect_ratio = h / w if w > 0 else 0
            
            # Use stricter threshold: if tall (> 1.2 ratio), likely a person/pole, NOT a car crash.
            if det['class_id'] == 10 and aspect_ratio > 1.2:
                # Force change to Person (0)
                det['class_id'] = 0
                det['class_name'] = 'Person'
            
            # C) SIZE FILTER (Accidents must be significant)
            # A car accident involves vehicles. Small boxes are likely noise/animals/people.
            box_area = w * h
            frame_area = self.width * self.height
            if det['class_id'] == 10 and (box_area / frame_area) < 0.015: 
                # Less than 1.5% of screen -> Ignore
                continue

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

    def draw_debug_zone(self, frame, mask=None):
        """Draws the danger zone (mask) on the frame for debugging."""
        if mask is not None:
             # Create a green overlay
             color_mask = np.zeros_like(frame)
             color_mask[mask > 0] = [0, 255, 0] # Green
             frame = cv2.addWeighted(frame, 1.0, color_mask, 0.3, 0)
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
                # We need to update history here first before checking approach
                # (History update code block was below, let's move it up or duplicate/reference)
                # Actually, let's just do the history update in the main block below, 
                # but we need 'dx' for the Alert generation which happens... 
                # modifying the structure slightly to update history FIRST.
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
            
            # BIKER
            if cls_id in BIKER_CLASSES:
                if in_lane:
                    alerts.append(f"BIKER AHEAD: {name}")
                elif is_approaching:
                    alerts.append(f"BIKER MERGING: {name}")
                else:
                     # Peripheral biker - minor alert or suppress? 
                     # User wants to know about object approaching, so maybe just "Biker (Left)"
                     pass # We will rely on the "Biker (Left)" generic or suppress if far?
                     # Let's keep existing behavior but maybe refine text
                     alerts.append(f"BIKER: {name}{direction_str}")

            # HAZARD (Animals/Peds)
            elif cls_id in HAZARD_CLASSES:
                if in_lane or is_approaching:
                    prefix = "HAZARD AHEAD" if in_lane else "HAZARD APPROACHING"
                    alerts.append(f"{prefix}: {name}")
                else:
                    # Still warn for Peds even if outside, but maybe less priority?
                    # kept for safety
                    alerts.append(f"HAZARD: {name}{direction_str}")

            # VEHICLES (Standard)
            # Only alert if: Fast Approach (in lane), Merging (approaching), or Crash
            # Standard "Car on left" is noise.
            elif cls_id in VEHICLE_CLASSES:
                 if is_approaching:
                     alerts.append(f"VEHICLE MERGING: {name}")

            # TRAFFIC SIGNS & LIGHTS
            elif cls_id in TRAFFIC_CLASSES:
                if in_lane:
                    if cls_id == 11:
                         alerts.append(f"STOP SIGN AHEAD")
                    elif cls_id == 9:
                         alerts.append(f"TRAFFIC LIGHT AHEAD")
            
            # GENERIC OBSTACLE (SAM / Unknown)
            elif (cls_id not in VEHICLE_CLASSES and 
                  cls_id not in BIKER_CLASSES and 
                  cls_id not in HAZARD_CLASSES and
                  cls_id not in TRAFFIC_CLASSES):
                  if area > 3000:
                      if in_lane:
                          alerts.append(f"OBSTACLE AHEAD")
                      elif is_approaching:
                          alerts.append(f"OBSTACLE MERGING")
            
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
                        
                        if growth > 0.30 and area > 1500:
                             alerts.append(f"CRASH IMMINENT (<1s): {disp_name}")
                        elif growth > 0.10 and area > 1500:
                             alerts.append(f"CRASH WARNING (<2s): {disp_name}")

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
                    
                    if iou > 0.15: # Significant overlap
                        alerts.append("POSSIBLE CRASH")
                        
        # 6. Lane Deviation (Flow Heuristic)
        # Calculate flow of all tracked objects
        # If all valid objects shift LEFT, we are drifting RIGHT (and vice versa)
        deltas = []
        for tid in current_ids:
            hist = self.track_history.get(tid, {}).get('center', [])
            if len(hist) >= 2:
                # Compare last two points
                dx = hist[-1][0] - hist[-2][0]
                deltas.append(dx)
        
        if len(deltas) > 2:
            avg_dx = sum(deltas) / len(deltas)
            # Threshold for drift (pixels per frame)
            # If everything moves left (negative dx) -> we drift right
            if avg_dx < -5: 
                alerts.append("LANE DEVIATION: RIGHT")
            elif avg_dx > 5:
                alerts.append("LANE DEVIATION: LEFT")

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
            if closest_dist < self.tailgating_distance:
                return "CRITICAL: BRAKE!", (0, 0, 255), closest_dist, closest_obj['box']
            elif closest_dist < self.critical_distance:
                return "WARNING: Too Close", (0, 165, 255), closest_dist, closest_obj['box']
            else:
                return f"Distance: {closest_dist:.1f}m", (0, 255, 0), closest_dist, closest_obj['box']
        
        return None, (0, 255, 0), None, None
