import cv2

# COCO Class Mapping (Standard YOLOv8/11)
# We can use the model's names dict, but having some constants here helps.

# Vulnerable Road Users and Animal Hazards
HAZARD_CLASSES = [
    0,  # person
    1,  # bicycle
    3,  # motorcycle
    14, # bird
    15, # cat
    16, # dog
    17, # horse
    18, # sheep
    19, # cow
    20, # elephant
    21, # bear
    22, # zebra
    23, # giraffe
]

# Vehicle Classes
VEHICLE_CLASSES = [
    2,  # car
    3,  # motorcycle
    5,  # bus
    7,  # truck
]

BIKER_CLASSES = [
    1, # bicycle
    3 # motorcycle
]

class HazardAnalyzer:
    def __init__(self):
        # Store history: {track_id: {'area': [float], 'center': [(x,y)], 'last_seen': timestamp}}
        self.track_history = {}
        # Thresholds
        self.fast_approach_threshold = 0.05 # 5% growth per frame approx
        self.side_aspect_ratio_threshold = 1.6 # Width / Height
        
    def analyze(self, boxes, names, frame_shape=None):
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
            # Define Lane roughly as 25% to 75% of width
            lane_x1 = 0
            lane_x2 = 0
            if frame_shape:
                w_frame = frame_shape[1]
                lane_x1 = w_frame * 0.25
                lane_x2 = w_frame * 0.75
                
            in_lane = False
            if frame_shape:
                if center_x > lane_x1 and center_x < lane_x2:
                    in_lane = True
            else:
                in_lane = True # Default to true if no shape to filter safely? Or False?
                # Let's default to True for safety if shape unknown
            
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
            if track_id != -1 and cls_id in VEHICLE_CLASSES + BIKER_CLASSES + HAZARD_CLASSES: # Track everything relevant
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
            
            # 3. Fast Approach (Only if in lane)
            if in_lane and track_id != -1:
                 # Logic ... (use existing history)
                 history = self.track_history.get(track_id)
                 if history and len(history['area']) >= 4:
                    prev_area = history['area'][-4]
                    if prev_area > 0:
                        growth = (area - prev_area) / prev_area
                        if growth > 0.30 and area > 1500:
                             alerts.append(f"CRASH IMMINENT (<1s): {name}")
                        elif growth > 0.10 and area > 1500:
                             alerts.append(f"CRASH WARNING (<2s): {name}")

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
