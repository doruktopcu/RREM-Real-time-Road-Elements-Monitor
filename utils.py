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
        
    def analyze(self, boxes, names):
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
            
            # 1. Biker Detection
            if cls_id in BIKER_CLASSES:
                alerts.append(f"BIKER: {name}")
                
            # 2. Vulnerable User / Animal (Existing Hazard)
            if cls_id in HAZARD_CLASSES and cls_id not in BIKER_CLASSES:
                alerts.append(f"HAZARD: {name}")
                
            # Logic requiring tracking
            if track_id != -1 and cls_id in VEHICLE_CLASSES:
                current_ids.append(track_id)
                
                # Update history
                if track_id not in self.track_history:
                    self.track_history[track_id] = {'area': [], 'center': []}
                
                history = self.track_history[track_id]
                history['area'].append(area)
                # Keep last 10 frames
                if len(history['area']) > 10:
                    history['area'].pop(0)
                    
                # 3. Fast Approach Detection
                # User Request: Compare with 3 frames ago, check for 20% size increase
                if len(history['area']) >= 4: # Current + 3 previous
                    prev_area = history['area'][-4] # 3 frames ago (1-based index from end: -1=curr, -2=prev1, -3=prev2, -4=prev3)
                    if prev_area > 0:
                        growth = (area - prev_area) / prev_area
                        if growth > 0.20: # 20% growth threshold
                             alerts.append(f"FAST APPROACH: {name}")

                # 4. Side View / Merging Detection
                # Heuristic: Wide aspect ratio + not too small (close enough to matter)
                if aspect_ratio > self.side_aspect_ratio_threshold and area > 5000:
                     alerts.append(f"SIDE TRAFFIC: {name}")
                     
        # Clean up old tracks
        # (Simple cleanup: if track_id not in current_ids, eventually remove. 
        # For simplicity in this script, we might skip complex cleanup or do it periodically)
        
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
