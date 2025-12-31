import os
# Enable MPS fallback for SAM2


import argparse
import time
import torch
import cv2
import os
import threading
from ultralytics import YOLO
from utils import HazardAnalyzer, draw_alert, LaneTracker, SimpleTracker, HazardStabilizer, DistanceMonitor



class RREMMonitor:
    def __init__(self, model_path="yolo11n.pt", conf_threshold=0.40):
        # Hardware Acceleration Check
        self.device = 'cpu'
        if torch.backends.mps.is_available():
            self.device = 'mps'
            print("Using Apple Metal (MPS) acceleration!")
        elif torch.cuda.is_available():
            self.device = 'cuda'
            print("Using CUDA acceleration!")
            
        self.conf_threshold = conf_threshold
        self.all_detected_hazards = set()
        self.prev_time = 0
        
        # Persistence & Voice State
        self.alert_display_until = 0
        self.current_display_message = ""
        self.last_speech_time = 0
        
        # State for capture
        self.cap = None
        self.source = None
        
        # Trackers and Analyzers
        self.lane_tracker = LaneTracker()
        self.tracker = SimpleTracker(max_missed=10)
        self.stabilizer = HazardStabilizer(buffer_frames=5) # Increased buffer for stability
        self.analyzer = HazardAnalyzer()
        self.distance_monitor = DistanceMonitor()
        


        
        # Load Model
        self.model = None
        self.load_model(model_path)
        
    def download_weights_if_needed(self, model_path):
        """Downloads standard YOLO weights if missing, using curl to avoid Python SSL issues."""
        if os.path.exists(model_path):
            return True
            
        # Only download standard YOLOv11 weights
        if model_path not in ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]:
            return False
            
        print(f"Weights {model_path} not found. Downloading...")
        url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{model_path}"
        
        try:
            import subprocess
            # Use curl -L (follow redirects) -k (insecure/skip SSL check if needed, though usually system curl allows it)
            # We'll try without -k first, but if user system has issues, -k might be needed. 
            # Given the error was "unable to get local issuer certificate" in python, system curl usually has its own CA bundle or works better.
            res = subprocess.run(["curl", "-L", "-o", model_path, url], check=True)
            if res.returncode == 0 and os.path.exists(model_path):
                print(f"Successfully downloaded {model_path}")
                return True
        except Exception as e:
            print(f"Failed to download with curl: {e}")
            
        return False

    def load_model(self, model_path):
        """Loads a new model. Returns True if successful, False otherwise."""
        print(f"Loading model: {model_path} on {self.device}...")
        try:
            # Ensure weights exist
            self.download_weights_if_needed(model_path)
            
            # Default to Ultralytics
            self.model = YOLO(model_path)
            # Basic validation
            if not self.model:
                raise ValueError("Model failed to load.")
                
            return True, "Success"
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            return False, str(e)
    
    def speak_warning(self, text):
        """Spawns a thread to speak the warning (MacOS 'say')."""
        def speak():
            # Clean text for command line safety
            safe_text = text.replace("'", "").replace('"', "")
            os.system(f'say "{safe_text}"')
            
        t = threading.Thread(target=speak, daemon=True)
        t.start()

    
    def start_capture(self, source):
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open source {source}")
            
    def stop_capture(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def process_frame(self):
        """
        Reads and processes a single frame.
        Returns: (frame, detections_list) or (None, []) if end of stream.
        """
        if not self.cap or not self.cap.isOpened():
            return None, []

        ret, frame = self.cap.read()
        if not ret:
            return None, []

        # Filter classes based on model type
        # COCO models have 80 classes. Our custom model has 11.
        relevant_classes = None
        if len(self.model.names) > 20:
             # Standard COCO Model - Filter irrelevant stuff
             # Person(0), Bike(1), Car(2), Motorcycle(3), Bus(5), Truck(7), Cat(15), Dog(16), Traffic Light(9), Stop Sign(11)
             relevant_classes = [0, 1, 2, 3, 5, 7, 9, 11, 15, 16]
        
        results = self.model.track(frame, persist=True, conf=self.conf_threshold, tracker="bytetrack.yaml", verbose=False, device=self.device, classes=relevant_classes)
        result = results[0]
        

        # --- HAZARD STABILIZER LOGIC ---
        h, w = frame.shape[:2]
        if self.stabilizer is None or self.stabilizer.width != w or self.stabilizer.height != h:
            self.stabilizer = HazardStabilizer(frame_width=w, frame_height=h)
            self.distance_monitor = DistanceMonitor(frame_width=w)
            
        # Convert YOLO results to dict list for stabilizer
        detections_dicts = []
        current_frame_ids = []
        
        if result.boxes and result.boxes.id is not None:
             track_ids = result.boxes.id.int().cpu().tolist()
             boxes = result.boxes.xyxy.cpu().tolist()
             clss = result.boxes.cls.int().cpu().tolist()
             confs = result.boxes.conf.cpu().tolist()
             
             for i, tid in enumerate(track_ids):
                 class_id = clss[i]
                 conf = confs[i]
                 
                 # --- FALSE POSITIVE FIX ---
                 # Accidents are rare. Require higher confidence to show them.
                 # Updated: Increased to 0.50 due to higher base confidence
                 if class_id == 10 and conf < 0.50:
                     continue
                     
                 current_frame_ids.append(tid)
                 detections_dicts.append({
                     'box': boxes[i],
                     'id': tid,
                     'class_id': class_id,
                     'class_name': result.names[class_id],
                     'conf': conf
                 })
        
        # Update stabilizer
        valid_hazards = self.stabilizer.update(detections_dicts, current_frame_ids)
        

        
        # Reconstruct BoxesShim for Analyzer
        class BoxShim:
            def __init__(self, data):
                self.cls = torch.tensor([float(data['class_id'])])
                self.id = torch.tensor([float(data['id'])])
                self.xyxy = torch.tensor([data['box']])
                self.conf = torch.tensor([data['conf']])
                self.zone = data.get('zone', 0) # Zone level (1=Green, 2=Yellow, 3=Red)
        
        shim_boxes_list = [BoxShim(d) for d in valid_hazards]
        
        class BoxesShim:
            def __init__(self, obj_list):
                self.obj_list = obj_list
                if obj_list:
                     self.id = torch.tensor([o.id for o in obj_list])
                else:
                     self.id = None
            def __iter__(self):
                return iter(self.obj_list)
            def __len__(self):
                return len(self.obj_list)
        
        boxes_shim = BoxesShim(shim_boxes_list)
        
        # Analyze using FILTERED boxes
        frame_shape = frame.shape
        
        # Since we pre-filter with HazardStabilizer (Red Zone), 
        # everything passed to Analyzer is "In Lane" roughly.
        # We pass full width as bounds so Analyzer considers them valid targets.
        current_alerts = self.analyzer.analyze(boxes_shim, result.names, frame_shape, lane_bounds=(0, w))
        

        
        # 4. Filter Hazards
        # Pass the mask to existing hazard stabilizer logic to ignore parked cars
        valid_hazards = self.stabilizer.update(detections_dicts, current_frame_ids)
        
        # 5. Distance Estimation & Safety Check
        safe_msg, safe_color, leading_dist, leading_box = self.distance_monitor.check_safe_distance(valid_hazards)
        
        # 6. Visualization
        # Draw Lane/Road (Segmentation Mask) - Green Overlay
        annotated_frame = result.plot()
        annotated_frame = self.stabilizer.draw_debug_zone(annotated_frame)
        
        # --- DISTANCE MONITOR ---
        dist_msg, dist_color, dist_val, dist_box = safe_msg, safe_color, leading_dist, leading_box
        if dist_msg:
             # Draw distance warning

            # Draw distance alert
            
            # Custom draw for Distance
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Draw top-right
            cv2.putText(annotated_frame, dist_msg, (w - 350, 50), font, 1.0, dist_color, 2)
            
            if dist_box:
                # Highlight the leading car
                cv2.rectangle(annotated_frame, (int(dist_box[0]), int(dist_box[1])), (int(dist_box[2]), int(dist_box[3])), dist_color, 4)
                
            # Voice Alert for critical
            current_time = time.time()
            if "CRITICAL" in dist_msg and (current_time - self.last_speech_time > 3.0):
                self.speak_warning("Brake!")
                self.last_speech_time = current_time
        
        # Draw Danger Zone
        annotated_frame = self.stabilizer.draw_debug_zone(annotated_frame)

        # 3. PERSISTENCE & VOICE LOGIC
        curr_time = time.time()
        
        # Define display_msg safely
        display_msg = ""
        
        if current_alerts:
            # Update persistent message
            msg = f"{', '.join(current_alerts[:1])}"
            self.current_display_message = msg
            self.alert_display_until = curr_time + 3.0
            
            # Voice Alert
            if curr_time - self.last_speech_time > 4.0:
                self.speak_warning(msg)
                self.last_speech_time = curr_time
        
        # Check persistence
        if curr_time < self.alert_display_until:
             display_msg = self.current_display_message
        
        # Draw and record
        if display_msg:
            # Update history
            if current_alerts:
                 self.all_detected_hazards.update(current_alerts)
            draw_alert(annotated_frame, f"ALERT: {display_msg}")

        # 4. VISUALIZATION (Lane, FPS)
        h, w = frame.shape[:2]
        # x1 = int(self.lane_tracker.avg_x1)
        # x2 = int(self.lane_tracker.avg_x2)
        # if x1 is not None and x2 is not None:
        #      cv2.line(annotated_frame, (int(x1), 0), (int(x1), h), (255, 255, 0), 2)
        #      cv2.line(annotated_frame, (int(x2), 0), (int(x2), h), (255, 255, 0), 2)
        #      cv2.putText(annotated_frame, "EGO LANE", ((int(x1)+int(x2))//2 - 60, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # FPS calculation
        fps = 1 / (curr_time - self.prev_time) if self.prev_time else 0
        self.prev_time = curr_time
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (annotated_frame.shape[1]-180, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return annotated_frame, current_alerts

    def get_video_info(self):
        """Returns (current_frame, total_frames, fps)"""
        if self.cap:
            curr = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            return curr, total, fps
        return 0, 0, 0

    def seek_frame(self, frame_index):
        if self.cap:
            # Clamp value
            total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            target = max(0, min(frame_index, total-1))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            
            # Reset analyzer history on seek to avoid jumpy alerts
            self.analyzer = HazardAnalyzer() 


def run_monitor_cli(source, model_path, conf_threshold, max_frames=None, output_image_path="hazard_detected.jpg"):
    # Wrapper for backward compatibility with CLI usage
    monitor = RREMMonitor(model_path, conf_threshold)
    monitor.start_capture(source)
    
    frame_count = 0
    hazard_saved = False
    
    try:
        while True:
            if max_frames and frame_count >= max_frames:
                print(f"Reached max frames: {max_frames}")
                break
                
            frame, alerts = monitor.process_frame()
            if frame is None:
                break
            
            frame_count += 1
            
            if alerts and not hazard_saved:
                 print(f"Hazard detected! Saving frame to {output_image_path}")
                 cv2.imwrite(output_image_path, frame)
                 hazard_saved = True
                 
            # If standard run, we usually save last frame if nothing detected
            last_frame = frame
            
    finally:
        monitor.stop_capture()
        
    if not hazard_saved and 'last_frame' in locals() and last_frame is not None:
         cv2.imwrite(output_image_path, last_frame)
         
    return list(monitor.all_detected_hazards)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Road Elements Monitor")
    parser.add_argument("--source", type=str, default="input_video.mp4", help="Video source (file path or camera index)")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--output", type=str, default="hazard_detected.jpg", help="Output path for detection image")
    
    args = parser.parse_args()
    
    detected = run_monitor_cli(args.source, args.model, args.conf, max_frames=args.max_frames, output_image_path=args.output)
    print(f"Final Detections: {detected}")
