import argparse
import time
import cv2
import os
import threading
from ultralytics import YOLO
from utils import HazardAnalyzer, draw_alert, LaneTracker

class RREMMonitor:
    def __init__(self, model_path="yolo11n.pt", conf_threshold=0.25):
        print(f"Loading model: {model_path}...")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.analyzer = HazardAnalyzer()
        self.lane_tracker = LaneTracker()
        self.all_detected_hazards = set()
        self.prev_time = 0
        
        # Persistence & Voice State
        self.alert_display_until = 0
        self.current_display_message = ""
        self.last_speech_time = 0
        
        # State for capture
        self.cap = None
        self.source = None
    
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

        # Run inference with tracking
        # Using ByteTrack as it generally handles lost/low-conf objects better than BoT-SORT
        results = self.model.track(frame, persist=True, conf=self.conf_threshold, tracker="bytetrack.yaml", verbose=False)
        result = results[0]

        # Process detections
        frame_shape = frame.shape
        lane_x1, lane_x2 = self.lane_tracker.update(frame)
        current_alerts = self.analyzer.analyze(result.boxes, result.names, frame_shape, lane_bounds=(lane_x1, lane_x2))
        
        # PERSISTENCE & VOICE
        curr_time = time.time()
        
        if current_alerts:
            # New alerts found
            # Join top alerts
            msg = f"{', '.join(current_alerts[:1])}" # Just top 1 for clean display
            self.current_display_message = msg
            self.alert_display_until = curr_time + 3.0 # Display for 3 seconds
            
            # Voice Alert (with cooldown)
            if curr_time - self.last_speech_time > 4.0:
                self.speak_warning(msg)
                self.last_speech_time = curr_time
        
        # Decide what to draw
        display_msg = ""
        if curr_time < self.alert_display_until:
             display_msg = self.current_display_message

        # Draw Output
        annotated_frame = result.plot()
        
        if display_msg:
            # We don't update all_detected_hazards here with persisted msg to avoid dupes in logs, 
            # assuming current_alerts handles the logical detection recording.
            if current_alerts:
                 self.all_detected_hazards.update(current_alerts)
            
            draw_alert(annotated_frame, f"ALERT: {display_msg}")

        # VISUALIZE LANE & DANGER ZONE
        h, w = frame.shape[:2]
        x1 = lane_x1
        x2 = lane_x2
        # Draw translucent overlay or lines? Lines are cleaner for speed.
        # Cyan lines
        cv2.line(annotated_frame, (x1, 0), (x1, h), (255, 255, 0), 2)
        cv2.line(annotated_frame, (x2, 0), (x2, h), (255, 255, 0), 2)
        # Helper Text
        cv2.putText(annotated_frame, "EGO LANE", ((x1+x2)//2 - 60, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time) if self.prev_time else 0
        self.prev_time = curr_time
        
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
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
