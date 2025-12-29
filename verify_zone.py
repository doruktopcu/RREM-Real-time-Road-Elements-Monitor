import cv2
from rrem_monitor import RREMMonitor

def verify_zone():
    monitor = RREMMonitor(model_path="yolo11n.pt")
    monitor.start_capture("input_video.mp4")
    
    # Seek to a frame with a road (e.g., 200)
    monitor.seek_frame(200)
    
    frame, alerts = monitor.process_frame()
    if frame is not None:
        cv2.imwrite("verification_zone_red.jpg", frame)
        print("Saved verification_zone_red.jpg")
    
    monitor.stop_capture()

if __name__ == "__main__":
    verify_zone()
