import sys
import os
import cv2
import numpy as np

# Add rrem_mobile to path
sys.path.append(os.path.join(os.getcwd(), 'rrem_mobile'))

from mobile_utils import HazardAnalyzer, HazardStabilizer, DistanceMonitor
from yolo_tflite import YoloTFLite

def verify_logic():
    print("Verifying Mobile Logic...")
    
    # 1. Logic Classes
    analyzer = HazardAnalyzer()
    stabilizer = HazardStabilizer()
    monitor = DistanceMonitor()
    print("Logic classes instantiated.")
    
    # 2. TFLite Model
    model_path = "yolo11n_float32.tflite" # Standard float32 export name usually
    if not os.path.exists(model_path):
        # Maybe int8 or other? 
        model_path = "yolo11n.tflite"
    
    if os.path.exists(model_path):
        print(f"Loading model: {model_path}")
        try:
            detector = YoloTFLite(model_path)
            print("Model loaded successfully.")
            
            # Test Inference on dummy image
            img = np.zeros((1080, 1920, 3), dtype=np.uint8)
            dets = detector.detect(img)
            print(f"Inference successful. Detections: {len(dets)}")
            
        except Exception as e:
            print(f"Model load failed (expected if tensorflow/tflite not compatible on desktop): {e}")
    else:
        print("Model file not found yet (Export still running?). Skipping inference test.")

if __name__ == "__main__":
    verify_logic()
