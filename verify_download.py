from rrem_monitor import RREMMonitor
import os

def test_download():
    # Ensure file is gone first
    if os.path.exists("yolo11s.pt"):
        os.remove("yolo11s.pt")
        
    print("Initializing monitor with yolo11s.pt...")
    # This should trigger download
    monitor = RREMMonitor(model_path="yolo11s.pt")
    
    if os.path.exists("yolo11s.pt"):
        print("Verification SUCCESS: yolo11s.pt exists.")
    else:
        print("Verification FAILED: yolo11s.pt not found.")

if __name__ == "__main__":
    test_download()
