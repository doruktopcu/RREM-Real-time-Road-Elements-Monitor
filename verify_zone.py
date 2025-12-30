import cv2
from utils import HazardStabilizer

def verify_zone():
    image_path = "dataset/raw_frames/NO20251028-132358-000067F_f000180.jpg"
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Error: Could not load {image_path}")
        return

    h, w = frame.shape[:2]
    stabilizer = HazardStabilizer(frame_width=w, frame_height=h)
    
    # Draw zones
    annotated_frame = stabilizer.draw_debug_zone(frame)
    
    cv2.imwrite("verification_zones_final.jpg", annotated_frame)
    print("Saved verification_zones_final.jpg")

if __name__ == "__main__":
    verify_zone()
