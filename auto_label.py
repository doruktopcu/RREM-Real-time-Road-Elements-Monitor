from ultralytics import YOLOWorld
import os
import glob
from tqdm import tqdm
import cv2
import ssl

# Disable SSL verification for CLIP model download
ssl._create_default_https_context = ssl._create_unverified_context

# Define the dataset path
DATASET_DIR = "dataset/unlabeled_datasets"

# Schema (from Implementation Plan):
# 0: Person, 1: Bicycle, 2: Car, 3: Motorcycle, 4: Bus, 5: Truck, 6: Cat, 7: Dog, 
# 8: Traffic Light, 9: Stop Sign, 10: Accident, 11: Pothole, 12: Fire Hazard, 
# 13: Fox, 14: Chicken, 15: Deer, 16: Horse, 17: Pigeon, 18: Sheep, 19: Cow

CATEGORY_CONFIG = {
    "potholes": {"classes": ["pothole"], "ids": [11], "type": "full_image"},
    "fire_hazard": {"classes": ["fire"], "ids": [12], "type": "full_image"},
    "fox": {"classes": ["fox"], "ids": [13], "type": "inference"},
    "chicken": {"classes": ["chicken"], "ids": [14], "type": "inference"},
    "deer": {"classes": ["deer"], "ids": [15], "type": "inference"},
    "horse": {"classes": ["horse"], "ids": [16], "type": "inference"},
    "pigeon": {"classes": ["pigeon"], "ids": [17], "type": "inference"},
    "sheep": {"classes": ["sheep"], "ids": [18], "type": "inference"},
    "cow": {"classes": ["cow"], "ids": [19], "type": "inference"},
    "cat": {"classes": ["cat"], "ids": [6], "type": "inference"},
    "dog": {"classes": ["dog"], "ids": [7], "type": "inference"},
    "unlabeled_people_cars": {
        "classes": ["person", "car", "motorcycle", "bus", "truck"],
        "ids": [0, 2, 3, 4, 5],
        "type": "inference"
    }
}

def auto_label():
    # Load YOLO-World model
    print("Loading YOLO-World model...")
    model = YOLOWorld("yolov8l-worldv2.pt")
    
    # Iterate through configured categories
    for folder, config in CATEGORY_CONFIG.items():
        folder_path = os.path.join(DATASET_DIR, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist. Skipping.")
            continue
            
        print(f"Processing folder: {folder}...")
        
        # Set custom classes only if doing inference
        if config.get("type") == "inference":
            model.set_classes(config["classes"])
        
        # Create labels directory
        labels_dir = os.path.join(folder_path, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        
        # extensions to look for
        valid_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in valid_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
            
        # Process images
        for img_path in tqdm(image_files, desc=f"Labeling {folder}"):
            # Prepare label file name
            basename = os.path.basename(img_path)
            filename_no_ext = os.path.splitext(basename)[0]
            label_path = os.path.join(labels_dir, f"{filename_no_ext}.txt")
            
            # For FULL IMAGE labeling: Overwrite if empty or strictly overwrite?
            # User said files are empty. We should overwrite them.
            # If type is inference: Skip if exists (Resume)
            
            is_full_label = config.get("type") == "full_image"
            
            if not is_full_label and os.path.exists(label_path):
                continue
            
            if is_full_label:
                # Blindly write full class label
                try:
                    with open(label_path, "w") as f:
                        # Center x, center y, width, height (all normalized 0-1)
                        # Full image = 0.5 0.5 1.0 1.0
                        class_id = config["ids"][0] 
                        f.write(f"{class_id} 0.500000 0.500000 1.000000 1.000000\n")
                except Exception as e:
                    print(f"Error writing label for {img_path}: {e}")
                continue

            try:
                # Perform inference
                results = model.predict(img_path, conf=0.25, verbose=False)
                
                result = results[0]
                
                with open(label_path, "w") as f:
                    for box in result.boxes:
                        # Get class index in the current model's list (0, 1, 2...)
                        cls_idx = int(box.cls.item())
                        
                        if cls_idx < len(config["ids"]):
                            global_id = config["ids"][cls_idx]
                            x, y, w, h = box.xywhn[0].tolist()
                            f.write(f"{global_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

    print("Auto-labeling complete!")

if __name__ == "__main__":
    auto_label()
