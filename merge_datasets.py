import os
import shutil
import random
import yaml
from pathlib import Path
from tqdm import tqdm
import glob

# Configuration
UNIFIED_DIR = "dataset/unified_dataset"
RAW_FRAMES_DIR = "dataset/raw_frames"
CUSTOM_CRASH_DIR = "dataset/custom_crash_dataset"
UNLABELED_DIR = "dataset/unlabeled_datasets"

# Final Class Names Mapping (Target Schema)
CLASS_NAMES = [
    "Person", "Bicycle", "Car", "Motorcycle", "Bus", "Truck", 
    "Cat", "Dog", "Traffic Light", "Stop Sign", "Accident", 
    "Pothole", "Fire Hazard", "Fox", "Chicken", "Deer", 
    "Horse", "Pigeon", "Sheep", "Cow"
]

# Mapping from Raw Frames (COCO 80) to Unified Schema
# Based on checking logic: COCO IDs need remixing to 0-19 range.
# Standard COCO: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck, 9=traffic light, 11=stop sign, 15=cat, 16=dog, ...
# We need to explicitly map the IDs found in raw_frames (which are COCO) to our new IDs.
# Assuming raw_frames used standard COCO numbering or the 'classes.txt' found earlier.
# The 'classes.txt' found earlier in step 25 seemed to list 80 COCO classes.
# Let's define the map. Keys are COCO IDs, Values are Target IDs.
COCO_TO_UNIFIED = {
    0: 0,   # Person -> Person
    1: 1,   # Bicycle -> Bicycle
    2: 2,   # Car -> Car
    3: 3,   # Motorcycle -> Motorcycle
    5: 4,   # Bus -> Bus
    7: 5,   # Truck -> Truck
    9: 8,   # Traffic Light -> Traffic Light
    11: 9,  # Stop Sign -> Stop Sign
    15: 6,  # Cat -> Cat
    16: 7,  # Dog -> Dog
    17: 16, # Horse -> Horse
    18: 18, # Sheep -> Sheep
    19: 19, # Cow -> Cow
    # Add others if they appear in raw_frames and we want to keep them
}

def setup_directories():
    if os.path.exists(UNIFIED_DIR):
        print(f"Removing existing {UNIFIED_DIR}...")
        shutil.rmtree(UNIFIED_DIR)
    
    for split in ['train', 'val']:
        os.makedirs(os.path.join(UNIFIED_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(UNIFIED_DIR, 'labels', split), exist_ok=True)

def copy_file(src_img, src_label, dst_split, unique_name, label_transform_func=None):
    if not os.path.exists(src_img):
        return
        
    dst_img_path = os.path.join(UNIFIED_DIR, 'images', dst_split, unique_name)
    dst_label_path = os.path.join(UNIFIED_DIR, 'labels', dst_split, os.path.splitext(unique_name)[0] + '.txt')
    
    # Copy image
    shutil.copy2(src_img, dst_img_path)
    
    # Process and copy label
    if src_label and os.path.exists(src_label):
        if label_transform_func:
            with open(src_label, 'r') as f_in, open(dst_label_path, 'w') as f_out:
                for line in f_in:
                    parts = line.strip().split()
                    if not parts: continue
                    class_id = int(parts[0])
                    new_id = label_transform_func(class_id)
                    if new_id is not None:
                        f_out.write(f"{new_id} {' '.join(parts[1:])}\n")
        else:
            shutil.copy2(src_label, dst_label_path)

def process_raw_frames(split_ratio=0.8):
    print("Processing raw_frames...")
    search_path = os.path.join(RAW_FRAMES_DIR, "**", "*.[jp][pn]g") 
    
    images = []
    for root, dirs, files in os.walk(RAW_FRAMES_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append(os.path.join(root, file))
                
    random.shuffle(images)
    split_idx = int(len(images) * split_ratio)
    
    def transform_coco(cid):
        return COCO_TO_UNIFIED.get(cid) 

    for idx, img_path in enumerate(tqdm(images, desc="Merging Raw Frames")):
        split = 'train' if idx < split_idx else 'val'
        
        rel_path = os.path.relpath(img_path, RAW_FRAMES_DIR)
        safe_name = "raw_" + rel_path.replace(os.sep, '_')
        
        label_path = None
        if '/images/' in img_path:
            possible = img_path.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
            if os.path.exists(possible):
                label_path = possible
        
        if not label_path:
            possible = img_path.rsplit('.', 1)[0] + '.txt'
            if os.path.exists(possible):
                label_path = possible
        
        copy_file(img_path, label_path, split, unique_name=safe_name, label_transform_func=transform_coco)

def process_custom_crash(split_ratio=0.8):
    print("Processing custom_crash_dataset...")
    
    images = []
    for root, dirs, files in os.walk(CUSTOM_CRASH_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append(os.path.join(root, file))
    
    random.shuffle(images)
    split_idx = int(len(images) * split_ratio)
    
    def transform_custom(cid):
        if cid == 4: return 10 
        return cid 

    for idx, img_path in enumerate(tqdm(images, desc="Merging Custom Crash")):
        split = 'train' if idx < split_idx else 'val'
        
        rel_path = os.path.relpath(img_path, CUSTOM_CRASH_DIR)
        safe_name = "crash_" + rel_path.replace(os.sep, '_')
        
        label_path = None
        if '/images/' in img_path:
            possible = img_path.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
            if os.path.exists(possible):
                label_path = possible
        
        if not label_path: 
             possible = img_path.rsplit('.', 1)[0] + '.txt'
             if os.path.exists(possible):
                label_path = possible

        copy_file(img_path, label_path, split, unique_name=safe_name, label_transform_func=transform_custom)

def process_unlabeled(split_ratio=0.8):
    print("Processing unlabeled_datasets (newly labeled)...")
    
    images = []
    # Walk through specific category folders
    for root, dirs, files in os.walk(UNLABELED_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append(os.path.join(root, file))
    
    random.shuffle(images)
    split_idx = int(len(images) * split_ratio)
    
    for idx, img_path in enumerate(tqdm(images, desc="Merging Auto-Labeled")):
        split = 'train' if idx < split_idx else 'val'
        
        rel_path = os.path.relpath(img_path, UNLABELED_DIR)
        safe_name = rel_path.replace(os.sep, '_') 
        # unlabeled subfolders are categories, so cat_img.jpg or cat_subdir_img.jpg is safe and descriptive
        
        parent = os.path.dirname(img_path)
        basename = os.path.basename(img_path)
        label_name = os.path.splitext(basename)[0] + '.txt'
        label_path = os.path.join(parent, 'labels', label_name)
        
        if os.path.exists(label_path):
            copy_file(img_path, label_path, split, unique_name=safe_name)

def create_yaml():
    data = {
        'path': os.path.abspath(UNIFIED_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(CLASS_NAMES),
        'names': CLASS_NAMES
    }
    
    with open(os.path.join(UNIFIED_DIR, 'data.yaml'), 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    print(f"Created data.yaml at {UNIFIED_DIR}/data.yaml")

def main():
    setup_directories()
    # process_raw_frames() 
    # process_custom_crash() 
    # process_unlabeled()
    # We will try to run all.
    
    # Catching potential errors during processing to ensure at least some data gets merged
    try: process_raw_frames()
    except Exception as e: print(f"Error processing raw frames: {e}")
    
    try: process_custom_crash()
    except Exception as e: print(f"Error processing custom crash: {e}")
            
    try: process_unlabeled()
    except Exception as e: print(f"Error processing unlabeled: {e}")

    create_yaml()
    print("Merge complete!")

if __name__ == "__main__":
    main()
