import os
import shutil
import random
import yaml
import glob
from tqdm import tqdm

def remap_and_copy(src_img_dir, src_label_dir, dst_img_dir, dst_label_dir, mapping, is_crash_dataset=False):
    """
    Copies images and creates remapped label files.
    """
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_label_dir, exist_ok=True)
    
    # Get all images
    # Supports common extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.MP4', '*.mov'] # Note: frames usually jpg
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(src_img_dir, ext)))
        # Also check uppercase
        images.extend(glob.glob(os.path.join(src_img_dir, ext.upper())))
        
    print(f"Processing {len(images)} images from {src_img_dir}...")
    
    for img_path in tqdm(images):
        filename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(filename)[0]
        
        # 1. Copy Image
        shutil.copy2(img_path, os.path.join(dst_img_dir, filename))
        
        # 2. Process Label
        # Expect label file to have same basename but .txt extension
        label_src = os.path.join(src_label_dir, name_no_ext + ".txt")
        label_dst = os.path.join(dst_label_dir, name_no_ext + ".txt")
        
        if os.path.exists(label_src):
            with open(label_src, 'r') as f_in, open(label_dst, 'w') as f_out:
                for line in f_in:
                    parts = line.strip().split()
                    if not parts: continue
                    
                    cls_id = int(parts[0])
                    
                    # Remap
                    new_id = -1
                    if is_crash_dataset:
                        # Crash Dataset Logic
                        # 0 (No Accident) -> 2 (Car)
                        # 1-4 (Accident) -> 10 (Accident)
                        if cls_id == 0:
                            new_id = 2 # Car
                        elif cls_id in [1, 2, 3, 4]:
                            new_id = 10 # Accident
                    else:
                        # Ankara Logic (COCO IDs)
                        # Map COCO IDs to new sequential IDs
                        # 0->0, 1->1, 2->2, 3->3, 5->4, 7->5, 15->6, 16->7, 9->8, 11->9
                        if cls_id in mapping:
                            new_id = mapping[cls_id]
                            
                    if new_id != -1:
                        # Write new line
                        f_out.write(f"{new_id} {' '.join(parts[1:])}\n")

def main():
    # --- CONFIG ---
    base_dir = "dataset"
    raw_frames_dir = os.path.join(base_dir, "raw_frames") # Combined img+txt
    crash_dataset_dir = os.path.join(base_dir, "custom_crash_dataset")
    output_dir = os.path.join(base_dir, "unified_dataset")
    
    # New Class Map
    # COCO Original -> New ID
    coco_map = {
        0: 0,   # Person
        1: 1,   # Bicycle
        2: 2,   # Car
        3: 3,   # Motorcycle
        5: 4,   # Bus
        7: 5,   # Truck
        15: 6,  # Cat
        16: 7,  # Dog
        9: 8,   # Traffic Light
        11: 9   # Stop Sign
    }
    
    class_names = [
        "Person", "Bicycle", "Car", "Motorcycle", "Bus", "Truck", 
        "Cat", "Dog", "Traffic Light", "Stop Sign", "Accident"
    ]
    
    # Clean output
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Structure
    dirs = {
        'train_img': os.path.join(output_dir, 'images', 'train'),
        'val_img': os.path.join(output_dir, 'images', 'val'),
        'train_lbl': os.path.join(output_dir, 'labels', 'train'),
        'val_lbl': os.path.join(output_dir, 'labels', 'val')
    }
    
    # --- 1. PROCESS ANKARA DATASET ---
    print("--- Processing Ankara Dataset ---")
    # Identify pairs
    ankara_files = glob.glob(os.path.join(raw_frames_dir, "*.jpg"))
    random.shuffle(ankara_files) # Shuffle for random split
    
    split_idx = int(len(ankara_files) * 0.8)
    train_ankara = ankara_files[:split_idx]
    val_ankara = ankara_files[split_idx:]
    
    # Helper for mixed folder (images and labels in same place)
    def process_mixed(file_list, dst_img, dst_lbl):
        os.makedirs(dst_img, exist_ok=True)
        os.makedirs(dst_lbl, exist_ok=True)
        for img_path in file_list:
            # Copy Image
            shutil.copy2(img_path, dst_img)
            
            # Label
            label_src = os.path.splitext(img_path)[0] + ".txt"
            label_dst = os.path.join(dst_lbl, os.path.basename(label_src))
            
            if os.path.exists(label_src):
                with open(label_src, 'r') as f_in, open(label_dst, 'w') as f_out:
                    for line in f_in:
                        parts = line.strip().split()
                        if not parts: continue
                        cls_id = int(parts[0])
                        if cls_id in coco_map:
                            f_out.write(f"{coco_map[cls_id]} {' '.join(parts[1:])}\n")

    process_mixed(train_ankara, dirs['train_img'], dirs['train_lbl'])
    process_mixed(val_ankara, dirs['val_img'], dirs['val_lbl'])
    
    # --- 2. PROCESS CRASH DATASET ---
    print("\n--- Processing Crash Dataset ---")
    # Assume standard structure: images/train, labels/train etc.
    remap_and_copy(
        os.path.join(crash_dataset_dir, 'images', 'train'),
        os.path.join(crash_dataset_dir, 'labels', 'train'),
        dirs['train_img'], dirs['train_lbl'],
        mapping=None, is_crash_dataset=True
    )
    
    remap_and_copy(
        os.path.join(crash_dataset_dir, 'images', 'val'),
        os.path.join(crash_dataset_dir, 'labels', 'val'),
        dirs['val_img'], dirs['val_lbl'],
        mapping=None, is_crash_dataset=True
    )
    
    # --- 3. CREATE DATA.YAML ---
    print("\n--- Creating data.yaml ---")
    yaml_data = {
        'path': output_dir, # Relative path (portable)
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(class_names),
        'names': class_names
    }
    
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
        
    print(f"\nSuccess! Unified dataset created at: {output_dir}")
    print(f"Classes: {class_names}")

if __name__ == "__main__":
    main()
