import cv2
import os
import glob
import random
import yaml

def visualize_samples(dataset_dir, output_dir, num_samples=10):
    # Load class names
    with open(os.path.join(dataset_dir, 'data.yaml'), 'r') as f:
        data = yaml.safe_load(f)
        class_names = data['names']
        
    img_dir = os.path.join(dataset_dir, 'images', 'train')
    label_dir = os.path.join(dataset_dir, 'labels', 'train')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all images
    img_files = glob.glob(os.path.join(img_dir, "*.jpg"))
    if not img_files:
        print("No images found to visualize.")
        return

    # Sample
    samples = random.sample(img_files, min(num_samples, len(img_files)))
    
    print(f"Visualizing {len(samples)} samples to {output_dir}...")
    
    for img_path in samples:
        basename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(basename)[0]
        label_path = os.path.join(label_dir, name_no_ext + ".txt")
        
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
                
                # Convert YOLO to pixels
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
                
                color = (0, 255, 0) # Green for regular
                if cls_id == 10: # Accident
                    color = (0, 0, 255) # Red for accident
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                label_text = class_names[cls_id]
                cv2.putText(img, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imwrite(os.path.join(output_dir, "vis_" + basename), img)
        
    print("Done.")

if __name__ == "__main__":
    visualize_samples("dataset/unified_dataset", "dataset/verification")
