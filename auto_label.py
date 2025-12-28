from ultralytics import YOLO
import os
import argparse
import shutil
import glob

def auto_label(source_dir, model_path, conf_threshold=0.40):
    """
    Auto-labels images in the source directory using a YOLO model.
    Moves generated labels to the same directory as images for LabelImg compatibility.
    """
    
    # 1. Load model
    print(f"Loading model: {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Define classes to KEEP (0=person, 1=bike, 2=car, 3=motorcycle, 5=bus, 7=truck, 15=cat, 16=dog)
    # Adjusting based on standard COCO classes for road elements relevant to RREM
    relevant_classes = [0, 1, 2, 3, 5, 7, 15, 16]
    
    print(f"Starting inference on {source_dir}...")
    print(f"Threshold: {conf_threshold}")
    
    # 3. Run inference
    # We use a temporary project folder to capture the output, then move it.
    temp_project = os.path.join(source_dir, "temp_inference")
    temp_name = "labels_generated"
    
    results = model.predict(
        source=source_dir, 
        save_txt=True,       
        save_conf=False,     
        classes=relevant_classes,
        conf=conf_threshold,           
        project=temp_project, 
        name=temp_name,
        stream=True # Use stream to avoid loading all results in memory if many images
    )
    
    # Execute the generator
    for _ in results:
        pass
        
    print("Inference complete. Organizing labels...")
    
    # 4. Move labels to be side-by-side with images (LabelImg preferred format)
    # YOLO saves txt files in: project/name/labels/
    generated_labels_dir = os.path.join(temp_project, temp_name, "labels")
    
    if os.path.exists(generated_labels_dir):
        txt_files = glob.glob(os.path.join(generated_labels_dir, "*.txt"))
        count = 0
        for txt_file in txt_files:
            filename = os.path.basename(txt_file)
            dest = os.path.join(source_dir, filename)
            shutil.move(txt_file, dest)
            count += 1
            
        print(f"Moved {count} label files to {source_dir}")
        
    # 5. Cleanup
    if os.path.exists(temp_project):
        shutil.rmtree(temp_project)
        print("Cleaned up temporary folders.")
        
    # 6. Create classes.txt if not exists (LabelImg needs this)
    classes_file = os.path.join(source_dir, "classes.txt")
    if not os.path.exists(classes_file):
        # We should list ALL classes that the user might want to annotate, 
        # but realistically LabelImg uses a predefined list or reads from txt.
        # Ideally, we write the names corresponding to the model's classes.
        # But for new labeling, usually we just want the ones we care about.
        # Let's write the names from the model.
        print("Creating classes.txt...")
        with open(classes_file, 'w') as f:
            for i in range(len(model.names)):
                f.write(f"{model.names[i]}\n")
                
    print("\nDone! Images and .txt labels are now paired in:")
    print(f"  {source_dir}")
    print("\nYou can now open this folder in LabelImg.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-label dataset using YOLO model.")
    parser.add_argument("--source", type=str, default="dataset/raw_frames", help="Folder containing images to label")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Path to YOLO model")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.source):
        print(f"Error: Source directory '{args.source}' does not exist.")
    else:
        auto_label(args.source, args.model, args.conf)
