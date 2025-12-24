import os
import glob
from rrem_monitor import run_monitor

def main():
    test_case_dir = "test_cases"
    output_dir = "batch_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all video files (mp4, MP4)
    files = glob.glob(os.path.join(test_case_dir, "*.[Mm][Pp]4"))
    files.sort()
    
    results = []
    
    print(f"Found {len(files)} files to process.")
    
    for i, file_path in enumerate(files):
        filename = os.path.basename(file_path)
        print(f"[{i+1}/{len(files)}] Processing {filename}...")
        
        output_image = os.path.join(output_dir, f"{filename}_result.jpg")
        
        # Run for max 300 frames (approx 10s) to be efficient
        # Adjust max_frames if longer duration is needed to see the event
        detections = run_monitor(
            source=file_path,
            model_path="yolo11n.pt",
            conf_threshold=0.5,
            max_frames=300,
            output_image_path=output_image
        )
        
        results.append({
            "file": filename,
            "hazards": detections,
            "image": output_image
        })
        print(f"  -> Found: {detections}")

    # Generate Report
    report_path = "batch_report.md"
    with open(report_path, "w") as f:
        f.write("# Batch Detection Results\n\n")
        f.write("| File | Hazards Detected | Result Image |\n")
        f.write("|---|---|---|\n")
        for res in results:
            hazards_str = ", ".join(res["hazards"]) if res["hazards"] else "None"
            # Relative path for markdown if viewing locally, or absolute for artifacts
            # We'll stick to a simple format
            f.write(f"| {res['file']} | {hazards_str} | [View Image]({res['image']}) |\n")
            
    print(f"\nBatch processing complete. Report saved to {report_path}")

if __name__ == "__main__":
    main()
