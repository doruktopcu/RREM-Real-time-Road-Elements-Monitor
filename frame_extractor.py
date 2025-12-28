import cv2
import os
import glob
import argparse

def extract_frames_from_folder(input_folder, output_folder, interval_seconds=3.0, limit=None):
    """
    Extracts frames from all videos in a folder at a set interval.
    
    Args:
        input_folder (str): Path to folder containing dashcam videos.
        output_folder (str): Path where images will be saved.
        interval_seconds (float): How many seconds to skip between captures.
        limit (int): Max number of videos to process.
    """
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # Find all video files (mp4, mov, avi, etc.)
    # Added uppercase extensions for robustness
    video_extensions = ['*.mp4', '*.MP4', '*.MOV', '*.mov', '*.avi', '*.AVI', '*.mkv', '*.MKV']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_folder, ext)))
    
    # Sort for consistent processing order
    video_files.sort()
    
    if limit:
        print(f"Limit applied: Processing first {limit} videos.")
        video_files = video_files[:limit]
    
    print(f"Found {len(video_files)} video files in {input_folder}")

    total_images_saved = 0

    for video_path in video_files:
        filename = os.path.basename(video_path)
        print(f"Processing: {filename}...")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error opening video: {filename}")
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            # Fallback if FPS can't be read
            fps = 30 
            print(f"  Warning: Could not read FPS for {filename}, defaulting to 30.")
        
        # Calculate how many frames to skip
        frames_to_skip = int(fps * interval_seconds)
        if frames_to_skip < 1:
            frames_to_skip = 1
        
        frame_count = 0
        saved_count = 0
        
        while True:
            # Read frame
            success, frame = cap.read()
            
            if not success:
                break # End of video
                
            # Only save if we hit the interval
            if frame_count % frames_to_skip == 0:
                # Create a unique filename: videoName_frameNumber.jpg
                # Using 6-digit padding for sortable filenames
                save_name = f"{os.path.splitext(filename)[0]}_f{frame_count:06d}.jpg"
                save_path = os.path.join(output_folder, save_name)
                
                cv2.imwrite(save_path, frame)
                saved_count += 1
                total_images_saved += 1
                
            frame_count += 1
            
        cap.release()
        print(f"  -> Saved {saved_count} images.")

    print(f"\nDone! Total images extracted: {total_images_saved}")
    print(f"Images are saved in: {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from dashcam videos for labeling.")
    parser.add_argument("--input", type=str, nargs='+', required=True, help="Input folder(s) containing videos")
    parser.add_argument("--output", type=str, default="dataset/raw_frames", help="Output folder for extracted images")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between extracted frames")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos to process per folder (for testing)")
    
    args = parser.parse_args()
    
    for input_folder in args.input:
        if not os.path.exists(input_folder):
            print(f"Warning: Input folder '{input_folder}' does not exist. Skipping.")
            continue
            
        print(f"\n--- Processing Folder: {input_folder} ---")
        extract_frames_from_folder(input_folder, args.output, args.interval, limit=args.limit)
