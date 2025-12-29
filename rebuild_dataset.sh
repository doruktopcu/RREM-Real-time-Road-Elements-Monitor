# 1. Extract Frames from External Drive (ALL VIDEOS)
# Interval 3.0s to keep total count reasonable (~10k-20k images)
echo "--- Step 1: Extracting Frames ---"
python3 frame_extractor.py \
    --input "/Volumes/Dodo 990EVO/Dashcam Recordings/Normal/Front" \
    --output "dataset/raw_frames" \
    --interval 30.0

# 2. Auto Label using YOLOv11 Medium for high quality labels
echo "--- Step 2: Auto Labeling ---"
# Check if model exists, if not it will download
python3 auto_label.py --model yolo11m.pt

# 3. Create Unified Dataset
echo "--- Step 3: Merging Datasets ---"
python3 prepare_dataset.py

# 4. Zip it for Colab
echo "--- Step 4: Zipping for Colab ---"
rm dataset.zip
zip -r -q dataset.zip dataset/unified_dataset

echo "--- DONE! Upload dataset.zip to Colab and train again. ---"
