# RREM: Real-time Road Elements Monitor

**Doruk Topcu**  
*Project Repository & Documentation*

---

## Abstract

**Real-time Road Elements Monitor (RREM)** is an advanced computer vision-based driver assistance system designed to democratize high-end safety features found in autonomous vehicles. Utilizing the **YOLO11** architecture for state-of-the-art object detection and a custom logic layer for hazard assessment, RREM detects, tracks, and evaluates road risks in real-time. This document serves as a comprehensive technical guide, detailing the system's architecture, detection logic, dataset preparation methodology, and operational workflow. The system distinguishes itself through a multi-zone hazard evaluation engine, heuristic-based collision warnings, and a unified dataset covering 20 distinct classes of road elements, from vulnerable road users to environmental hazards.

---

## 1. Introduction

Traffic accidents remain a leading cause of mortality worldwide. While modern luxury vehicles come equipped with Advanced Driver Assistance Systems (ADAS), a vast majority of vehicles on the road lack these capabilities. RREM aims to bridge this gap by providing a software-based solution that transforms a standard dashcam feed into an intelligent safety guardian.

The core objectives of RREM are:
1.  **Accurate Detection**: identifying a wide range of road elements, including vehicles, pedestrians, animals, and infrastructure.
2.  **Intelligent Risk Assessment**: Distinguishing between benign objects and actual hazards using spatial and temporal analysis.
3.  **Real-time Performance**: Operating efficiently on standard hardware (consumer GPUs/CPUs) with optimization for Apple Metal (MPS) and CUDA.

---

## 2. System Architecture

The RREM system is built on a modular pipeline architecture, separating perception, tracking, and decision-making into distinct stages.

### 2.1 Hardware Acceleration
To ensure real-time latency (target > 30 FPS), the system automatically detects and leverages hardware acceleration:
*   **Apple Silicon (M1/M2/M3)**: Uses Metal Performance Shaders (MPS) for high-efficiency inference.
*   **NVIDIA GPUs**: Uses CUDA execution providers.
*   **CPU Fallback**: Optimized OpenVINO or standard PyTorch execution as a backup.

### 2.2 Software Stack
*   **Perception Engine**: Ultralytics YOLO11 (v11n/v11s/v11m) for object detection.
*   **Image Processing**: OpenCV for pre-processing, lane finding, and visualization.
*   **Logic Layer**: Python-based temporal tracking and heuristic analysis (`utils.py`).

---

## 3. Methodology

### 3.1 Object Detection Model
The system uses the **YOLO11** (You Only Look Once) architecture. This single-stage detector is chosen for its superior speed-accuracy trade-off.
*   **Model Variants**: The system supports `yolo11n` (nano) for edge devices and `yolo11m` (medium) for higher accuracy.
*   **Classes**: The model is trained on a custom **Unified Schema** of 20 classes (detailed in Section 4).

### 3.2 Multi-Stage Tracking
Raw detections are unstable. RREM implements a multi-stage tracking approach:
1.  **ByteTrack**: An efficient association method that tracks objects across frames, handling occlusions and associating low-confidence detections.
2.  **SimpleTracker (Custom)**: A backup IOU-based greedy matcher implemented in `utils.py` to handle edge cases or specific hazard tracking without heavy overhead.

### 3.3 Hazard Assessment Logic (The "Brain")
The core innovation of RREM lies in its `HazardAnalyzer` and `HazardStabilizer` modules (located in `utils.py`). Instead of alerting on every detection, the system applies rigorous filters:

#### A. Spatial Zoning (Danger Zones)
The field of view is logically divided into three primary zones defined by polygon geometry:
*   **Green Zone (Far)**: Top-central region. Objects here trigger "Awareness" logs but no audio alerts.
*   **Yellow Zone (Medium)**: Middle band. Objects here trigger "Caution" warnings.
*   **Red Zone (Close)**: Bottom-central region immediately in front of the ego-vehicle. Objects here trigger "CRITICAL BRAKE" alerts.

Side zones (Left/Right) are monitored to detect cutting-in vehicles ("Side Traffic").

#### B. Temporal Stabilization
To prevent flickering alerts (false positives from single-frame glitches), the `HazardStabilizer` requires a hazard to be present for a minimum buffer of frames (default: 5) before validating it. It creates a `BoxShim` object to maintain persistence across the pipeline.

#### C. Distance Estimation
A pinhole camera model is approximately applied in the `DistanceMonitor`.
*   **Formula**: $D = \frac{W_{real} \times f}{W_{image}}$
*   **Parameters**: $W_{real}$ is the assumed physical width of the class (e.g., Car=1.8m, Person=0.5m). $f$ is the focal length (calibrated or estimated).
*   **Tailgating Logic**: If a vehicle in the center lane is closer than the braking distance (e.g., <8m), a "Tailgating" warning is issued.

#### D. Fast Approach Detection (Time-to-Collision Heuristic)
The system calculates the expansion rate of bounding boxes. An object that is rapidly increasing in area (Visual Looming) implies a decreasing Time-to-Collision (TTC).
*   **Metric**: Area Growth Rate $\Delta A / A_{prev}$.
*   **Threshold**: A growth rate > 30% over 4 frames generally indicates an imminent collision (<1s), triggering a "CRASH IMMINENT" alert regardless of zone.

---

## 4. Dataset Preparation

A robust model requires diverse data. We created a **Unified Dataset** by merging multiple sources.

### 4.1 Data Sources
1.  **Raw Frames**: Extracted from high-resolution dashcam footage (COCO-labeled).
2.  **Custom Crash Dataset**: Specialized frames containing accidents and wreckage.
3.  **Unlabeled Datasets**: A collected set of "edge cases" (Potholes, Animals, Fire Hazards) that were initially unlabeled.

### 4.2 Auto-Labeling Pipeline
To prepare the `unlabeled_datasets`, we employed a semi-supervised learning approach using **YOLO-World (v2)**, an open-vocabulary detector.
*   **Inference Mode**: For classes like `Cat`, `Dog`, `Horse`, and `Fox`, YOLO-World detected objects based on text prompts.
*   **Full-Image Labeling**: For `Potholes` and `Fire Hazards`, where specific object localization was difficult but the image context was known, we applied full-frame bounding boxes to ensure the model learns the global context of the hazard.

### 4.3 Unified Schema (20 Classes)
All datasets were remapped to a single consistent index:

| ID | Class Name | ID | Class Name |
| :--- | :--- | :--- | :--- |
| 0 | Person | 10 | Accident |
| 1 | Bicycle | 11 | Pothole |
| 2 | Car | 12 | Fire Hazard |
| 3 | Motorcycle | 13 | Fox |
| 4 | Bus | 14 | Chicken |
| 5 | Truck | 15 | Deer |
| 6 | Cat | 16 | Horse |
| 7 | Dog | 17 | Pigeon |
| 8 | Traffic Light | 18 | Sheep |
| 9 | Stop Sign | 19 | Cow |

### 4.4 Merging Strategy
The script `merge_datasets.py` handles the physical merging. Crucially, it resolves **filename collisions** (e.g., `frame1.jpg` existing in both source datasets) by prepending source identifiers (`raw_frame1.jpg`, `crash_frame1.jpg`) during the copy process to ensure zero data loss.

---

## 5. Implementation Details

### 5.1 Directory Structure
```
RREM-Real-time-Road-Elements-Monitor/
├── rrem_monitor.py       # Main Application Entry Point
├── rrem_gui.py           # Graphical User Interface Wrapper
├── utils.py              # Core Logic (Analyzers, Trackers, Drawing)
├── auto_label.py         # Tool: YOLO-World Auto-labeling
├── merge_datasets.py     # Tool: Dataset Merging & Schema Remapping
├── requirements.txt      # Dependencies
└── dataset/
    └── unified_dataset/  # Final Training Data (Train/Val Split)
        ├── data.yaml     # Model Configuration
        ├── images/
        └── labels/
```

### 5.2 Key Classes (`utils.py`)
*   `RREMMonitor`: Orchestrates the camera feed, model inference, and alert system.
*   `LaneTracker`: A computer-vision based lane finding algorithm (Hough Transform + Polyfit) to determine the ego-lane.
*   `HazardAnalyzer`: Evaluates the threat level of tracked objects based on position and trajectory.

---

## 6. Installation & Usage

### Prerequisites
*   Python 3.8+
*   PyTorch (with MPS or CUDA support recommended)
*   Ultralytics YOLO
*   OpenCV (`opencv-python`)

### Installation
```bash
git clone https://github.com/doruktopcu/RREM.git
cd RREM
pip install -r requirements.txt
```

### Modes of Operation

**1. Standard Monitor (CLI)**
Running on a video file:
```bash
python3 rrem_monitor.py --source input_video.mp4 --conf 0.5
```
Running on Webcam:
```bash
python3 rrem_monitor.py --source 0
```

**2. Dataset Tools**
To regenerate the unified dataset:
```bash
# 1. Label unlabelled images
python3 auto_label.py

# 2. Merge all sources
python3 merge_datasets.py
```

---

## 7. Results & Performance

The system achieves robust performance on standard hardware:
*   **M2 Pro (Mac)**: ~45 FPS (YOLO11n), ~28 FPS (YOLO11m)
*   **RTX 3060**: ~60+ FPS (YOLO11m)

The integration of the "Red Zone" logic significantly reduces false positives compared to raw YOLO detection, effectively suppressing warnings for distant or non-threatening objects while maintaining high recall for immediate hazards.

---

## 8. Future Roadmap

1.  **Integration of SAM (Segment Anything Model)**: For pixel-perfect road segmentation and drivable area analysis.
2.  **Stereo Vision Support**: For accurate depth perception without single-camera estimation errors.
3.  **V2X Communication**: Broadcasting hazard alerts to nearby RREM-enabled vehicles.

---

## References

1.  Redmon, J., et al. "You Only Look Once: Unified, Real-Time Object Detection." *(CVPR 2016)*
2.  Ultralytics YOLOv8/v11 Docs: https://docs.ultralytics.com
3.  Zhang, Y., et al. "ByteTrack: Multi-Object Tracking by Associating Every Detection Box." *(ECCV 2022)*
4.  Wang, T., et al. "YOLO-World: Real-Time Open-Vocabulary Object Detection." *(CVPR 2024)*
