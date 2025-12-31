# RREM: Real-time Road Elements Monitor

**Doruk Topcu**  
*Project Repository & Documentation*

---


## Abstract

**Real-time Road Elements Monitor (RREM)** represents a significant evolution in the field of Advanced Driver Assistance Systems (ADAS), specifically targeting the safety gaps prevalent in unstructured driving environments. While traditional ADAS solutions focus heavily on vehicle and pedestrian tracking in regulated urban settings, they often fail to account for the chaotic nature of rural or developing road infrastructure. RREM addresses this by enhancing road safety through the detection of a wide array of environmental hazards often overlooked by conventional systems. Leveraging the cutting-edge **YOLO11** object detection architecture and optimized for heterogeneous computing environments—specifically Apple Silicon (MPS) and NVIDIA CUDA platforms—RREM provides robust, real-time alerts for 20 distinct classes. These classes encompass vulnerable road users (pedestrians, cyclists), unexpected animal hazards (cows, sheep, deer), and critical infrastructure anomalies (potholes, fire hazards). This document details the system's modular architecture, the curation of a novel, unified 20-class dataset generated via a semi-supervised YOLO-World auto-labeling pipeline, and the implementation of a multi-zone hazard assessment logic designed to prioritize driver attention. Experimental results validate the system's robustness, achieving a mean Average Precision (mAP@50) of **0.835** and a recall of **0.778**. These metrics demonstrate the system's high efficacy and reliability in diverse, real-world scenarios where reaction time is paramount.

---

## 1. Introduction

The automotive industry has seen a paradigm shift with the integration of **Advanced Driver Assistance Systems (ADAS)** into modern vehicles. These systems have undeniably reduced accident rates; however, a majority of current implementations are trained and tested within highly structured environments—marked by clear lane markings, predictable traffic flow, and standard signage. This limitation leaves a significant safety gap when vehicles operate in unstructured scenarios typical of rural areas or developing regions. In these environments, drivers frequently encounter "long-tail" distribution hazards that standard datasets do not adequately represent, including stray livestock, deteriorating road surfaces (potholes), and erratic vulnerable road users.

RREM addresses this critical gap by proposing a comprehensive, low-latency monitoring system capable of identifying a broader spectrum of road elements. Unlike proprietary "black-box" solutions found in high-end vehicles, RREM is designed with accessibility and edge-deployment in mind.

Our contribution to the field is threefold:
1.  **Hardware-Agnostic Perception Engine**: A real-time detection pipeline that is compatible with consumer-grade hardware, bridging the gap between high-performance workstations and accessible edge devices.
2.  **Unified Anomaly Dataset**: A unified 20-class dataset that aggregates diverse data sources and is enriched via a novel open-vocabulary auto-labeling strategy.
3.  **Temporal Hazard Logic**: A temporal stabilization and varying-zone logic that effectively minimizes false positives caused by sensory noise while ensuring critical responsiveness during imminent collision scenarios.

---

## 2. System Architecture

The RREM system is built on a modular pipeline architecture, separating perception, tracking, and decision-making into distinct stages.

### 2.1 Hardware Acceleration
To ensure the system meets the strict latency requirements of ADAS (typically requiring response times under 100ms), RREM implements a hardware abstraction layer that dynamically selects the optimal execution path:
*   **Apple Metal (MPS)**: The system is heavily optimized for macOS devices, utilizing the Metal Performance Shaders (MPS) graph to offload matrix operations to the Neural Engine on M-series chips.
*   **CUDA**: For standard desktop and embedded GPU environments, the system utilizes NVIDIA's CUDA libraries for parallel processing.
*   **CPU Fallback**: A highly optimized OpenVINO-compatible execution path ensures universally compatible deployment.

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

### 3.2 Hazard Assessment Logic (The "Brain")
The core innovation of RREM lies in its `HazardAnalyzer` and `HazardStabilizer` modules (located in `utils.py`). Instead of alerting on every detection, the system applies rigorous filters:

#### A. Spatial Zoning (Danger Zones)
A raw 2D bounding box detection is insufficient for generating meaningful safety warnings. A system that alerts the driver to every detected object would quickly lead to "alert fatigue." RREM introduces a "Hazard Analyzer" module that projects a 3D spatial logic onto the 2D image plane. The camera's field of view is segmented into three static, polygon-based zones:
*   **Green Zone (Far Field)**: Objects detected in the peripheral or distant field are tracked for situational awareness but do not trigger audio alerts, reducing driver distraction.
*   **Yellow Zone (Mid Field)**: This represents the caution area. Detections entering this zone initiate visual "Caution" warnings, preparing the driver for potential action.
*   **Red Zone (Near Field)**: This is the critical braking area immediately in front of the ego-vehicle. Any hazard entering this zone triggers an immediate, high-priority "CRITICAL BRAKE" audio-visual alert.

#### B. Temporal Stabilization
To mitigate false positives caused by camera sensor noise, motion blur, or single-frame detection glitches, we implement a specific \textit{Hazard Stabilizer}. This module employs a temporal buffer of length $N$ (default $N=5$ frames). An object is not classified as a confirmed hazard until it persists within a danger zone for $N$ consecutive frames. This logic acts as a low-pass filter for the detection stream, significantly reducing flickering alerts without compromising reaction time for sustained, genuine threats.

#### C. Distance Estimation
A pinhole camera model is approximately applied in the `DistanceMonitor`.
*   **Formula**: $D = \frac{W_{real} \times f}{W_{image}}$
*   **Parameters**: $W_{real}$ is the assumed physical width of the class (e.g., Car=1.8m, Person=0.5m). $f$ is the focal length (calibrated or estimated).
*   **Tailgating Logic**: If a vehicle in the center lane is closer than the braking distance (e.g., <8m), a "Tailgating" warning is issued.

#### D. Fast Approach Detection (Time-to-Collision Heuristic)
The system calculates the expansion rate of bounding boxes. An object that is rapidly increasing in area (Visual Looming) implies a decreasing Time-to-Collision (TTC). A rapid positive change in area (**>30%** inter-frame expansion) signifies a "Visual Looming" effect, triggering a fast-approach warning even if the object is currently outside the defined Red Zone.

---

## 4. Dataset Preparation

A robust model requires diverse data. We created a **Unified Dataset** by merging multiple sources.

### 4.1 Data Sources
### 4.1 Data Sources
1.  **Car Accidents and Deformation Dataset**: Bounding box–annotated car images showing varying levels of damage (Source: [Kaggle/Marslan Arshad](https://www.kaggle.com/datasets/marslanarshad/car-accidents-and-deformation-datasetannotated)).
2.  **Karlsruhe Dataset**: Labeled Cars and Pedestrians, originally used for part-based object detection (Geiger et al., NIPS 2011).
3.  **Animals Datasets**:
    - **Animals-10** (Source: [Kaggle/Alessio Corrado](https://www.kaggle.com/datasets/alessiocorrado99/animals10))
    - **Animals-90** (Subset) (Source: [Kaggle/Sourav Banerjee](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals))
4.  **Hazard Datasets**:
    - **FIRE Dataset** (Source: [Kaggle/phylake1337](https://www.kaggle.com/datasets/phylake1337/fire-dataset))
    - **Pothole Detection Dataset** (Source: [Kaggle/atulyakumar98](https://www.kaggle.com/datasets/atulyakumar98/pothole-detection-dataset))
5.  **Proprietary Data**: Self-collected footage using a **70mai A800SE** dashcam mounted on a Fiat Egea (Front Window), capturing local road conditions and edge cases. [**Download Unified Dataset**](https://drive.google.com/file/d/1PioXMpZysQ8eAN1ewytGi8kr7uTgaWLx/view?usp=sharing)

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
git clone https://github.com/doruktopcu/RREM-Real-time-Road-Elements-Monitor.git
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

Running with GUI:
```bash
python3 rrem_gui.py
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

This section presents a quantitative evaluation of the RREM system's object detection capabilities, based on a rigorous training regimen.

### 7.1 Experimental Setup
The model was trained using the following hyperparameters, selected to prevent overfitting while ensuring convergence:
*   **Architecture**: YOLO11m (Medium parameter count).
*   **Epochs**: 50, with early stopping enabled.
*   **Batch Size**: 128 images per batch.
*   **Optimizer**: Auto (adaptive selection between SGD and AdamW) with Momentum=0.937 and Weight Decay=0.0005.
*   **Data Augmentation**: Aggressive augmentations including Mosaic (1.0) and Random Erasing (0.4) were employed to enhance robustness against partial occlusions and varying lighting conditions.

### 7.2 Quantitative Metrics
The training process concluded with the model converging to a high degree of accuracy. The final evaluation metrics on the held-out validation set are as follows:

| Metric | Value | Description |
| :--- | :--- | :--- |
| **mAP @ 0.50** | **0.835** | Mean Average Precision at 50% IoU threshold, indicating robust detection capabilities. |
| **mAP @ 0.50-0.95** | **0.739** | Mean Average Precision averaged over multiple IoU thresholds, reflecting high localization accuracy. |
| **Precision** | **0.823** | The ratio of true positive detections to total positive detections. |
| **Recall** | **0.778** | The capability of the model to find all relevant objects in the scene. |

The class-wise performance analysis reveals that the model performs exceptionally well on distinct, rigid objects such as **Potholes (mAP@50: 0.995)** and **Fire Hazards (mAP@50: 0.995)**, validating the efficacy of the full-image auto-labeling strategy for environmental hazards. Vehicle classes such as **Cars** also showed strong performance (**mAP@50: 0.934**), ensuring reliable forward collision warnings.

### 7.3 System Latency
Real-time inference tests were conducted on an **Apple M1 Pro MacBook** using the Metal Performance Shaders (MPS) backend. The system achieved a variable frame rate of **15-30 FPS** for the YOLO11m model. This performance demonstrates that RREM meets the latency requirements for real-time driver assistance on accessible, consumer-grade hardware. It is worth noting that while the model training was accelerated using an enterprise-grade **NVIDIA A100 GPU**, the inference performance reported here is exclusively based on the Apple Silicon platform, reflecting the target deployment environment.

Trained on [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1N8frlROTmB2yyXLA4JnXkHSZwQfkLorx?usp=sharing) 

### 7.4 Limitations & Future Work
While the current system demonstrates robust performance, we acknowledge several limitations that will guide future development:

**Current Dataset Limitations**:
The current Unified Dataset lacks specific samples for local traffic signs and traffic lights. Consequently, the system does not currently detect traffic control signals reliably in the deployment region. Future versions will explicitly incorporate a dedicated dataset for **Turkish Road Standards**, including local traffic signage and light configurations, to ensure full compliance with local traffic regulations.

**Future Roadmap**:
1.  **Advanced Segmentation**: We plan to integrate the Segment Anything Model (SAM) for pixel-level drivable area analysis, moving beyond simple bounding boxes to understand road boundaries.
2.  **Stereo Vision Support**: We intend to implement dual-camera support for true depth perception, replacing the current monocular estimation heuristic with accurate triangulation.
3.  **V2X Connectivity**: A key future milestone is enabling the system to broadcast detected hazards (e.g., potholes) to a cloud server. This would facilitate the creation of a crowd-sourced map of road conditions, alerting other drivers to hazards before they are even within visual range.
4.  **Rear-View Monitoring**: Implementing rear tracking to provide alerts for potential rear-end collisions.
5.  **General Obstacle Segmentation**: Detecting random items/debris on the road that don't fit into standard classes but pose a driving risk.

---

## References

1.  Redmon, J., et al. "You Only Look Once: Unified, Real-Time Object Detection." *(CVPR 2016)*
2.  Ultralytics YOLOv8/v11 Docs: https://docs.ultralytics.com
3.  Wang, T., et al. "YOLO-World: Real-Time Open-Vocabulary Object Detection." *(CVPR 2024)*
