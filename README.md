# RREM: Real-time Road Elements Monitor

**RREM** is a computer vision-based driving assistant designed to enhance road safety by detecting and alerting drivers to hazardous elements in real-time. It aims to provide the advanced monitoring capabilities found in high-end dashcams and autonomous vehicles to any driver using a standard camera setup.

## 🚀 Current Capabilities
The system currently uses **YOLO11** for real-time object detection and tracking, featuring:

*   **Vulnerable Road User Detection**: Explicit recognition and alerts for **Bicycles**, **Motorcycles**, and **Pedestrians**.
*   **Animal Hazard Detection**: Identifies animals on the road (Dogs, Cows, Sheep, etc.) to prevent collisions.
*   **Collision Risk Assessment (Fast Approach)**: heuristic-based detection of rapidly approaching vehicles (20% expansion over 3 frames) to warn of potential rear-end or head-on collisions.
*   **Side/Merge Traffic Alert**: Monitors for vehicles with a wide aspect ratio, indicating side traffic at intersections or aggressive mergers.

## 🔮 Future Roadmap (Final Stages)
The ultimate goal is to create a comprehensive driving safety guardian. Planned features include:

### Advanced Detection
*   **Crash Detection**: Real-time identification of accidents occurring ahead.
*   **Emergency Vehicle Detection**: Recognition of Ambulances, Firetrucks, and Police cars to facilitate yielding.
*   **Foreign Object Debris (FOD)**: Spotting obstacles/debris on the road.
*   **People with Objects**: Detecting pedestrians carrying large items that might obstruct movement.
*   **Tailgating/Tail Tracking**: Monitoring vehicles following too closely.

### Environmental Awareness
*   **Weather Condition Analysis**: Detecting rain, snow, and fog.
*   **Road Surface Monitoring**: Identifying slippery conditions, oil leaks, or potholes.
*   **Night Time Enhancement**: Specialized models for low-light detection.

### Traffic & Infrastructure
*   **Traffic Sign & Billboard Recognition**: Relay traffic alerts and speed limits.
*   **Red Light & Signal Change**: Early warnings for traffic light changes.
*   **Roadblock & Lane Closure**: Detection of construction zones or blocked lanes.

## 🛠️ Installation & Usage

### Prerequisites
*   Python 3.8+
*   `ultralytics` (YOLO)
*   `opencv-python`

### Setup
```bash
pip install ultralytics opencv-python
```

### Running the Monitor
To run detection on a video file:
```bash
python3 rrem_monitor.py --source input_video.mp4
```

To run on a webcam (index 0):
```bash
python3 rrem_monitor.py --source 0
```

### Batch Processing
To process a folder of test cases:
```bash
python3 batch_runner.py
```
