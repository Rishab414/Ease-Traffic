# EaseTraffic: 4-Grid Intelligent Traffic Control 🚦

EaseTraffic is an AI-powered traffic management system designed to optimize signal timings at 4-way intersections. By utilizing real-time computer vision and vehicle tracking, the system dynamically adjusts green light durations based on actual vehicle density, reducing congestion and improving transit efficiency.

The project implements the **YOLOv8** object detection model combined with the **SORT (Simple Online and Realtime Tracking)** algorithm to analyze live video feeds from four directions (North, South, East, West) simultaneously.

## Features

* **4-Grid Real-time Monitoring:** Simultaneous processing of four camera feeds in a unified dashboard.
* **YOLOv8 Detection:** High-accuracy detection of cars, motorbikes, buses, and trucks.
* **Intelligent Vehicle Tracking:** Uses Kalman Filters (via SORT) to maintain unique vehicle IDs and prevent double-counting.
* **Two-Phase Operation:**
* **Analysis Mode:** An initial 10-second window to collect baseline traffic density.
* **Running Mode:** Dynamic signal rotation based on vehicle counts.


* **Priority-Based Scheduling:** Automatically ranks directions from highest to lowest density and assigns green light durations accordingly.
* **Dynamic UI Overlay:** Polished visual indicators showing signal status (Red/Green), vehicle counts, and mode status.
* **Automated Looping:** Handles video stream resets automatically for continuous simulation.

## Technologies Used

* **Python:** Core application logic.
* **OpenCV:** Image processing, video handling, and UI rendering.
* **YOLOv8 (Ultralytics):** Real-time object detection.
* **SORT Algorithm:** Multi-object tracking implementation.
* **NumPy:** Efficient numerical array processing for coordinates and matrices.
* **FilterPy:** Kalman Filter implementation for vehicle trajectory prediction.
* **Cvzone:** Enhanced UI elements and corner overlays.

## Output
<img width="1598" height="922" alt="Screenshot 2026-01-12 224305" src="https://github.com/user-attachments/assets/49b9a6b1-bee7-4814-88be-1bcb4affd7b0" />

<img width="1600" height="916" alt="Screenshot 2026-01-12 224246" src="https://github.com/user-attachments/assets/6547fff5-e910-41cd-bb17-6b728b3211a7" />


## Project Structure

```
EaseTraffic/
├── Ease.py             # Main control loop and Traffic Management Logic
├── sort.py             # SORT Tracking algorithm implementation
├── n1.mp4              # North direction video feed
├── s.mp4               # South direction video feed
├── e1.mp4              # East direction video feed
├── w1.mp4              # West direction video feed
└── Yolo-Weights/
    └── yolov8m.pt      # Pre-trained YOLOv8 weights

```

## How It Works

### 1. Detection & Tracking

For every frame, the system identifies vehicles using YOLOv8. These detections are passed to the `Sort` class in `sort.py`, which uses a **Kalman Filter** to predict where the vehicle will be in the next frame. This ensures that even if a vehicle is briefly obscured, its ID remains constant.

### 2. Line Crossing Logic

A virtual "detection line" is placed in the center of each grid. When a vehicle's center point crosses this line (either Top-to-Bottom or Bottom-to-Top), the `counting_set` is updated, and the UI flashes green to indicate a successful count.

### 3. Signal Calculation

The system follows a ranking logic:

1. The direction with the **most** vehicles gets the longest green light (**15s**).
2. The direction with the **least** vehicles gets the shortest green light (**8s**).
3. This priority list is recalculated at the end of every 4-direction cycle based on the "Cycle Counts" gathered.

## Getting Started

### Prerequisites

* **Python 3.8+**
* **Conda or Pip** for environment management
* **GPU (Optional):** Recommended for higher FPS (CUDA-enabled)

### Installation

**1) Clone the Repository:**

```bash
git clone https://github.com/yourusername/EaseTraffic.git
cd EaseTraffic

```

**2) Install Dependencies:**

```bash
pip install ultralytics opencv-python numpy filterpy scikit-image matplotlib cvzone

```

**3) Download YOLO Weights:**
Ensure you have `yolov8m.pt` inside a folder named `Yolo-Weights` in the parent directory, or update the `YOLO_WEIGHTS` path in `Ease.py`.

**4) Run the Application:**

```bash
python Ease.py

```

## Usage

1. **Analysis Mode:** Upon startup, the system will spend 10 seconds (default) observing all 4 feeds. Do not interrupt this phase.
2. **Running Mode:** After analysis, the system starts the green light rotation.
3. **Controls:**
* Press **'q'** to quit the application.
* Press **'r'** to reset the system and restart the 10-second analysis phase.



## Algorithm Details

### YOLOv8 (You Only Look Once)

* Used for localized object detection.
* Specifically filtered to detect `car`, `motorbike`, `bus`, and `truck`.

### SORT (Simple Online and Realtime Tracking)

* **Kalman Filter:** Predicts future state of vehicle boxes.
* **Hungarian Algorithm:** Efficiently associates new detections with existing tracks.
* **IOU (Intersection over Union):** Used as a metric to match detections to trackers.

## Future Enhancements

* **Density Estimation:** Instead of just counting crossings, measure the total area occupied by vehicles.
* **Emergency Vehicle Priority:** Use audio or visual cues to give instant green lights to ambulances/fire trucks.
* **IoT Integration:** Connect the software to physical LED signal hardware via Arduino/Raspberry Pi.
* **Weather Adaptation:** Adjust detection thresholds dynamically for rain or night-time conditions.

## Troubleshooting

**Issue: Low FPS**

* **Solution:** Use a smaller YOLO model (e.g., `yolov8n.pt`) or ensure OpenCV is using your GPU.

**Issue: ID Switching**

* **Solution:** Increase the `max_age` parameter in the `Sort` initialization to keep track of disappeared objects for longer.

**Issue: Videos not loading**

* **Solution:** Check the file names in the `caps` dictionary within `Ease.py` to match your local `.mp4` files.

## License

This project is licensed under the MIT License.

## Acknowledgments

* **Ultralytics** for the YOLOv8 implementation.
* **Alex Bewley** for the original SORT algorithm research.
* **OpenCV Community** for the robust visualization tools.

---

**Version:** 1.0.0

**Last Updated:** March 2025

**Status:** Functional Prototype
