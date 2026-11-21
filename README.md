# Human Detection, Tracking and Counting System

Real-time people detection, persistent multi-object tracking (`P1`, `P2`, `P3`...), and unique ID counting in video streams using **Python**, **Ultralytics YOLO (`yolo26n.pt`)**, and **OpenCV**.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO26-111111)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Arrays-013243?logo=numpy&logoColor=white)

---

## Key Features

- **Person-Only Detection:** Uses `yolo26n.pt` filtered to the `person` class (`class = 0`) for fast, accurate detection.
- **Persistent ID Labels (`P1`, `P2`, `P3`...):** Assigns readable labels backed by tracker IDs across consecutive frames.
- **Unique Tracked Count:** Tracks unique IDs seen during playback rather than counting the same person repeatedly on every frame.
- **Test Script Included:** Includes `test_yolo.py` to verify model inference and video playback before running the full pipeline.
- **Interactive Exit:** Click the OpenCV display window and press `q` to stop playback cleanly.

---

## Tech Stack, Models and Techniques

| Layer | Technology / Technique | Role in Project |
| :--- | :--- | :--- |
| **Language** | **Python** | Application logic, video capture loop, and ID tracking |
| **Detection Model** | **YOLO26n (`yolo26n.pt`)** | Real-time object detection locating people (`class 0`) |
| **Tracker** | **ByteTrack** *(via Ultralytics)* | Associates detections across frames using motion and overlap |
| **Vision and Math** | **OpenCV (`cv2`) and NumPy** | Frame reading, bounding-box/label rendering, array operations |
| **Legacy / Reference** | **Haar Cascades (`*.xml`)** | Included Haar cascade XML files (`haarcascade_frontalface_default.xml`, `haarcascade_fullbody.xml`) |

### How the Techniques Work Together

1. **Frame Extraction:** OpenCV reads frames sequentially from `people.mp4` or a connected camera.
2. **Object Detection:** `yolo26n.pt` detects people (`class 0`) and returns bounding boxes and confidence scores.
3. **Multi-Object Tracking (MOT):** ByteTrack links detections across frames so tracked individuals keep consistent IDs.
4. **Unique ID Counting:** Each new tracker ID is mapped to a label (`P1`, `P2`, ...) and recorded in a set or dictionary of seen IDs.

---

## Project Structure

```text
human-detector-and-counter/
├── .gitignore                        # Git ignore rules
├── haarcascade_frontalface_default.xml # Haar cascade model for frontal faces
├── haarcascade_fullbody.xml            # Haar cascade model for full body
├── image.png                         # Sample image / preview asset
├── main.py                           # Main detection, tracking, and counting script
├── people.mp4                        # Sample input video
├── README.md                         # Project documentation
├── test_yolo.py                      # YOLO testing script
└── yolo26n.pt                        # Ultralytics YOLO26 nano weights



```

## Installation and Usage

### Step 1 — Install Dependencies

Install Ultralytics YOLO, OpenCV, NumPy, and PyTorch:

```
pip install ultralytics opencv-python numpy torch torchvision

```

### Step 2 — Run the YOLO Test Script

Verify model loading (`yolo26n.pt`) and frame inference on `people.mp4`:

```bash
python test_yolo.py

```

### Step 3 — Run the Main Application

Start human detection, tracking, and unique ID counting:

```bash
python main.py

```

> **Window Control:** Click inside the OpenCV video window to focus it, then press `q` to exit.

---

## Limitations and Future Improvements

### Current Limitations

| Area | Details | Impact on Output |
| --- | --- | --- |
| **Occlusions and Re-entry** | Heavy overlap or exiting and re-entering the scene can trigger a new track ID | Count reflects unique tracked IDs rather than strict headcount |

### Planned Enhancements

* **Entry and Exit Zone Counting:** Track directional movement across a line or region of interest.
* **Video and Report Export:** Save annotated frames with `cv2.VideoWriter` and export summary logs to CSV or JSON.
* **Environment Lockfile:** Add a `requirements.txt` file for one-command reproducible setup.

