# Physics-Based Vision: Shadow Depth Action Recognition

This project was developed during the Winter Internship'25 at console.success 

This project detects **hand-to-mouth actions** using a combination of **geometry** and **physics-inspired shadow analysis** instead of relying only on keypoints.  

It estimates the **3D distance** between the hand and mouth by:
- Tracking face and hand landmarks (MediaPipe)
- Detecting shadow intensity on the face
- Generating a shadow heatmap
- Combining geometric distance with shadow-based depth
- Classifying actions in real time

---

## ✨ Features

- Real-time webcam processing
- Face + hand landmark detection
- Mouth region tracking
- Shadow detection using intensity loss
- Heatmap visualization
- Distance estimation (cm)
- Action classification:
  - **None**
  - **Approaching Mouth**
  - **Touching Face / Eating**

---

## 🛠 Tech Stack

- Python  
- OpenCV  
- MediaPipe  
- NumPy  

---

## ⚙️ Installation

### 1. Create and activate a virtual environment (Windows example)

```bash
python -m venv .venv
.venv\Scripts\activate
```
### 2. Install dependencies
```bash
pip install opencv-python mediapipe numpy
```
## Run the Project
```bash
python shadow_depth_final.py
```
The app opens a window titled “Final Shadow Depth System” and starts your default webcam.
Press Esc to quit.

---

## How It Works🔍
- Detects face and hand landmarks using MediaPipe.
- Finds the mouth region (bounding box + center).
- Tracks fingertip (index tip).
- Measures geometric distance (pixels → cm).
- Detects shadow in the mouth ROI by comparing observed intensity to a baseline.
- Creates a shadow intensity heatmap overlay.
- Combines geometry + shadow physics to estimate final distance.
- Classifies the action based on thresholds.

## 🎯 Output
- Distance shown in cm (final blended estimate).
- Action label displayed (“None”, “Approaching Mouth”, “Touching Face / Eating”).
- Heatmap overlay on mouth region (shadow intensity loss).
- Real-time tracking with fingertip and mouth center visualization.

## 🌐 Applications
- Health monitoring (face-touching awareness)
- Eating/smoking detection
- Human–computer interaction
- Behavioral analysis
- Computer vision research

## 📹 Demo
👉 [Watch the YouTube Demo](https://youtu.be/B8xPWeGXofk)
In the demo, you can see:
- Mouth region detection
- Fingertip tracking
- Distance estimation in centimeters
- Shadow intensity heatmap overlay
- Action classification (None, Approaching, Touching Face / Eating)

## 👩‍💻 Author
Prathiksha Vasudevan
