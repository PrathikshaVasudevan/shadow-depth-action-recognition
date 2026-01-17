# Physics-Based Shadow Depth Action Recognition

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

### 2. Install Dependencies

```bash
pip install opencv-python mediapipe numpy