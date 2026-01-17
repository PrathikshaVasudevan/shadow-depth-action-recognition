import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# -----------------------------
# Config & constants
# -----------------------------
WINDOW_NAME = "Final Shadow Depth System"
PIXELS_PER_CM = 30.0            # geometric scale (approx; adjust if you calibrate)
SHADOW_K = 5000.0               # calibration constant for shadow-based depth (k / area)
SHADOW_THRESH = 0.7             # observed < thresh * baseline => shadow
SMOOTH_ALPHA = 0.2              # blend weight for shadow depth correction
APPROACH_BUFFER_FRAMES = 15     # ~0.5s at 30fps

# Assume single dominant light from camera direction (into the scene)
LIGHT_DIR = np.array([0.0, 0.0, -1.0])

# Mouth landmark IDs (MediaPipe Face Mesh)
MOUTH_IDS = [13, 14, 78, 308]   # upper/lower lip center + corners

# -----------------------------
# Helpers
# -----------------------------
def euclidean(p1, p2):
    return float(np.linalg.norm(np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)))

def compute_mouth_box_and_center(landmarks, w, h):
    xs = [int(landmarks[i].x * w) for i in MOUTH_IDS]
    ys = [int(landmarks[i].y * h) for i in MOUTH_IDS]
    x1, x2 = max(0, min(xs) - 20), min(w, max(xs) + 20)
    y1, y2 = max(0, min(ys) - 20), min(h, max(ys) + 20)
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    return (x1, y1, x2, y2), center

def shadow_mask_and_loss(roi_bgr):
    """
    Compute a simple physics-inspired shadow mask:
    - baseline intensity: local average (proxy for predicted Lambertian intensity)
    - shadow where observed < SHADOW_THRESH * baseline
    Returns: mask (uint8), loss (float32)
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    baseline = float(np.mean(gray))  # crude baseline; fast and robust for demo
    pred = np.full_like(gray, baseline, dtype=np.float32)
    loss = np.clip(pred - gray, 0, None)  # intensity loss due to occlusion
    mask = (gray < SHADOW_THRESH * pred).astype(np.uint8) * 255
    # clean up mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask, loss

def shadow_depth_from_area(mask):
    """
    Estimate depth from shadow area: depth_cm ≈ k / area
    Larger shadow area => smaller distance (hand closer to face).
    """
    area = float(np.sum(mask > 0))
    if area <= 1.0:
        return None
    depth_cm = SHADOW_K / area
    return depth_cm

def draw_heatmap_on_roi(frame, box, loss):
    x1, y1, x2, y2 = box
    H, W = y2 - y1, x2 - x1
    if H <= 0 or W <= 0:
        return frame
    # Normalize loss to [0,255] and colorize
    loss_norm = cv2.normalize(loss, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(loss_norm, cv2.COLORMAP_INFERNO)
    # Blend with original ROI
    roi = frame[y1:y2, x1:x2]
    blended = cv2.addWeighted(roi, 0.5, heatmap, 0.5, 0)
    frame[y1:y2, x1:x2] = blended
    return frame

def classify_action(z_cm, mouth_shadow, approach_buffer):
    touching = (z_cm is not None) and (z_cm < 2.0) and mouth_shadow
    approaching = (z_cm is not None) and (2.0 <= z_cm <= 5.0) and (sum(approach_buffer) > 3)
    if touching:
        return "Touching Face / Eating"
    if approaching:
        return "Approaching Mouth"
    return "None"

# -----------------------------
# Init MediaPipe & OpenCV
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

approach_buffer = deque(maxlen=APPROACH_BUFFER_FRAMES)

# -----------------------------
# Main loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face_mesh.process(rgb)
    hand_results = hands.process(rgb)

    mouth_center = None
    mouth_box = None
    fingertip = None
    action = "None"

    # --- Face / mouth ROI ---
    if face_results.multi_face_landmarks:
        fl = face_results.multi_face_landmarks[0]
        mouth_box, mouth_center = compute_mouth_box_and_center(fl.landmark, w, h)
        x1, y1, x2, y2 = mouth_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.circle(frame, mouth_center, 5, (255, 0, 0), -1)

    # --- Hand / fingertip ---
    if hand_results.multi_hand_landmarks:
        hl = hand_results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
        tip = hl.landmark[8]  # index fingertip
        fingertip = (int(tip.x * w), int(tip.y * h))
        cv2.circle(frame, fingertip, 8, (0, 255, 0), -1)

    # --- Distance & shadow physics ---
    z_geom_cm = None
    z_shadow_cm = None
    z_final_cm = None
    mouth_contains_shadow = False

    if mouth_center and fingertip:
        # Geometric distance (pixel -> cm)
        dist_pixels = euclidean(mouth_center, fingertip)
        z_geom_cm = dist_pixels / PIXELS_PER_CM

        # Draw line between fingertip and mouth center
        cv2.line(frame, mouth_center, fingertip, (0, 255, 255), 2)

        # Shadow analysis in mouth ROI
        if mouth_box:
            x1, y1, x2, y2 = mouth_box
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                mask, loss = shadow_mask_and_loss(roi)
                # Shadow-based depth
                z_shadow_cm = shadow_depth_from_area(mask)
                # Mouth contains shadow?
                mouth_contains_shadow = (np.sum(mask > 0) > 500)
                # Heatmap overlay
                frame = draw_heatmap_on_roi(frame, mouth_box, loss)

        # Combine cues: geometry + shadow correction
        if z_geom_cm is not None:
            if z_shadow_cm is not None:
                z_final_cm = max(0.0, (1.0 - SMOOTH_ALPHA) * z_geom_cm + SMOOTH_ALPHA * z_shadow_cm)
            else:
                z_final_cm = z_geom_cm

        # Classification
        approach_buffer.append(1 if (z_final_cm is not None and z_final_cm < 5.0) else 0)
        action = classify_action(z_final_cm, mouth_contains_shadow, approach_buffer)

    # --- Overlays ---
    if z_final_cm is not None:
        cv2.putText(frame, f"Distance: {z_final_cm:.2f} cm",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    elif z_geom_cm is not None:
        cv2.putText(frame, f"Distance (geom): {z_geom_cm:.2f} cm",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)

    cv2.putText(frame, f"Action: {action}",
                (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow(WINDOW_NAME, frame)

    # Exit conditions
    if cv2.waitKey(1) == 27:
        break
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()