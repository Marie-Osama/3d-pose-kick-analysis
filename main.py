import cv2
import mediapipe as mp
import numpy as np
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque

# ────────────────────────────────────────────────
# Setup MediaPipe Pose Landmarker
# ────────────────────────────────────────────────
options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path="pose_landmarker_lite.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_segmentation_masks=False
)
landmarker = vision.PoseLandmarker.create_from_options(options)

# ────────────────────────────────────────────────
# Video I/O
# ────────────────────────────────────────────────
video_path = "input/Koryo poomsae by world champion in 2023. #고려 #koryo #poomsae.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened(): raise Exception("Error opening video")

width, height = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("output/output_taekwondo_3d_analysis.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,13),(13,15),(12,14),(14,16),
    (11,12),(11,23),(12,24),(23,24),
    (23,25),(25,27),(24,26),(26,28),
    (27,29),(27,31),(28,30),(28,32),(29,31),(30,32)
]

# ────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────
def get_3d(world, i):
    lm = world[i]
    return np.array([lm.x, lm.y, lm.z])

def draw_landmarks(frame, lm_list):
    for i, j in POSE_CONNECTIONS:
        if lm_list[i].visibility > 0.25 and lm_list[j].visibility > 0.25:
            p1 = int(lm_list[i].x * width), int(lm_list[i].y * height)
            p2 = int(lm_list[j].x * width), int(lm_list[j].y * height)
            cv2.line(frame, p1, p2, (200, 100, 0), 3)

    for i, lm in enumerate(lm_list):
        x, y = int(lm.x * width), int(lm.y * height)
        cv2.circle(frame, (x, y), 6 if i in [0,11,12,23,24,25,26,27,28] else 4,
                   (50, 255, 50), -1)

def put_text(frame, text, y, color=(255,255,255), scale=1, thickness=2):
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness)

# ────────────────────────────────────────────────
# Kick detection state
# ────────────────────────────────────────────────
kick_history = deque(maxlen=3)
frame_time = 0
total_kicks = 0

# ────────────────────────────────────────────────
# Main Loop
# ────────────────────────────────────────────────
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame_time += 1000 / fps

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = landmarker.detect_for_video(mp_image, int(frame_time))
    annotated = frame.copy()

    y_offset = 150

    if results.pose_world_landmarks:
        world = results.pose_world_landmarks[0]

        # Extract important joints
        left_hip  = get_3d(world, 23)
        right_hip = get_3d(world, 24)
        left_knee = get_3d(world, 25)
        right_knee= get_3d(world, 26)

        avg_hip = (left_hip + right_hip) / 2
        vec_l = left_knee - avg_hip
        vec_r = right_knee - avg_hip

        # Compute 3D leg angle
        if np.linalg.norm(vec_l) > 1e-6 and np.linalg.norm(vec_r) > 1e-6:
            cos_t = np.dot(vec_l, vec_r) / (np.linalg.norm(vec_l)*np.linalg.norm(vec_r))
            angle = math.degrees(math.acos(np.clip(cos_t, -1, 1)))

            high = angle > 130
            kick_history.append(high)

            put_text(annotated, f"Leg Angle: {angle:.1f}°", y_offset,
                     (0,255,0) if high else (0,165,255))
            y_offset += 28

            # Detect NEW kick (not repeated)
            if high and not any(kick_history[:-1]):
                total_kicks += 1
                put_text(annotated, "HIGH KICK! (new)", y_offset, (0,0,255), 1.3, 3)
            elif high:
                put_text(annotated, "HIGH KICK! (continuing)", y_offset, (100,100,255))
            y_offset += 28

            put_text(annotated, f"Total Kicks: {total_kicks}", y_offset, (0,255,255))

        # Draw 2D skeleton overlay
        if results.pose_landmarks:
            draw_landmarks(annotated, results.pose_landmarks[0])

    cv2.imshow("Taekwondo 3D Analysis", annotated)
    out.write(annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Done!")
print("Total Kicks:", total_kicks)