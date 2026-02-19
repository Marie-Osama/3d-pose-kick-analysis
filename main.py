import cv2
import mediapipe as mp
import numpy as np
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque

# ────────────────────────────────────────────────
# MediaPipe Pose Landmarker Setup
# ────────────────────────────────────────────────
BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

model_path = "pose_landmarker_lite.task"

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,                        # Assume single person (poomsae)
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_segmentation_masks=False
)

landmarker = PoseLandmarker.create_from_options(options)

# ────────────────────────────────────────────────
# Video I/O
# ────────────────────────────────────────────────
video_path = "input/Koryo poomsae by world champion in 2023. #고려 #koryo #poomsae.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video file")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output/output_taekwondo_3d_analysis.mp4", fourcc, fps, (width, height))

# ────────────────────────────────────────────────
# Analysis variables
# ────────────────────────────────────────────────
prev_landmarks_3d = None
left_was_high = False
right_was_high = False
left_kick_count = 0
right_kick_count = 0 
total_kicks = 0
kick_history = deque(maxlen=3)   # remembers last 3 frames


frame_timestamp_ms = 0.0

print("Processing video... Press 'q' to quit preview.")

# MediaPipe Pose connections (same as the old POSE_CONNECTIONS)
POSE_CONNECTIONS = [
    # Head & face
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    # Upper body
    (9, 10),
    (11, 13), (13, 15),          # left arm
    (12, 14), (14, 16),          # right arm
    (11, 12),                    # shoulders
    (11, 23), (12, 24),          # torso to hips
    (23, 24),                    # hips
    # Legs
    (23, 25), (25, 27),          # left leg
    (24, 26), (26, 28),          # right leg
    # Feet (optional - some versions include more foot points)
    (27, 29), (27, 31), (28, 30), (28, 32),
    (29, 31), (30, 32),
]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_timestamp_ms += 1000 / fps  # in milliseconds

    # Prepare MediaPipe image (RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Run pose landmarker
    results = landmarker.detect_for_video(mp_image, int(frame_timestamp_ms))

    annotated_frame = frame.copy()

    if results.pose_world_landmarks:
        # ────────────────────────────────────────────────
        # Get 3D world landmarks (meters, origin at hips approx, y up)
        # ────────────────────────────────────────────────
        world_lm = results.pose_world_landmarks[0]  # first (only) person

        def get_3d(idx):
            lm = world_lm[idx]
            return np.array([lm.x, lm.y, lm.z])

        # MediaPipe indices
        nose        = get_3d(0)
        left_shoulder = get_3d(11)
        right_shoulder = get_3d(12)
        left_hip    = get_3d(23)
        right_hip   = get_3d(24)
        left_knee   = get_3d(25)
        right_knee  = get_3d(26)
        left_ankle  = get_3d(27)
        right_ankle = get_3d(28)

        # ────────────────────────────────────────────────
        # Text position (using 2D image landmarks for stable overlay)
        # ────────────────────────────────────────────────
        text_pos = (20, 150)

        text_offset = 0
        line_height = 28

        offset_container = [0]  # list with one int

        def put_text(text, color=(255, 255, 255), thickness=2, scale=1.0):
            cv2.putText(annotated_frame, text,
                        (text_pos[0], text_pos[1] + offset_container[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
            offset_container[0] += line_height

        # ────────────────────────────────────────────────
        # Leg angle at hips (3D)
        # ────────────────────────────────────────────────
        avg_hip = (left_hip + right_hip) / 2
        vec_l = left_knee - avg_hip
        vec_r = right_knee - avg_hip
        norm_l = np.linalg.norm(vec_l)
        norm_r = np.linalg.norm(vec_r)

        high_kick_detected_this_frame = False

        if norm_l > 1e-6 and norm_r > 1e-6:
            cos_theta = np.clip(np.dot(vec_l, vec_r) / (norm_l * norm_r), -1, 1)
            leg_angle = math.degrees(math.acos(cos_theta))
            color = (0, 255, 0) if leg_angle > 130 else (0, 165, 255)
            put_text(f"Leg Angle: {leg_angle:.1f}°", color)

            if leg_angle > 130:
                high_kick_detected_this_frame = True
                put_text("HIGH KICK!", (0, 0, 255), thickness=3, scale=1.0)

        kick_history.append(high_kick_detected_this_frame)

        previous_were_high = any(kick_history[i] for i in range(len(kick_history)-1)) \
                     if len(kick_history) > 1 else False
                     
        if high_kick_detected_this_frame and not previous_were_high:  # not any of the previous ones
            total_kicks += 1
            put_text("HIGH KICK! (new)", (0, 0, 255), thickness=3, scale=1.5)
        elif high_kick_detected_this_frame:
            put_text("HIGH KICK! (continuing)", (100, 100, 255), scale=1.0)  # optional debug

        put_text(f"Total Kicks: {total_kicks}", (0, 255, 255))
    
        # ────────────────────────────────────────────────
        # Draw 2D landmarks & connections
        # ────────────────────────────────────────────────
        if results.pose_landmarks:
            landmarks_2d = results.pose_landmarks[0]

            # Lines first (background layer)
            for conn in POSE_CONNECTIONS:
                i, j = conn
                if i < len(landmarks_2d) and j < len(landmarks_2d):
                    start = landmarks_2d[i]
                    end   = landmarks_2d[j]
                    if start.visibility > 0.25 and end.visibility > 0.25:
                        x1, y1 = int(start.x * width), int(start.y * height)
                        x2, y2 = int(end.x   * width), int(end.y   * height)
                        cv2.line(annotated_frame, (x1, y1), (x2, y2), (200, 100, 0), 3)  # orange lines

            # Circles on top
            for i, lm in enumerate(landmarks_2d):
                x = int(lm.x * width)
                y = int(lm.y * height)
                radius = 6 if i in [0,11,12,23,24,25,26,27,28] else 4   # bigger for main joints
                cv2.circle(annotated_frame, (x, y), radius, (50, 255, 50), -1)

    cv2.imshow("Taekwondo 3D Analysis", annotated_frame)
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Done!")
print(f"Output saved: output_taekwondo_3d_analysis.mp4")
print(f"Total Kicks: {total_kicks}")
