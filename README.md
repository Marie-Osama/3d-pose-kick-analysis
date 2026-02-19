# 3D Pose-Based Kick Analysis Using MediaPipe

This project demonstrates **3D pose estimation**, **joint-angle computation**, and **movement detection** using MediaPipe’s Pose Landmarker.  
The system analyzes leg movement in a video, detects high kicks, counts repetitions, and overlays 3D/2D pose information on the output video.
<img width="367" height="703" alt="Screenshot 2026-02-19 at 3 02 14 PM" src="https://github.com/user-attachments/assets/d258ecdd-4c4b-4ffd-9a5d-4b9e64329b0b" />

---

## ✨ Features
- Extracts **3D world landmarks** (x, y, z in meters)
- Computes **leg angle at the hip** using vector math
- Detects **high kicks** using an angle threshold
- Counts **unique kicks** with temporal smoothing
- Visualizes:
  - full skeleton overlay  
  - text statistics (angle, kick count)
- Outputs final annotated video (`output_taekwondo_3d_analysis.mp4`)

---

## 📦 Requirements
Install dependencies:

```bash
pip install -r requirements.txt

You will also need a MediaPipe model file:

Download:
pose_landmarker_lite.task
from:
https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

Place the file in the same folder as main.py.
```
---

## ▶️ How to Run
python main.py

Input video:
Set video_path in the code to your file.

Output:
A fully annotated video is saved as:
output_taekwondo_3d_analysis.mp4

--- 

## 📁 Project Structure

project/
│ main.py
| input/
| output/
│ pose_landmarker_lite.task
│ requirements.txt
│ README.md
