#!/usr/bin/env python3
import os
import subprocess
import time
import sys

DIRECTORY = os.path.dirname(os.path.abspath(__file__))
CLIPS_DIR = os.path.join(DIRECTORY, "demo_clips")

def generate_mock_videos():
    import cv2
    import numpy as np
    os.makedirs(CLIPS_DIR, exist_ok=True)
    
    videos = [
        {"name": "stirring_sauce.mp4", "shape": "circle"},
        {"name": "slicing_onion.mp4", "shape": "square"},
        {"name": "peeling_mango.mp4", "shape": "triangle"}
    ]

    for v in videos:
        out_path = os.path.join(CLIPS_DIR, v["name"])
        if os.path.exists(out_path):
            print(f"File {v['name']} already exists, skipping generation.")
            continue
            
        print(f"Generating mock video: {v['name']}...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, 24.0, (640, 360))
        
        for f in range(120):  # 5 seconds at 24 fps
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            x_pos = 100 + f * 4
            if v["shape"] == "circle":
                cv2.circle(frame, (x_pos, 180), 40, (16, 185, 129), -1)  # Emerald
            elif v["shape"] == "square":
                cv2.rectangle(frame, (x_pos - 30, 150), (x_pos + 30, 210), (59, 130, 246), -1)  # Blue
            else:
                pts = np.array([[x_pos, 140], [x_pos - 30, 210], [x_pos + 30, 210]], np.int32)
                cv2.fillPoly(frame, [pts], (168, 85, 247))  # Purple
            out.write(frame)
        out.release()
        print(f"Successfully generated {v['name']}")

def main():
    generate_mock_videos()
    
    print("\nStarting Action Recognition & Discovery Dashboard...")
    server_script = os.path.join(DIRECTORY, "dashboard_app.py")
    
    # Run the server
    p = subprocess.Popen([sys.executable, server_script])
    try:
        # Give it a second to start
        time.sleep(2)
        print("\nDashboard is active! Access it at http://localhost:8000")
        print("Keep this process running. Press Ctrl+C to terminate.")
        p.wait()
    except KeyboardInterrupt:
        print("\nTerminating dashboard server...")
        p.terminate()
        p.wait()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()
