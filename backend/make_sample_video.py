import os
import numpy as np
import cv2

OUTPUT_PATH = os.path.join("..", "sample_media", "traffic_4k.mp4")
WIDTH, HEIGHT = 1280, 720
FPS = 30
DURATION_S = 10

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (WIDTH, HEIGHT))

if not writer.isOpened():
    raise RuntimeError(
        "Could not open VideoWriter. If this fails on your machine, try "
        "changing fourcc from 'mp4v' to 'XVID' and OUTPUT_PATH's extension "
        "to .avi instead of .mp4."
    )

num_frames = FPS * DURATION_S
for i in range(num_frames):
    frame = np.full((HEIGHT, WIDTH, 3), (40, 40, 40), dtype=np.uint8)

    t = i / num_frames
    x1 = int((t * (WIDTH + 200)) % (WIDTH + 200)) - 100
    cv2.rectangle(frame, (x1, 300), (x1 + 120, 360), (60, 120, 220), -1)

    x2 = WIDTH - int(((t * 1.3) * (WIDTH + 200)) % (WIDTH + 200)) + 100
    cv2.rectangle(frame, (x2, 450), (x2 + 100, 500), (60, 200, 120), -1)

    cv2.putText(frame, f"frame {i}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2, cv2.LINE_AA)

    writer.write(frame)

writer.release()
print(f"Wrote {num_frames} frames to {OUTPUT_PATH}")