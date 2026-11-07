import cv2
from PIL import Image
import numpy as np
from config import NUM_FRAMES

def sample_video_frames(video_path, num_frames=NUM_FRAMES):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"Video has no readable frames: {video_path}")
    indices = np.linspace(0, max(total - 1, 0), num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames extracted from {video_path}")
    # Repeat last valid frame if short/corrupt clips yield fewer frames.
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    return frames[:num_frames]
