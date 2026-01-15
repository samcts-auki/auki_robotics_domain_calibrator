#!/usr/bin/env python3
"""
Live QR detection with smooth video preview.

Runs detection in a background thread to avoid freezing the UI.
"""

import cv2
import time
import threading
import queue
import sys
from pathlib import Path

# Ensure parent directory is on sys.path for local imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from domain_calibrator import DomainCalibratorPy


def main():
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Failed to open camera index {camera_index}")
        raise SystemExit(1)

    calibrator = DomainCalibratorPy()

    frame_queue = queue.Queue(maxsize=1)
    result_lock = threading.Lock()
    last_status = "No valid markers detected"
    last_pose = None
    running = True

    def detection_worker():
        nonlocal last_status, last_pose, running
        while running:
            try:
                frame = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            result = calibrator.detect_and_calibrate(frame)
            with result_lock:
                if result is None:
                    last_status = "No valid markers detected"
                    last_pose = None
                else:
                    last_status = "QR detected"
                    last_pose = result

    worker = threading.Thread(target=detection_worker, daemon=True)
    worker.start()

    print("Live detection running. Press 'q' to quit.")

    last_submit = 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        # Downscale for faster detection
        small = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)

        # Submit a frame at ~10 Hz
        now = time.time()
        if now - last_submit >= 0.1:
            if frame_queue.empty():
                frame_queue.put(small)
                last_submit = now

        # Overlay status
        with result_lock:
            status = last_status

        color = (0, 255, 0) if status == "QR detected" else (0, 0, 255)
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        cv2.imshow("Live QR Detection (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    running = False
    worker.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
