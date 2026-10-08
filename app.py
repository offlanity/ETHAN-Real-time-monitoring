import sys
import os
import webbrowser
import cv2

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from recognizer import FaceRecognizer
from kinematics import KinematicsTracker
from dashboard import DashboardServer, DashboardState
from utils import (
    FPSCounter,
    draw_kinematics_face,
    draw_fps,
    draw_info_bar,
    draw_add_prompt,
    draw_name_input,
)

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_PATH         = "dataset"
SIMILARITY_THRESHOLD = 0.45

# CAMERA SOURCE:
# Use "0" for laptop webcam, or paste your DroidCam URL: "http://<PHONE_IP>:4747/video"
CAMERA_SOURCE        = "0"
WINDOW_TITLE         = "C.L.A.I.V.E. - Physics & Facial Recognition Engine"
FRAME_WIDTH          = 1280
FRAME_HEIGHT         = 720
DET_SIZE             = (320, 320)
SKIP_FRAMES          = 2
DASHBOARD_HOST       = "127.0.0.1"
DASHBOARD_PORT       = 8765


def main() -> None:
    if not os.path.isdir(DATASET_PATH):
        os.makedirs(DATASET_PATH, exist_ok=True)
        print(f"[INFO] Created dataset folder at: {DATASET_PATH}")

    print("[INFO] Initializing InsightFace recognition pipeline...")
    recognizer = FaceRecognizer(
        dataset_path=DATASET_PATH,
        threshold=SIMILARITY_THRESHOLD,
        det_size=DET_SIZE,
    )

    # Initialize Physics & Kinematics Engine
    tracker = KinematicsTracker()

    source = int(CAMERA_SOURCE) if CAMERA_SOURCE.isdigit() else CAMERA_SOURCE
    print(f"[INFO] Connecting to camera: {source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera source: {source}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    dashboard_state = DashboardState(DATASET_PATH)
    dashboard = DashboardServer(dashboard_state, host=DASHBOARD_HOST, port=DASHBOARD_PORT)
    dashboard_url = dashboard.start()
    print(f"[INFO] Local classroom dashboard: {dashboard_url}")
    webbrowser.open(dashboard_url, new=2)

    fps_counter = FPSCounter(window=30)
    frame_idx = 0
    cached_detections = []

    typing_mode = False
    name_buffer = ""
    pending_snapshot = None
    pending_bbox = None

    print("\n[READY] System operational. Press ENTER on unknown faces to register.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] Stream disconnected.")
            break

        if not typing_mode:
            # Run deep learning face inference on every Nth frame to preserve FPS
            if frame_idx % SKIP_FRAMES == 0:
                cached_detections = recognizer.process_frame(frame)
                
                # Check for any unrecognized faces for live registration
                unknown = next((r for r in cached_detections if not r["recognized"]), None)
                if unknown is not None:
                    pending_snapshot = frame.copy()
                    pending_bbox = unknown["bbox"]
                else:
                    pending_snapshot = None
                    pending_bbox = None
            frame_idx += 1

        # ── Update Kinematics Tracker with current detections ─────────────────
        kinematics_results = tracker.update(cached_detections)

        # ── Draw Physics & Face Annotations ───────────────────────────────────
        for item in kinematics_results:
            draw_kinematics_face(frame, item)

        fps_counter.tick()
        draw_fps(frame, fps_counter.fps)
        draw_info_bar(frame, recognizer.num_registered)

        if typing_mode:
            draw_name_input(frame, name_buffer)
        elif pending_bbox is not None:
            draw_add_prompt(frame)

        dashboard_state.update(kinematics_results, recognizer.registered_names, fps_counter.fps)
        dashboard_state.set_frame(frame)

        cv2.imshow(WINDOW_TITLE, frame)
        key = cv2.waitKey(1) & 0xFF

        # Handle UI events
        if typing_mode:
            if key in (13, 10):  # Enter key confirms
                typed_name = name_buffer.strip()
                if typed_name and pending_snapshot is not None:
                    recognizer.add_person(typed_name, pending_snapshot, pending_bbox)
                typing_mode = False
                name_buffer = ""
                pending_snapshot = None
                pending_bbox = None
            elif key == 27:      # ESC cancels
                typing_mode = False
                name_buffer = ""
            elif key == 8:       # Backspace
                name_buffer = name_buffer[:-1]
            elif 32 <= key <= 126 and len(name_buffer) < 40:
                name_buffer += chr(key)
            continue

        if key in (13, 10) and pending_bbox is not None:
            typing_mode = True
            name_buffer = ""
        elif key in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    dashboard.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()