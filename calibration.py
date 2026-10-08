import cv2
import math
import os

from calibration_config import save_pixels_per_meter

CAMERA_SOURCE = os.environ.get("CAMERA_SOURCE", "0") # Change to DroidCam URL if needed
REFERENCE_LENGTH_METERS = 1.0

clicks = []

def mouse_callback(event, x, y, flags, param):
    global clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))
        print(f"[CLICK] Point registered at pixel coordinates: ({x}, {y})")

def main():
    source = int(CAMERA_SOURCE) if CAMERA_SOURCE.isdigit() else CAMERA_SOURCE
    cap = cv2.VideoCapture(source)
    cv2.namedWindow("Linear Calibration Tool")
    cv2.setMouseCallback("Linear Calibration Tool", mouse_callback)

    print("\n" + "="*55)
    print(" LINEAR CALIBRATION TOOL")
    print(" 1. Keep the camera resolution/zoom fixed and use a 1-meter reference")
    print("    at approximately the same depth/plane as the tracked subject.")
    print(" 2. Click the START and END of the reference.")
    print(" 3. Press 's' to save the px/m scale, 'r' to reset, or 'q' to quit.")
    print("="*55 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw clicked points and connecting line
        for pt in clicks:
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)

        if len(clicks) >= 2:
            pt1, pt2 = clicks[0], clicks[1]
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

            # Euclidean pixel distance formula: √((x2 - x1)² + (y2 - y1)²)
            pixel_distance = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])

            cv2.putText(
                frame,
                f"Pixel Distance: {pixel_distance:.2f} px",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
            )
            pixels_per_meter = pixel_distance / REFERENCE_LENGTH_METERS
            cv2.putText(
                frame,
                f"Scale: {pixels_per_meter:.2f} px/m (press 's' to save)",
                (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Linear Calibration Tool", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            clicks.clear()
            print("[INFO] Reset points.")
        elif key == ord('s') and len(clicks) >= 2:
            pt1, pt2 = clicks[0], clicks[1]
            pixel_distance = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
            pixels_per_meter = pixel_distance / REFERENCE_LENGTH_METERS
            save_pixels_per_meter(pixels_per_meter)
            print(
                f"[INFO] Saved calibration: {pixels_per_meter:.2f} px/m. "
                "Restart the tracker to load it."
            )
        elif key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()