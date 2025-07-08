import os
from dotenv import load_dotenv
import cv2
import time
from datetime import datetime

# Set environment variables to handle Wayland and display issues
os.environ["QT_QPA_PLATFORM"] = "xcb"  # Force X11 instead of Wayland
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp"  # Use TCP for RTSP
)

# Ensure the 'pictures' directory exists
PICTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "pictures")
os.makedirs(PICTURES_DIR, exist_ok=True)

# Load environment variables from .env file
load_dotenv()
USER = os.getenv("CAM_USER")
PASS = os.getenv("CAM_PASS")
IP = os.getenv("CAM_IP")

# Validate that all required environment variables are set
if not USER or not PASS or not IP:
    raise ValueError(
        "Missing environment variables: CAM_USER, CAM_PASS or CAM_IP"
    )

# Initialize HOG (Histogram of Oriented Gradients) person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Construct RTSP URL for the IP camera
rtsp_url = f"rtsp://{USER}:{PASS}@{IP}/live/ch0"

# Configure video capture with FFMPEG backend for better RTSP support
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

# Set additional properties for better RTSP handling and performance
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size for lower latency
cap.set(cv2.CAP_PROP_FPS, 15)  # Reduce FPS to lower CPU usage

# Verify that the camera stream opened successfully
if not cap.isOpened():
    raise Exception(f"Could not open RTSP stream: {rtsp_url}")

print("Press 'q' to exit.")
print("Motion detection active. Person detection only runs on motion.")
print("Images with detected people+motion will be saved in 'pictures'.")


def filter_detections(
    boxes, weights, min_confidence=0.4, min_height=120, max_aspect_ratio=2.5
):
    """
    Filter detections to reduce false positives from objects like bags, bottles,
    chairs, etc.

    This function applies several filters to ensure only realistic person detections
    are returned:
    - Confidence threshold: Only high-confidence detections
    - Minimum height: People should be reasonably tall
    - Aspect ratio: People are typically taller than wide
    - Width ratio: Width should be proportional to height
    - Area ratio: Detection area should be reasonable for a person

    Args:
        boxes: List of detection boxes (x, y, w, h)
        weights: List of confidence scores from the HOG detector
        min_confidence: Minimum confidence threshold (0.0 to 1.0)
        min_height: Minimum height in pixels for a detection to be considered a
                   person
        max_aspect_ratio: Maximum width/height ratio
                         (people are typically taller than wide)

    Returns:
        Filtered list of (box, weight) tuples containing only realistic person
        detections
    """
    filtered_detections = []

    for box, weight in zip(boxes, weights):
        x, y, w, h = box

        # Filter 1: Check confidence threshold (increased from 0.3 to 0.4)
        # Only accept detections with high confidence to reduce false positives
        if weight < min_confidence:
            continue

        # Filter 2: Check minimum height (increased from 100 to 120)
        # People should be reasonably tall in the frame
        if h < min_height:
            continue

        # Filter 3: Check aspect ratio (reduced from 3.0 to 2.5)
        # People are typically taller than wide, so width/height should be <
        # max_ratio
        aspect_ratio = w / h
        if aspect_ratio > max_aspect_ratio:
            continue

        # Filter 4: Check width relative to height (more strict)
        # Width should be at least 35% of height to avoid thin objects like chairs
        if w < h * 0.35:
            continue

        # Filter 5: Check maximum width relative to height
        # Width should not be too large compared to height (avoid wide objects)
        if w > h * 0.8:
            continue

        # Filter 6: Check area ratio (new filter)
        # The detection area should be reasonable for a person
        # Too small areas might be noise, too large might be furniture
        area = w * h
        frame_area = 1920 * 1080  # Assuming typical frame size
        area_ratio = area / frame_area

        # Area should be between 0.01% and 15% of frame area
        if area_ratio < 0.0001 or area_ratio > 0.15:
            continue

        # Filter 7: Check if detection is too close to frame edges (new filter)
        # People are usually not detected right at the edges
        margin = 20
        if (
            x < margin
            or y < margin
            or x + w > 1920 - margin
            or y + h > 1080 - margin
        ):
            continue

        # If all filters pass, add to filtered detections
        filtered_detections.append((box, weight))

    return filtered_detections


# Performance optimization variables
frame_count = 0
consecutive_errors = 0
detection_interval = 3
last_detections = []
fps_counter = 0
fps_start_time = time.time()
current_fps = 0

# Motion detection variables
motion_detected = False
motion_min_area = 5000  # Minimum area in pixels to consider as motion
prev_gray = None
motion_box = None

print("Starting video processing loop...")

while True:
    ret, frame = cap.read()
    frame_count += 1
    fps_counter += 1

    if not ret:
        consecutive_errors += 1
        print(
            f"Could not read frame {frame_count}. "
            f"Consecutive errors: {consecutive_errors}"
        )
        if consecutive_errors > 10:
            print("Too many consecutive errors. Attempting to reconnect...")
            cap.release()
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 15)
            consecutive_errors = 0
            frame_count = 0
            continue
        cv2.waitKey(100)
        continue
    consecutive_errors = 0

    # Calculate FPS every second
    if time.time() - fps_start_time >= 1.0:
        current_fps = fps_counter / (time.time() - fps_start_time)
        fps_counter = 0
        fps_start_time = time.time()

    # --- MOTION DETECTION ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    motion_detected = False
    motion_box = None
    if prev_gray is not None:
        frame_delta = cv2.absdiff(prev_gray, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for c in contours:
            if cv2.contourArea(c) < motion_min_area:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            motion_box = (x, y, w, h)
            motion_detected = True
            # Draw red rectangle for motion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            break  # Only mark the first large motion area
    prev_gray = gray

    # --- PERSON DETECTION ONLY IF MOTION ---
    people_detected = 0
    if motion_detected and frame_count % detection_interval == 0:
        height, width = frame.shape[:2]
        detection_width = width // 2
        detection_height = height // 2
        detection_frame = cv2.resize(
            frame, (detection_width, detection_height)
        )
        boxes, weights = hog.detectMultiScale(
            detection_frame,
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05,
            hitThreshold=0,
        )
        scaled_boxes = []
        for x, y, w, h in boxes:
            scaled_x = int(x * 2)
            scaled_y = int(y * 2)
            scaled_w = int(w * 2)
            scaled_h = int(h * 2)
            scaled_boxes.append((scaled_x, scaled_y, scaled_w, scaled_h))
        last_detections = filter_detections(scaled_boxes, weights)

        # Save image if a person is detected
        if last_detections:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"person_{now}.jpg"
            filepath = os.path.join(PICTURES_DIR, filename)
            cv2.imwrite(filepath, frame)
            print(f"Saved image: {filepath}")

    # Draw bounding boxes for detected people (use cached results)
    for (x, y, w, h), confidence in last_detections:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Person ({confidence:.2f})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        people_detected += 1

    # --- UI OVERLAYS ---
    detection_text = f"People detected: {people_detected}"
    cv2.putText(
        frame,
        detection_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    fps_part = f"FPS: {current_fps:.1f}"
    frame_part = f"Frame: {frame_count}"
    performance_text = f"{fps_part} | {frame_part}"
    cv2.putText(
        frame,
        performance_text,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    opt_text = "CPU Optimized: Detection every 3 frames, 50% resolution"
    cv2.putText(
        frame,
        opt_text,
        (10, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    filter_text = "Strict filtering: confidence>0.4, height>120px, aspect<2.5"
    cv2.putText(
        frame,
        filter_text,
        (10, frame.shape[0] - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
    )
    if motion_detected:
        cv2.putText(
            frame,
            "MOTION DETECTED",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    cv2.imshow("Live View - IP Camera (Person & Motion Detection)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Exiting...")
        break

print("Cleaning up...")
cap.release()
cv2.destroyAllWindows()
print("Application closed successfully.")
