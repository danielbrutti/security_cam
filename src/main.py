import os
from dotenv import load_dotenv
import cv2
import time

# Set environment variables to handle Wayland and display issues
os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Force X11 instead of Wayland
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
    'rtsp_transport;tcp'  # Use TCP for RTSP
)

# Load environment variables from .env file
load_dotenv()
USER = os.getenv('CAM_USER')
PASS = os.getenv('CAM_PASS')
IP = os.getenv('CAM_IP')

# Validate that all required environment variables are set
if not USER or not PASS or not IP:
    raise ValueError(
        "Missing environment variables: CAM_USER, CAM_PASS or CAM_IP"
    )

# Initialize HOG (Histogram of Oriented Gradients) person detector
# This is a pre-trained SVM classifier for detecting people in images
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
print("Person detection is active - green boxes show detected people.")
print("CPU optimized: detection every 3 frames, reduced resolution")


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
        if (x < margin or y < margin or 
                x + w > 1920 - margin or y + h > 1080 - margin):
            continue
            
        # If all filters pass, add to filtered detections
        filtered_detections.append((box, weight))
    
    return filtered_detections


# Performance optimization variables
frame_count = 0  # Track total frames processed
consecutive_errors = 0  # Track consecutive frame read errors
detection_interval = 3  # Run detection every 3 frames to reduce CPU usage
last_detections = []  # Store last detection results to avoid recalculation
fps_counter = 0  # Count frames for FPS calculation
fps_start_time = time.time()  # Start time for FPS calculation
current_fps = 0  # Current FPS value

print("Starting video processing loop...")

while True:
    # Read frame from RTSP stream
    ret, frame = cap.read()
    frame_count += 1
    fps_counter += 1
    
    # Handle frame read errors
    if not ret:
        consecutive_errors += 1
        print(f"Could not read frame {frame_count}. "
              f"Consecutive errors: {consecutive_errors}")
        
        # If too many consecutive errors, try to reconnect to the stream
        if consecutive_errors > 10:
            print("Too many consecutive errors. Attempting to reconnect...")
            cap.release()
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 15)
            consecutive_errors = 0
            frame_count = 0
            continue
        
        # Wait a bit before trying again to avoid busy waiting
        cv2.waitKey(100)
        continue
    
    # Reset error counter on successful frame read
    consecutive_errors = 0
    
    # Calculate FPS every second for performance monitoring
    if time.time() - fps_start_time >= 1.0:
        current_fps = fps_counter / (time.time() - fps_start_time)
        fps_counter = 0
        fps_start_time = time.time()
    
    # Run detection only every N frames to reduce CPU usage
    # This is a key optimization that significantly reduces processing load
    if frame_count % detection_interval == 0:
        # Resize frame for faster detection (reduce by 50%)
        # Smaller images process much faster while maintaining detection accuracy
        height, width = frame.shape[:2]
        detection_width = width // 2
        detection_height = height // 2
        detection_frame = cv2.resize(
            frame, (detection_width, detection_height)
        )
        
        # Detect people in the resized frame using HOG detector
        # The detector looks for human-like patterns in the image
        boxes, weights = hog.detectMultiScale(
            detection_frame, 
            winStride=(8, 8),  # Step size for detection window
            padding=(4, 4),    # Padding around detection windows
            scale=1.05,        # Scale factor for multi-scale detection
            hitThreshold=0     # Detection threshold
        )
        
        # Scale detection boxes back to original frame size
        # Since we detected on a smaller image, we need to scale coordinates
        scaled_boxes = []
        for (x, y, w, h) in boxes:
            scaled_x = int(x * 2)  # Scale back to original width
            scaled_y = int(y * 2)  # Scale back to original height
            scaled_w = int(w * 2)  # Scale back to original width
            scaled_h = int(h * 2)  # Scale back to original height
            scaled_boxes.append((scaled_x, scaled_y, scaled_w, scaled_h))
        
        # Apply filtering to reduce false positives
        last_detections = filter_detections(scaled_boxes, weights)
    
    # Draw bounding boxes around detected people (use cached results)
    # This avoids redrawing the same detections every frame
    for (x, y, w, h), confidence in last_detections:
        # Draw green rectangle around detected person
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Add confidence score label above the box
        cv2.putText(
            frame, f'Person ({confidence:.2f})', (x, y - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
    
    # Add detection status to frame (top-left corner)
    detection_text = f"People detected: {len(last_detections)}"
    cv2.putText(
        frame, detection_text, (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )
    
    # Add performance monitoring info (FPS and frame count)
    fps_part = f"FPS: {current_fps:.1f}"
    frame_part = f"Frame: {frame_count}"
    performance_text = f"{fps_part} | {frame_part}"
    cv2.putText(
        frame, performance_text, (10, 60), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    
    # Add optimization info at bottom of frame
    opt_text = "CPU Optimized: Detection every 3 frames, 50% resolution"
    cv2.putText(
        frame, opt_text, (10, frame.shape[0] - 20), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
    )
    
    # Add filtering info
    filter_text = "Strict filtering: confidence>0.4, height>120px, aspect<2.5"
    cv2.putText(
        frame, filter_text, (10, frame.shape[0] - 40), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
    )
    
    # Display the processed frame
    cv2.imshow('Live View - IP Camera (Person Detection)', frame)
    
    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Clean up resources
print("Cleaning up...")
cap.release()
cv2.destroyAllWindows()
print("Application closed successfully.") 