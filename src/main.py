import os
from dotenv import load_dotenv
import cv2

# Load environment variables
load_dotenv()
USER = os.getenv('CAM_USER')
PASS = os.getenv('CAM_PASS')
IP = os.getenv('CAM_IP')

if not USER or not PASS or not IP:
    raise ValueError(
        "Missing environment variables: CAM_USER, CAM_PASS or CAM_IP"
    )

rtsp_url = f"rtsp://{USER}:{PASS}@{IP}/live/ch0"

cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    raise Exception(f"Could not open RTSP stream: {rtsp_url}")

print("Press 'q' to exit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not read frame from stream.")
        break
    cv2.imshow('Live View - IP Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 