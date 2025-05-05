import cv2
import numpy as np
import torch
from pathlib import Path

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_objects(frame):
    """
    Detect vehicles and traffic signs in the frame using YOLOv5.
    """
    results = model(frame)  # Perform detection
    return results.xyxy[0].numpy()  # Extract bounding boxes and labels

def classify_movement(prev_frame, curr_frame, detections):
    """
    Classify whether the detected objects are moving or stationary.
    """
    movement_status = []

    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    for *xyxy, _, _ in detections:
        x1, y1, x2, y2 = map(int, xyxy)
        # Extract region of interest (ROI) from current frame
        roi = curr_gray[y1:y2, x1:x2]
        # Extract ROI from previous frame
        prev_roi = prev_gray[y1:y2, x1:x2]
        # Compute absolute difference between current and previous ROI
        diff = cv2.absdiff(roi, prev_roi)
        # Threshold the difference
        _, diff_thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        # Calculate percentage of non-zero pixels
        nonzero_percent = np.count_nonzero(diff_thresh) / (roi.shape[0] * roi.shape[1])
        # If percentage of non-zero pixels is above a threshold, consider it moving
        if nonzero_percent > 0.3:
            movement_status.append("Moving")
        else:
            movement_status.append("Stationary")

    return movement_status

def draw_detections(frame, detections):
    """
    Draw bounding boxes around detected objects.
    """
    for *xyxy, _, _ in detections:
        x1, y1, x2, y2 = map(int, xyxy)
        color = (0, 255, 0)  # Green color for all objects
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

def main(video_path, output_path):
    # Initialize the video capture object
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Read the first frame
    ret, prev_frame = cap.read()

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        # Perform object detection
        detections = detect_objects(curr_frame)

        # Classify movement of detected objects
        movement_status = classify_movement(prev_frame, curr_frame, detections)

        # Draw detections
        draw_detections(curr_frame, detections)

        # Display movement status
        for status, (x1, y1, _, _, _, _) in zip(movement_status, detections):
            color = (0, 255, 255) if status == "Moving" else (0, 0, 255)
            cv2.putText(curr_frame, str(status), (int(x1), int(y1) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write the frame to the output video file
        out.write(curr_frame)

        # Update previous frame
        prev_frame = curr_frame.copy()

    # Release resources
    cap.release()
    out.release()  # Release video writer

# Run the main function on the provided video
video_path = 'S:\Files\Work\College\Year3\Sem-6\Sem-Project\DataSet\TestVideo-4.mp4'
output_path = 'S:\Files\Work\College\Year3\Sem-6\Sem-Project\Outputs\Detection_Distance_OutputVideo-4.mp4'
main(video_path, output_path)