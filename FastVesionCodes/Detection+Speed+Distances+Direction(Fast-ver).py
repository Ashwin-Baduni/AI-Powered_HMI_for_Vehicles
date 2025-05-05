import cv2
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_objects(frame):
    """
    Detect vehicles and traffic signs in the frame using YOLOv5.
    """
    results = model(frame)  # Perform detection
    detections = results.xyxy[0].numpy()  # Extract bounding boxes and labels

    # Calculate centers of each bounding box
    centers = []
    for *xyxy, _, _ in detections:
        x1, y1, x2, y2 = map(int, xyxy)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        centers.append((center_x, center_y))

    return detections, centers

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
        if nonzero_percent > 0.1:
            movement_status.append("Moving")
        else:
            movement_status.append("Stationary")

    return movement_status

def calculate_distance(center1, center2):
    """
    Calculate Euclidean distance between two centers.
    """
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

def calculate_speed(distance, fps):
    """
    Calculate speed (pixels per second) given distance and frame rate.
    """
    return distance * fps

def estimate_direction(centers):
    """
    Estimate the direction of movement based on centroids of tracked objects.
    """
    directions = []
    num_objects = len(centers)

    for i in range(num_objects):
        if i > 0:
            prev_center = centers[i - 1]
            curr_center = centers[i]
            # Calculate vector from previous to current center
            direction_vector = (curr_center[0] - prev_center[0], curr_center[1] - prev_center[1])
            # Calculate angle (in degrees) of direction vector
            angle = np.degrees(np.arctan2(direction_vector[1], direction_vector[0]))
            directions.append(angle)
        else:
            directions.append(0.0)  # Assuming no movement for the first frame

    return directions

def calculate_turning_radius(centers):
    """
    Calculate the turning radius based on centroids of tracked objects.
    """
    turning_radii = []

    num_objects = len(centers)

    for i in range(num_objects):
        if i > 1:
            prev_center = centers[i - 2]
            curr_center = centers[i]
            next_center = centers[i - 1]

            # Calculate vectors from previous, current, and next centers
            vector1 = (prev_center[0] - curr_center[0], prev_center[1] - curr_center[1])
            vector2 = (next_center[0] - curr_center[0], next_center[1] - curr_center[1])

            # Calculate angle between vector1 and vector2
            angle = np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))
            # Calculate turning radius
            turning_radius = np.linalg.norm(vector1) / (2 * np.sin(angle))
            turning_radii.append(turning_radius)
        else:
            turning_radii.append(0.0)  # Assume no turning radius for the first frame

    return turning_radii

def draw_detections(frame, detections, movement_status, centers, speeds, directions, turning_radii):
    """
    Draw bounding boxes around detected objects and display movement status, speeds, directions, and turning radii.
    """
    for (x1, y1, x2, y2, _, _), status, speed, direction, turning_radius in zip(detections, movement_status, speeds, directions, turning_radii):
        # Convert coordinates to integers
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        # Draw bounding box
        color = (0, 255, 0)  # Green color for all objects
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Display movement status
        text_color = (0, 255, 255) if status == "Moving" else (0, 0, 255)
        cv2.putText(frame, str(status), (int(x1), int(y1) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        # Display speeds
        cv2.putText(frame, f"Speed: {speed:.2f} px/sec", (int(x1), int(y1) + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Display directions
        cv2.putText(frame, f"Direction: {direction:.2f} deg", (int(x1), int(y1) + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # Display turning radii
        cv2.putText(frame, f"Turn Radius: {turning_radius:.2f} px", (int(x1), int(y1) + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

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

    # Initialize variables for tracking objects
    prev_detections, prev_centers = detect_objects(prev_frame)
    object_speeds = defaultdict(list)

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        # Perform object detection
        detections, centers = detect_objects(curr_frame)

        # Classify movement of detected objects
        movement_status = classify_movement(prev_frame, curr_frame, detections)

        # Calculate distances and speeds for each object
        for i, (center1, (x1, y1, x2, y2, _, _)) in enumerate(zip(prev_centers, prev_detections)):
            for j, (center2, _) in enumerate(zip(centers, detections)):
                if i == j:  # Same object, skip
                    continue
                distance = calculate_distance(center1, center2)
                if distance > 0:  # Calculate speed only if distance is positive
                    speed = calculate_speed(distance, fps)
                    object_speeds[i].append(speed)

        # Calculate average speed for each object
        speeds = []
        for i in range(len(detections)):
            avg_speed = np.mean(object_speeds[i]) if i in object_speeds else 0.0
            speeds.append(avg_speed)

        # Estimate direction and turning radii
        directions = estimate_direction(centers)
        turning_radii = calculate_turning_radius(centers)

        # Draw detections and display movement status, speeds, directions, and turning radii
        draw_detections(curr_frame, detections, movement_status, centers, speeds, directions, turning_radii)

        # Write the frame to the output video file
        out.write(curr_frame)

        # Update previous frame information
        prev_frame = curr_frame.copy()
        prev_detections = detections
        prev_centers = centers

    # Release resources
    cap.release()
    out.release()  # Release video writer

video_path = 'S:\Files\Work\College\Year3\Sem-6\Sem-Project\DataSet\TestVideo-1.mp4'
output_path = 'S:\Files\Work\College\Year3\Sem-6\Sem-Project\FastOutputs\Detection_Distance_Speed_Direction_OutputVideo-1.mp4'
main(video_path, output_path)