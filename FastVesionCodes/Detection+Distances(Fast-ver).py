import cv2
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor

# Load YOLOv5 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

def detect_objects(frame):
    """
    Detect vehicles and traffic signs in the frame using YOLOv5.
    """
    results = model(frame)  # Perform detection
    return results.xyxy[0].cpu().numpy()  # Extract bounding boxes and labels

def classify_movement(prev_frame, curr_frame, detections):
    """
    Classify whether the detected objects are moving or stationary.
    """
    movement_status = []

    for *xyxy, _, _ in detections:
        x1, y1, x2, y2 = map(int, xyxy)
        roi_curr = curr_frame[y1:y2, x1:x2]
        roi_prev = prev_frame[y1:y2, x1:x2]
        diff = cv2.absdiff(roi_curr, roi_prev)
        _, diff_thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        nonzero_percent = np.count_nonzero(diff_thresh) / (roi_curr.shape[0] * roi_curr.shape[1])
        movement_status.append("Moving" if nonzero_percent > 0.1 else "Stationary")

    return movement_status

def draw_detections(frame, detections):
    """
    Draw bounding boxes around detected objects.
    """
    for *xyxy, _, _ in detections:
        x1, y1, x2, y2 = map(int, xyxy)
        color = (0, 255, 0)  # Green color for all objects
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

def process_frame(frame, prev_frame):
    detections = detect_objects(frame)
    movement_status = classify_movement(prev_frame, frame, detections)
    draw_detections(frame, detections)
    for status, (x1, y1, _, _, _, _) in zip(movement_status, detections):
        color = (0, 255, 255) if status == "Moving" else (0, 0, 255)
        cv2.putText(frame, str(status), (int(x1), int(y1) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def main(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    ret, prev_frame = cap.read()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break
            futures.append(executor.submit(process_frame, curr_frame.copy(), prev_frame))
            prev_frame = curr_frame

        for future in futures:
            frame = future.result()
            out.write(frame)

    cap.release()
    out.release()

# Run the main function on the provided video
video_path = 'S:\Files\Work\College\Year3\Sem-6\Sem-Project\DataSet\TestVideo-4.mp4'
output_path = 'S:\Files\Work\College\Year3\Sem-6\Sem-Project\FastOutputs\Detection_Distance_OutputVideo-4.mp4'
main(video_path, output_path)
