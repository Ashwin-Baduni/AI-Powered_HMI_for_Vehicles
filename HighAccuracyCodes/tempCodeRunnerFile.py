def draw_detections(frame, detections, movement_status, centers, speeds):
    """
    Draw bounding boxes around detected objects and display movement status, distances, and speeds.
    """
    for (x1, y1, x2, y2, _, _), status, speed in zip(detections, movement_status, speeds):
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

        # Display distances between objects (optional)
        num_objects = len(centers)
        for i in range(num_objects):
            for j in range(i + 1, num_objects):
                # Calculate distance between centers i and j
                distance = calculate_distance(centers[i], centers[j])
                # Display distance on the frame
                cv2.putText(frame, f"Dist: {distance:.2f}", (int(centers[i][0]), int(centers[i][1]) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)