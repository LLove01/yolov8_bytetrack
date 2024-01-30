import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

# Load a pre-trained or custom-trained YOLO model
model = YOLO('yolov8n.pt')  # Adjust the model path as needed

# Initialize ByteTrack
byte_tracker = sv.ByteTrack()


def callback(frame: np.ndarray, index: int) -> np.ndarray:
    # Run object detection
    results = model(frame)[0]

    # Convert detections to supervision format
    detections = sv.Detections.from_ultralytics(results)

    # Filter detections by confidence
    CONFIDENCE_THRESHOLD = 0.5  # Set your confidence threshold
    detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]

    # Update ByteTrack with filtered detections
    detections = byte_tracker.update_with_detections(detections)

    # Initialize an annotator for visualizing the detections
    annotator = sv.BoxAnnotator()

    labels = []
    for det in detections:
        # Unpack the tuple
        bbox, mask, confidence, class_id, tracker_id, _ = det

        # Use these values to construct the label
        label = f"#{tracker_id} {model.names[class_id]} {confidence:0.2f}"
        labels.append(label)

    annotated_frame = annotator.annotate(
        scene=frame.copy(), detections=detections, labels=labels)

    return annotated_frame


# Define paths for your input and output videos
VIDEO_PATH = 'test.mp4'
TARGET_PATH = 'result.mp4'

# Process the video
sv.process_video(source_path=VIDEO_PATH,
                 target_path=TARGET_PATH, callback=callback)
