import mediapipe as mp
import numpy as np
from PIL import Image
import io

# Initialize Mediapipe face detection and drawing utilities
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


def process_image(image_data):
    """
    Process and encode the image for face recognition using Mediapipe.
    Returns the bounding box data as encoding.
    """
    try:
        # Convert the raw image data to a PIL image
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)

        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            # Process the image to detect faces
            results = face_detection.process(image_np)

            if results.detections:
                # For simplicity, use the first detected face's bounding box as encoding
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                encoding = [bbox.xmin, bbox.ymin, bbox.width, bbox.height]
                return encoding

        return None  # No face detected
    except Exception as e:
        print(f"Error processing the image: {str(e)}")
        return None
