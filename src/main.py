"""
TensorFlow Object Detection Script
Use the default model and first available camera:
  $ python main.py
Use a USB camera with a custom model:
  $ python main.py --model-id "custom_model_id" --usbcam
"""

import cv2
import argparse
import tensorflow as tf
import numpy as np
from picamera2 import Picamera2


# Constants for camera resolution
IM_WIDTH = 1280
IM_HEIGHT = 720


def check_available_cameras():
    """Check which camera indices are available."""
    available_cameras = []
    for i in range(10):  # Try first 10 indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
        cap.release()
    return available_cameras


def load_model(model_id):
    """
    Load the TensorFlow SavedModel for object detection.
    
    Args: 
        model_id (str): Identifier for the model directory.
    
    Returns:
        model: Loaded TensorFlow model.
    """
    try:
        # Assumes the model is stored in the directory "models/{model_id}"
        model = tf.saved_model.load(f"models/{model_id}")
        print(f"Model '{model_id}' loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model '{model_id}': {e}")
        exit()


def process_frame(model, frame):
    """
    Process a single frame for object detection using TensorFlow.
    
    Args:
        model: TensorFlow detection model.
        frame (numpy.ndarray): BGR image frame.
    """
    # Convert BGR to RGB (TensorFlow models usually expect RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert image to float32 and normalize to [0,1]
    input_tensor = tf.convert_to_tensor(frame_rgb, dtype=tf.float32)
    input_tensor = input_tensor[tf.newaxis, ...] / 255.0  # add batch dimension

    # Run inference
    detections = model(input_tensor)

    # Process the outputs:
    # Many TensorFlow detection models return a dict with keys like:
    # 'detection_boxes', 'detection_scores', 'detection_classes', 'num_detections'
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() 
                  for key, value in detections.items()}
    detection_boxes = detections['detection_boxes']
    detection_scores = detections['detection_scores']
    detection_classes = detections['detection_classes'].astype(np.int32)

    height, width, _ = frame.shape

    # Annotate detections on the original frame (BGR)
    for i in range(num_detections):
        score = detection_scores[i]
        if score < 0.5:
            continue  # skip detections with low confidence
        
        # Box coordinates are normalized [ymin, xmin, ymax, xmax]
        ymin, xmin, ymax, xmax = detection_boxes[i]
        left, top = int(xmin * width), int(ymin * height)
        right, bottom = int(xmax * width), int(ymax * height)
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        label = f"ID:{detection_classes[i]} {score:.2f}"
        cv2.putText(frame, label, (left, top - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("TensorFlow Object Detection", frame)
    

def run_inference(model, camera_idx, use_usb):
    """
    Run inference on the selected camera.
    
    Args:
        model: TensorFlow detection model.
        camera_idx (int): Camera index to use.
        use_usb (bool): Whether to use a USB camera.
    """
    if use_usb:
        # Initialize USB webcam feed
        cap = cv2.VideoCapture(camera_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, IM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IM_HEIGHT)
    else:
        # Initialize Picamera2
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (IM_WIDTH, IM_HEIGHT)})
        picam2.configure(config)
        picam2.start()

    print(f"Successfully connected to {'USB camera' if use_usb else 'Picamera2'}. Press 'q' to exit.")

    try:
        if use_usb:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame from camera.")
                    break
                process_frame(model, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting...")
                    break
        else:
            while True:
                frame = picam2.capture_array()
                process_frame(model, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting...")
                    break
    finally:
        if use_usb:
            cap.release()
        else:
            picam2.stop()
        cv2.destroyAllWindows()


def main():
    """Main function to parse arguments and run the program."""
    parser = argparse.ArgumentParser(description="TensorFlow Object Detection Script")
    parser.add_argument(
        "--model-id", 
        type=str, 
        default="tf_model", 
        help="Model ID to load (e.g. 'tf_model', 'custom_tf_model')"
    )
    parser.add_argument(
        "--camera-idx", 
        type=int, 
        default=None, 
        help="Camera index to use (default: first available camera)"
    )
    parser.add_argument(
        "--usbcam", 
        action="store_true", 
        help="Use a USB webcam instead of Picamera2"
    )
    args = parser.parse_args()

    # Debug override (if needed)
    # args.usbcam = True

    # Check available cameras if USB camera is requested
    available_cameras = check_available_cameras()
    if args.usbcam and not available_cameras:
        print("No USB cameras found. Please check your camera connections.")
        exit()

    # Use the specified camera index or the first available one (for USB)
    camera_idx = args.camera_idx if args.camera_idx is not None else (available_cameras[0] if args.usbcam else None)
    if args.usbcam and camera_idx not in available_cameras:
        print(f"Camera index {camera_idx} is not available. Available cameras: {available_cameras}")
        exit()

    # Load the TensorFlow model and run inference
    model = load_model(args.model_id)
    run_inference(model, camera_idx, args.usbcam)

if __name__ == "__main__":
    main()
