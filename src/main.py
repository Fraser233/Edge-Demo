"""
YOLOv11 Object Detection Script
Use the default model and first available camera:
  $ python main.py
Use a USB camera with a custom model:
  $ python main.py --model-id "custom_model_id" --usbcam
"""

import cv2
import argparse
from ultralytics import YOLO
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
        else:
            cap.release()

    return available_cameras


def load_model(model_id):
    """
    Load the YOLO model.
    
    Args: 
        model_id (str): Model ID to load.
    
    Returns:
        model (YOLO): Loaded YOLO model.
    """
    try:
        model = YOLO(f"models/{model_id}.pt")
        print(f"Model '{model_id}' loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model '{model_id}': {e}")
        exit()


def process_frame(model, frame):
    """
    Process a single frame for object detection.
    
    Args:
        model (YOLOv11): YOLOv11 model to use for inference.
        frame (numpy.ndarray): Frame to process.
    """
    # Perform prediction on the current frame
    results = model.predict(source=frame, stream=True, save=False, show=False)

    # Iterate over the streaming results
    for result in results:
        # Annotate the frame with predictions
        annotated_frame = result.plot()

        # Display the frame
        cv2.imshow("YOLOv11 Detection", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def run_inference(model, camera_idx, use_usb):
    """
    Run inference on the selected camera.
    
    Args:
        model (YOLOv11): YOLOv11 model to use for inference.
        camera_idx (int): Camera index to use.
        use_usb (bool): Whether to use a USB camera.
    """
    if use_usb:
        # Initialize USB webcam feed
        cap = cv2.VideoCapture(camera_idx)
        cap.set(3, IM_WIDTH)
        cap.set(4, IM_HEIGHT)
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
    parser = argparse.ArgumentParser(description="YOLOv11 Object Detection Script")
    parser.add_argument(
        "--model-id", 
        type=str, 
        default="yolo11n", 
        help="Model ID to load (options: 'yolo11n', 'yolo11s')"
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

    # Debug
    args.usbcam = True

    # Check available cameras
    available_cameras = check_available_cameras()

    if not available_cameras and args.usbcam:
        print("No USB cameras found. Please check your camera connections.")
        exit()

    # Use the specified camera index or the first available one
    camera_idx = args.camera_idx \
        if args.camera_idx is not None \
            else (available_cameras[0] if args.usbcam else None)
    
    if args.usbcam and camera_idx not in available_cameras:
        print(f"Camera index {camera_idx} is not available. Available cameras: {available_cameras}")
        exit()

    # Load the model and run inference
    model = load_model(args.model_id)
    run_inference(model, camera_idx, args.usbcam)


if __name__ == "__main__":
    main()
    