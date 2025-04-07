import cv2
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from picamera2 import Picamera2

# Constants for camera resolution
IM_WIDTH = 1280
IM_HEIGHT = 720

# Optional: Partial COCO labels mapping for better display (expand as needed)
COCO_LABELS = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    # Add additional mappings as needed
}

def check_available_cameras():
    available_cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
        cap.release()
    return available_cameras

def load_online_model(model_url):
    """
    Load a TF2-compatible object detection model from TensorFlow Hub.
    For example, SSD MobileNet V2 trained on COCO.
    """
    detector = hub.load(model_url)
    print(f"Online model loaded from: {model_url}")
    return detector

def process_frame(detector, frame, use_usb=True):
    """
    Process a single frame using the online detector.
    Assumes the model expects a uint8 image of shape [1, H, W, 3] and returns:
      - detection_boxes: [1, num_detections, 4] in normalized [ymin, xmin, ymax, xmax]
      - detection_scores: [1, num_detections]
      - detection_classes: [1, num_detections]
      - num_detections: [1]
    """
    # Prepare input image
    if use_usb:
        # USB frames are in BGR; convert to RGB.
        input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        # Assume Picamera2 frames are in RGB.
        input_img = frame.copy()
    
    # If the image has an extra alpha channel, drop it.
    if input_img.shape[-1] == 4:
        input_img = input_img[..., :3]

    # Many TF2 detection models expect a batch dimension and uint8 input
    input_tensor = tf.convert_to_tensor(np.expand_dims(input_img, axis=0), dtype=tf.uint8)
    
    # Run inference
    results = detector(input_tensor)
    
    detection_boxes = results['detection_boxes'].numpy()[0]
    detection_scores = results['detection_scores'].numpy()[0]
    detection_classes = results['detection_classes'].numpy()[0].astype(np.int32)
    num_detections = int(results['num_detections'].numpy()[0])
    
    score_threshold = 0.5
    height, width, _ = frame.shape

    # For display, ensure we show a BGR image
    if use_usb:
        display_frame = frame.copy()  # already in BGR
    else:
        display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Process and draw detections
    for i in range(num_detections):
        if detection_scores[i] < score_threshold:
            continue
        ymin, xmin, ymax, xmax = detection_boxes[i]
        left = int(xmin * width)
        top = int(ymin * height)
        right = int(xmax * width)
        bottom = int(ymax * height)
        
        cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
        class_id = detection_classes[i]
        class_name = COCO_LABELS.get(class_id, "N/A")
        label = f"{class_name} ({detection_scores[i]:.2f})"
        cv2.putText(display_frame, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow("Online Model Object Detection", display_frame)

def run_inference(detector, camera_idx, use_usb):
    if use_usb:
        cap = cv2.VideoCapture(camera_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, IM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IM_HEIGHT)
    else:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (IM_WIDTH, IM_HEIGHT)})
        picam2.configure(config)
        picam2.start()

    print(f"Connected to {'USB camera' if use_usb else 'Picamera2'}. Press 'q' to exit.")

    try:
        while True:
            frame = cap.read()[1] if use_usb else picam2.capture_array()
            process_frame(detector, frame, use_usb)
            key = cv2.waitKey(10)  # increased delay for better key capture
            if key & 0xFF == ord('q'):
                print("Exiting...")
                break
    finally:
        if use_usb:
            cap.release()
        else:
            picam2.stop()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Online Model Object Detection Script using TensorFlow Hub")
    parser.add_argument("--model-url", type=str,
                        default="https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2",
                        help="TF Hub model URL")
    parser.add_argument("--camera-idx", type=int, default=None,
                        help="Camera index to use")
    parser.add_argument("--usbcam", action="store_true",
                        help="Use USB webcam instead of Picamera2")
    args = parser.parse_args()

    available_cameras = check_available_cameras()
    if args.usbcam and not available_cameras:
        print("No USB cameras found.")
        exit()

    camera_idx = args.camera_idx if args.camera_idx is not None else (available_cameras[0] if args.usbcam else None)
    if args.usbcam and camera_idx not in available_cameras:
        print(f"Camera index {camera_idx} is not available. Available: {available_cameras}")
        exit()

    detector = load_online_model(args.model_url)
    run_inference(detector, camera_idx, args.usbcam)

if __name__ == "__main__":
    main()
