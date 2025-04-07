import cv2
import argparse
import numpy as np
from picamera2 import Picamera2
import tensorflow as tf

# Constants for camera resolution
IM_WIDTH = 1280
IM_HEIGHT = 720

def check_available_cameras():
    available_cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
        cap.release()
    return available_cameras

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    print(f"TFLite model loaded from: {model_path}")
    return interpreter

def process_frame(interpreter, frame):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    input_tensor = resized.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])

    if predictions.ndim == 3:
        predictions = predictions[0]

    score_threshold = 0.5
    height, width, _ = frame.shape

    for detection in predictions:
        x_min, y_min, x_max, y_max, score, class_id = detection
        if score < score_threshold:
            continue

        left = int(x_min * width)
        top = int(y_min * height)
        right = int(x_max * width)
        bottom = int(y_max * height)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        label = f"ID:{int(class_id)} {score:.2f}"
        cv2.putText(frame, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("TFLite YOLO Object Detection", frame)

def run_inference(interpreter, camera_idx, use_usb):
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
            process_frame(interpreter, frame)
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
    parser = argparse.ArgumentParser(description="TFLite YOLO Object Detection Script")
    parser.add_argument("--model-path", type=str,
                        default="models/yolo11n_tf/yolo11n_float32.tflite",
                        help="Path to .tflite model file")
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

    interpreter = load_tflite_model(args.model_path)
    run_inference(interpreter, camera_idx, args.usbcam)

if __name__ == "__main__":
    main()
