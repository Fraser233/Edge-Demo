import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

# Specify your image path here
image_path = "data.png"  # Replace with your actual image file

# Load and preprocess the image
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)
# Many TF2 detection models expect a batch of images
input_tensor = tf.expand_dims(image_np, axis=0)

# Load a TF2-compatible object detection model from TensorFlow Hub
# This example uses the SSD MobileNet V2 model trained on the COCO dataset.
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

# Run inference
results = detector(input_tensor)

# The returned results is a dict with the following keys:
#   'detection_boxes', 'detection_scores', 'detection_classes', 'num_detections'
detection_boxes = results['detection_boxes'].numpy()[0]
detection_scores = results['detection_scores'].numpy()[0]
detection_classes = results['detection_classes'].numpy()[0].astype(np.int32)
num_detections = int(results['num_detections'].numpy()[0])

# Optional: COCO labels mapping (partial list for illustration)
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

print("Number of detections:", num_detections)
for i in range(num_detections):
    score = detection_scores[i]
    # Filter out detections with low confidence
    if score < 0.5:
        continue
    box = detection_boxes[i]
    class_id = detection_classes[i]
    class_name = COCO_LABELS.get(class_id, "N/A")
    print(f"Detection {i}: Class ID {class_id} ({class_name}), Score: {score:.2f}, Box: {box}")
