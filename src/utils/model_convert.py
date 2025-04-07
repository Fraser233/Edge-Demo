# from ultralytics import YOLO

# # Load your trained YOLOv8 model
# model = YOLO('models/yolo11n.pt')

# # Export the model to ONNX format
# model.export(format='onnx')

# onnx-tf convert -i models/yolo11n.onnx -o models
import onnx
from onnx_tf.backend import prepare

# Load your ONNX model
onnx_model = onnx.load("models/yolo11n.onnx")

# Convert to TensorFlow
tf_rep = prepare(onnx_model)

# Export as a TensorFlow SavedModel
tf_rep.export_graph("models/yolo11n_savedmodel")

