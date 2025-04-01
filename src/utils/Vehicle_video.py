from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes
import cv2

# Use the exact video file path, you should modify the file path on your own device
video_path = "data/videos/Vehicles_Highway.mp4"
print(f"Using video file: {video_path}")

# Custom callback function to display results
def display_detections(predictions, frame):
    # Process and display the frame with detections
    frame_with_boxes = render_boxes(predictions, frame)
    
    # Display the frame with detections
    cv2.imshow("YOLOv8 Object Detection", frame_with_boxes)
    
    # Slow down playback slightly to make detections more visible
    # and check for 'q' key to quit
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        return False  # Signal to stop processing
    
    return True  # Continue processing

try:
    # Verify the video file exists
    import os
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        exit()
        
    # Initialize the pipeline with the specific video file
    pipeline = InferencePipeline.init(
        model_id="yolov8n-640",
        video_reference=video_path,
        on_prediction=display_detections
    )
    print(f"Successfully initialized pipeline with video: {video_path}")
    print("Processing video and displaying results...")
    print("Press 'q' to quit the video display")
    
    # Start the pipeline
    pipeline.start()
    
    # Join the pipeline (wait for it to finish processing)
    pipeline.join()
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()
    print("Video processing completed")
    
except Exception as e:
    print(f"An error occurred: {e}")
    cv2.destroyAllWindows()