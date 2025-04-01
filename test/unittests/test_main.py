"""
Run the tests using the following command from the project root:
    $ python -m unittest discover -s test/unittests -p "test_*.py"

If all tests pass, you should see:

.....
----------------------------------------------------------------------
Ran 5 tests in 0.XXXs

OK
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import cv2
import numpy as np

# Import the main module from src directory
sys.path.insert(0, 'src')
import main


class TestMain(unittest.TestCase):

    @patch('main.cv2.VideoCapture')
    def test_check_available_cameras(self, mock_video_capture):
        """Test the check_available_cameras function."""
        # Mock behavior of cv2.VideoCapture
        mock_video_capture.return_value.isOpened.side_effect = [True, False, True, False]

        # Call the function
        available_cameras = main.check_available_cameras()

        # Assert the result
        self.assertEqual(available_cameras, [0, 2])


    @patch('main.YOLO')
    def test_load_model(self, mock_yolo):
        """Test the load_model function."""
        # Mock the YOLO model loading
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        # Call the function
        model = main.load_model("test_model_id")

        # Assert the result
        mock_yolo.assert_called_once_with("models/test_model_id.pt")
        self.assertEqual(model, mock_model)


    @patch('main.cv2.VideoCapture')
    @patch('main.process_frame')
    def test_run_inference_usb(self, mock_process_frame, mock_video_capture):
        """Test the run_inference function with a USB camera."""
        # Mock the USB camera behavior
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.read.side_effect = [(True, np.zeros((480, 640, 3), dtype=np.uint8)), (False, None)]

        # Mock the YOLO model
        mock_model = MagicMock()

        # Call the function
        main.run_inference(mock_model, camera_idx=0, use_usb=True)

        # Assert the behavior
        mock_video_capture.assert_called_once_with(0)
        mock_cap.read.assert_called()
        mock_process_frame.assert_called_once_with(mock_model, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cap.release.assert_called_once()


    @patch('main.process_frame')
    def test_run_inference_picamera2(self, mock_process_frame):
        """Test the run_inference function with Picamera2."""
        # Mock the Picamera2 behavior
        mock_picam2 = MagicMock()
        with patch('main.Picamera2', return_value=mock_picam2):
            mock_picam2.capture_array.side_effect = [np.zeros((480, 640, 3), dtype=np.uint8), KeyboardInterrupt]

            # Mock the YOLO model
            mock_model = MagicMock()

            # Call the function
            main.run_inference(mock_model, camera_idx=None, use_usb=False)

            # Assert the behavior
            mock_picam2.start.assert_called_once()
            mock_picam2.capture_array.assert_called()
            mock_process_frame.assert_called_once_with(mock_model, np.zeros((480, 640, 3), dtype=np.uint8))
            mock_picam2.stop.assert_called_once()


    @patch('main.check_available_cameras')
    @patch('main.load_model')
    @patch('main.run_inference')
    @patch('sys.argv', ['main.py', '--model-id', 'test_model', '--usbcam'])
    def test_main(self, mock_run_inference, mock_load_model, mock_check_available_cameras):
        """Test the main function."""
        # Mock available cameras
        mock_check_available_cameras.return_value = [0]

        # Mock the YOLO model
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        # Call the main function
        main.main()

        # Assert the behavior
        mock_check_available_cameras.assert_called_once()
        mock_load_model.assert_called_once_with("test_model")
        mock_run_inference.assert_called_once_with(mock_model, 0, True)


if __name__ == "__main__":
    unittest.main()
    