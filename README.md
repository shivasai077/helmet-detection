# Helmet Detection System

A real-time computer vision system that detects safety helmets using YOLOv3 deep learning model and OpenCV. This project combines helmet detection with face detection capabilities for enhanced safety monitoring in industrial and construction environments.

## 🎯 Features

- **Real-time Helmet Detection**: Uses YOLOv3 neural network for accurate helmet detection
- **Face Detection**: Integrated Haar Cascade classifier for face detection
- **Live Camera Feed**: Processes video stream from webcam in real-time
- **Visual Feedback**: Draws bounding boxes around detected helmets and faces
- **Audio Alerts**: Optional sound notifications for safety violations
- **Confidence Scoring**: Displays detection confidence levels
- **Performance Metrics**: Shows inference time for each frame

## 🏗️ Project Structure

```
helmet/
├── code/
│   ├── helmet_detection.py    # Main helmet detection script
│   ├── face.py               # Standalone face detection script
│   ├── utils.py              # Utility functions for drawing and post-processing
│   ├── obj.names             # Object class names for YOLO model
│   ├── yolov3-obj.cfg        # YOLOv3 network configuration
│   ├── yolov3-obj_2400.weights # Pre-trained model weights
│   ├── haarcascade_frontalface_default.xml # Haar cascade for face detection
│   └── pavan.jpg.jpg         # Sample test image
├── requirements.txt          # Python package dependencies
└── README.md                 # Project documentation
```

## 🔧 Requirements

- Python 3.7+
- OpenCV 4.x (with GUI support)
- NumPy
- YOLOv3 pre-trained weights
- Webcam or video input device

## 📦 Installation

1. **Clone or download the project**
   ```bash
   cd helmet/code
   ```

2. **Install required packages**
   
   Option A - Using requirements.txt (Recommended):
   ```bash
   pip install -r requirements.txt
   ```
   
   Option B - Manual installation:
   ```bash
   pip install opencv-python numpy
   ```

   **Important**: Make sure you have `opencv-python` (NOT `opencv-python-headless`) installed for GUI support.

3. **Verify OpenCV installation**
   ```bash
   python -c "import cv2; print('OpenCV version:', cv2.__version__)"
   ```

## 🚀 Usage

### Helmet Detection (Main Application)

Run the main helmet detection system:

```bash
python helmet_detection.py
```

**Controls:**
- Press `q` to quit the application
- The system will display:
  - Live camera feed with detection boxes
  - Face detection (yellow rectangles)
  - Helmet detection (white/colored rectangles)
  - Confidence scores and inference time

### Face Detection Only

To run standalone face detection:

```bash
python face.py
```

**Controls:**
- Press `ESC` key to exit
- Detected faces and eyes will be highlighted
- Face images are automatically saved as `face.jpg`

## ⚙️ Configuration

### Detection Parameters

You can modify these parameters in `helmet_detection.py`:

```python
confThreshold = 0.5    # Confidence threshold (0.0 - 1.0)
nmsThreshold = 0.4     # Non-maximum suppression threshold
inpWidth = 416         # Input image width
inpHeight = 416        # Input image height
```

### Camera Settings

To use a different camera or video file:

```python
# For webcam (change index for different cameras)
cap = cv.VideoCapture(0)  # 0 for default camera, 1 for second camera, etc.

# For video file
# cap = cv.VideoCapture('path/to/your/video.mp4')
```

## 🎛️ Model Information

- **Architecture**: YOLOv3 (You Only Look Once v3)
- **Input Size**: 416x416 pixels
- **Classes**: Helmet detection (single class)
- **Framework**: OpenCV DNN module
- **Backend**: OpenCV (CPU-optimized)

## 🔧 Troubleshooting

### Common Issues

1. **OpenCV GUI Error**: `The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support`
   
   **Solution**: 
   ```bash
   pip uninstall opencv-python-headless
   pip install --upgrade opencv-python
   ```

2. **Camera Not Found**: `Cannot open camera`
   
   **Solution**: 
   - Check camera permissions
   - Try different camera indices (0, 1, 2...)
   - Ensure no other applications are using the camera

3. **Model Files Missing**: 
   
   **Solution**: Ensure these files are in the `code/` directory:
   - `yolov3-obj_2400.weights`
   - `yolov3-obj.cfg`
   - `haarcascade_frontalface_default.xml`

4. **Poor Detection Performance**:
   
   **Solution**:
   - Adjust `confThreshold` (lower for more detections)
   - Ensure good lighting conditions
   - Position camera at appropriate distance

### Performance Optimization

- **CPU Usage**: The system runs on CPU by default. For better performance:
  ```python
  net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
  net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
  ```
  (Requires OpenCV compiled with CUDA support)

## 📊 Detection Logic

The system implements a multi-stage detection process:

1. **Frame Capture**: Captures video frames from camera
2. **Face Detection**: Uses Haar Cascade for face detection
3. **Helmet Detection**: Uses YOLOv3 for helmet detection
4. **Post-processing**: Applies Non-Maximum Suppression (NMS)
5. **Visualization**: Draws bounding boxes and labels
6. **Alert System**: Triggers alerts based on detection results

## 🔮 Future Enhancements

- [ ] Support for multiple helmet types and colors
- [ ] Integration with database for logging violations
- [ ] Web interface for remote monitoring
- [ ] Mobile app integration
- [ ] Advanced analytics and reporting
- [ ] Support for video file batch processing
- [ ] GPU acceleration optimization
- [ ] Person tracking across frames

## 📝 Notes

- The system is designed for safety monitoring in industrial environments
- Detection accuracy depends on lighting conditions and camera quality
- For production use, consider fine-tuning the model with domain-specific data
- Audio alerts can be uncommented in the code for sound notifications

## 🤝 Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features  
- Improving documentation
- Optimizing performance

## 📄 License

This project is for educational and safety monitoring purposes. Please ensure compliance with local privacy and surveillance regulations when deploying in real environments. 