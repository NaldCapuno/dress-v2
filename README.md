# Person Detection Flask App with YOLOv8n

A modern web application for detecting persons in images using YOLOv8n (You Only Look Once version 8 nano) model.

## Features

- **Real-time Person Detection**: Upload images and detect persons with bounding boxes
- **Modern Web Interface**: Beautiful, responsive UI with drag-and-drop functionality
- **High Accuracy**: Uses YOLOv8n model with 50% confidence threshold
- **Multiple Image Formats**: Supports JPG, PNG, GIF, BMP formats
- **Visual Results**: Shows detection results with confidence scores and bounding boxes

## Installation

1. **Clone or download this project**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Open your browser** and go to `http://localhost:5000`

## Usage

1. **Upload an Image**: 
   - Drag and drop an image onto the upload area, or
   - Click "Choose File" to browse and select an image

2. **View Results**: 
   - The app will process the image and show:
     - Number of persons detected
     - Confidence scores for each detection
     - Image with bounding boxes around detected persons

3. **Upload Another**: Click "Upload Another Image" to process more images

## Technical Details

- **Model**: YOLOv8n (nano version for fast inference)
- **Detection Class**: Person (class ID 0 in COCO dataset)
- **Confidence Threshold**: 50%
- **Framework**: Flask web framework
- **Computer Vision**: OpenCV for image processing
- **Deep Learning**: PyTorch backend via Ultralytics

## File Structure

```
dress-v2/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── LICENSE                # License file
├── README.md              # This file
├── src/                   # Source code modules
│   ├── __init__.py
│   ├── botsort_tracker.py # Bot-SORT tracking implementation
│   ├── config.py          # Database configuration and helpers
│   └── rfid_scanner.py    # RFID scanner module
├── scripts/               # Utility scripts
│   └── create_admin.py    # Admin account creation script
├── models/                # ML model files
│   ├── best.pt            # Dress code detection model
│   └── yolov8n.pt         # Person detection model
├── database/              # Database schema files
│   ├── dress_clean.sql    # Clean database schema
│   └── dummy_data.sql     # Sample data
├── templates/             # Flask HTML templates
│   ├── index.html
│   ├── login.html
│   ├── dean_dashboard.html
│   ├── guidance_dashboard.html
│   └── osas_dashboard.html
├── static/                # Static assets (CSS, JS, images)
│   ├── style.css
│   └── images/
├── uploads/               # Temporary upload directory (auto-created)
└── results/               # Processed images with detections (auto-created)
```

## API Endpoints

- `GET /` - Main web interface
- `POST /upload` - Upload image for person detection
- `POST /detect` - Detect persons from image URL (JSON API)
- `GET /results/<filename>` - Serve processed result images

## Requirements

- Python 3.8+
- Modern web browser
- Internet connection (for downloading YOLOv8n model on first run)

## Troubleshooting

1. **Model Download**: On first run, the app will download the YOLOv8n model (~6MB)
2. **Memory**: Ensure sufficient RAM for image processing
3. **File Permissions**: Make sure the app has write permissions for uploads/ and results/ folders

## Customization

- **Confidence Threshold**: Modify the threshold in `app.py` (currently 0.5)
- **Detection Classes**: Change class ID to detect other objects (see COCO dataset classes)
- **UI Styling**: Edit CSS in `templates/index.html`

## License

This project is open source and available under the MIT License.
