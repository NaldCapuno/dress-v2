from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import cv2
import numpy as np
from ultralytics import YOLO
import os
import base64
import threading
import time
from botsort_tracker import BotSORT

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load YOLOv8n model for person detection
person_model = YOLO('yolov8n.pt')

# Load best.pt model for dress code detection
dress_model = YOLO('best.pt')

# Initialize BotSort tracker
tracker = BotSORT()

# Global variables for webcam
camera = None
detection_enabled = False
current_frame = None
frame_lock = threading.Lock()
selected_camera_id = 0

# Auto-initialize camera
def initialize_camera():
    global camera, selected_camera_id
    try:
        camera = cv2.VideoCapture(selected_camera_id)  # Use selected camera
        if camera.isOpened():
            print(f"Camera {selected_camera_id} initialized successfully")
            return True
        else:
            print(f"Failed to initialize camera {selected_camera_id}")
            camera = None
            return False
    except Exception as e:
        print(f"Error initializing camera: {e}")
        camera = None
        return False

# Initialize camera on startup
initialize_camera()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def format_dress_class(class_name):
    """Format dress class names for better readability"""
    class_mapping = {
        'polo_shirt': 'Polo Shirt',
        'pants': 'Pants',
        'shoes': 'Shoes',
        'blouse': 'Blouse',
        'skirt': 'Skirt',
        'doll_shoes': 'Doll Shoes'
    }
    return class_mapping.get(class_name, class_name.replace('_', ' ').title())

def detect_dress_code(person_crop):
    """Detect dress code items for a person crop using best.pt model"""
    try:
        # Run dress code detection on person crop
        results = dress_model(person_crop)
        
        dress_items = []
        dress_detections = []
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    if confidence > 0.85:  # High threshold for dress detection
                        # Get class name from model
                        class_name = dress_model.names[class_id]
                        dress_detections.append({
                            'class': class_name,
                            'confidence': round(confidence, 2)
                        })
        
        # Group by class and get best confidence for each dress item
        class_confidences = {}
        for detection in dress_detections:
            class_name = detection['class']
            confidence = detection['confidence']
            if class_name not in class_confidences or confidence > class_confidences[class_name]:
                class_confidences[class_name] = confidence
        
        # Convert to list of dress items with formatted names
        for class_name, confidence in class_confidences.items():
            dress_items.append({
                'class': format_dress_class(class_name),
                'confidence': confidence
            })
        
        # Sort by confidence (highest first)
        dress_items.sort(key=lambda x: x['confidence'], reverse=True)
        
        return dress_items
            
    except Exception as e:
        print(f"Error in dress detection: {e}")
        return []

def detect_persons_with_dress(image_path):
    """Two-stage detection: first detect persons, then detect dress code"""
    try:
        # Read image for tracking
        image = cv2.imread(image_path)
        
        # Stage 1: Detect persons using YOLOv8n
        results = person_model(image_path)
        
        # Process person detections
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    # Get class ID and confidence
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Check if it's a person (class 0 in COCO dataset)
                    if class_id == 0 and confidence > 0.5:  # 0.5 confidence threshold
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': round(confidence, 2),
                            'class': 'person'
                        })
        
        # Stage 2: Detect dress code for each person
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            
            # Extract person crop with some padding
            padding = 10
            crop_x1 = max(0, x1 - padding)
            crop_y1 = max(0, y1 - padding)
            crop_x2 = min(image.shape[1], x2 + padding)
            crop_y2 = min(image.shape[0], y2 + padding)
            
            person_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Detect dress code for this person
            dress_items = detect_dress_code(person_crop)
            detection['dress_items'] = dress_items
            # Create a summary string for display
            if dress_items:
                dress_summary = ", ".join([f"{item['class']} ({item['confidence']:.2f})" for item in dress_items[:3]])  # Show top 3 items
                detection['dress_summary'] = dress_summary
            else:
                detection['dress_summary'] = "No dress items detected"
        
        # Update tracker with detections for static image
        if detections:
            # Convert detections to tracker format
            dets = []
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                dets.append([x1, y1, x2, y2, det['confidence']])
            
            dets = np.array(dets)
            
            # Update tracker
            tracked_objects = tracker.update(dets, image)
            
            # Add tracking IDs to detections
            for i, track in enumerate(tracked_objects):
                if i < len(detections):
                    detections[i]['track_id'] = int(track[4])
                    detections[i]['bbox'] = [int(track[0]), int(track[1]), int(track[2]), int(track[3])]
        
        return detections
    except Exception as e:
        print(f"Error in two-stage detection: {e}")
        return []

def draw_detections(image_path, detections, output_path):
    """Draw bounding boxes on the image with tracking IDs and dress code"""
    try:
        # Read image
        image = cv2.imread(image_path)
        
        # Draw bounding boxes
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            track_id = detection.get('track_id', 'N/A')
            dress_summary = detection.get('dress_summary', 'No dress items detected')
            
            # Choose color based on track ID for better visualization
            color = (0, 255, 0)  # Default green
            if track_id != 'N/A':
                # Generate different colors for different track IDs
                color_int = track_id * 50 % 255
                color = (color_int, 255, 255 - color_int)
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with tracking ID and dress items as a list
            label1 = f"ID:{track_id} Person: {confidence:.2f}"
            
            # Draw first label (person info)
            label_size1 = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size1[1] - 10), 
                         (x1 + label_size1[0], y1), color, -1)
            cv2.putText(image, label1, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Draw dress items as a list
            dress_items = detection.get('dress_items', [])
            if dress_items:
                current_y = y1 - label_size1[1] - 15
                cv2.putText(image, "Dress Items:", (x1, current_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                current_y -= 20
                
                for i, item in enumerate(dress_items[:3]):  # Show top 3 items
                    item_text = f"• {item['class']} ({item['confidence']:.2f})"
                    text_size = cv2.getTextSize(item_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    
                    # Draw background for each item
                    cv2.rectangle(image, (x1, current_y - text_size[1] - 5), 
                                 (x1 + text_size[0] + 5, current_y + 5), color, -1)
                    cv2.putText(image, item_text, (x1 + 2, current_y - 2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    current_y -= 25
            else:
                # No dress items detected
                no_items_text = "No dress items detected"
                text_size = cv2.getTextSize(no_items_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(image, (x1, y1 - label_size1[1] - 35), 
                             (x1 + text_size[0] + 5, y1 - label_size1[1] - 10), color, -1)
                cv2.putText(image, no_items_text, (x1 + 2, y1 - label_size1[1] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Save result
        cv2.imwrite(output_path, image)
        return True
    except Exception as e:
        print(f"Error drawing detections: {e}")
        return False

def detect_persons_frame_with_dress(frame):
    """Two-stage detection for video frames: persons + dress code"""
    try:
        # Stage 1: Detect persons using YOLOv8n
        results = person_model(frame)
        
        # Process person detections
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    # Get class ID and confidence
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Check if it's a person (class 0 in COCO dataset)
                    if class_id == 0 and confidence > 0.5:  # 0.5 confidence threshold
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': round(confidence, 2),
                            'class': 'person'
                        })
        
        # Stage 2: Detect dress code for each person
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            
            # Extract person crop with some padding
            padding = 10
            crop_x1 = max(0, x1 - padding)
            crop_y1 = max(0, y1 - padding)
            crop_x2 = min(frame.shape[1], x2 + padding)
            crop_y2 = min(frame.shape[0], y2 + padding)
            
            person_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Detect dress code for this person
            dress_items = detect_dress_code(person_crop)
            detection['dress_items'] = dress_items
            # Create a summary string for display
            if dress_items:
                dress_summary = ", ".join([f"{item['class']} ({item['confidence']:.2f})" for item in dress_items[:3]])  # Show top 3 items
                detection['dress_summary'] = dress_summary
            else:
                detection['dress_summary'] = "No dress items detected"
        
        # Update tracker with detections
        if detections:
            # Convert detections to tracker format
            dets = []
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                dets.append([x1, y1, x2, y2, det['confidence']])
            
            dets = np.array(dets)
            
            # Update tracker
            tracked_objects = tracker.update(dets, frame)
            
            # Add tracking IDs to detections
            for i, track in enumerate(tracked_objects):
                if i < len(detections):
                    detections[i]['track_id'] = int(track[4])
                    detections[i]['bbox'] = [int(track[0]), int(track[1]), int(track[2]), int(track[3])]
        
        return detections
    except Exception as e:
        print(f"Error in frame detection: {e}")
        return []

def draw_detections_frame(frame, detections):
    """Draw bounding boxes on a video frame with tracking IDs and dress code"""
    try:
        # Draw bounding boxes
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            track_id = detection.get('track_id', 'N/A')
            dress_summary = detection.get('dress_summary', 'No dress items detected')
            
            # Choose color based on track ID for better visualization
            color = (0, 255, 0)  # Default green
            if track_id != 'N/A':
                # Generate different colors for different track IDs
                color_int = track_id * 50 % 255
                color = (color_int, 255, 255 - color_int)
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with tracking ID and dress items as a list
            label1 = f"ID:{track_id} Person: {confidence:.2f}"
            
            # Draw first label (person info)
            label_size1 = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size1[1] - 10), 
                         (x1 + label_size1[0], y1), color, -1)
            cv2.putText(frame, label1, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Draw dress items as a list
            dress_items = detection.get('dress_items', [])
            if dress_items:
                current_y = y1 - label_size1[1] - 15
                cv2.putText(frame, "Dress Items:", (x1, current_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                current_y -= 20
                
                for i, item in enumerate(dress_items[:3]):  # Show top 3 items
                    item_text = f"• {item['class']} ({item['confidence']:.2f})"
                    text_size = cv2.getTextSize(item_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    
                    # Draw background for each item
                    cv2.rectangle(frame, (x1, current_y - text_size[1] - 5), 
                                 (x1 + text_size[0] + 5, current_y + 5), color, -1)
                    cv2.putText(frame, item_text, (x1 + 2, current_y - 2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    current_y -= 25
            else:
                # No dress items detected
                no_items_text = "No dress items detected"
                text_size = cv2.getTextSize(no_items_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame, (x1, y1 - label_size1[1] - 35), 
                             (x1 + text_size[0] + 5, y1 - label_size1[1] - 10), color, -1)
                cv2.putText(frame, no_items_text, (x1 + 2, y1 - label_size1[1] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame
    except Exception as e:
        print(f"Error drawing detections on frame: {e}")
        return frame

def generate_frames():
    """Generate video frames for streaming"""
    global camera, detection_enabled, current_frame, frame_lock
    
    while True:
        if camera is not None:
            success, frame = camera.read()
            if success:
                with frame_lock:
                    current_frame = frame.copy()
                
                # Perform detection if enabled
                if detection_enabled:
                    detections = detect_persons_frame_with_dress(frame)
                    frame = draw_detections_frame(frame, detections)
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                break
        else:
            # Send a black frame if no camera
            black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(black_frame, "Camera Turned Off", (150, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', black_frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.03)  # ~30 FPS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Detect persons with dress code
        detections = detect_persons_with_dress(file_path)
        
        # Generate result filename
        result_filename = f"result_{filename}"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        
        # Draw detections on image
        success = draw_detections(file_path, detections, result_path)
        
        if success:
            # Convert result image to base64 for display
            with open(result_path, 'rb') as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
            return jsonify({
                'success': True,
                'detections': detections,
                'count': len(detections),
                'image': img_base64,
                'filename': result_filename
            })
        else:
            return jsonify({'error': 'Failed to process image'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/detect', methods=['POST'])
def detect_from_url():
    """Detect persons from image URL"""
    try:
        data = request.get_json()
        image_url = data.get('image_url')
        
        if not image_url:
            return jsonify({'error': 'No image URL provided'}), 400
        
        # For this example, we'll assume the URL points to a local file
        # In production, you'd download the image from the URL
        if os.path.exists(image_url):
            detections = detect_persons_with_dress(image_url)
            return jsonify({
                'success': True,
                'detections': detections,
                'count': len(detections)
            })
        else:
            return jsonify({'error': 'Image file not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/results/<filename>')
def uploaded_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    """Start webcam (or return status if already running)"""
    global camera, selected_camera_id
    try:
        # Try to get camera_id from JSON, fallback to selected_camera_id
        camera_id = selected_camera_id
        try:
            if request.json:
                camera_id = request.json.get('camera_id', selected_camera_id)
        except:
            # If JSON parsing fails, use the global selected_camera_id
            pass
        
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(camera_id)
            if camera.isOpened():
                selected_camera_id = camera_id  # Update global selected camera ID
                return jsonify({'success': True, 'message': f'Camera {camera_id} started successfully'})
            else:
                camera = None
                return jsonify({'success': False, 'message': f'Failed to open camera {camera_id}'}), 500
        else:
            return jsonify({'success': True, 'message': 'Camera already running'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error starting camera: {str(e)}'}), 500

@app.route('/change_camera', methods=['POST'])
def change_camera():
    """Change to a different camera"""
    global camera, detection_enabled, selected_camera_id
    try:
        data = request.get_json()
        camera_id = data.get('camera_id', 0)
        
        # Stop current camera
        if camera is not None:
            camera.release()
            camera = None
        
        # Start new camera
        camera = cv2.VideoCapture(camera_id)
        if camera.isOpened():
            selected_camera_id = camera_id  # Update global selected camera ID
            detection_enabled = False  # Reset detection when changing camera
            return jsonify({'success': True, 'message': f'Switched to camera {camera_id}'})
        else:
            camera = None
            return jsonify({'success': False, 'message': f'Failed to open camera {camera_id}'}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error changing camera: {str(e)}'}), 500

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop webcam"""
    global camera, detection_enabled
    try:
        if camera is not None:
            camera.release()
            camera = None
            detection_enabled = False  # Also disable detection when camera stops
            return jsonify({'success': True, 'message': 'Camera stopped successfully'})
        else:
            return jsonify({'success': True, 'message': 'Camera was not running'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error stopping camera: {str(e)}'}), 500

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    """Toggle person detection on/off"""
    global detection_enabled
    try:
        detection_enabled = not detection_enabled
        status = "enabled" if detection_enabled else "disabled"
        return jsonify({'success': True, 'detection_enabled': detection_enabled, 'message': f'Detection {status}'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error toggling detection: {str(e)}'}), 500

@app.route('/capture_frame', methods=['POST'])
def capture_frame():
    """Capture current frame and return detection results with tracking"""
    global current_frame, frame_lock
    try:
        with frame_lock:
            if current_frame is not None:
                # Perform detection on current frame
                detections = detect_persons_frame_with_dress(current_frame)
                
                # Draw detections on frame
                frame_with_detections = draw_detections_frame(current_frame.copy(), detections)
                
                # Encode frame as base64
                ret, buffer = cv2.imencode('.jpg', frame_with_detections)
                if ret:
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    return jsonify({
                        'success': True,
                        'detections': detections,
                        'count': len(detections),
                        'image': frame_base64
                    })
                else:
                    return jsonify({'success': False, 'message': 'Failed to encode frame'}), 500
            else:
                return jsonify({'success': False, 'message': 'No frame available'}), 400
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error capturing frame: {str(e)}'}), 500

@app.route('/get_cameras', methods=['GET'])
def get_cameras():
    """Get list of available cameras"""
    try:
        cameras = []
        # Test cameras 0-9
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cameras.append({
                    'id': i,
                    'name': f'Camera {i}'
                })
                cap.release()
        
        return jsonify({'success': True, 'cameras': cameras})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error getting cameras: {str(e)}'}), 500

@app.route('/reset_tracker', methods=['POST'])
def reset_tracker():
    """Reset the tracker to clear all tracking IDs"""
    global tracker
    try:
        tracker = BotSORT()
        return jsonify({'success': True, 'message': 'Tracker reset successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error resetting tracker: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting Flask app for person detection with Bot-SORT tracking...")
    print("Camera will auto-start when the app launches")
    print("Make sure you have installed: pip install ultralytics opencv-python flask pillow scipy")
    app.run(debug=True, host='0.0.0.0', port=5000)
