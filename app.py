from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import cv2
import numpy as np
from ultralytics import YOLO
import os
import base64
import threading
import time
from botsort_tracker import BotSORT
try:
    from rfid_scanner import (
        get_rfid_uid, start_rfid_monitoring, stop_rfid_monitoring, 
        subscribe_to_rfid_events, unsubscribe_from_rfid_events,
        get_rfid_status, _rfid_is_present
    )
    RFID_AVAILABLE = True
except ImportError as e:
    print(f"Warning: RFID scanner not available: {e}")
    RFID_AVAILABLE = False
    # Define dummy functions
    def get_rfid_uid(*args, **kwargs):
        return None, "RFID not available"
    def start_rfid_monitoring():
        pass
    def stop_rfid_monitoring():
        pass
    def subscribe_to_rfid_events():
        return None
    def unsubscribe_from_rfid_events(*args):
        pass
    def get_rfid_status():
        return {'available': False, 'present': False}
    def _rfid_is_present():
        return False

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

# Global variables for RFID integration
rfid_event_queue = None
rfid_last_uid = None
rfid_present = False
rfid_lock = threading.Lock()

# Global variable for test mode
test_mode = False
test_mode_lock = threading.Lock()

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

def initialize_rfid():
    """Initialize RFID monitoring"""
    global rfid_event_queue
    try:
        # Start RFID monitoring
        start_rfid_monitoring()
        
        # Subscribe to RFID events
        rfid_event_queue = subscribe_to_rfid_events()
        
        print("RFID monitoring initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing RFID: {e}")
        return False

def rfid_event_handler():
    """Handle RFID events in background thread"""
    global rfid_last_uid, rfid_present, detection_enabled, rfid_lock
    
    while True:
        try:
            if rfid_event_queue:
                event = rfid_event_queue.get(timeout=1.0)
                if event['type'] == 'uid':
                    with rfid_lock:
                        rfid_last_uid = event['uid']
                        rfid_present = True
                        detection_enabled = True
                    print(f"RFID Card detected: {event['uid']} - Detection ENABLED")
                else:
                    with rfid_lock:
                        rfid_present = False
                        detection_enabled = False
                    print("RFID Card removed - Detection DISABLED")
        except:
            # Timeout or no events, check current status
            current_present = _rfid_is_present()
            with rfid_lock:
                if rfid_present != current_present:
                    rfid_present = current_present
                    detection_enabled = current_present
                    if current_present:
                        print("RFID Card present - Detection ENABLED")
                    else:
                        print("RFID Card removed - Detection DISABLED")
        time.sleep(0.1)

# Initialize camera and RFID on startup
initialize_camera()
initialize_rfid()

# Start RFID event handler thread
rfid_thread = threading.Thread(target=rfid_event_handler, daemon=True)
rfid_thread.start()

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

def validate_dress_code(detected_items, gender='male'):
    """Validate dress code compliance based on gender requirements"""
    
    # Define gender-specific dress code requirements
    if gender == 'male':
        required_items = ['polo_shirt', 'pants', 'shoes']
        item_names = {
            'polo_shirt': 'Polo Shirt',
            'pants': 'Pants', 
            'shoes': 'Shoes'
        }
    else:  # female
        required_items = ['blouse', 'skirt', 'doll_shoes']
        item_names = {
            'blouse': 'Blouse',
            'skirt': 'Skirt',
            'doll_shoes': 'Doll Shoes'
        }
    
    # Get detected item classes
    detected_classes = [item['class'].lower().replace(' ', '_') for item in detected_items]
    
    # Check compliance
    compliance_status = {}
    for required_item in required_items:
        if required_item in detected_classes:
            compliance_status[required_item] = {
                'present': True,
                'name': item_names[required_item],
                'status': '✅'
            }
        else:
            compliance_status[required_item] = {
                'present': False,
                'name': item_names[required_item],
                'status': '❌'
            }
    
    # Calculate compliance percentage
    present_count = sum(1 for status in compliance_status.values() if status['present'])
    compliance_percentage = (present_count / len(required_items)) * 100
    
    # Determine overall status
    if compliance_percentage == 100:
        overall_status = "COMPLIANT"
        status_color = "success"
    elif compliance_percentage >= 66:
        overall_status = "PARTIALLY COMPLIANT"
        status_color = "warning"
    else:
        overall_status = "NON-COMPLIANT"
        status_color = "danger"
    
    return {
        'compliance_status': compliance_status,
        'compliance_percentage': compliance_percentage,
        'overall_status': overall_status,
        'status_color': status_color,
        'required_items': required_items,
        'detected_items': detected_classes
    }

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
        
        # Validate dress code compliance (using male for now)
        validation_result = validate_dress_code(dress_items, gender='male')
        
        return validation_result
        
    except Exception as e:
        print(f"Error in dress code detection: {e}")
        return {
            'compliance_status': {},
            'compliance_percentage': 0,
            'overall_status': 'ERROR',
            'status_color': 'danger',
            'required_items': [],
            'detected_items': []
        }
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
            dress_validation = detect_dress_code(person_crop)
            detection['dress_validation'] = dress_validation
            
            # Create a compliance summary for display
            compliance_status = dress_validation['compliance_status']
            compliance_items = []
            for item_key, item_status in compliance_status.items():
                compliance_items.append(f"{item_status['status']} {item_status['name']}")
            
            detection['dress_summary'] = f"{dress_validation['overall_status']} ({dress_validation['compliance_percentage']:.0f}%)"
            detection['dress_details'] = " | ".join(compliance_items)
        
        # Update tracker with detections for static image
        if detections:
            # Convert detections to tracker format [x, y, w, h, confidence]
            dets = []
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                w, h = x2 - x1, y2 - y1
                dets.append([x1, y1, w, h, det['confidence']])
            
            dets = np.array(dets)
            
            # Update tracker
            tracked_objects = tracker.update(dets, image)
            
            # Add tracking IDs to detections
            for i, track in enumerate(tracked_objects):
                if i < len(detections):
                    detections[i]['track_id'] = int(track[4])
                    # Convert back to [x1, y1, x2, y2] format
                    x, y, w, h = track[0], track[1], track[2], track[3]
                    detections[i]['bbox'] = [int(x), int(y), int(x + w), int(y + h)]
        
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
            dress_validation = detect_dress_code(person_crop)
            detection['dress_validation'] = dress_validation
            
            # Create a compliance summary for display
            compliance_status = dress_validation['compliance_status']
            compliance_items = []
            for item_key, item_status in compliance_status.items():
                compliance_items.append(f"{item_status['status']} {item_status['name']}")
            
            detection['dress_summary'] = f"{dress_validation['overall_status']} ({dress_validation['compliance_percentage']:.0f}%)"
            detection['dress_details'] = " | ".join(compliance_items)
        
        # Update tracker with detections
        if detections:
            # Convert detections to tracker format [x, y, w, h, confidence]
            dets = []
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                w, h = x2 - x1, y2 - y1
                dets.append([x1, y1, w, h, det['confidence']])
            
            dets = np.array(dets)
            
            # Update tracker
            tracked_objects = tracker.update(dets, frame)
            
            # Add tracking IDs to detections
            for i, track in enumerate(tracked_objects):
                if i < len(detections):
                    detections[i]['track_id'] = int(track[4])
                    # Convert back to [x1, y1, x2, y2] format
                    x, y, w, h = track[0], track[1], track[2], track[3]
                    detections[i]['bbox'] = [int(x), int(y), int(x + w), int(y + h)]
        
        return detections
    except Exception as e:
        print(f"Error in frame detection: {e}")
        return []

def draw_detections_frame(frame, detections):
    """Draw bounding boxes on a video frame with tracking IDs and dress code compliance"""
    try:
        # Draw bounding boxes
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            track_id = detection.get('track_id', 'N/A')
            dress_summary = detection.get('dress_summary', 'No dress items detected')
            dress_details = detection.get('dress_details', '')
            dress_validation = detection.get('dress_validation', {})
            
            # Choose color based on dress code compliance
            if dress_validation.get('status_color') == 'success':
                color = (0, 255, 0)  # Green for compliant
            elif dress_validation.get('status_color') == 'warning':
                color = (0, 255, 255)  # Yellow for partially compliant
            elif dress_validation.get('status_color') == 'danger':
                color = (0, 0, 255)  # Red for non-compliant
            else:
                color = (128, 128, 128)  # Gray for unknown
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with tracking ID and person confidence
            label1 = f"ID:{track_id} Person: {confidence:.2f}"
            
            # Draw first label (person info)
            label_size1 = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size1[1] - 10), 
                         (x1 + label_size1[0], y1), color, -1)
            cv2.putText(frame, label1, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Draw dress code compliance status
            current_y = y1 - label_size1[1] - 15
            compliance_text = f"Dress Code: {dress_summary}"
            text_size = cv2.getTextSize(compliance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw background for compliance status
            cv2.rectangle(frame, (x1, current_y - text_size[1] - 5), 
                         (x1 + text_size[0] + 5, current_y + 5), color, -1)
            cv2.putText(frame, compliance_text, (x1 + 2, current_y - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Draw individual item status
            if dress_details:
                current_y -= 25
                items_text = f"Items: {dress_details}"
                items_size = cv2.getTextSize(items_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                
                # Draw background for items
                cv2.rectangle(frame, (x1, current_y - items_size[1] - 5), 
                             (x1 + items_size[0] + 5, current_y + 5), color, -1)
                cv2.putText(frame, items_text, (x1 + 2, current_y - 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
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
                
                # Check if detection should be enabled (RFID card present OR test mode active)
                with test_mode_lock:
                    test_mode_active = test_mode
                
                with rfid_lock:
                    rfid_detection_enabled = detection_enabled and rfid_present
                
                detection_enabled_for_frame = rfid_detection_enabled or test_mode_active
                
                if detection_enabled_for_frame:
                    detections = detect_persons_frame_with_dress(frame)
                    frame = draw_detections_frame(frame, detections)
                    
                    # Add status overlay based on mode
                    if test_mode_active:
                        cv2.putText(frame, "TEST MODE: Detection Always Active", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)  # Orange color
                    else:
                        cv2.putText(frame, f"RFID: ACTIVE - {rfid_last_uid}", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # Add RFID status overlay
                    cv2.putText(frame, "RFID: WAITING - Scan Card to Enable Detection", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
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
    """Toggle person detection on/off (only works if RFID card is present)"""
    global detection_enabled, rfid_present, rfid_lock
    try:
        with rfid_lock:
            if not rfid_present:
                return jsonify({'success': False, 'message': 'RFID card must be present to enable detection'}), 400
            
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
                # Check if detection should be enabled (RFID card present OR test mode active)
                with test_mode_lock:
                    test_mode_active = test_mode
                
                with rfid_lock:
                    rfid_detection_enabled = detection_enabled and rfid_present
                
                detection_enabled_for_capture = rfid_detection_enabled or test_mode_active
                
                if detection_enabled_for_capture:
                    # Perform detection on current frame
                    detections = detect_persons_frame_with_dress(current_frame)
                    
                    # Draw detections on frame
                    frame_with_detections = draw_detections_frame(current_frame.copy(), detections)
                    
                    # Add status overlay based on mode
                    if test_mode_active:
                        cv2.putText(frame_with_detections, "TEST MODE: Detection Always Active", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)  # Orange color
                    else:
                        cv2.putText(frame_with_detections, f"RFID: ACTIVE - {rfid_last_uid}", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # No detection, just return original frame
                    detections = []
                    frame_with_detections = current_frame.copy()
                    
                    # Add RFID status overlay
                    cv2.putText(frame_with_detections, "RFID: WAITING - Scan Card to Enable Detection", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
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

# RFID Status Endpoints
@app.route('/rfid/status', methods=['GET'])
def rfid_status():
    """Get RFID scanner status"""
    global rfid_last_uid, rfid_present, detection_enabled, rfid_lock
    try:
        with rfid_lock:
            status = get_rfid_status()
            status.update({
                'last_uid': rfid_last_uid,
                'present': rfid_present,
                'detection_enabled': detection_enabled
            })
        return jsonify({'success': True, 'status': status})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error getting RFID status: {str(e)}'}), 500

@app.route('/rfid/read', methods=['POST'])
def rfid_read():
    """Read RFID card data once"""
    try:
        uid, error = get_rfid_uid(timeout_seconds=3)
        if uid:
            return jsonify({'success': True, 'uid': uid})
        else:
            return jsonify({'success': False, 'message': error or 'No card detected'}), 404
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error reading RFID card: {str(e)}'}), 500

@app.route('/toggle_test_mode', methods=['POST'])
def toggle_test_mode():
    """Toggle test mode on/off"""
    global test_mode, detection_enabled
    try:
        data = request.get_json()
        test_mode_enabled = data.get('test_mode', False)
        
        with test_mode_lock:
            test_mode = test_mode_enabled
            
        if test_mode:
            # In test mode, always enable detection
            detection_enabled = True
            return jsonify({'success': True, 'test_mode': True, 'message': 'Test mode activated - Detection always enabled'})
        else:
            # Exit test mode, return to RFID-based detection
            with rfid_lock:
                detection_enabled = rfid_present
            return jsonify({'success': True, 'test_mode': False, 'message': 'Test mode deactivated - Detection requires RFID card'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error toggling test mode: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting Flask app for person detection with Bot-SORT tracking...")
    print("Camera will auto-start when the app launches")
    print("RFID monitoring will start automatically")
    print("Detection will only work when RFID card is present")
    print("Make sure you have installed: pip install ultralytics opencv-python flask pillow scipy pyscard")
    app.run(debug=True, host='0.0.0.0', port=5000)
