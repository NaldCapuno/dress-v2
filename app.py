from flask import Flask, render_template, request, jsonify, send_from_directory, Response, session, redirect, url_for
import cv2
import numpy as np
from ultralytics import YOLO
import os
import base64
import threading
import time
from botsort_tracker import BotSORT
from werkzeug.security import check_password_hash
try:
    from config import get_connection, find_student_by_rfid, insert_rfid_log, insert_violation
except Exception as e:
    get_connection = None
    find_student_by_rfid = None
    insert_rfid_log = None
    insert_violation = None
    print(f"Warning: Database config not available: {e}")

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
# Secret key for session management (can be overridden via environment variable)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'change-this-in-production')

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
rfid_last_student = None  # Holds last looked-up student dict or None
rfid_last_violation_ts = 0  # last violation timestamp to throttle duplicates
rfid_current_uid_checks = 0  # number of detection checks for current RFID UID
rfid_current_uid_violated = False  # whether a violation has been issued for current UID session
rfid_consecutive_non_compliant = 0  # counter for consecutive non-compliant detections
rfid_last_compliance_status = None  # track last compliance status to detect changes

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
    global rfid_last_uid, rfid_present, detection_enabled, rfid_lock, rfid_last_student, rfid_consecutive_non_compliant, rfid_last_compliance_status
    
    while True:
        try:
            if rfid_event_queue:
                event = rfid_event_queue.get(timeout=1.0)
                if event['type'] == 'uid':
                    with rfid_lock:
                        rfid_last_uid = event['uid']
                        rfid_present = True
                        detection_enabled = True
                        # Reset per-scan counters
                        rfid_current_uid_checks = 0
                        rfid_current_uid_violated = False
                        rfid_consecutive_non_compliant = 0
                        rfid_last_compliance_status = None
                        # Perform DB lookup and log
                        try:
                            student = None
                            if get_connection is not None and rfid_last_uid:
                                student = find_student_by_rfid(rfid_last_uid) if find_student_by_rfid else None
                                if student and insert_rfid_log:
                                    insert_rfid_log(rfid_last_uid, student.get('student_id'), 'valid')
                                elif insert_rfid_log:
                                    insert_rfid_log(rfid_last_uid, None, 'unregistered')
                            rfid_last_student = student
                        except Exception as e:
                            print(f"RFID DB handling error: {e}")
                    print(f"RFID Card detected: {event['uid']} - Detection ENABLED")
                else:
                    with rfid_lock:
                        rfid_present = False
                        detection_enabled = False
                        rfid_last_student = None
                        rfid_current_uid_checks = 0
                        rfid_current_uid_violated = False
                        rfid_consecutive_non_compliant = 0
                        rfid_last_compliance_status = None
                    print("RFID Card removed - Detection DISABLED")
        except:
            # Timeout or no events, check current status
            current_present = _rfid_is_present()
            with rfid_lock:
                if rfid_present != current_present:
                    rfid_present = current_present
                    detection_enabled = current_present
                    if not current_present:
                        rfid_last_student = None
                        rfid_current_uid_checks = 0
                        rfid_current_uid_violated = False
                        rfid_consecutive_non_compliant = 0
                        rfid_last_compliance_status = None
                    if current_present:
                        print("RFID Card present - Detection ENABLED")
                    else:
                        print("RFID Card removed - Detection DISABLED")
        time.sleep(0.1)

# Initialize RFID on startup (camera will be initialized when user starts it)
# initialize_camera()  # Commented out to keep camera off by default
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

def detect_dress_code(person_crop, gender: str = 'male'):
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
        
        # Validate dress code compliance per provided gender
        validation_result = validate_dress_code(dress_items, gender=(gender or 'male').lower())
        
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
        
        # Determine gender context (from last RFID student if present)
        with rfid_lock:
            current_gender = (rfid_last_student or {}).get('gender')
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
            dress_validation = detect_dress_code(person_crop, gender=current_gender or 'male')
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
        
        # Determine gender context from current RFID student if available
        with rfid_lock:
            current_gender = (rfid_last_student or {}).get('gender')
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
            dress_validation = detect_dress_code(person_crop, gender=current_gender or 'male')
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


def _maybe_record_violation(frame, detections, admin_user):
    """Record violation after 3 consecutive non-compliant detections for current RFID student.
    
    - Only records once per RFID scan session
    - Requires 3 consecutive non-compliant detections before recording
    - Resets counter if compliance status changes
    """
    global rfid_last_student, rfid_last_violation_ts, rfid_consecutive_non_compliant, rfid_last_compliance_status, rfid_current_uid_violated
    
    try:
        if not rfid_last_student:
            print("DEBUG: No RFID student found, skipping violation check")
            return None
            
        # Determine if any detection for this frame indicates NON-COMPLIANT
        non_compliant = False
        violation_details = []
        current_compliance_status = None
        
        for det in detections or []:
            dv = det.get('dress_validation') or {}
            overall_status = dv.get('overall_status')
            if overall_status == 'NON-COMPLIANT':
                non_compliant = True
                comp = dv.get('compliance_status') or {}
                for key, val in comp.items():
                    if not val.get('present'):
                        violation_details.append(val.get('name') or key)
            current_compliance_status = overall_status
        
        print(f"DEBUG: Frame analysis - Non-compliant: {non_compliant}, Status: {current_compliance_status}, Detections: {len(detections)}")
        
        # Only process if RFID card is present
        with rfid_lock:
            if not rfid_present:
                print("DEBUG: RFID not present, skipping violation check")
                return None
                
            # If we already issued a violation for this UID session, stop
            if rfid_current_uid_violated:
                print("DEBUG: Violation already recorded for this RFID session")
                return None
            
            # Check if compliance status changed (reset counter if it did)
            if rfid_last_compliance_status is not None and rfid_last_compliance_status != current_compliance_status:
                rfid_consecutive_non_compliant = 0
                print(f"DEBUG: Compliance status changed from {rfid_last_compliance_status} to {current_compliance_status}, resetting counter")
            
            # Update last compliance status
            rfid_last_compliance_status = current_compliance_status
            
            # Increment counter only for non-compliant detections
            if non_compliant:
                rfid_consecutive_non_compliant += 1
                print(f"DEBUG: Non-compliant detection #{rfid_consecutive_non_compliant} for student {rfid_last_student.get('student_id')}")
            else:
                # Reset counter if compliant
                rfid_consecutive_non_compliant = 0
                print(f"DEBUG: Compliant detection, resetting counter for student {rfid_last_student.get('student_id')}")
            
            # Only proceed if we have 3 consecutive non-compliant detections
            if rfid_consecutive_non_compliant < 3:
                print(f"DEBUG: Need {3 - rfid_consecutive_non_compliant} more consecutive non-compliant detections")
                return None

        # Throttle: avoid spamming the same student too frequently (10 seconds)
        now_ts = time.time()
        if now_ts - rfid_last_violation_ts < 10:
            print(f"DEBUG: Throttled - last violation was {now_ts - rfid_last_violation_ts:.1f}s ago")
            return None

        print(f"DEBUG: Recording violation for student {rfid_last_student.get('student_id')} after {rfid_consecutive_non_compliant} consecutive detections")
        print(f"DEBUG: Admin user: {admin_user is not None}")

        # Save enhanced proof image with annotations
        proof_name = f"violation_{int(now_ts)}_{rfid_last_student.get('student_id', 'unknown')}.jpg"
        proof_path = os.path.join(RESULT_FOLDER, proof_name)
        print(f"DEBUG: Proof image path: {proof_path}")
        
        try:
            os.makedirs(RESULT_FOLDER, exist_ok=True)
            print(f"DEBUG: Created/verified results folder: {RESULT_FOLDER}")
            
            # Create an enhanced proof image with violation details
            proof_frame = frame.copy()
            print(f"DEBUG: Created proof frame copy, shape: {proof_frame.shape}")
            
            # Build violation type text (gender-aware) for image annotation
            with rfid_lock:
                current_gender = (rfid_last_student or {}).get('gender')
            gender_label = str(current_gender or 'unknown').lower()
            missing = ", ".join(violation_details) if violation_details else "Missing required items"
            violation_type_for_image = f"{gender_label} dress code violation: {missing}"
            
            # Add violation information overlay
            violation_text = f"VIOLATION RECORDED - {violation_type_for_image}"
            timestamp_text = f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now_ts))}"
            student_text = f"Student ID: {rfid_last_student.get('student_id', 'Unknown')}"
            rfid_text = f"RFID: {rfid_last_uid}"
            
            print(f"DEBUG: Adding text overlays to proof image")
            
            # Draw semi-transparent background for text
            overlay = proof_frame.copy()
            cv2.rectangle(overlay, (10, 10), (proof_frame.shape[1] - 10, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, proof_frame, 0.3, 0, proof_frame)
            
            # Add violation text
            cv2.putText(proof_frame, violation_text, (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)  # Red text
            cv2.putText(proof_frame, timestamp_text, (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # White text
            cv2.putText(proof_frame, student_text, (20, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # White text
            cv2.putText(proof_frame, rfid_text, (20, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # White text
            
            print(f"DEBUG: Added text overlays, now adding bounding boxes")
            
            # Draw bounding boxes and violation details on detected persons
            for det in detections or []:
                dv = det.get('dress_validation') or {}
                if dv.get('overall_status') == 'NON-COMPLIANT':
                    x1, y1, x2, y2 = det.get('bbox', [0, 0, 0, 0])
                    
                    # Draw red bounding box for non-compliant person
                    cv2.rectangle(proof_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    
                    # Add violation details
                    comp = dv.get('compliance_status') or {}
                    missing_items = []
                    for key, val in comp.items():
                        if not val.get('present'):
                            missing_items.append(val.get('name') or key)
                    
                    if missing_items:
                        missing_text = f"Missing: {', '.join(missing_items)}"
                        cv2.putText(proof_frame, missing_text, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            print(f"DEBUG: Added bounding boxes, attempting to save image")
            
            # Save the enhanced proof image
            success = cv2.imwrite(proof_path, proof_frame)
            print(f"DEBUG: cv2.imwrite returned: {success}")
            print(f"DEBUG: Proof image saved: {proof_path}")
            
            # Verify file was created
            if os.path.exists(proof_path):
                file_size = os.path.getsize(proof_path)
                print(f"DEBUG: File exists, size: {file_size} bytes")
            else:
                print(f"DEBUG: ERROR - File was not created!")
            
        except Exception as e:
            print(f"DEBUG: Error saving proof image: {e}")
            import traceback
            traceback.print_exc()
            proof_path = None

        # Use the violation type that was already created for the image
        violation_type = violation_type_for_image

        recorded_by = None
        if admin_user and isinstance(admin_user, dict):
            try:
                recorded_by = int(admin_user.get('admin_id'))
            except Exception:
                recorded_by = None

        # Store only filename in DB; serve via /results/<filename>
        rel_path = proof_name if proof_path else None
        print(f"DEBUG: Database path: {rel_path}")

        if insert_violation:
            print(f"DEBUG: Attempting database insertion...")
            vid = insert_violation(rfid_last_student.get('student_id'), violation_type, rel_path, recorded_by)
            print(f"DEBUG: Database insertion returned: {vid}")
            if vid:
                rfid_last_violation_ts = now_ts
                with rfid_lock:
                    rfid_current_uid_violated = True
                
                # Create violation summary for logging
                violation_summary = {
                    'violation_id': vid,
                    'student_id': rfid_last_student.get('student_id'),
                    'student_name': rfid_last_student.get('name', 'Unknown'),
                    'rfid_uid': rfid_last_uid,
                    'violation_type': violation_type,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now_ts)),
                    'proof_image': proof_name,
                    'missing_items': violation_details,
                    'consecutive_detections': rfid_consecutive_non_compliant
                }
                
                print(f"VIOLATION RECORDED:")
                print(f"  - Violation ID: {vid}")
                print(f"  - Student: {rfid_last_student.get('name')} (ID: {rfid_last_student.get('student_id')})")
                print(f"  - RFID: {rfid_last_uid}")
                print(f"  - Type: {violation_type}")
                print(f"  - Proof Image: {proof_name}")
                print(f"  - Consecutive Non-Compliant Detections: {rfid_consecutive_non_compliant}")
                
                return vid
            else:
                print(f"DEBUG: Database insertion failed - no violation ID returned")
        else:
            print(f"DEBUG: insert_violation function not available")
        return None
    except Exception as e:
        print(f"Violation record error: {e}")
        return None

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
                    
                    # Attempt to record violation if non-compliant (for live monitoring)
                    # Note: admin_user is None in background thread, will be handled in violation function
                    _maybe_record_violation(frame, detections, None)
                    
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
    # Only allow SECURITY role to access the main index/dashboard
    admin = session.get('admin') or {}
    role = str(admin.get('role') or '').lower()
    if role != 'security':
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Alias for the main dashboard; restricted to security role."""
    admin = session.get('admin') or {}
    role = str(admin.get('role') or '').lower()
    if role != 'security':
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/osas', methods=['GET'])
def osas_dashboard():
    """OSAS dashboard - only accessible to admins with role 'osas'."""
    admin = session.get('admin') or {}
    role = str(admin.get('role') or '').lower()
    if role != 'osas':
        return redirect(url_for('login'))
    return render_template('osas_dashboard.html')

@app.route('/dean', methods=['GET'])
def dean_dashboard():
    """Dean dashboard - only accessible to admins with role 'dean'."""
    admin = session.get('admin') or {}
    role = str(admin.get('role') or '').lower()
    if role != 'dean':
        return redirect(url_for('login'))
    return render_template('dean_dashboard.html', college=admin.get('college'))

@app.route('/dean/programs', methods=['GET'])
def dean_programs():
    """Return distinct programs for the dean's college."""
    college = (session.get('admin') or {}).get('college')
    if not college:
        return jsonify({'success': True, 'programs': []})
    conn = get_connection() if get_connection else None
    if conn is None:
        return jsonify({'success': False, 'error': 'DB not configured'}), 500
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT COALESCE(course,'') AS course FROM students WHERE college=%s AND COALESCE(course,'')<>'' ORDER BY course ASC",
                (college,)
            )
            rows = cur.fetchall() or []
        programs = [r.get('course') for r in rows if r.get('course')]
        return jsonify({'success': True, 'programs': programs})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        conn.close()


# ------------------ Dean endpoints (college-level accountability) ------------------
@app.route('/dean/violations', methods=['GET'])
def dean_get_violations():
    """List violations for dean review (defaults to cases forwarded to dean)."""
    try:
        status_filter = request.args.get('status', 'forwarded_dean')
        start_dt = request.args.get('start')
        end_dt = request.args.get('end')
        academic_year = request.args.get('academic_year')
        semester = request.args.get('semester')
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 50))
        offset = max(0, (page - 1) * page_size)
        college = request.args.get('college') or ((session.get('admin') or {}).get('college'))
        program = request.args.get('program')

        conn = get_connection() if get_connection else None
        if conn is None:
            return jsonify({'success': False, 'error': 'DB not configured'}), 500

        where = []
        params = []
        if status_filter:
            where.append("v.status = %s")
            params.append(status_filter)
        # Enforce dean operates per-college: require college filter
        if not college:
            return jsonify({'success': True, 'rows': [], 'total': 0})
        where.append("s.college = %s")
        params.append(college)
        if program:
            where.append("s.course = %s")
            params.append(program)
        if start_dt:
            where.append("v.timestamp >= %s")
            params.append(start_dt)
        if end_dt:
            where.append("v.timestamp <= %s")
            params.append(end_dt)
        if academic_year and semester in {"1", "2"}:
            try:
                start_year = int(academic_year.split('-')[0])
                if semester == "1":
                    ay_start = f"{start_year}-08-01 00:00:00"
                    ay_end = f"{start_year}-12-31 23:59:59"
                else:
                    ay_start = f"{start_year+1}-01-01 00:00:00"
                    ay_end = f"{start_year+1}-05-31 23:59:59"
                where.append("v.timestamp BETWEEN %s AND %s")
                params.extend([ay_start, ay_end])
            except Exception:
                pass
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""

        base_select = (
            "SELECT v.violation_id, v.student_id, v.recorded_by, v.violation_type, v.timestamp, v.image_proof, v.status, "
            "s.name, s.gender, s.course, s.college "
            "FROM violations v LEFT JOIN students s ON v.student_id = s.student_id"
        )

        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) AS cnt FROM violations v LEFT JOIN students s ON v.student_id = s.student_id{where_sql}", params)
            total = (cur.fetchone() or {}).get('cnt', 0)

            cur.execute(
                f"{base_select}{where_sql} ORDER BY v.timestamp DESC LIMIT %s OFFSET %s",
                params + [page_size, offset]
            )
            rows = cur.fetchall() or []
        conn.close()
        return jsonify({'success': True, 'rows': rows, 'total': total})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/dean/violation/<int:violation_id>/status', methods=['POST'])
def dean_update_violation_status(violation_id: int):
    """Dean can forward to guidance, set pending, or resolve."""
    try:
        data = request.get_json(silent=True) or {}
        status = str(data.get('status') or '').strip().lower()
        allowed = {"pending", "forwarded_guidance", "resolved"}
        if status not in allowed:
            return jsonify({'success': False, 'error': 'Invalid status'}), 400
        conn = get_connection() if get_connection else None
        if conn is None:
            return jsonify({'success': False, 'error': 'DB not configured'}), 500
        with conn.cursor() as cur:
            cur.execute("UPDATE violations SET status=%s WHERE violation_id=%s", (status, violation_id))
        conn.close()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/dean/analytics', methods=['GET'])
def dean_analytics():
    """Aggregate analytics for dean view (college-level)."""
    try:
        start_dt = request.args.get('start')
        end_dt = request.args.get('end')
        academic_year = request.args.get('academic_year')
        semester = request.args.get('semester')
        status_filter = request.args.get('status', 'forwarded_dean')
        college = request.args.get('college') or ((session.get('admin') or {}).get('college'))
        program = request.args.get('program')

        conn = get_connection() if get_connection else None
        if conn is None:
            return jsonify({'success': False, 'error': 'DB not configured'}), 500

        where = []
        params = []
        if status_filter:
            where.append("v.status = %s")
            params.append(status_filter)
        # Enforce dean operates per-college: require college filter
        if not college:
            return jsonify({'success': True, 'total': 0, 'by_program': [], 'by_gender': [], 'by_status': []})
        where.append("s.college = %s")
        params.append(college)
        if program:
            where.append("s.course = %s")
            params.append(program)
        if start_dt:
            where.append("v.timestamp >= %s")
            params.append(start_dt)
        if end_dt:
            where.append("v.timestamp <= %s")
            params.append(end_dt)
        if academic_year and semester in {"1", "2"}:
            try:
                start_year = int(academic_year.split('-')[0])
                if semester == "1":
                    ay_start = f"{start_year}-08-01 00:00:00"
                    ay_end = f"{start_year}-12-31 23:59:59"
                else:
                    ay_start = f"{start_year+1}-01-01 00:00:00"
                    ay_end = f"{start_year+1}-05-31 23:59:59"
                where.append("v.timestamp BETWEEN %s AND %s")
                params.extend([ay_start, ay_end])
            except Exception:
                pass
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""

        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) AS total FROM violations v LEFT JOIN students s ON v.student_id=s.student_id{where_sql}", params)
            total = (cur.fetchone() or {}).get('total', 0)

            cur.execute(
                f"SELECT COALESCE(s.course,'Unknown') AS label, COUNT(*) AS cnt FROM violations v LEFT JOIN students s ON v.student_id=s.student_id{where_sql} GROUP BY label ORDER BY cnt DESC",
                params,
            )
            by_program = cur.fetchall() or []

            cur.execute(
                f"SELECT LOWER(COALESCE(s.gender,'')) AS label, COUNT(*) AS cnt FROM violations v LEFT JOIN students s ON v.student_id=s.student_id{where_sql} GROUP BY label",
                params,
            )
            by_gender = cur.fetchall() or []

            cur.execute(
                f"SELECT v.status AS label, COUNT(*) AS cnt FROM violations v{where_sql} GROUP BY v.status",
                params,
            )
            by_status = cur.fetchall() or []
        conn.close()
        return jsonify({'success': True, 'total': total, 'by_program': by_program, 'by_gender': by_gender, 'by_status': by_status})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/dean/trend', methods=['GET'])
def dean_trend():
    try:
        start_dt = request.args.get('start')
        end_dt = request.args.get('end')
        academic_year = request.args.get('academic_year')
        semester = request.args.get('semester')
        status_filter = request.args.get('status', 'forwarded_dean')
        group_by = request.args.get('group_by', 'day')
        college = request.args.get('college') or ((session.get('admin') or {}).get('college'))
        program = request.args.get('program')

        conn = get_connection() if get_connection else None
        if conn is None:
            return jsonify({'success': False, 'error': 'DB not configured'}), 500

        where = []
        params = []
        if status_filter:
            where.append("v.status = %s")
            params.append(status_filter)
        # Enforce dean operates per-college: require college filter
        if not college:
            return jsonify({'success': True, 'series': []})
        where.append("s.college = %s")
        params.append(college)
        if program:
            where.append("s.course = %s")
            params.append(program)
        if start_dt:
            where.append("timestamp >= %s")
            params.append(start_dt)
        if end_dt:
            where.append("timestamp <= %s")
            params.append(end_dt)
        if academic_year and semester in {"1", "2"}:
            try:
                start_year = int(academic_year.split('-')[0])
                if semester == "1":
                    ay_start = f"{start_year}-08-01 00:00:00"
                    ay_end = f"{start_year}-12-31 23:59:59"
                else:
                    ay_start = f"{start_year+1}-01-01 00:00:00"
                    ay_end = f"{start_year+1}-05-31 23:59:59"
                where.append("timestamp BETWEEN %s AND %s")
                params.extend([ay_start, ay_end])
            except Exception:
                pass
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""

        if group_by == 'month':
            group_expr = "DATE_FORMAT(timestamp, '%Y-%m')"
            order_expr = "DATE_FORMAT(timestamp, '%Y-%m')"
        elif group_by == 'week':
            group_expr = "YEARWEEK(timestamp, 3)"
            order_expr = "YEARWEEK(timestamp, 3)"
        else:
            group_expr = "DATE(timestamp)"
            order_expr = "DATE(timestamp)"

        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {group_expr} AS label, COUNT(*) AS cnt FROM violations v LEFT JOIN students s ON v.student_id=s.student_id{where_sql} GROUP BY label ORDER BY {order_expr} ASC",
                params,
            )
            rows = cur.fetchall() or []
        conn.close()
        return jsonify({'success': True, 'series': rows})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/logout', methods=['POST'])
def logout():
    """Clear session and log out the current user."""
    try:
        session.clear()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')

    # POST: JSON { username, password }
    try:
        data = request.get_json(force=True, silent=True) or {}
        username = (data.get('username') or '').strip()
        password = data.get('password') or ''

        if not username or not password:
            return jsonify({'success': False, 'error': 'Username and password are required.'}), 400

        if get_connection is None:
            return jsonify({'success': False, 'error': 'Database not configured.'}), 500

        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT admin_id, username, password_hash, role, college, created_at
                    FROM admins
                    WHERE username = %s
                    LIMIT 1
                    """,
                    (username,)
                )
                admin = cur.fetchone()

            if not admin or not check_password_hash(admin.get('password_hash', ''), password):
                return jsonify({'success': False, 'error': 'Invalid username or password.'}), 401

            # Remove sensitive field and persist minimal session
            admin.pop('password_hash', None)
            session['admin'] = {
                'admin_id': admin.get('admin_id'),
                'username': admin.get('username'),
                'role': admin.get('role'),
                'college': admin.get('college'),
            }

            return jsonify({'success': True, 'admin': session['admin']})
        finally:
            conn.close()
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

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

@app.route('/violation_proof/<filename>')
def violation_proof(filename):
    """Serve violation proof images with proper headers"""
    try:
        return send_from_directory(RESULT_FOLDER, filename, as_attachment=False)
    except Exception as e:
        return f"Error serving proof image: {e}", 404

@app.route('/violation_log')
def violation_log():
    """Display recent violations with proof images"""
    try:
        if not get_connection:
            return jsonify({'error': 'Database not available'}), 500
            
        conn = get_connection()
        with conn.cursor() as cur:
            # Get recent violations (last 50)
            cur.execute("""
                SELECT v.violation_id, v.student_id, v.violation_type, v.timestamp, v.image_proof,
                       s.name as student_name, s.gender, s.course, s.college
                FROM violations v 
                LEFT JOIN students s ON v.student_id = s.student_id
                ORDER BY v.timestamp DESC 
                LIMIT 50
            """)
            violations = cur.fetchall() or []
            
        return jsonify({
            'success': True, 
            'violations': violations,
            'total_count': len(violations)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        if 'conn' in locals():
            conn.close()

@app.route('/debug_state')
def debug_state():
    """Debug endpoint to check current system state"""
    try:
        with rfid_lock:
            state = {
                'rfid_present': rfid_present,
                'rfid_last_uid': rfid_last_uid,
                'rfid_last_student': rfid_last_student,
                'rfid_consecutive_non_compliant': rfid_consecutive_non_compliant,
                'rfid_last_compliance_status': rfid_last_compliance_status,
                'rfid_current_uid_violated': rfid_current_uid_violated,
                'detection_enabled': detection_enabled,
                'current_frame_available': current_frame is not None,
                'result_folder_exists': os.path.exists(RESULT_FOLDER),
                'result_folder_path': os.path.abspath(RESULT_FOLDER)
            }
        
        return jsonify({
            'success': True,
            'state': state,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/test_violation')
def test_violation():
    """Test violation recording with current frame"""
    try:
        with frame_lock:
            if current_frame is None:
                return jsonify({'success': False, 'error': 'No frame available'})
            
            print("DEBUG: Starting test violation...")
            
            # Create a test student for violation testing
            test_student = {
                'student_id': 'TEST-001',
                'name': 'Test Student',
                'gender': 'male'
            }
            
            # Temporarily set RFID student for testing
            global rfid_last_student, rfid_consecutive_non_compliant, rfid_last_compliance_status, rfid_current_uid_violated, rfid_present
            rfid_last_student = test_student
            rfid_consecutive_non_compliant = 3  # Force violation recording
            rfid_last_compliance_status = 'NON-COMPLIANT'
            rfid_current_uid_violated = False  # Reset violation flag
            rfid_present = True  # Simulate RFID present
            
            print(f"DEBUG: Test student set: {test_student}")
            print(f"DEBUG: Consecutive count: {rfid_consecutive_non_compliant}")
            
            # Perform detection on current frame
            detections = detect_persons_frame_with_dress(current_frame)
            print(f"DEBUG: Detections found: {len(detections)}")
            
            # Force non-compliant status for testing
            for det in detections:
                det['dress_validation'] = {
                    'overall_status': 'NON-COMPLIANT',
                    'compliance_status': {
                        'polo_shirt': {'present': False, 'name': 'Polo Shirt'},
                        'pants': {'present': True, 'name': 'Pants'},
                        'shoes': {'present': False, 'name': 'Shoes'}
                    }
                }
            
            print("DEBUG: Forced non-compliant status on detections")
            
            # Record violation
            admin_user = session.get('admin') or {}
            print(f"DEBUG: Admin user available: {admin_user is not None}")
            
            violation_id = _maybe_record_violation(current_frame, detections, admin_user)
            
            print(f"DEBUG: Violation ID returned: {violation_id}")
            
            # Reset test state
            rfid_last_student = None
            rfid_consecutive_non_compliant = 0
            rfid_last_compliance_status = None
            rfid_current_uid_violated = False
            rfid_present = False
            
            if violation_id:
                return jsonify({
                    'success': True, 
                    'violation_id': violation_id,
                    'message': 'Test violation recorded successfully',
                    'detections': len(detections)
                })
            else:
                return jsonify({
                    'success': False, 
                    'message': 'Failed to record test violation',
                    'detections': len(detections)
                })
                
    except Exception as e:
        print(f"DEBUG: Test violation error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/violation_report')
def violation_report():
    """Generate comprehensive violation report"""
    try:
        if not get_connection:
            return jsonify({'error': 'Database not available'}), 500
            
        # Get date range from query parameters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        conn = get_connection()
        with conn.cursor() as cur:
            # Build query with optional date filtering
            where_clause = ""
            params = []
            
            if start_date and end_date:
                where_clause = "WHERE v.timestamp BETWEEN %s AND %s"
                params = [start_date, end_date]
            
            # Get violations with student details
            query = f"""
                SELECT v.violation_id, v.student_id, v.violation_type, v.timestamp, v.image_proof,
                       s.name as student_name, s.gender, s.course, s.college, s.student_id as student_number
                FROM violations v 
                LEFT JOIN students s ON v.student_id = s.student_id
                {where_clause}
                ORDER BY v.timestamp DESC
            """
            
            cur.execute(query, params)
            violations = cur.fetchall() or []
            
            # Get summary statistics
            cur.execute(f"""
                SELECT 
                    COUNT(*) as total_violations,
                    COUNT(DISTINCT v.student_id) as unique_students,
                    COUNT(CASE WHEN s.gender = 'male' THEN 1 END) as male_violations,
                    COUNT(CASE WHEN s.gender = 'female' THEN 1 END) as female_violations
                FROM violations v 
                LEFT JOIN students s ON v.student_id = s.student_id
                {where_clause}
            """, params)
            
            stats = cur.fetchone() or {}
            
        return jsonify({
            'success': True,
            'violations': violations,
            'statistics': stats,
            'report_generated': time.strftime('%Y-%m-%d %H:%M:%S'),
            'date_range': {'start': start_date, 'end': end_date}
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        if 'conn' in locals():
            conn.close()

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

                    # Attempt to record violation if non-compliant
                    admin_user = session.get('admin') or {}
                    _maybe_record_violation(current_frame, detections, admin_user)
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
        import platform
        
        # Try to get system-specific camera names
        system = platform.system().lower()
        
        # Test cameras 0-5 (reduced range to avoid errors)
        for i in range(6):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    # Try to get camera properties for better naming
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    # Try to get camera backend info
                    backend = cap.getBackendName()
                    
                    # Try to get more descriptive names based on system
                    camera_name = None
                    
                    if system == "windows":
                        # On Windows, try to get device names
                        try:
                            import subprocess
                            result = subprocess.run(['wmic', 'path', 'win32_pnpentity', 'where', 'name like "%camera%"', 'get', 'name'], 
                                                  capture_output=True, text=True, timeout=3)
                            if result.returncode == 0:
                                lines = result.stdout.strip().split('\n')
                                for line in lines[1:]:  # Skip header
                                    if line.strip() and 'camera' in line.lower():
                                        camera_name = line.strip()
                                        break
                        except:
                            pass
                    
                    # Fallback to generic naming with more details
                    if not camera_name:
                        if i == 0:
                            camera_name = f"Default Camera ({width}x{height})"
                        else:
                            camera_name = f"Camera {i} ({width}x{height})"
                    
                    # Add backend info if available
                    if backend and backend != "UNKNOWN":
                        camera_name += f" [{backend}]"
                    
                    cameras.append({
                        'id': i,
                        'name': camera_name,
                        'resolution': f"{width}x{height}",
                        'fps': fps,
                        'backend': backend
                    })
                    cap.release()
            except Exception as e:
                # Skip cameras that cause errors
                continue
        
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
    global rfid_last_uid, rfid_present, detection_enabled, rfid_lock, rfid_last_student
    try:
        with rfid_lock:
            status = get_rfid_status()
            status.update({
                'last_uid': rfid_last_uid,
                'present': rfid_present,
                'detection_enabled': detection_enabled,
                'student': rfid_last_student
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
            # Lookup and log immediately
            student = None
            if get_connection is not None:
                try:
                    student = find_student_by_rfid(uid) if find_student_by_rfid else None
                    if student and insert_rfid_log:
                        insert_rfid_log(uid, student.get('student_id'), 'valid')
                    elif insert_rfid_log:
                        insert_rfid_log(uid, None, 'unregistered')
                except Exception as e:
                    print(f"RFID read DB error: {e}")
            return jsonify({'success': True, 'uid': uid, 'student': student})
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
