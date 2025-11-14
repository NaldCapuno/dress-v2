from flask import Flask, render_template, request, jsonify, send_from_directory, Response, session, redirect, url_for
from flask_mail import Mail, Message
import cv2
import numpy as np
from ultralytics import YOLO
import os
import base64
import threading
import time
import re
from io import BytesIO
from datetime import datetime
from src.botsort_tracker import BotSORT
from src.email_templates import generate_violation_email_body
from werkzeug.security import check_password_hash

# Diagnostic: Print Python path for debugging
import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
    print("✓ reportlab library loaded successfully - PDF generation enabled")
except ImportError as e:
    REPORTLAB_AVAILABLE = False
    print(f"✗ Warning: reportlab not available. PDF generation will be disabled.")
    print(f"  Error: {e}")
    print(f"  Python path: {sys.executable}")
    print(f"  Try running: {sys.executable} -m pip install reportlab")
except Exception as e:
    REPORTLAB_AVAILABLE = False
    print(f"✗ Warning: Error loading reportlab. PDF generation will be disabled.")
    print(f"  Error: {e}")
    print(f"  Python path: {sys.executable}")

# College abbreviation mapping
COLLEGE_ABBREVIATIONS = {
    'College of Sciences': 'CS',
    'College of Engineering': 'COE',
    'College of Business and Accountancy': 'CBA',
    'College of Nursing and Health Sciences': 'CNHS',
    'College of Architecture and Design': 'CAD',
    'College of Hospitality Management and Tourism': 'CHTM',
    'College of Criminal Justice and Education': 'CCJE',
    'College of Teacher Education': 'CTE',
    'College of Arts And Humanities': 'CAH'
}

def get_college_abbreviation(college_name):
    """Get the abbreviation for a college name, or return the original if not found."""
    return COLLEGE_ABBREVIATIONS.get(college_name, college_name)
try:
    from src.config import (
        get_connection,
        find_student_by_rfid,
        insert_rfid_log,
        insert_violation,
        has_student_violation_today,
    )
except Exception as e:
    get_connection = None
    find_student_by_rfid = None
    insert_rfid_log = None
    insert_violation = None
    has_student_violation_today = None
    print(f"Warning: Database config not available: {e}")


RFID_AVAILABLE = False

try:
    from src.rfid_scanner import (
        get_rfid_uid,
        start_rfid_monitoring,
        stop_rfid_monitoring,
        subscribe_to_rfid_events,
        unsubscribe_from_rfid_events,
        get_rfid_status,
        _rfid_is_present,
        set_rfid_enabled,
        is_rfid_enabled,
    )
except Exception as e:
    print(f"Warning: RFID scanner not available: {e}")

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
        return {"available": False, "present": False, "enabled": False}

    def _rfid_is_present():
        return False

    def set_rfid_enabled(enabled):
        pass

    def is_rfid_enabled():
        return False
else:
    RFID_AVAILABLE = True

app = Flask(__name__)
# Secret key for session management (can be overridden via environment variable)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'change-this-in-production')
# Flask-Mail configuration
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', '587'))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'true').lower() in {'1','true','yes','on'}
app.config['MAIL_USE_SSL'] = os.getenv('MAIL_USE_SSL', 'false').lower() in {'1','true','yes','on'}
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME', 'dress.psu@gmail.com')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD', 'ckyvhuudtqhleqkw')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER', app.config['MAIL_USERNAME'])

mail = Mail(app)

# Add charset=utf-8 to all JSON responses
@app.after_request
def add_charset_to_json(response):
    """Automatically add charset=utf-8 to JSON responses"""
    if response.content_type == 'application/json':
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response

# Register Blueprints
from routes import auth_bp, dashboards_bp, violations_bp, files_bp, camera_bp, rfid_bp, debug_bp, students_bp
app.register_blueprint(auth_bp)
app.register_blueprint(dashboards_bp)
app.register_blueprint(violations_bp)
app.register_blueprint(files_bp)
app.register_blueprint(camera_bp)
app.register_blueprint(rfid_bp)
app.register_blueprint(debug_bp)
app.register_blueprint(students_bp)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
VIOLATION_SUBDIR = 'violations'
VIOLATION_FOLDER = os.path.join(RESULT_FOLDER, VIOLATION_SUBDIR)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(VIOLATION_FOLDER, exist_ok=True)

# Alerts cache for deans (per college)
dean_alerts_cache = {}

# Load YOLOv8n model for person detection
person_model = YOLO('models/yolov8n.pt')

# Load best.pt model for dress code detection
dress_model = YOLO('models/best.pt')

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
rfid_current_uid_snapshot_saved = False  # whether a clean snapshot was saved for current UID
rfid_consecutive_non_compliant = 0  # counter for consecutive non-compliant detections
rfid_last_compliance_status = None  # track last compliance status to detect changes
rfid_enabled = False  # flag to completely disable RFID processing
rfid_violation_timeout = 30  # seconds after which violation flag resets (allows re-recording)

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
    global rfid_last_uid, rfid_present, detection_enabled, rfid_lock, rfid_last_student, rfid_consecutive_non_compliant, rfid_last_compliance_status, rfid_last_violation_ts, camera, rfid_enabled, tracker
    
    while True:
        try:
            if rfid_event_queue and rfid_enabled and camera is not None and camera.isOpened():
                event = rfid_event_queue.get(timeout=1.0)
                if event['type'] == 'uid':
                    with rfid_lock:
                        incoming_uid = event['uid']
                        is_same_uid = (rfid_present and rfid_last_uid == incoming_uid)
                        rfid_last_uid = incoming_uid
                        rfid_present = True
                        # Only reset per-scan counters on NEW UID (or after removal), not on repeated same-UID events
                        if not is_same_uid:
                            rfid_current_uid_checks = 0
                            rfid_current_uid_violated = False
                            rfid_consecutive_non_compliant = 0
                            rfid_last_compliance_status = None
                            rfid_current_uid_snapshot_saved = False
                            rfid_last_violation_ts = 0
                            # Reset tracker for new RFID scan
                            tracker = BotSORT()
                            print("DEBUG: Tracker reset for new RFID scan")
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
                    
                    # Only enable detection and capture images if RFID has a record in database
                    # Don't re-enable detection if violation was already recorded for this UID or if student already has violation today
                    if rfid_last_student is not None:
                        student_id = rfid_last_student.get('student_id')
                        # Check if student already has a violation today
                        has_violation_today = False
                        if has_student_violation_today and student_id:
                            try:
                                has_violation_today = has_student_violation_today(student_id)
                            except Exception as e:
                                print(f"DEBUG: Error checking violation today: {e}")
                        
                        with rfid_lock:
                            # If student already has violation today, mark as violated and disable detection
                            if has_violation_today:
                                rfid_current_uid_violated = True
                                detection_enabled = False
                                print(f"RFID Card detected: {event['uid']} - Student found: {rfid_last_student.get('name', 'Unknown')} - Detection DISABLED (student already has violation today)")
                            # Only enable detection if it's a new UID, no violation was recorded yet, and no violation today
                            elif not is_same_uid or not rfid_current_uid_violated:
                                detection_enabled = True
                                print(f"RFID Card detected: {event['uid']} - Student found: {rfid_last_student.get('name', 'Unknown')} - Detection ENABLED")
                            else:
                                print(f"RFID Card detected: {event['uid']} - Student found: {rfid_last_student.get('name', 'Unknown')} - Detection DISABLED (violation already recorded in this session)")
                        # Capture a clean snapshot on each RFID scan (no overlays/bounding boxes) - only once per UID
                        try:
                            snapshot = None
                            with frame_lock:
                                if current_frame is not None:
                                    snapshot = current_frame.copy()
                            save_snapshot = False
                            with rfid_lock:
                                if not rfid_current_uid_snapshot_saved:
                                    save_snapshot = True
                                    rfid_current_uid_snapshot_saved = True
                            if snapshot is not None and save_snapshot:
                                ts = int(time.time())
                                sid = (rfid_last_student or {}).get('student_id', 'unknown')
                                snap_name = f"scan_{ts}_{sid}.jpg"
                                os.makedirs(RESULT_FOLDER, exist_ok=True)
                                snap_path = os.path.join(RESULT_FOLDER, snap_name)
                                cv2.imwrite(snap_path, snapshot)
                                print(f"DEBUG: Saved clean RFID snapshot (non-violation): {snap_path}")

                                # Create a duplicate with dress code bounding boxes (no labels/text)
                                try:
                                    boxed = snapshot.copy()
                                    # Run dress model directly on the full snapshot
                                    results = dress_model(snapshot)
                                    for r in results:
                                        boxes = r.boxes
                                        if boxes is None:
                                            continue
                                        for box in boxes:
                                            conf = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                                            if conf < 0.50:
                                                continue
                                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                            # Use a constant blue box for dress items to distinguish from violations
                                            cv2.rectangle(boxed, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                                            # Label with class name and confidence
                                            try:
                                                class_id = int(box.cls[0]) if hasattr(box, 'cls') else None
                                                class_name = dress_model.names[class_id] if class_id is not None else 'item'
                                            except Exception:
                                                class_name = 'item'
                                            label_text = f"{class_name} {conf*100:.0f}%"
                                            label_scale = 0.4
                                            label_thickness = 1
                                            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, label_scale, label_thickness)
                                            pad = 3
                                            bx1, by1 = int(x1), int(y1)
                                            # Draw filled background above the box if room, otherwise inside
                                            rect_top = by1 - th - pad*2 if by1 - th - pad*2 > 0 else by1 + pad
                                            rect_bottom = rect_top + th + pad*2
                                            cv2.rectangle(boxed, (bx1, rect_top), (bx1 + tw + pad*2, rect_bottom), (255, 0, 0), -1)
                                            cv2.putText(boxed, label_text, (bx1 + pad, rect_bottom - pad), cv2.FONT_HERSHEY_SIMPLEX, label_scale, (255, 255, 255), label_thickness)
                                    boxed_name = f"scan_{ts}_{sid}_dress.jpg"
                                    boxed_path = os.path.join(RESULT_FOLDER, boxed_name)
                                    cv2.imwrite(boxed_path, boxed)
                                    print(f"DEBUG: Saved dress-boxed RFID snapshot: {boxed_path}")
                                except Exception as e:
                                    print(f"DEBUG: Error creating dress-boxed RFID snapshot: {e}")
                            else:
                                print("DEBUG: No current_frame to save RFID snapshot")
                        except Exception as e:
                            print(f"DEBUG: Error saving RFID snapshot: {e}")
                    else:
                        print(f"RFID Card detected: {event['uid']} - No student record found in database - Detection DISABLED")
                        # Disable detection if no student record found
                        with rfid_lock:
                            detection_enabled = False
                else:
                    with rfid_lock:
                        rfid_present = False
                        detection_enabled = False
                        rfid_last_student = None
                        rfid_current_uid_checks = 0
                        rfid_current_uid_violated = False
                        rfid_consecutive_non_compliant = 0
                        rfid_last_compliance_status = None
                        rfid_current_uid_snapshot_saved = False
                        rfid_last_violation_ts = 0
                        # Reset tracker when RFID card is removed
                        tracker = BotSORT()
                    print("RFID Card removed - Detection DISABLED - Tracker reset")
            elif rfid_event_queue:
                # RFID disabled or camera off, just consume events without processing
                try:
                    rfid_event_queue.get(timeout=1.0)
                except:
                    pass
        except:
            # Timeout or no events, check current status
            # Only check RFID status if RFID is enabled and camera is active
            if rfid_enabled and camera is not None and camera.isOpened():
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
                            # Reset tracker when RFID card is removed
                            tracker = BotSORT()
                        if current_present:
                            print("RFID Card present - Detection ENABLED")
                        else:
                            print("RFID Card removed - Detection DISABLED - Tracker reset")
            else:
                # RFID disabled or camera off, ensure RFID is inactive
                with rfid_lock:
                    if rfid_present:
                        rfid_present = False
                        detection_enabled = False
                        rfid_last_student = None
                        rfid_current_uid_checks = 0
                        rfid_current_uid_violated = False
                        rfid_consecutive_non_compliant = 0
                        rfid_last_compliance_status = None
                        rfid_current_uid_snapshot_saved = False
                        rfid_last_uid = None
                        # Reset tracker when RFID is disabled or camera is off
                        tracker = BotSORT()
                        print("RFID disabled or camera off - RFID forced inactive - Tracker reset")
        time.sleep(0.1)

# Initialize RFID on startup (camera will be initialized when user starts it)
# initialize_camera()  # Commented out to keep camera off by default
# initialize_rfid()  # Commented out - RFID will start with camera

# Start RFID event handler thread (but RFID monitoring won't start until camera is on)
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
                'status': 'compliant:'
            }
        else:
            compliance_status[required_item] = {
                'present': False,
                'name': item_names[required_item],
                'status': 'non-compliant:'
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
        
        # Update tracker with detections for static image FIRST to get track IDs
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
        
        # Determine gender context (from last RFID student if present)
        with rfid_lock:
            current_gender = (rfid_last_student or {}).get('gender')
        # Stage 2: Detect dress code ONLY for person with track_id == 1
        for detection in detections:
            # Only process dress detection for track_id == 1
            if detection.get('track_id') != 1:
                # Set empty dress validation for other IDs
                detection['dress_validation'] = {
                    'compliance_status': {},
                    'overall_status': 'Not tracked',
                    'compliance_percentage': 0,
                    'status_color': 'info'
                }
                detection['dress_summary'] = "Not tracked (ID != 1)"
                detection['dress_details'] = ""
                continue
            
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
            
            # Group items by status
            compliant_items = []
            noncompliant_items = []
            for item_key, item_status in compliance_status.items():
                if item_status['present']:
                    compliant_items.append(item_status['name'].lower().replace(' ', '_'))
                else:
                    noncompliant_items.append(item_status['name'].lower().replace(' ', '_'))
            
            # Format grouped details (only show categories that have items)
            detail_parts = []
            if compliant_items:
                detail_parts.append(f"compliant: {', '.join(compliant_items)}")
            if noncompliant_items:
                detail_parts.append(f"non-compliant: {', '.join(noncompliant_items)}")
            
            detection['dress_summary'] = f"{dress_validation['overall_status']} ({dress_validation['compliance_percentage']:.0f}%)"
            detection['dress_details'] = "\n".join(detail_parts)
        
        return detections
    except Exception as e:
        print(f"Error in two-stage detection: {e}")
        return []

def draw_detections(image_path, detections, output_path):
    """Draw bounding boxes on the image with tracking IDs and detailed dress code compliance"""
    try:
        # Read image
        image = cv2.imread(image_path)
        
        # Draw bounding boxes
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            track_id = detection.get('track_id', 'N/A')
            dress_summary = detection.get('dress_summary', 'No dress items detected')
            dress_validation = detection.get('dress_validation', {})
            
            # Choose color based on dress code compliance
            if dress_validation.get('status_color') == 'success':
                color = (0, 255, 0)  # Green for compliant
            elif dress_validation.get('status_color') == 'warning':
                color = (0, 255, 255)  # Yellow for partially compliant
            elif dress_validation.get('status_color') == 'danger':
                color = (0, 0, 255)  # Red for non-compliant
            else:
                # Fallback to track ID based color
                color = (0, 255, 0)  # Default green
                if track_id != 'N/A':
                    # Generate different colors for different track IDs
                    color_int = track_id * 50 % 255
                    color = (color_int, 255, 255 - color_int)
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with tracking ID and person confidence
            label1 = f"ID:{track_id} Person: {confidence:.2f}"
            
            # Draw first label (person info)
            label_size1 = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size1[1] - 10), 
                         (x1 + label_size1[0], y1), color, -1)
            cv2.putText(image, label1, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Draw dress code compliance status
            current_y = y1 - label_size1[1] - 15
            compliance_text = f"Dress Code: {dress_summary}"
            text_size = cv2.getTextSize(compliance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw background for compliance status
            cv2.rectangle(image, (x1, current_y - text_size[1] - 5), 
                         (x1 + text_size[0] + 5, current_y + 5), color, -1)
            cv2.putText(image, compliance_text, (x1 + 2, current_y - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Draw detailed dress code items with present/non-compliant status
            compliance_status = dress_validation.get('compliance_status', {})
            if compliance_status:
                current_y -= 25
                
                # Determine gender for context
                with rfid_lock:
                    current_gender = (rfid_last_student or {}).get('gender', 'male')
                
                gender_text = f"Gender: {current_gender.title()}"
                gender_size = cv2.getTextSize(gender_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                cv2.rectangle(image, (x1, current_y - gender_size[1] - 5), 
                             (x1 + gender_size[0] + 5, current_y + 5), color, -1)
                cv2.putText(image, gender_text, (x1 + 2, current_y - 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                
                # Group items by status and display
                compliant_items = []
                noncompliant_items = []
                for item_key, item_status in compliance_status.items():
                    if item_status['present']:
                        compliant_items.append(item_status['name'].lower().replace(' ', '_'))
                    else:
                        noncompliant_items.append(item_status['name'].lower().replace(' ', '_'))
                
                # Draw has items (only if present)
                if compliant_items:
                    has_text = f"has: {', '.join(compliant_items)}"
                    current_y -= 20
                    has_size = cv2.getTextSize(has_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                    cv2.rectangle(image, (x1, current_y - has_size[1] - 5), 
                                 (x1 + has_size[0] + 5, current_y + 5), color, -1)
                    cv2.putText(image, has_text, (x1 + 2, current_y - 2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                
                # Draw non-compliant items (only if present)
                if noncompliant_items:
                    noncompliant_text = f"non-compliant: {', '.join(noncompliant_items)}"
                    current_y -= 20
                    noncompliant_size = cv2.getTextSize(noncompliant_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                    cv2.rectangle(image, (x1, current_y - noncompliant_size[1] - 5), 
                                 (x1 + noncompliant_size[0] + 5, current_y + 5), color, -1)
                    cv2.putText(image, noncompliant_text, (x1 + 2, current_y - 2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            else:
                # No dress code validation available
                no_items_text = "No dress code validation available"
                text_size = cv2.getTextSize(no_items_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                cv2.rectangle(image, (x1, current_y - text_size[1] - 5), 
                             (x1 + text_size[0] + 5, current_y + 5), color, -1)
                cv2.putText(image, no_items_text, (x1 + 2, current_y - 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
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
        
        # Update tracker with detections FIRST to get track IDs
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
        
        # Determine gender context from current RFID student if available
        with rfid_lock:
            current_gender = (rfid_last_student or {}).get('gender')
        # Stage 2: Detect dress code ONLY for person with track_id == 1
        for detection in detections:
            # Only process dress detection for track_id == 1
            if detection.get('track_id') != 1:
                # Set empty dress validation for other IDs
                detection['dress_validation'] = {
                    'compliance_status': {},
                    'overall_status': 'Not tracked',
                    'compliance_percentage': 0,
                    'status_color': 'info'
                }
                detection['dress_summary'] = "Not tracked (ID != 1)"
                detection['dress_details'] = ""
                continue
            
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
            
            # Group items by status
            compliant_items = []
            noncompliant_items = []
            for item_key, item_status in compliance_status.items():
                if item_status['present']:
                    compliant_items.append(item_status['name'].lower().replace(' ', '_'))
                else:
                    noncompliant_items.append(item_status['name'].lower().replace(' ', '_'))
            
            # Format grouped details (only show categories that have items)
            detail_parts = []
            if compliant_items:
                detail_parts.append(f"compliant: {', '.join(compliant_items)}")
            if noncompliant_items:
                detail_parts.append(f"non-compliant: {', '.join(noncompliant_items)}")
            
            detection['dress_summary'] = f"{dress_validation['overall_status']} ({dress_validation['compliance_percentage']:.0f}%)"
            detection['dress_details'] = "\n".join(detail_parts)
        
        return detections
    except Exception as e:
        print(f"Error in frame detection: {e}")
        return []

def draw_detections_frame(frame, detections):
    """Draw bounding boxes on a video frame with tracking IDs and detailed dress code compliance"""
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
            
            # Draw detailed dress code items with present/non-compliant status
            compliance_status = dress_validation.get('compliance_status', {})
            if compliance_status:
                current_y -= 25
                
                # Determine gender for context
                with rfid_lock:
                    current_gender = (rfid_last_student or {}).get('gender', 'male')
                
                gender_text = f"Gender: {current_gender.title()}"
                gender_size = cv2.getTextSize(gender_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                cv2.rectangle(frame, (x1, current_y - gender_size[1] - 5), 
                             (x1 + gender_size[0] + 5, current_y + 5), color, -1)
                cv2.putText(frame, gender_text, (x1 + 2, current_y - 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                
                # Group items by status and display
                compliant_items = []
                noncompliant_items = []
                for item_key, item_status in compliance_status.items():
                    if item_status['present']:
                        compliant_items.append(item_status['name'].lower().replace(' ', '_'))
                    else:
                        noncompliant_items.append(item_status['name'].lower().replace(' ', '_'))
                
                # Draw has items (only if present)
                if compliant_items:
                    has_text = f"has: {', '.join(compliant_items)}"
                    current_y -= 20
                    has_size = cv2.getTextSize(has_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                    cv2.rectangle(frame, (x1, current_y - has_size[1] - 5), 
                                 (x1 + has_size[0] + 5, current_y + 5), color, -1)
                    cv2.putText(frame, has_text, (x1 + 2, current_y - 2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                
                # Draw non-compliant items (only if present)
                if noncompliant_items:
                    noncompliant_text = f"non-compliant: {', '.join(noncompliant_items)}"
                    current_y -= 20
                    noncompliant_size = cv2.getTextSize(noncompliant_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                    cv2.rectangle(frame, (x1, current_y - noncompliant_size[1] - 5), 
                                 (x1 + noncompliant_size[0] + 5, current_y + 5), color, -1)
                    cv2.putText(frame, noncompliant_text, (x1 + 2, current_y - 2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
        return frame
    except Exception as e:
        print(f"Error drawing detections on frame: {e}")
        return frame


def _maybe_record_violation(frame, detections, admin_user):
    """Record violation after 3 consecutive violation detections (non-compliant or partially compliant) for current RFID student.
    
    - Only records once per RFID scan session
    - Requires 3 consecutive violation detections before recording
    - Counts both NON-COMPLIANT and PARTIALLY COMPLIANT as violations
    - Only resets counter when status changes from violation to COMPLIANT
    - Does not reset counter on temporary NO_DETECTION (person out of frame)
    """
    global rfid_last_student, rfid_last_violation_ts, rfid_consecutive_non_compliant, rfid_last_compliance_status, rfid_current_uid_violated
    
    try:
        if not rfid_last_student:
            print("DEBUG: No RFID student found, skipping violation check")
            return None
            
        # Determine if any detection for this frame indicates NON-COMPLIANT or PARTIALLY COMPLIANT
        non_compliant = False
        partially_compliant = False
        violation_details = []
        current_compliance_status = None
        
        # Process detections to determine overall compliance status
        # Check ALL detections and use the worst status found
        if detections:
            for det in detections:
                dv = det.get('dress_validation') or {}
                overall_status = dv.get('overall_status')
                
                # Track the worst status found (NON-COMPLIANT > PARTIALLY COMPLIANT > COMPLIANT)
                if overall_status == 'NON-COMPLIANT':
                    non_compliant = True
                    current_compliance_status = 'NON-COMPLIANT'
                    comp = dv.get('compliance_status') or {}
                    for key, val in comp.items():
                        if not val.get('present'):
                            violation_details.append(val.get('name') or key)
                elif overall_status == 'PARTIALLY COMPLIANT' and not non_compliant:
                    # Only set partially compliant if we haven't found a non-compliant yet
                    partially_compliant = True
                    if current_compliance_status != 'NON-COMPLIANT':
                        current_compliance_status = 'PARTIALLY COMPLIANT'
                        comp = dv.get('compliance_status') or {}
                        for key, val in comp.items():
                            if not val.get('present'):
                                violation_details.append(val.get('name') or key)
                elif current_compliance_status is None:
                    # Only set to COMPLIANT if no violations found yet
                    current_compliance_status = overall_status
        else:
            # No detections - consider this as "no compliance status"
            current_compliance_status = 'NO_DETECTION'
        
        # Consider both NON-COMPLIANT and PARTIALLY COMPLIANT as violations
        has_violation = non_compliant or partially_compliant
        
        print(f"DEBUG: Frame analysis - Non-compliant: {non_compliant}, Partially compliant: {partially_compliant}, Status: {current_compliance_status}, Detections: {len(detections)}")
        
        # Only process if RFID card is present
        with rfid_lock:
            if not rfid_present:
                print("DEBUG: RFID not present, skipping violation check")
                return None
                
            # If we already issued a violation for this UID session, check timeout
            if rfid_current_uid_violated:
                # Check if enough time has passed to allow re-recording
                time_since_violation = time.time() - rfid_last_violation_ts
                if time_since_violation < rfid_violation_timeout:
                    print(f"DEBUG: Violation already recorded for this RFID session ({time_since_violation:.1f}s ago, timeout: {rfid_violation_timeout}s)")
                    return None
                else:
                    # Reset violation flag after timeout
                    rfid_current_uid_violated = False
                    print(f"DEBUG: Violation timeout reached ({time_since_violation:.1f}s), resetting violation flag")
            
            # Only reset counter if status changes from violation state (NON-COMPLIANT/PARTIALLY COMPLIANT) to COMPLIANT
            # Don't reset if changing between NON-COMPLIANT and PARTIALLY COMPLIANT (both are violations)
            if rfid_last_compliance_status is not None:
                was_violation = rfid_last_compliance_status in ['NON-COMPLIANT', 'PARTIALLY COMPLIANT']
                is_violation = current_compliance_status in ['NON-COMPLIANT', 'PARTIALLY COMPLIANT']
                
                # Reset counter only if we went from violation to compliant (or no detection)
                if was_violation and not is_violation:
                    rfid_consecutive_non_compliant = 0
                    print(f"DEBUG: Compliance status changed from violation ({rfid_last_compliance_status}) to {current_compliance_status}, resetting counter")
                elif not was_violation and is_violation:
                    # Starting a new violation sequence
                    print(f"DEBUG: Compliance status changed from {rfid_last_compliance_status} to violation ({current_compliance_status}), starting violation sequence")
            
            # Update last compliance status
            rfid_last_compliance_status = current_compliance_status
            
            # Increment counter for both non-compliant and partially compliant detections
            if has_violation:
                rfid_consecutive_non_compliant += 1
                print(f"DEBUG: Violation detection #{rfid_consecutive_non_compliant} for student {rfid_last_student.get('student_id')} (Status: {current_compliance_status})")
            else:
                # Only reset counter if fully compliant or no detections
                if current_compliance_status == 'NO_DETECTION':
                    # Don't reset counter on no detection - might be temporary
                    print(f"DEBUG: No detections for student {rfid_last_student.get('student_id')}, keeping counter at {rfid_consecutive_non_compliant}")
                elif current_compliance_status == 'COMPLIANT':
                    # Reset counter only when fully compliant
                    rfid_consecutive_non_compliant = 0
                    # Also reset the violation flag so they can be recorded again if they become non-compliant
                    rfid_current_uid_violated = False
                    print(f"DEBUG: Fully compliant detection, resetting counter and violation flag for student {rfid_last_student.get('student_id')}")
            
            # Only proceed if we have 3 consecutive violation detections (non-compliant or partially compliant)
            if rfid_consecutive_non_compliant < 3:
                print(f"DEBUG: Need {3 - rfid_consecutive_non_compliant} more consecutive violation detections")
                return None

        # Throttle: avoid spamming the same student too frequently (10 seconds)
        now_ts = time.time()
        if now_ts - rfid_last_violation_ts < 10:
            print(f"DEBUG: Throttled - last violation was {now_ts - rfid_last_violation_ts:.1f}s ago")
            return None

        # Daily limit: only one recorded violation per student per day
        student_id = (rfid_last_student or {}).get('student_id')
        if has_student_violation_today and student_id:
            if has_student_violation_today(student_id):
                print(f"DEBUG: Daily limit reached - violation already recorded today for student {student_id}")
                return None

        print(f"DEBUG: Recording violation for student {rfid_last_student.get('student_id')} after {rfid_consecutive_non_compliant} consecutive detections")
        print(f"DEBUG: Admin user: {admin_user is not None}")

        # Save enhanced proof image with annotations
        proof_name = f"violation_{int(now_ts)}_{rfid_last_student.get('student_id', 'unknown')}.jpg"
        proof_path = os.path.join(VIOLATION_FOLDER, proof_name)
        print(f"DEBUG: Proof image path: {proof_path}")
        
        try:
            os.makedirs(VIOLATION_FOLDER, exist_ok=True)
            print(f"DEBUG: Created/verified violation folder: {VIOLATION_FOLDER}")
            
            # Create an enhanced proof image with violation details
            proof_frame = frame.copy()
            print(f"DEBUG: Created proof frame copy, shape: {proof_frame.shape}")
            
            # Build violation type text for image annotation
            noncompliant = ", ".join(violation_details) if violation_details else "non-compliant items"
            violation_type_for_image = f"non-compliant: {noncompliant}"
            
            # Add violation information overlay (with word wrapping)
            violation_text = f"VIOLATION RECORDED - {violation_type_for_image}"
            timestamp_text = f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now_ts))}"
            student_text = f"Student ID: {rfid_last_student.get('student_id', 'Unknown')}"
            rfid_text = f"RFID: {rfid_last_uid}"

            print(f"DEBUG: Adding text overlays to proof image")

            # Prepare wrapping
            margin_left = 20
            margin_top = 20
            margin_right = 20
            line_spacing = 6
            title_scale = 0.7
            info_scale = 0.5
            thickness = 2
            max_text_width = proof_frame.shape[1] - (margin_left + margin_right)

            def wrap_lines(text, scale):
                words = (text or '').split()
                lines = []
                current = ''
                for w in words:
                    trial = (current + ' ' + w).strip()
                    size = cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
                    if size[0] <= max_text_width or not current:
                        current = trial
                    else:
                        lines.append(current)
                        current = w
                if current:
                    lines.append(current)
                return lines

            wrapped_title = wrap_lines(violation_text, title_scale)
            # Timestamp, student, RFID are short; keep one line each
            info_lines = [timestamp_text, student_text, rfid_text]

            # Calculate total background height
            y = margin_top
            total_height = 0
            # Title block
            for ln in wrapped_title:
                sz = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, title_scale, thickness)[0]
                total_height += sz[1] + line_spacing
            # Info lines
            for ln in info_lines:
                sz = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, info_scale, thickness)[0]
                total_height += sz[1] + line_spacing
            total_height += 5  # bottom padding

            # Draw semi-transparent background rectangle sized to content
            overlay = proof_frame.copy()
            cv2.rectangle(overlay, (10, 10), (proof_frame.shape[1] - 10, 10 + total_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, proof_frame, 0.3, 0, proof_frame)

            # Draw wrapped title lines
            y = margin_top
            for ln in wrapped_title:
                cv2.putText(proof_frame, ln, (margin_left, y + cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, title_scale, thickness)[0][1]),
                           cv2.FONT_HERSHEY_SIMPLEX, title_scale, (0, 0, 255), thickness)
                y += cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, title_scale, thickness)[0][1] + line_spacing
            # Draw info lines
            for ln in info_lines:
                cv2.putText(proof_frame, ln, (margin_left, y + cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, info_scale, thickness)[0][1]),
                           cv2.FONT_HERSHEY_SIMPLEX, info_scale, (255, 255, 255), thickness)
                y += cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, info_scale, thickness)[0][1] + line_spacing
            
            print(f"DEBUG: Added text overlays, now adding bounding boxes")
            
            # Draw bounding boxes and violation details on detected persons
            for det in detections or []:
                dv = det.get('dress_validation') or {}
                overall_status = dv.get('overall_status')
                # Draw bounding box for both NON-COMPLIANT and PARTIALLY COMPLIANT
                if overall_status in ['NON-COMPLIANT', 'PARTIALLY COMPLIANT']:
                    x1, y1, x2, y2 = det.get('bbox', [0, 0, 0, 0])
                    
                    # Draw red bounding box for non-compliant or partially compliant person
                    # Use slightly different color for partially compliant (orange) vs non-compliant (red)
                    if overall_status == 'PARTIALLY COMPLIANT':
                        box_color = (0, 165, 255)  # Orange color for partially compliant
                    else:
                        box_color = (0, 0, 255)  # Red color for non-compliant
                    
                    cv2.rectangle(proof_frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 3)
                    
                    # Add violation details
                    comp = dv.get('compliance_status') or {}
                    noncompliant_items = []
                    for key, val in comp.items():
                        if not val.get('present'):
                            noncompliant_items.append(val.get('name') or key)
                    
                    # Removed per-person non-compliant text near the bounding box as requested
            
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


        # Store relative path in DB; serve via /results/violations/<filename>
        rel_path = os.path.join(VIOLATION_SUBDIR, proof_name) if proof_path else None
        print(f"DEBUG: Database path: {rel_path}")

        if insert_violation:
            print(f"DEBUG: Attempting database insertion...")
            vid = insert_violation(rfid_last_student.get('student_id'), violation_type, rel_path)
            print(f"DEBUG: Database insertion returned: {vid}")
            if vid:
                rfid_last_violation_ts = now_ts
                with rfid_lock:
                    rfid_current_uid_violated = True
                    # Disable detection after violation is recorded
                    detection_enabled = False
                    print(f"✓ DETECTION STOPPED: Violation recorded for student {rfid_last_student.get('student_id')} - Detection is now DISABLED")
                    print(f"DEBUG: rfid_current_uid_violated={rfid_current_uid_violated}, detection_enabled={detection_enabled}")
                
                # Create violation summary for logging
                violation_summary = {
                    'violation_id': vid,
                    'student_id': rfid_last_student.get('student_id'),
                    'student_name': rfid_last_student.get('name', 'Unknown'),
                    'rfid_uid': rfid_last_uid,
                    'violation_type': violation_type,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now_ts)),
                    'proof_image': proof_name,
                    'noncompliant_items': violation_details,
                    'consecutive_detections': rfid_consecutive_non_compliant
                }
                
                print(f"VIOLATION RECORDED:")
                print(f"  - Violation ID: {vid}")
                print(f"  - Student: {rfid_last_student.get('name')} (ID: {rfid_last_student.get('student_id')})")
                print(f"  - RFID: {rfid_last_uid}")
                print(f"  - Type: {violation_type}")
                print(f"  - Proof Image: {proof_name}")
                print(f"  - Consecutive Non-Compliant Detections: {rfid_consecutive_non_compliant}")
                # Attempt to send email notification to student (with strike count)
                print(f"DEBUG: Starting email notification process...")
                try:
                    student_email = (rfid_last_student or {}).get('email')
                    student_name = (rfid_last_student or {}).get('name', 'Student')
                    student_id = (rfid_last_student or {}).get('student_id')
                    print(f"DEBUG: Initial email check - email: {student_email}, student_id: {student_id}")
                    if not student_email and (rfid_last_student or {}).get('student_id'):
                        print(f"DEBUG: No email found in rfid_last_student, attempting lookup by student_id...")
                        try:
                            from src.config import find_student_by_id as _find_student_by_id
                        except Exception as import_err:
                            print(f"DEBUG: Failed to import find_student_by_id: {import_err}")
                            _find_student_by_id = None
                        if _find_student_by_id:
                            stu = _find_student_by_id(student_id)
                            if stu:
                                student_email = stu.get('email') or student_email
                                student_name = stu.get('name') or student_name
                                student_id = stu.get('student_id') or student_id
                                print(f"DEBUG: Found student via lookup - email: {student_email}")
                            else:
                                print(f"DEBUG: Student lookup returned None for student_id: {student_id}")
                        else:
                            print(f"DEBUG: find_student_by_id function not available")
                    if student_email:
                        print(f"DEBUG: Student email found: {student_email}, proceeding with email composition...")
                        # Determine strike number (cap at 3)
                        strike_num = 1
                        try:
                            from src.config import get_student_violation_count as _get_v_cnt, get_student_violations as _get_v_list
                        except Exception:
                            _get_v_cnt = None
                            _get_v_list = None
                        if _get_v_cnt and student_id:
                            total = int(_get_v_cnt(student_id) or 1)
                            strike_num = max(1, min(3, total))
                        # Build violation list lines
                        violation_lines = []
                        if _get_v_list and student_id:
                            try:
                                vlist = _get_v_list(student_id) or []
                                for v in vlist:
                                    timestamp = v.get('timestamp')
                                    if timestamp:
                                        # Parse and format timestamp
                                        try:
                                            # Handle both datetime objects and string timestamps
                                            if isinstance(timestamp, str):
                                                # Try parsing common datetime formats
                                                from datetime import datetime
                                                try:
                                                    # Try ISO format first
                                                    dt = datetime.fromisoformat(str(timestamp).replace('Z', '+00:00'))
                                                except:
                                                    try:
                                                        # Try MySQL datetime format
                                                        dt = datetime.strptime(str(timestamp), '%Y-%m-%d %H:%M:%S')
                                                    except:
                                                        # Fallback to current time if parsing fails
                                                        dt = datetime.now()
                                            else:
                                                # Assume it's already a datetime object
                                                dt = timestamp
                                            tstr = dt.strftime('%a, %d %b %Y %I:%M %p')
                                        except Exception:
                                            # Fallback to string representation if formatting fails
                                            tstr = str(timestamp)
                                    else:
                                        tstr = 'Unknown date'
                                    vtype = str(v.get('violation_type') or '')
                                    violation_lines.append(f"{tstr} – {vtype}")
                            except Exception:
                                pass
                        dt_str = time.strftime('%a, %d %b %Y %I:%M %p', time.localtime(now_ts))
                        # Compose strike-specific subject/body
                        if strike_num == 1:
                            subject = 'Dress Code Violation - 1st Offense (Warning)'
                            offense_line = '1st Offense'
                        elif strike_num == 2:
                            subject = 'Dress Code Violation - 2nd Offense (5-day Suspension)'
                            offense_line = '2nd Offense'
                        else:
                            subject = 'Dress Code Violation - 3rd Offense (Up to 1 month Suspension)'
                            offense_line = '3rd Offense'

                        # Join violation lines outside f-string to avoid backslash issue
                        violation_text = '\n'.join(violation_lines) if violation_lines else 'No history available'
                        
                        # Prepare image attachment with CID for inline display
                        image_cid = None
                        if proof_path and os.path.exists(proof_path):
                            # Generate a unique Content-ID for the inline image
                            image_cid = f"violation_proof_{int(now_ts)}"
                            print(f"DEBUG: Image CID generated: {image_cid}")
                        
                        # Generate email body using HTML template
                        html_body = generate_violation_email_body(
                            student_name=student_name,
                            violation_datetime=dt_str,
                            strike_num=strike_num,
                            offense_line=offense_line,
                            violation_history=violation_text,
                            image_cid=image_cid
                        )
                        
                        # Create plain text fallback
                        image_attachment_text = "\n\nPROOF OF VIOLATION\nA proof image is attached to this email.\n" if image_cid else ""
                        plain_text_body = f"""DRESS CODE VIOLATION NOTIFICATION

Dear {student_name},

This is to inform you that the DRESS (Dress-code Recognition Surveillance System) detected a dress code violation on your part on {dt_str}.

Please remember that following the university dress code is part of maintaining discipline and professionalism. We ask that you correct your attire and comply on your next visit.

VIOLATION DETAILS
Current Strike Count: {strike_num} of 3
Your Current Offense: {offense_line}

Recorded Violations:
{violation_text}{image_attachment_text}
UNIVERSITY GUIDELINES
• 1st Offense – Warning
• 2nd Offense – 5-day suspension
• 3rd Offense – 2-week to 1-month suspension

ACTION REQUIRED
Please report to the Guidance Office to address this matter and complete the required procedures.

Thank you for your cooperation.

Respectfully,
DRESS Monitoring Team
Palawan State University

This is an automated notification. Please do not reply to this email."""
                        
                        try:
                            print(f"DEBUG: Creating email message...")
                            print(f"DEBUG: Mail config - Server: {app.config.get('MAIL_SERVER')}, Username: {app.config.get('MAIL_USERNAME')}")
                            print(f"DEBUG: Sender: {app.config.get('MAIL_DEFAULT_SENDER', app.config.get('MAIL_USERNAME'))}")
                            print(f"DEBUG: Recipient: {student_email}")
                            msg = Message(
                                subject=subject, 
                                recipients=[student_email], 
                                html=html_body,
                                body=plain_text_body,
                                sender=app.config.get('MAIL_DEFAULT_SENDER', app.config.get('MAIL_USERNAME'))
                            )
                            
                            # Attach proof image as inline attachment if available
                            if image_cid and proof_path and os.path.exists(proof_path):
                                try:
                                    with open(proof_path, 'rb') as img_file:
                                        msg.attach(
                                            filename=proof_name,
                                            content_type='image/jpeg',
                                            data=img_file.read(),
                                            disposition='inline',
                                            headers={'Content-ID': f'<{image_cid}>'}
                                        )
                                    print(f"DEBUG: Proof image attached with CID: {image_cid}")
                                except Exception as attach_err:
                                    print(f"DEBUG: Error attaching image: {attach_err}")
                            
                            print(f"DEBUG: Attempting to send email to {student_email}...")
                            # Flask-Mail requires application context, especially when called from background threads
                            with app.app_context():
                                mail.send(msg)
                            print(f"✓ SUCCESS: Violation email sent to {student_email}")
                        except Exception as _em:
                            print(f"✗ ERROR: Failed to send violation email to {student_email}")
                            print(f"✗ ERROR DETAILS: {type(_em).__name__}: {_em}")
                            import traceback
                            print(f"✗ TRACEBACK:\n{traceback.format_exc()}")
                    else:
                        print(f"⚠ WARNING: No student email available; skipping email notification")
                        print(f"⚠ DEBUG: student_email was: {student_email}, student_id was: {student_id}")
                except Exception as _e:
                    print(f"✗ CRITICAL ERROR in email notification process: {type(_e).__name__}: {_e}")
                    import traceback
                    print(f"✗ TRACEBACK:\n{traceback.format_exc()}")
                
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
                    _present = rfid_present
                    _student_set = (rfid_last_student is not None)
                    _violated = rfid_current_uid_violated
                    # Detection is enabled only if: detection flag is True, RFID is present, student is set, AND no violation was recorded yet
                    rfid_detection_enabled = detection_enabled and _present and _student_set and not _violated
                
                # In test mode, detection always runs regardless of RFID
                detection_enabled_for_frame = rfid_detection_enabled or test_mode_active
                
                if detection_enabled_for_frame:
                    # Keep a clean copy of the frame (without live overlays) for proof image
                    clean_frame_for_proof = frame.copy()
                    
                    detections = detect_persons_frame_with_dress(frame)
                    frame = draw_detections_frame(frame, detections)
                    
                    # Attempt to record violation using the CLEAN frame (no overlay text)
                    # Note: admin_user is None in background thread, will be handled in violation function
                    _maybe_record_violation(clean_frame_for_proof, detections, None)
                    
                    # Status overlay removed as requested
                    pass
                else:
                    # Status overlay removed as requested
                    pass
                
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

def get_programs_by_college(college):
    """Return all programs for a given college based on the program-to-college mapping.
    Uses the exact college names from the database enum."""
    college_program_map = {
        'College of Sciences': [
            'Bachelor of Science in Biology',
            'Bachelor of Science in Marine Biology',
            'Bachelor of Science in Computer Science',
            'Bachelor of Science in Environmental Science',
            'Bachelor of Science in Information Technology'
        ],
        'College of Arts and Humanities': [
            'Bachelor of Arts in Communication',
            'Bachelor of Arts in Political Science',
            'Bachelor of Arts in Philippine Studies',
            'Bachelor of Science in Social Work',
            'Bachelor of Science in Psychology'
        ],
        'College of Business and Accountancy': [
            'Bachelor of Science in Accountancy',
            'Bachelor of Science in Management Accounting',
            'Bachelor of Science in Business Administration',
            'Bachelor of Science in Entrepreneurship',
            'Bachelor of Science in Public Administration'
        ],
        'College of Criminal Justice Education': [
            'Bachelor of Science in Criminology'
        ],
        'College of Engineering': [
            'Bachelor of Science in Civil Engineering',
            'Bachelor of Science in Electrical Engineering',
            'Bachelor of Science in Mechanical Engineering',
            'Bachelor of Science in Petroleum Engineering'
        ],
        'College of Architecture and Design': [
            'Bachelor of Science in Architecture'
        ],
        'College of Hospitality Management and Tourism': [
            'Bachelor of Science in Hospitality Management',
            'Bachelor of Science in Tourism Management'
        ],
        'College of Nursing and Health Sciences': [
            'Bachelor of Science in Nursing',
            'Bachelor of Science in Midwifery'
        ],
        'College of Teacher Education': [
            'Bachelor of Elementary Education',
            'Bachelor of Secondary Education',
            'Bachelor of Physical Education'
        ]
    }
    # Handle case-insensitive matching and common variations
    college_lower = college.lower() if college else ''
    for key, programs in college_program_map.items():
        if key.lower() == college_lower:
            return programs
    # Also handle some common variations
    if 'arts' in college_lower and 'humanities' in college_lower:
        return college_program_map.get('College of Arts and Humanities', [])
    if 'criminal' in college_lower and 'justice' in college_lower:
        return college_program_map.get('College of Criminal Justice Education', [])
    if 'nursing' in college_lower or 'health' in college_lower:
        return college_program_map.get('College of Nursing and Health Sciences', [])
    return college_program_map.get(college, [])

if __name__ == '__main__':
    print("Starting Flask app for person detection with Bot-SORT tracking...")
    print("Camera will auto-start when the app launches")
    print("RFID monitoring will start automatically")
    print("Detection will only work when RFID card is present")
    print("Make sure you have installed: pip install ultralytics opencv-python flask pillow scipy pyscard")
    # Start alerts checker in background
    try:
        import threading as _t
        _t.Thread(target=(lambda: __import__('time') or None), daemon=True)
    except Exception:
        pass
    app.run(debug=True, host='0.0.0.0', port=5000)
