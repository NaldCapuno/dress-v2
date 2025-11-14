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
from src.botsort_tracker import BotSORT
from werkzeug.security import check_password_hash
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

def generate_violation_email_body(student_name, violation_datetime, strike_num, offense_line, violation_history, image_cid=None):
    """
    Generate HTML email body for dress code violation notification.
    Mobile-responsive design with inline CSS.
    
    Args:
        student_name (str): Name of the student
        violation_datetime (str): Date and time of the violation
        strike_num (int): Current strike number (1-3)
        offense_line (str): Text description of the offense (e.g., "1st Offense")
        violation_history (str): Formatted list of previous violations
        image_cid (str, optional): Content-ID (CID) for inline image attachment
    
    Returns:
        str: HTML formatted email body
    """
    # Format violation history as HTML list (matching web app colors)
    if violation_history and violation_history != 'No history available':
        history_items = violation_history.split('\n')
        formatted_history = '<ul style="margin: 10px 0; padding-left: 20px;">'
        for item in history_items:
            if item.strip():
                formatted_history += f'<li style="margin: 5px 0; color: #374151; font-size: 14px;">{item.strip()}</li>'
        formatted_history += '</ul>'
    else:
        formatted_history = '<p style="color: #9ca3af; font-style: italic; font-size: 14px;">No history available</p>'
    
    # Determine strike color based on number (matching web app colors)
    if strike_num == 1:
        strike_color = '#f59e0b'  # Warning (matches web app)
    elif strike_num == 2:
        strike_color = '#ef4444'  # Error (matches web app)
    else:
        strike_color = '#e55100'  # Accent-dark (matches web app)
    
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dress Code Violation Notification</title>
</head>
<body style="margin: 0; padding: 0; font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif; background-color: #f8fafc;">
    <table role="presentation" style="width: 100%; border-collapse: collapse; background-color: #f8fafc; padding: 20px 0;">
        <tr>
            <td align="center" style="padding: 20px 10px;">
                <table role="presentation" style="max-width: 600px; width: 100%; border-collapse: collapse; background-color: #ffffff; border-radius: 8px; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);">
                    <!-- Header -->
                    <tr>
                        <td style="background: linear-gradient(135deg, #2ca9e1 0%, #1e7bb8 100%); padding: 30px 20px; text-align: center; border-radius: 8px 8px 0 0;">
                            <h1 style="margin: 0; color: #ffffff; font-size: 24px; font-weight: 600; letter-spacing: 0.5px;">
                                DRESS CODE VIOLATION
                            </h1>
                            <p style="margin: 10px 0 0 0; color: #ffffff; font-size: 14px; opacity: 0.95;">
                                Dress-code Recognition Surveillance System
                            </p>
                        </td>
                    </tr>
                    
                    <!-- Content -->
                    <tr>
                        <td style="padding: 30px 20px;">
                            <p style="margin: 0 0 15px 0; color: #374151; font-size: 16px; line-height: 1.6;">
                                Dear <strong style="color: #2ca9e1;">{student_name}</strong>,
                            </p>
                            <p style="margin: 0 0 20px 0; color: #4b5563; font-size: 15px; line-height: 1.6;">
                                This is to inform you that the DRESS (Dress-code Recognition Surveillance System) detected a dress code violation on your part on <strong style="color: #374151;">{violation_datetime}</strong>.
                            </p>
                            <p style="margin: 0 0 25px 0; color: #4b5563; font-size: 15px; line-height: 1.6;">
                                Please remember that following the university dress code is part of maintaining discipline and professionalism. We ask that you correct your attire and comply on your next visit.
                            </p>
                            
                            <!-- Violation Details Box -->
                            <div style="background-color: #f8fafc; border-left: 4px solid {strike_color}; padding: 20px; margin: 25px 0; border-radius: 4px;">
                                <h2 style="margin: 0 0 15px 0; color: #1f2937; font-size: 18px; font-weight: 600;">
                                    Violation Details
                                </h2>
                                <table role="presentation" style="width: 100%; border-collapse: collapse;">
                                    <tr>
                                        <td style="padding: 8px 0; color: #6b7280; font-size: 14px; width: 50%;">Current Strike Count:</td>
                                        <td style="padding: 8px 0; color: {strike_color}; font-size: 16px; font-weight: 600;">{strike_num} of 3</td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 8px 0; color: #6b7280; font-size: 14px;">Your Current Offense:</td>
                                        <td style="padding: 8px 0; color: #374151; font-size: 14px; font-weight: 500;">{offense_line}</td>
                                    </tr>
                                </table>
                                <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #e5e7eb;">
                                    <p style="margin: 0 0 10px 0; color: #6b7280; font-size: 14px; font-weight: 500;">Recorded Violations:</p>
                                    {violation_history}
                                </div>
                            </div>
                            
                            <!-- Proof Image -->
                            {proof_image_section}
                            
                            <!-- Guidelines Box -->
                            <div style="background-color: #fff7ed; border: 1px solid #f25a04; padding: 20px; margin: 25px 0; border-radius: 4px;">
                                <h2 style="margin: 0 0 15px 0; color: #1f2937; font-size: 18px; font-weight: 600;">
                                    University Guidelines
                                </h2>
                                <ul style="margin: 0; padding-left: 20px; color: #4b5563; font-size: 14px; line-height: 1.8;">
                                    <li style="margin: 5px 0;"><strong style="color: #f25a04;">1st Offense</strong> – Warning</li>
                                    <li style="margin: 5px 0;"><strong style="color: #f25a04;">2nd Offense</strong> – 5-day suspension</li>
                                    <li style="margin: 5px 0;"><strong style="color: #f25a04;">3rd Offense</strong> – 2-week to 1-month suspension</li>
                                </ul>
                            </div>
                            
                            <!-- Action Required -->
                            <div style="background-color: #fef2f2; border-left: 4px solid #ef4444; padding: 15px 20px; margin: 25px 0; border-radius: 4px;">
                                <p style="margin: 0; color: #991b1b; font-size: 15px; font-weight: 600;">
                                    ⚠️ Action Required
                                </p>
                                <p style="margin: 10px 0 0 0; color: #7f1d1d; font-size: 14px; line-height: 1.6;">
                                    Please report to the Guidance Office to address this matter and complete the required procedures.
                                </p>
                            </div>
                            
                            <p style="margin: 25px 0 0 0; color: #4b5563; font-size: 15px; line-height: 1.6;">
                                Thank you for your cooperation.
                            </p>
                        </td>
                    </tr>
                    
                    <!-- Footer -->
                    <tr>
                        <td style="background-color: #f8fafc; padding: 25px 20px; text-align: center; border-radius: 0 0 8px 8px; border-top: 1px solid #e5e7eb;">
                            <p style="margin: 0 0 10px 0; color: #374151; font-size: 15px; font-weight: 500;">
                                Respectfully,
                            </p>
                            <p style="margin: 0 0 5px 0; color: #2ca9e1; font-size: 14px; font-weight: 600;">
                                DRESS Monitoring Team
                            </p>
                            <p style="margin: 0; color: #6b7280; font-size: 13px;">
                                Palawan State University
                            </p>
                            <p style="margin: 20px 0 0 0; color: #9ca3af; font-size: 12px; font-style: italic;">
                                This is an automated notification. Please do not reply to this email.
                            </p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""
    
    # Generate proof image section if image CID is provided
    if image_cid:
        proof_image_section = f"""
                            <div style="background-color: #f8fafc; padding: 20px; margin: 25px 0; border-radius: 4px; border: 1px solid #e5e7eb;">
                                <h2 style="margin: 0 0 15px 0; color: #1f2937; font-size: 18px; font-weight: 600;">
                                    Proof of Violation
                                </h2>
                                <div style="text-align: center; margin: 15px 0;">
                                    <img src="cid:{image_cid}" alt="Violation Proof Image" style="max-width: 100%; height: auto; border-radius: 4px; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); border: 1px solid #e5e7eb;" />
                                </div>
                                <p style="margin: 10px 0 0 0; color: #6b7280; font-size: 12px; text-align: center; font-style: italic;">
                                    Proof image attached to this email
                                </p>
                            </div>"""
    else:
        proof_image_section = ""
    
    return html_template.format(
        student_name=student_name,
        violation_datetime=violation_datetime,
        strike_num=strike_num,
        offense_line=offense_line,
        violation_history=formatted_history,
        strike_color=strike_color,
        proof_image_section=proof_image_section
    )

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

@app.route('/osas/colleges', methods=['GET'])
def osas_colleges():
    """Return distinct colleges for OSAS filtering."""
    conn = get_connection() if get_connection else None
    if conn is None:
        return jsonify({'success': False, 'error': 'DB not configured'}), 500
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT COALESCE(college,'') AS college FROM students WHERE COALESCE(college,'')<>'' ORDER BY college ASC"
            )
            rows = cur.fetchall() or []
        colleges = [r.get('college') for r in rows if r.get('college')]
        return jsonify({'success': True, 'colleges': colleges})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        conn.close()

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

@app.route('/osas/programs', methods=['GET'])
def osas_programs():
    """Return distinct programs for OSAS filtering (optionally filtered by college).
    Returns all enum values from the database schema, optionally filtered by college mapping."""
    college = request.args.get('college')
    conn = get_connection() if get_connection else None
    if conn is None:
        return jsonify({'success': False, 'error': 'DB not configured'}), 500
    try:
        # Get all enum values from the database schema
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COLUMN_TYPE 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = DATABASE() 
                AND TABLE_NAME = 'students' 
                AND COLUMN_NAME = 'program'
            """)
            enum_row = cur.fetchone()
            
            if enum_row and enum_row.get('COLUMN_TYPE'):
                # Parse enum values from format: enum('value1','value2',...)
                enum_str = enum_row.get('COLUMN_TYPE', '')
                enum_values = re.findall(r"'([^']+)'", enum_str)
                all_programs = sorted(enum_values)
            else:
                # Fallback: query from students table
                cur.execute(
                    "SELECT DISTINCT COALESCE(program,'') AS program FROM students WHERE COALESCE(program,'')<>'' ORDER BY program ASC"
                )
                rows = cur.fetchall() or []
                all_programs = [r.get('program') for r in rows if r.get('program')]
            
            # If college filter is provided, return all programs for that college based on mapping
            if college:
                college_programs = get_programs_by_college(college)
                # Filter to only include programs that are in both the enum and the college mapping
                programs = [p for p in all_programs if p in college_programs]
            else:
                programs = all_programs
                
        return jsonify({'success': True, 'programs': programs})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/guidance', methods=['GET'])
def guidance_dashboard():
    """Guidance dashboard - only accessible to admins with role 'guidance'."""
    admin = session.get('admin') or {}
    role = str(admin.get('role') or '').lower()
    if role != 'guidance':
        return redirect(url_for('login'))
    return render_template('guidance_dashboard.html')

@app.route('/guiance', methods=['GET'])
def guidance_alias():
    """Alias path for guidance (handles common misspelling)."""
    admin = session.get('admin') or {}
    role = str(admin.get('role') or '').lower()
    if role != 'guidance':
        return redirect(url_for('login'))
    return redirect(url_for('guidance_dashboard'))

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
    
    # Hardcoded programs by college
    if college == 'College of Sciences':
        programs = [
            'Bachelor of Science in Biology',
            'Bachelor of Science in Marine Biology',
            'Bachelor of Science in Computer Science',
            'Bachelor of Science in Environmental Science',
            'Bachelor of Science in Information Technology'
        ]
        return jsonify({'success': True, 'programs': programs})
    
    elif college == 'College of Arts And Humanities':
        programs = [
            'Bachelor of Arts in Communication',
            'Bachelor of Arts in Political Science',
            'Bachelor of Arts in Philippine Studies',
            'Bachelor of Science in Social Work',
            'Bachelor of Science in Psychology'
        ]
        return jsonify({'success': True, 'programs': programs})
    
    elif college == 'College of Business and Accountancy':
        programs = [
            'Bachelor of Science in Accountancy',
            'Bachelor of Science in Management Accounting',
            'Bachelor of Science in Business Administration',
            'Bachelor of Science in Entrepreneurship',
            'Bachelor of Science in Public Administration'
        ]
        return jsonify({'success': True, 'programs': programs})
    
    elif college == 'Criminal Justice and Education':
        programs = [
            'Bachelor of Science in Criminology'
        ]
        return jsonify({'success': True, 'programs': programs})
    
    elif college == 'College of Engineering':
        programs = [
            'Bachelor of Science in Civil Engineering',
            'Bachelor of Science in Electrical Engineering',
            'Bachelor of Science in Mechanical Engineering',
            'Bachelor of Science in Petroleum Engineering'
        ]
        return jsonify({'success': True, 'programs': programs})
    
    elif college == 'College of Architecture and Design':
        programs = [
            'Bachelor of Science in Architecture'
        ]
        return jsonify({'success': True, 'programs': programs})
    
    elif college == 'College of Hospitality Management and Tourism':
        programs = [
            'Bachelor of Science in Hospitality Management',
            'Bachelor of Science in Tourism Management'
        ]
        return jsonify({'success': True, 'programs': programs})
    
    elif college == 'Nursing and Health Sciences':
        programs = [
            'Bachelor of Science in Nursing',
            'Bachelor of Science in Midwifery'
        ]
        return jsonify({'success': True, 'programs': programs})
    
    elif college == 'College of Teacher Education':
        programs = [
            'Bachelor of Elementary Education',
            'Bachelor of Secondary Education',
            'Bachelor of Physical Education'
        ]
        return jsonify({'success': True, 'programs': programs})
    
    conn = get_connection() if get_connection else None
    if conn is None:
        return jsonify({'success': False, 'error': 'DB not configured'}), 500
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT COALESCE(program,'') AS program FROM students WHERE college=%s AND COALESCE(program,'')<>'' ORDER BY program ASC",
                (college,)
            )
            rows = cur.fetchall() or []
        programs = [r.get('program') for r in rows if r.get('program')]
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
        # Show all statuses by default; only filter if provided
        status_filter = request.args.get('status')
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
            # For dean view, treat 'forwarded_dean' as 'pending' in filters
            if status_filter == 'pending':
                where.append("(v.status = 'pending' OR v.status = 'forwarded_dean')")
            else:
                where.append("v.status = %s")
                params.append(status_filter)
        # Enforce dean operates per-college: require college filter
        if not college:
            return jsonify({'success': True, 'rows': [], 'total': 0})
        where.append("s.college = %s")
        params.append(college)
        if program:
            where.append("s.program = %s")
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
            "SELECT v.violation_id, v.student_id, v.violation_type, v.timestamp, v.image_proof, "
            "CASE WHEN v.status = 'forwarded_dean' THEN 'pending' ELSE v.status END AS status, "
            "s.name, s.gender, s.program, s.college "
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
        print(f"Dean status update: violation_id={violation_id}, status={status}, data={data}")
        allowed = {"pending", "forwarded_guidance", "resolved"}
        if status not in allowed:
            return jsonify({'success': False, 'error': 'Invalid status'}), 400
        conn = get_connection() if get_connection else None
        if conn is None:
            return jsonify({'success': False, 'error': 'DB not configured'}), 500
        with conn.cursor() as cur:
            cur.execute("UPDATE violations SET status=%s WHERE violation_id=%s", (status, violation_id))
            print(f"Updated violation {violation_id} to status {status}")
        conn.close()
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error updating violation status: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/dean/notifications', methods=['GET'])
def dean_notifications():
    """Recent violations for the dean's college, newest first."""
    try:
        # Default to pending > 3 days to surface actionable notifications
        status_filter = request.args.get('status', 'pending')
        college = request.args.get('college') or ((session.get('admin') or {}).get('college'))
        if not college:
            return jsonify({'success': True, 'rows': []})
        conn = get_connection() if get_connection else None
        if conn is None:
            return jsonify({'success': False, 'error': 'DB not configured'}), 500
        where = ["s.college = %s"]
        params = [college]
        if status_filter:
            # For dean view, treat 'pending' filter to include both 'pending' and 'forwarded_dean'
            if status_filter == 'pending':
                where.append("(v.status = 'pending' OR v.status = 'forwarded_dean')")
            else:
                where.append("v.status = %s")
                params.append(status_filter)
        # Older than 3 days by default
        where.append("v.timestamp < NOW() - INTERVAL 3 DAY")
        where_sql = " WHERE " + " AND ".join(where)
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT v.violation_id, v.student_id, v.violation_type, v.timestamp, v.image_proof,
                       CASE WHEN v.status = 'forwarded_dean' THEN 'pending' ELSE v.status END AS status,
                       s.name, s.gender, s.program, s.college
                FROM violations v
                LEFT JOIN students s ON v.student_id = s.student_id
                {where_sql}
                ORDER BY v.timestamp DESC
                LIMIT 50
                """,
                params,
            )
            rows = cur.fetchall() or []
        conn.close()
        return jsonify({'success': True, 'rows': rows, 'total': len(rows)})
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
        # Show all statuses by default; only filter if provided
        status_filter = request.args.get('status')
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
            where.append("s.program = %s")
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
                f"SELECT COALESCE(s.program,'Unknown') AS label, COUNT(*) AS cnt FROM violations v LEFT JOIN students s ON v.student_id=s.student_id{where_sql} GROUP BY label ORDER BY cnt DESC",
                params,
            )
            by_program = cur.fetchall() or []

            cur.execute(
                f"SELECT LOWER(COALESCE(s.gender,'')) AS label, COUNT(*) AS cnt FROM violations v LEFT JOIN students s ON v.student_id=s.student_id{where_sql} GROUP BY label",
                params,
            )
            by_gender = cur.fetchall() or []

            # by_status must include the students join because filters may reference s.*
            # For dean view, transform 'forwarded_dean' to 'pending' in status counts
            cur.execute(
                f"SELECT CASE WHEN v.status = 'forwarded_dean' THEN 'pending' ELSE v.status END AS label, COUNT(*) AS cnt FROM violations v LEFT JOIN students s ON v.student_id=s.student_id{where_sql} GROUP BY CASE WHEN v.status = 'forwarded_dean' THEN 'pending' ELSE v.status END",
                params,
            )
            by_status = cur.fetchall() or []

            # Provide by_college as well for overview usage (will be a single bucket for the dean's college)
            cur.execute(
                f"SELECT COALESCE(s.college,'Unknown') AS label, COUNT(*) AS cnt FROM violations v LEFT JOIN students s ON v.student_id=s.student_id{where_sql} GROUP BY label ORDER BY cnt DESC",
                params,
            )
            by_college = cur.fetchall() or []
        conn.close()
        return jsonify({'success': True, 'total': total, 'by_program': by_program, 'by_gender': by_gender, 'by_status': by_status, 'by_college': by_college})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/dean/alerts', methods=['GET'])
def dean_alerts():
    """Return alert info for the dean's college when any students have pending >3 days."""
    try:
        college = request.args.get('college') or ((session.get('admin') or {}).get('college'))
        if not college:
            return jsonify({'success': True, 'alert': False, 'num_students': 0, 'sample': [], 'notification_triggered': False})
        # Compute live to ensure up-to-date results (cache may be empty)
        conn = get_connection() if get_connection else None
        if conn is None:
            # Fallback to cache if DB not available
            info = dean_alerts_cache.get(college) or {'alert': False, 'num_students': 0, 'sample': [], 'notification_triggered': False}
            return jsonify({'success': True, **info})
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(DISTINCT v.student_id) AS num_students
                FROM violations v
                LEFT JOIN students s ON v.student_id = s.student_id
                WHERE (v.status = 'pending' OR v.status = 'forwarded_dean') AND v.timestamp < NOW() - INTERVAL 3 DAY AND s.college = %s
                """,
                (college,)
            )
            row = cur.fetchone() or {}
            num_students = int(row.get('num_students') or 0)
            
            # Trigger notification when there is at least 1 student
            notification_triggered = num_students >= 1
            
            # Show notification if it was ever triggered (current logic uses live count)
            alert = notification_triggered
            
            # Do not auto-update statuses in dean notifications

            sample = []
            if alert:
                cur.execute(
                    """
                    SELECT DISTINCT v.student_id, s.name, s.program
                    FROM violations v
                    LEFT JOIN students s ON v.student_id = s.student_id
                    WHERE (v.status = 'pending' OR v.status = 'forwarded_dean') AND v.timestamp < NOW() - INTERVAL 3 DAY AND s.college = %s
                    ORDER BY s.name ASC
                    LIMIT 10
                    """,
                    (college,)
                )
                sample = cur.fetchall() or []
        conn.close()
        return jsonify({'success': True, 'alert': alert, 'num_students': num_students, 'sample': sample, 'notification_triggered': notification_triggered})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/dean/alerts/students', methods=['GET'])
def dean_alert_students():
    """Return distinct students with pending violations older than 3 days for the dean's college."""
    try:
        college = request.args.get('college') or ((session.get('admin') or {}).get('college'))
        limit = int(request.args.get('limit', '500'))
        if not college:
            return jsonify({'success': True, 'rows': []})
        conn = get_connection() if get_connection else None
        if conn is None:
            return jsonify({'success': False, 'error': 'DB not configured'}), 500
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT v.student_id, s.name, s.program
                FROM violations v
                LEFT JOIN students s ON v.student_id = s.student_id
                WHERE (v.status = 'pending' OR v.status = 'forwarded_dean') AND v.timestamp < NOW() - INTERVAL 3 DAY AND s.college = %s
                ORDER BY s.name ASC
                LIMIT %s
                """,
                (college, limit)
            )
            rows = cur.fetchall() or []
        conn.close()
        return jsonify({'success': True, 'rows': rows})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/dean/trend', methods=['GET'])
def dean_trend():
    try:
        start_dt = request.args.get('start')
        end_dt = request.args.get('end')
        academic_year = request.args.get('academic_year')
        semester = request.args.get('semester')
        # Show all statuses by default; only filter if provided
        status_filter = request.args.get('status')
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
            where.append("s.program = %s")
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


# ------------------ OSAS endpoints (university-wide oversight) ------------------
@app.route('/osas/violations', methods=['GET'])
def osas_get_violations():
    """List violations for OSAS review (university-wide)."""
    try:
        # Show all statuses by default; only filter if provided
        status_filter = request.args.get('status')
        start_dt = request.args.get('start')
        end_dt = request.args.get('end')
        academic_year = request.args.get('academic_year')
        semester = request.args.get('semester')
        college = request.args.get('college')
        program = request.args.get('program')
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 50))
        offset = max(0, (page - 1) * page_size)

        conn = get_connection() if get_connection else None
        if conn is None:
            return jsonify({'success': False, 'error': 'DB not configured'}), 500

        where = []
        params = []
        if status_filter:
            where.append("v.status = %s")
            params.append(status_filter)
        if college:
            where.append("s.college = %s")
            params.append(college)
        if program:
            where.append("s.program = %s")
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
            "SELECT v.violation_id, v.student_id, v.violation_type, v.timestamp, v.image_proof, v.status, "
            "s.name, s.gender, s.program, s.college "
            "FROM violations v LEFT JOIN students s ON v.student_id = s.student_id"
        )

        with conn.cursor() as cur:
            # Auto-forward pending >3 days to dean when OSAS views violations
            try:
                cur.execute(
                    """
                    UPDATE violations
                    SET status = 'forwarded_dean'
                    WHERE status = 'pending' AND timestamp < NOW() - INTERVAL 3 DAY
                    """
                )
                if cur.rowcount:
                    try:
                        conn.commit()
                    except Exception as _ce:
                        print(f"OSAS auto-forward commit failed: {_ce}")
            except Exception as _e:
                print(f"OSAS auto-forward failed: {_e}")
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


# ------------------ Guidance endpoints (university-wide counseling) ------------------
@app.route('/guidance/violations', methods=['GET'])
def guidance_get_violations():
    """List violations for Guidance review (university-wide)."""
    try:
        status_filter = request.args.get('status')
        start_dt = request.args.get('start')
        end_dt = request.args.get('end')
        academic_year = request.args.get('academic_year')
        semester = request.args.get('semester')
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 50))
        offset = max(0, (page - 1) * page_size)

        conn = get_connection() if get_connection else None
        if conn is None:
            return jsonify({'success': False, 'error': 'DB not configured'}), 500

        where = []
        params = []
        if status_filter:
            where.append("v.status = %s")
            params.append(status_filter)
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
            "SELECT v.violation_id, v.student_id, v.violation_type, v.timestamp, v.image_proof, v.status, "
            "s.name, s.gender, s.program, s.college "
            "FROM violations v LEFT JOIN students s ON v.student_id = s.student_id"
        )

        with conn.cursor() as cur:
            # Auto-forward pending >3 days to dean when Guidance views violations
            try:
                cur.execute(
                    """
                    UPDATE violations
                    SET status = 'forwarded_dean'
                    WHERE status = 'pending' AND timestamp < NOW() - INTERVAL 3 DAY
                    """
                )
                if cur.rowcount:
                    try:
                        conn.commit()
                    except Exception as _ce:
                        print(f"Guidance auto-forward commit failed: {_ce}")
            except Exception as _e:
                print(f"Guidance auto-forward failed: {_e}")
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


@app.route('/guidance/violation/<int:violation_id>/status', methods=['POST'])
def guidance_update_violation_status(violation_id: int):
    """Guidance can set pending or resolved."""
    try:
        data = request.get_json(silent=True) or {}
        status = str(data.get('status') or '').strip().lower()
        print(f"Guidance status update: violation_id={violation_id}, status={status}, data={data}")
        allowed = {"pending", "resolved"}
        if status not in allowed:
            return jsonify({'success': False, 'error': 'Invalid status'}), 400
        conn = get_connection() if get_connection else None
        if conn is None:
            return jsonify({'success': False, 'error': 'DB not configured'}), 500
        with conn.cursor() as cur:
            cur.execute("UPDATE violations SET status=%s WHERE violation_id=%s", (status, violation_id))
            print(f"Updated violation {violation_id} to status {status}")
        conn.close()
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error updating violation status: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/guidance/analytics', methods=['GET'])
def guidance_analytics():
    """Aggregate analytics for Guidance view (university-wide)."""
    try:
        start_dt = request.args.get('start')
        end_dt = request.args.get('end')
        academic_year = request.args.get('academic_year')
        semester = request.args.get('semester')
        status_filter = request.args.get('status')

        conn = get_connection() if get_connection else None
        if conn is None:
            return jsonify({'success': False, 'error': 'DB not configured'}), 500

        where = []
        params = []
        if status_filter:
            where.append("v.status = %s")
            params.append(status_filter)
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
                f"SELECT COALESCE(s.college,'Unknown') AS label, COUNT(*) AS cnt FROM violations v LEFT JOIN students s ON v.student_id=s.student_id{where_sql} GROUP BY label ORDER BY cnt DESC",
                params,
            )
            by_college = cur.fetchall() or []

            cur.execute(
                f"SELECT COALESCE(s.program,'Unknown') AS label, COUNT(*) AS cnt FROM violations v LEFT JOIN students s ON v.student_id=s.student_id{where_sql} GROUP BY label ORDER BY cnt DESC",
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
        return jsonify({'success': True, 'total': total, 'by_college': by_college, 'by_program': by_program, 'by_gender': by_gender, 'by_status': by_status})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/guidance/trend', methods=['GET'])
def guidance_trend():
    try:
        start_dt = request.args.get('start')
        end_dt = request.args.get('end')
        academic_year = request.args.get('academic_year')
        semester = request.args.get('semester')
        status_filter = request.args.get('status')
        group_by = request.args.get('group_by', 'day')

        conn = get_connection() if get_connection else None
        if conn is None:
            return jsonify({'success': False, 'error': 'DB not configured'}), 500

        where = []
        params = []
        if status_filter:
            where.append("v.status = %s")
            params.append(status_filter)
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

        if group_by == 'month':
            group_expr = "DATE_FORMAT(v.timestamp, '%Y-%m')"
            order_expr = "DATE_FORMAT(v.timestamp, '%Y-%m')"
        elif group_by == 'week':
            group_expr = "YEARWEEK(v.timestamp, 3)"
            order_expr = "YEARWEEK(v.timestamp, 3)"
        else:
            group_expr = "DATE(v.timestamp)"
            order_expr = "DATE(v.timestamp)"

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

@app.route('/osas/violation/<int:violation_id>/status', methods=['POST'])
def osas_update_violation_status(violation_id: int):
    """OSAS can forward to dean, guidance, or resolve."""
    try:
        data = request.get_json(silent=True) or {}
        status = str(data.get('status') or '').strip().lower()
        print(f"OSAS status update: violation_id={violation_id}, status={status}, data={data}")
        # OSAS cannot forward to guidance (only dean). Remove forwarded_guidance from allowed.
        allowed = {"pending", "forwarded_dean", "resolved"}
        if status not in allowed:
            return jsonify({'success': False, 'error': 'Invalid status'}), 400
        conn = get_connection() if get_connection else None
        if conn is None:
            return jsonify({'success': False, 'error': 'DB not configured'}), 500
        with conn.cursor() as cur:
            cur.execute("UPDATE violations SET status=%s WHERE violation_id=%s", (status, violation_id))
            print(f"Updated violation {violation_id} to status {status}")
        conn.close()
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error updating violation status: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/osas/analytics', methods=['GET'])
def osas_analytics():
    """Aggregate analytics for OSAS view (university-wide)."""
    try:
        start_dt = request.args.get('start')
        end_dt = request.args.get('end')
        academic_year = request.args.get('academic_year')
        semester = request.args.get('semester')
        # Show all statuses by default; only filter if provided
        status_filter = request.args.get('status')
        college = request.args.get('college')
        program = request.args.get('program')

        conn = get_connection() if get_connection else None
        if conn is None:
            return jsonify({'success': False, 'error': 'DB not configured'}), 500

        where = []
        params = []
        if status_filter:
            where.append("v.status = %s")
            params.append(status_filter)
        if college:
            where.append("s.college = %s")
            params.append(college)
        if program:
            where.append("s.program = %s")
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
                f"SELECT COALESCE(s.college,'Unknown') AS label, COUNT(*) AS cnt FROM violations v LEFT JOIN students s ON v.student_id=s.student_id{where_sql} GROUP BY label ORDER BY cnt DESC",
                params,
            )
            by_college = cur.fetchall() or []

            cur.execute(
                f"SELECT COALESCE(s.program,'Unknown') AS label, COUNT(*) AS cnt FROM violations v LEFT JOIN students s ON v.student_id=s.student_id{where_sql} GROUP BY label ORDER BY cnt DESC",
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
        return jsonify({'success': True, 'total': total, 'by_college': by_college, 'by_program': by_program, 'by_gender': by_gender, 'by_status': by_status})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/osas/trend', methods=['GET'])
def osas_trend():
    try:
        start_dt = request.args.get('start')
        end_dt = request.args.get('end')
        academic_year = request.args.get('academic_year')
        semester = request.args.get('semester')
        # Show all statuses by default; only filter if provided
        status_filter = request.args.get('status')
        group_by = request.args.get('group_by', 'day')
        college = request.args.get('college')
        program = request.args.get('program')

        conn = get_connection() if get_connection else None
        if conn is None:
            return jsonify({'success': False, 'error': 'DB not configured'}), 500

        where = []
        params = []
        if status_filter:
            where.append("v.status = %s")
            params.append(status_filter)
        if college:
            where.append("s.college = %s")
            params.append(college)
        if program:
            where.append("s.program = %s")
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

        if group_by == 'month':
            group_expr = "DATE_FORMAT(v.timestamp, '%Y-%m')"
            order_expr = "DATE_FORMAT(v.timestamp, '%Y-%m')"
        elif group_by == 'week':
            group_expr = "YEARWEEK(v.timestamp, 3)"
            order_expr = "YEARWEEK(v.timestamp, 3)"
        else:
            group_expr = "DATE(v.timestamp)"
            order_expr = "DATE(v.timestamp)"

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

@app.route('/results/violations/<filename>')
def uploaded_violation_file(filename):
    return send_from_directory(VIOLATION_FOLDER, filename)

@app.route('/violation_proof/<filename>')
def violation_proof(filename):
    """Serve violation proof images with proper headers"""
    try:
        # Serve from violations subfolder by default
        return send_from_directory(VIOLATION_FOLDER, filename, as_attachment=False)
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
                       s.name as student_name, s.gender, s.program, s.college
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

@app.route('/debug_rfid')
def debug_rfid():
    """Debug endpoint to check RFID status"""
    try:
        with rfid_lock:
            rfid_state = {
                'rfid_available': RFID_AVAILABLE,
                'rfid_present': rfid_present,
                'rfid_last_uid': rfid_last_uid,
                'rfid_event_queue': rfid_event_queue is not None,
                'detection_enabled': detection_enabled,
                'rfid_last_student': rfid_last_student
            }
        
        # Get actual RFID status from scanner
        if RFID_AVAILABLE:
            try:
                scanner_status = get_rfid_status()
                rfid_state['scanner_status'] = scanner_status
            except Exception as e:
                rfid_state['scanner_error'] = str(e)
        
        return jsonify({
            'success': True,
            'rfid_state': rfid_state,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

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

@app.route('/test_email', methods=['POST'])
def test_email():
    """Test email sending functionality"""
    try:
        data = request.get_json() or {}
        test_email_address = data.get('email', app.config.get('MAIL_USERNAME', 'dress.psu@gmail.com'))
        
        print(f"DEBUG: Testing email to {test_email_address}")
        print(f"DEBUG: Mail config - Server: {app.config.get('MAIL_SERVER')}, Username: {app.config.get('MAIL_USERNAME')}")
        
        try:
            msg = Message(
                subject='DRESS Test Email',
                recipients=[test_email_address],
                body='This is a test email from the DRESS system. If you receive this, email configuration is working correctly.'
            )
            with app.app_context():
                mail.send(msg)
            print(f"✓ SUCCESS: Test email sent to {test_email_address}")
            return jsonify({'success': True, 'message': f'Test email sent to {test_email_address}'})
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"✗ ERROR: Failed to send test email: {error_msg}")
            import traceback
            print(f"✗ TRACEBACK:\n{traceback.format_exc()}")
            return jsonify({'success': False, 'error': error_msg, 'details': str(e)}), 500
    except Exception as e:
        print(f"✗ ERROR in test_email endpoint: {e}")
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
                       s.name as student_name, s.gender, s.program, s.college, s.student_id as student_number
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
    """Start webcam and RFID monitoring (or return status if already running)"""
    global camera, selected_camera_id, rfid_event_queue, rfid_enabled
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
                
                # Initialize and start RFID monitoring when camera starts
                if RFID_AVAILABLE:
                    try:
                        # Initialize RFID if not already done
                        if rfid_event_queue is None:
                            initialize_rfid()
                        else:
                            start_rfid_monitoring()
                        
                        # Enable RFID processing
                        rfid_enabled = True
                        set_rfid_enabled(True)  # Enable RFID polling
                        print(f"DEBUG: RFID enabled set to True, rfid_enabled: {rfid_enabled}")
                        print("RFID monitoring started with camera")
                    except Exception as e:
                        print(f"Warning: Could not start RFID monitoring: {e}")
                
                return jsonify({'success': True, 'message': f'Camera {camera_id} and RFID monitoring started successfully'})
            else:
                camera = None
                return jsonify({'success': False, 'message': f'Failed to open camera {camera_id}'}), 500
        else:
            return jsonify({'success': True, 'message': 'Camera already running'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error starting camera: {str(e)}'}), 500

@app.route('/change_camera', methods=['POST'])
def change_camera():
    """Change to a different camera and enable RFID monitoring"""
    global camera, detection_enabled, selected_camera_id, rfid_enabled, rfid_event_queue
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
            
            # Enable RFID monitoring when camera is started via switching
            if RFID_AVAILABLE:
                try:
                    if rfid_event_queue is None:
                        initialize_rfid()
                    else:
                        start_rfid_monitoring()
                    
                    # Enable RFID processing
                    rfid_enabled = True
                    set_rfid_enabled(True)  # Enable RFID polling
                    print(f"DEBUG: RFID enabled via camera switch, rfid_enabled: {rfid_enabled}")
                    print("RFID monitoring started with camera switch")
                except Exception as e:
                    print(f"Warning: Could not start RFID monitoring during camera switch: {e}")
            
            return jsonify({'success': True, 'message': f'Switched to camera {camera_id} and enabled RFID monitoring'})
        else:
            camera = None
            return jsonify({'success': False, 'message': f'Failed to open camera {camera_id}'}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error changing camera: {str(e)}'}), 500

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop webcam and RFID monitoring"""
    global camera, detection_enabled, rfid_event_queue, rfid_enabled
    try:
        if camera is not None:
            camera.release()
            camera = None
            detection_enabled = False  # Also disable detection when camera stops
            
            # Stop RFID monitoring when camera stops
            if RFID_AVAILABLE:
                try:
                    # Disable RFID processing first
                    rfid_enabled = False
                    set_rfid_enabled(False)  # Disable RFID polling
                    print(f"DEBUG: RFID enabled set to False, rfid_enabled: {rfid_enabled}")
                    
                    stop_rfid_monitoring()
                    
                    # Unsubscribe from RFID events to completely stop monitoring
                    if rfid_event_queue is not None:
                        unsubscribe_from_rfid_events(rfid_event_queue)
                        rfid_event_queue = None
                    
                    print("RFID monitoring completely stopped with camera")
                    
                    # Reset RFID state
                    with rfid_lock:
                        rfid_present = False
                        rfid_last_student = None
                        rfid_consecutive_non_compliant = 0
                        rfid_last_compliance_status = None
                        rfid_current_uid_violated = False
                        rfid_last_uid = None
                        
                except Exception as e:
                    print(f"Warning: Could not stop RFID monitoring: {e}")
            
            return jsonify({'success': True, 'message': 'Camera and RFID monitoring stopped successfully'})
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
                    _present = rfid_present
                    _student_set = (rfid_last_student is not None)
                    _violated = rfid_current_uid_violated
                    # Detection is enabled only if: detection flag is True, RFID is present, student is set, AND no violation was recorded yet
                    rfid_detection_enabled = detection_enabled and _present and _student_set and not _violated
                
                # In test mode, detection always runs regardless of RFID
                detection_enabled_for_capture = rfid_detection_enabled or test_mode_active
                
                if detection_enabled_for_capture:
                    # Perform detection on current frame
                    detections = detect_persons_frame_with_dress(current_frame)
                    
                    # Draw detections on frame
                    frame_with_detections = draw_detections_frame(current_frame.copy(), detections)
                    
                    # Status overlay removed as requested
                    pass

                    # Attempt to record violation if non-compliant
                    admin_user = session.get('admin') or {}
                    _maybe_record_violation(current_frame, detections, admin_user)
                else:
                    # No detection, just return original frame
                    detections = []
                    frame_with_detections = current_frame.copy()
                    
                    # Status overlay removed as requested
                    pass
                
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
    global rfid_last_uid, rfid_present, detection_enabled, rfid_lock, rfid_last_student, rfid_enabled
    try:
        with rfid_lock:
            camera_active = camera is not None and camera.isOpened()
            print(f"DEBUG: RFID status check - rfid_enabled: {rfid_enabled}, rfid_present: {rfid_present}")
            
            # If RFID is disabled, return inactive status
            if not rfid_enabled or not camera_active:
                status = {
                    'available': RFID_AVAILABLE,
                    'present': False,
                    'last_uid': None,
                    'detection_enabled': False,
                    'student': None,
                    'enabled': False,
                    'camera_active': camera_active,
                }
                print("DEBUG: RFID disabled or camera inactive, returning inactive status")
            else:
                # RFID is enabled, get actual status
                status = get_rfid_status()
                status.update({
                    'last_uid': rfid_last_uid,
                    'present': rfid_present,
                    'detection_enabled': detection_enabled,
                    'student': rfid_last_student,
                    'enabled': True,
                    'camera_active': camera_active,
                })
                print(f"DEBUG: RFID enabled, returning status: {status}")
        return jsonify({'success': True, 'status': status})
    except Exception as e:
        print(f"DEBUG: RFID status error: {e}")
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
    # Start alerts checker in background
    try:
        import threading as _t
        _t.Thread(target=(lambda: __import__('time') or None), daemon=True)
    except Exception:
        pass
    app.run(debug=True, host='0.0.0.0', port=5000)