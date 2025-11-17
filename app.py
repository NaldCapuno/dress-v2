from flask import Flask, render_template, request, jsonify, send_from_directory, Response, session, redirect, url_for
from flask_mail import Mail, Message
import cv2
import numpy as np
from ultralytics import YOLO
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, skip loading .env file
    pass
import base64
import threading
import time
import re
import queue
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
# All email settings must be configured in .env file
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', '587'))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'true').lower() in {'1','true','yes','on'}
app.config['MAIL_USE_SSL'] = os.getenv('MAIL_USE_SSL', 'false').lower() in {'1','true','yes','on'}
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER', os.getenv('MAIL_USERNAME'))
# Email timeout settings to prevent hanging
app.config['MAIL_TIMEOUT'] = int(os.getenv('MAIL_TIMEOUT', '10'))  # 10 seconds timeout
app.config['MAIL_CONNECT_TIMEOUT'] = int(os.getenv('MAIL_CONNECT_TIMEOUT', '5'))  # 5 seconds connection timeout

# Validate required email settings
if not app.config['MAIL_USERNAME'] or not app.config['MAIL_PASSWORD']:
    print("⚠ WARNING: MAIL_USERNAME and/or MAIL_PASSWORD not set in .env file. Email notifications will not work.")

mail = Mail(app)

# Add charset=utf-8 to all JSON responses
@app.after_request
def add_charset_to_json(response):
    """Automatically add charset=utf-8 to JSON responses"""
    if response.content_type == 'application/json':
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response

# Register Blueprints
from routes import auth_bp, dashboards_bp, violations_bp, files_bp, camera_bp, rfid_bp, debug_bp, students_bp, settings_bp
app.register_blueprint(auth_bp)
app.register_blueprint(dashboards_bp)
app.register_blueprint(violations_bp)
app.register_blueprint(files_bp)
app.register_blueprint(camera_bp)
app.register_blueprint(rfid_bp)
app.register_blueprint(debug_bp)
app.register_blueprint(students_bp)
app.register_blueprint(settings_bp)

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

# Global variables for smooth detection (async processing)
detection_queue = None  # Queue for frames to be processed
latest_detections = None  # Latest detection results
latest_detection_frame = None  # Frame that was used for latest detections
detection_results_lock = threading.Lock()  # Lock for detection results
detection_thread = None  # Background detection thread

# Global variables for RFID integration
rfid_event_queue = None
rfid_last_uid = None
rfid_present = False
rfid_lock = threading.Lock()
rfid_last_student = None  # Holds last looked-up student dict or None

# Global variable for auto-sync control
auto_sync_enabled = True
auto_sync_lock = threading.Lock()
rfid_last_violation_ts = 0  # last violation timestamp to throttle duplicates
rfid_last_violation_uid = None  # last UID that had a violation (for throttle check - only throttle same card)
rfid_current_uid_checks = 0  # number of detection checks for current RFID UID
rfid_current_uid_violated = False  # whether a violation has been issued for current UID session
rfid_current_uid_compliant = False  # whether compliant status was detected for current UID session (stops detection)
rfid_current_uid_snapshot_saved = False  # whether a clean snapshot was saved for current UID
rfid_consecutive_non_compliant = 0  # counter for consecutive non-compliant detections
rfid_consecutive_compliant = 0  # counter for consecutive compliant detections
rfid_last_compliance_status = None  # track last compliance status to detect changes
rfid_enabled = False  # flag to completely disable RFID processing
rfid_violation_timeout = 30  # seconds after which violation flag resets (allows re-recording)
compliant_monitor_frame_counter = 0  # counter to periodically check for violations even when compliant

# Global variable for test mode
test_mode = False
test_mode_lock = threading.Lock()

# Schedule checking function
def is_system_scheduled_active():
    """Check if the system should be active based on the schedule settings"""
    try:
        if get_connection is None:
            # If DB not configured, allow system to run
            return True
        
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT setting_value FROM settings WHERE setting_key = 'system_schedule'"
                )
                result = cur.fetchone()
                
                if not result or not result.get('setting_value'):
                    # No schedule set, system is always active
                    return True
                
                import json
                schedule = json.loads(result['setting_value'])
                
                if not schedule or len(schedule) == 0:
                    # Empty schedule, system is always active
                    return True
                
                # Get current day and time
                now = datetime.now()
                current_day = now.strftime('%A')  # e.g., 'Monday'
                current_time = now.strftime('%H:%M')  # e.g., '14:30'
                
                # Check if current time falls within any schedule entry for today
                for entry in schedule:
                    if entry['day'] == current_day:
                        if entry['start_time'] <= current_time < entry['end_time']:
                            return True
                
                # Not within any schedule
                return False
        finally:
            conn.close()
    except Exception as e:
        error_msg = str(e)
        if "Access denied" in error_msg or "1045" in error_msg:
            print(f"⚠️ Database authentication error: {error_msg}")
            print("⚠️ Please check:")
            print("   1. DB_PASSWORD environment variable is set correctly")
            print("   2. Your IP address is whitelisted in Aiven")
            print("   3. SSL certificate is properly configured")
        else:
            print(f"Error checking schedule: {e}")
        # On error, allow system to run (fail open)
        return True

# Auto-initialize camera
def initialize_camera():
    global camera, selected_camera_id, detection_queue, detection_thread, latest_detections, latest_detection_frame
    try:
        camera = cv2.VideoCapture(selected_camera_id)  # Use selected camera
        if camera.isOpened():
            print(f"Camera {selected_camera_id} initialized successfully")
            
            # Initialize detection queue and start detection thread if not already running
            if detection_queue is None:
                detection_queue = queue.Queue(maxsize=2)  # Small queue to prevent lag
                latest_detections = None
                latest_detection_frame = None
                
                # Start detection worker thread
                if detection_thread is None or not detection_thread.is_alive():
                    detection_thread = threading.Thread(target=detection_worker, daemon=True)
                    detection_thread.start()
                    print("Detection worker thread started")
            
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

def update_rfid_enabled_based_on_schedule():
    """Update rfid_enabled flag based on schedule and test mode"""
    global rfid_enabled, test_mode, test_mode_lock, camera
    
    # Check test mode first
    with test_mode_lock:
        test_mode_active = test_mode
    
    # If test mode is active, RFID should be disabled
    if test_mode_active:
        if rfid_enabled:
            rfid_enabled = False
            set_rfid_enabled(False)
            print("RFID disabled: Test mode is active")
        return
    
    # Check schedule
    is_active = is_system_scheduled_active()
    
    if is_active:
        # Within scheduled hours - enable RFID if camera is running
        if camera is not None and camera.isOpened():
            if not rfid_enabled:
                rfid_enabled = True
                set_rfid_enabled(True)
                print("RFID enabled: Within scheduled hours")
        else:
            # Camera not running, disable RFID
            if rfid_enabled:
                rfid_enabled = False
                set_rfid_enabled(False)
                print("RFID disabled: Camera is not running")
    else:
        # Outside scheduled hours - disable RFID
        if rfid_enabled:
            rfid_enabled = False
            set_rfid_enabled(False)
            print("RFID disabled: Outside scheduled hours")

def rfid_event_handler():
    """Handle RFID events in background thread"""
    global rfid_last_uid, rfid_present, detection_enabled, rfid_lock, rfid_last_student, rfid_consecutive_non_compliant, rfid_last_compliance_status, rfid_last_violation_ts, camera, rfid_enabled, tracker, test_mode, test_mode_lock
    
    while True:
        try:
            # Update RFID enabled status based on schedule and test mode
            update_rfid_enabled_based_on_schedule()
            
            # Skip RFID processing if test mode is active
            with test_mode_lock:
                test_mode_active = test_mode
            
            if test_mode_active:
                time.sleep(0.1)  # Small delay to avoid busy waiting
                continue
            
            # Check if system is scheduled to be active - RFID only works during scheduled hours
            if not is_system_scheduled_active():
                time.sleep(0.1)  # Small delay to avoid busy waiting
                continue
            
            if rfid_event_queue and rfid_enabled and camera is not None and camera.isOpened():
                try:
                    event = rfid_event_queue.get(timeout=1.0)
                except queue.Empty:
                    # Timeout - no event in queue, check current RFID status
                    current_present = _rfid_is_present()
                    with rfid_lock:
                        # Always check if card is not present and reset state accordingly
                        if not current_present:
                            # Card is not present - always reset state if we have any stale data
                            if rfid_present or rfid_last_uid is not None or rfid_last_student is not None:
                                # Card was present but now removed - reset all state
                                rfid_present = False
                                detection_enabled = False
                                rfid_last_student = None
                                rfid_last_uid = None  # Reset UID so next scan is treated as new
                                rfid_current_uid_checks = 0
                                rfid_current_uid_violated = False
                                rfid_current_uid_compliant = False
                                rfid_just_violated = False
                                rfid_just_compliant = False
                                rfid_consecutive_non_compliant = 0
                                rfid_consecutive_compliant = 0
                                rfid_last_compliance_status = None
                                rfid_current_uid_snapshot_saved = False
                                rfid_last_violation_ts = 0
                                rfid_last_violation_uid = None
                                # Reset tracker when RFID card is removed
                                tracker = BotSORT()
                                print("RFID Card removed - Detection DISABLED - Tracker reset - All state cleared")
                        elif current_present and not rfid_present:
                            # Card just appeared
                            rfid_present = True
                            print("RFID Card present - Waiting for UID event")
                    continue  # Skip to next iteration
                
                if event['type'] == 'uid':
                    with rfid_lock:
                        incoming_uid = event['uid']
                        is_same_uid = (rfid_present and rfid_last_uid == incoming_uid)
                        old_uid = rfid_last_uid
                        
                        # Safety check: if we have a different UID or no previous UID, treat as new card
                        if rfid_last_uid is not None and rfid_last_uid != incoming_uid:
                            # Different card detected - reset everything first
                            print(f"DEBUG: Different RFID card detected - Old UID: {rfid_last_uid}, New UID: {incoming_uid}")
                            rfid_current_uid_checks = 0
                            rfid_current_uid_violated = False
                            rfid_current_uid_compliant = False
                            rfid_consecutive_non_compliant = 0
                            rfid_consecutive_compliant = 0
                            rfid_last_compliance_status = None
                            rfid_current_uid_snapshot_saved = False
                            rfid_last_violation_ts = 0
                            rfid_last_violation_uid = None
                            tracker = BotSORT()
                        
                        rfid_last_uid = incoming_uid
                        rfid_present = True
                        
                        # Only reset per-scan counters on NEW UID (or after removal), not on repeated same-UID events
                        if not is_same_uid:
                            print(f"DEBUG: New RFID card detected - Old UID: {old_uid}, New UID: {incoming_uid}")
                            rfid_current_uid_checks = 0
                            rfid_current_uid_violated = False
                            rfid_current_uid_compliant = False
                            rfid_consecutive_non_compliant = 0
                            rfid_consecutive_compliant = 0
                            rfid_last_compliance_status = None
                            rfid_current_uid_snapshot_saved = False
                            rfid_last_violation_ts = 0
                            rfid_last_violation_uid = None  # Reset violation UID tracking for new card
                            compliant_monitor_frame_counter = 0  # Reset compliant monitoring counter
                            # Reset tracker for new RFID scan
                            tracker = BotSORT()
                            print("DEBUG: Tracker reset for new RFID scan - All flags reset")
                        else:
                            print(f"DEBUG: Same RFID card still present - UID: {incoming_uid}")
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
                    # Only disable detection if student already has a violation recorded today (not based on session flag)
                    if rfid_last_student is not None:
                        student_id = rfid_last_student.get('student_id')
                        # Check if student already has a violation today (check database, not session flag)
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
                                # The flag will be reset in the /rfid/status endpoint after frontend reads it
                                detection_enabled = False
                                print(f"RFID Card detected: {event['uid']} - Student found: {rfid_last_student.get('name', 'Unknown')} - Detection DISABLED (student already has violation today)")
                            # Enable detection for new UID
                            elif not is_same_uid:
                                # New card - always enable detection (flags were reset above)
                                detection_enabled = True
                                print(f"RFID Card detected: {event['uid']} - Student found: {rfid_last_student.get('name', 'Unknown')} - Detection ENABLED (new card)")
                                print(f"DEBUG: detection_enabled={detection_enabled}, rfid_current_uid_violated={rfid_current_uid_violated}, rfid_current_uid_compliant={rfid_current_uid_compliant}")
                            # Same card - only disable if violation today OR if compliant detected
                            elif rfid_current_uid_compliant:
                                # Same card, but compliant already detected - keep detection disabled
                                detection_enabled = False
                                print(f"RFID Card detected: {event['uid']} - Student found: {rfid_last_student.get('name', 'Unknown')} - Detection DISABLED (student is compliant)")
                                print(f"DEBUG: detection_enabled={detection_enabled}, rfid_current_uid_violated={rfid_current_uid_violated}, rfid_current_uid_compliant={rfid_current_uid_compliant}")
                            else:
                                # Same card, no violation today, and not compliant - enable detection
                                # This allows re-detection if the same card is scanned again (unless violation today)
                                detection_enabled = True
                                print(f"RFID Card detected: {event['uid']} - Student found: {rfid_last_student.get('name', 'Unknown')} - Detection ENABLED (same card, no violation today)")
                                print(f"DEBUG: detection_enabled={detection_enabled}, has_violation_today={has_violation_today}, rfid_current_uid_violated={rfid_current_uid_violated}, rfid_current_uid_compliant={rfid_current_uid_compliant}")
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
                    # Event received but not a 'uid' event (shouldn't happen, but handle it)
                    with rfid_lock:
                        rfid_present = False
                        detection_enabled = False
                        rfid_last_student = None
                        rfid_last_uid = None  # Reset UID so next scan is treated as new
                        rfid_current_uid_checks = 0
                        rfid_current_uid_violated = False
                        rfid_current_uid_compliant = False
                        rfid_consecutive_non_compliant = 0
                        rfid_consecutive_compliant = 0
                        rfid_last_compliance_status = None
                        rfid_current_uid_snapshot_saved = False
                        rfid_last_violation_ts = 0
                        rfid_last_violation_uid = None  # Reset violation UID tracking when card removed
                        # Reset tracker when RFID card is removed
                        tracker = BotSORT()
                    print("RFID Card removed - Detection DISABLED - Tracker reset - All state cleared")
            elif rfid_event_queue:
                # RFID disabled or camera off, just consume events without processing
                try:
                    rfid_event_queue.get(timeout=1.0)
                except queue.Empty:
                    pass
        except Exception as e:
            # Other exceptions - log and continue
            print(f"Error in RFID event handler: {e}")
            import traceback
            traceback.print_exc()
        
        # Check if RFID is disabled or camera is off - ensure RFID is inactive
        if not rfid_enabled or camera is None or not camera.isOpened():
            with rfid_lock:
                if rfid_present:
                    rfid_present = False
                    detection_enabled = False
                    rfid_last_student = None
                    rfid_last_uid = None
                    rfid_current_uid_checks = 0
                    rfid_current_uid_violated = False
                    rfid_current_uid_compliant = False
                    rfid_consecutive_non_compliant = 0
                    rfid_consecutive_compliant = 0
                    rfid_last_compliance_status = None
                    rfid_current_uid_snapshot_saved = False
                    rfid_last_violation_ts = 0
                    rfid_last_violation_uid = None
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
                    
                    if confidence > 0.70:  # High threshold for dress detection
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
    global rfid_last_student, rfid_last_violation_ts, rfid_last_violation_uid, rfid_consecutive_non_compliant, rfid_consecutive_compliant, rfid_last_compliance_status, rfid_current_uid_violated, rfid_current_uid_compliant, detection_enabled, rfid_last_uid
    
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
        
        # Collect compliant items for debug output
        compliant_items_list = []
        if detections:
            for det in detections:
                dv = det.get('dress_validation') or {}
                comp = dv.get('compliance_status') or {}
                for key, val in comp.items():
                    if val.get('present'):
                        compliant_items_list.append(val.get('name') or key)
        
        print(f"DEBUG: Frame analysis - Non-compliant: {non_compliant}, Partially compliant: {partially_compliant}, Status: {current_compliance_status}, Detections: {len(detections)}")
        if current_compliance_status == 'COMPLIANT':
            print(f"DEBUG: COMPLIANT items detected: {', '.join(compliant_items_list) if compliant_items_list else 'None'}")
        elif current_compliance_status in ['NON-COMPLIANT', 'PARTIALLY COMPLIANT']:
            print(f"DEBUG: Violation items missing: {', '.join(violation_details) if violation_details else 'None'}")
        
        # Only process if RFID card is present
        with rfid_lock:
            if not rfid_present:
                print("DEBUG: RFID not present, skipping violation check")
                return None
                
            # Check if student already has a violation today (daily limit check)
            # This check happens early to prevent detection
            student_id = (rfid_last_student or {}).get('student_id')
            if has_student_violation_today and student_id:
                if has_student_violation_today(student_id):
                    print(f"DEBUG: Daily limit reached - violation already recorded today for student {student_id}")
                    # Keep violation flag True to prevent detection
                    with rfid_lock:
                        rfid_current_uid_violated = True  # Keep it True to prevent detection
                    # Don't return here - let the function continue to check threshold
            
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
                # If person was previously marked as compliant, don't record violations or re-enable detection
                # This prevents false positives when detection was stopped after 2 compliant detections
                if rfid_current_uid_compliant:
                    print(f"DEBUG: Violation detected but person was previously COMPLIANT - ignoring to prevent false positives (Status: {current_compliance_status})")
                    return None  # Exit early, don't process or record this violation
                
                rfid_consecutive_non_compliant += 1
                print(f"DEBUG: Violation detection #{rfid_consecutive_non_compliant} for student {rfid_last_student.get('student_id')} (Status: {current_compliance_status})")
            else:
                # Only reset counter if fully compliant or no detections
                if current_compliance_status == 'NO_DETECTION':
                    # Don't reset counter on no detection - might be temporary
                    print(f"DEBUG: No detections for student {rfid_last_student.get('student_id')}, keeping counter at {rfid_consecutive_non_compliant}")
                elif current_compliance_status == 'COMPLIANT':
                    # Reset violation counter when fully compliant
                    rfid_consecutive_non_compliant = 0
                    # Also reset the violation flag so they can be recorded again if they become non-compliant
                    rfid_current_uid_violated = False
                    
                    # Increment compliant counter
                    rfid_consecutive_compliant += 1
                    student_id = rfid_last_student.get('student_id') if rfid_last_student else 'Unknown'
                    print(f"DEBUG: [COMPLIANT] Detection #{rfid_consecutive_compliant} for student {student_id}")
                    print(f"DEBUG: [COMPLIANT] Compliant items: {', '.join(compliant_items_list) if compliant_items_list else 'None'}")
                    print(f"DEBUG: [COMPLIANT] Counter: {rfid_consecutive_compliant}/2 (need 2 consecutive for auto-stop)")
                    
                    # Stop detection after 2 consecutive compliant detections (to avoid false positives)
                    if rfid_consecutive_compliant >= 2 and not rfid_current_uid_compliant:
                        # Only set flags once per detection event
                        rfid_current_uid_compliant = True
                        with rfid_lock:
                            detection_enabled = False
                        print(f"DEBUG: [COMPLIANT] ✓ Student {student_id} reached 2 consecutive COMPLIANT detections - DETECTION STOPPED")
                        print(f"DEBUG: [COMPLIANT] Flags: rfid_current_uid_compliant={rfid_current_uid_compliant}, detection_enabled={detection_enabled}")
                    else:
                        remaining = 2 - rfid_consecutive_compliant
                        print(f"DEBUG: [COMPLIANT] Need {remaining} more consecutive compliant detection(s) to stop")
                else:
                    # Reset compliant counter if status is not COMPLIANT
                    rfid_consecutive_compliant = 0
                    rfid_current_uid_compliant = False
            
            # Only proceed if we have 3 consecutive violation detections (non-compliant or partially compliant)
            if rfid_consecutive_non_compliant < 3:
                print(f"DEBUG: Need {3 - rfid_consecutive_non_compliant} more consecutive violation detections")
                return None

        # Daily limit: only one recorded violation per student per day
        student_id = (rfid_last_student or {}).get('student_id')
        has_violation_today = False
        if has_student_violation_today and student_id:
            if has_student_violation_today(student_id):
                has_violation_today = True
                print(f"DEBUG: Daily limit reached - violation already recorded today for student {student_id}")

        # Throttle: only apply if the SAME card is scanned consecutively (within 10 seconds)
        # Different cards can be scanned immediately without throttle
        now_ts = time.time()
        is_same_card = (rfid_last_uid is not None and rfid_last_violation_uid is not None and rfid_last_uid == rfid_last_violation_uid)
        is_throttled = False
        if is_same_card and now_ts - rfid_last_violation_ts < 10:
            is_throttled = True
            print(f"DEBUG: Throttled - same card scanned within {now_ts - rfid_last_violation_ts:.1f}s (violation NOT recorded)")
        
        # If daily limit reached or throttled, don't record
        if has_violation_today or is_throttled:
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

        # Get violation count BEFORE inserting the new violation to calculate correct strike number
        student_id = rfid_last_student.get('student_id')
        previous_violation_count = 0
        try:
            from src.config import get_student_violation_count as _get_v_cnt
        except Exception:
            _get_v_cnt = None
        if _get_v_cnt and student_id:
            previous_violation_count = int(_get_v_cnt(student_id) or 0)
            print(f"DEBUG: Previous violation count for student {student_id}: {previous_violation_count}")

        # Store relative path in DB; serve via /results/violations/<filename>
        rel_path = os.path.join(VIOLATION_SUBDIR, proof_name) if proof_path else None
        print(f"DEBUG: Database path: {rel_path}")

        if insert_violation:
            print(f"DEBUG: Attempting database insertion...")
            vid = insert_violation(rfid_last_student.get('student_id'), violation_type, rel_path)
            print(f"DEBUG: Database insertion returned: {vid}")
            if vid:
                rfid_last_violation_ts = now_ts
                rfid_last_violation_uid = rfid_last_uid  # Track which UID had the violation for throttle check
                with rfid_lock:
                    # Set flags for violation event
                    rfid_current_uid_violated = True
                    # Violation is recorded
                    print(f"✓ VIOLATION RECORDED: Violation recorded for student {rfid_last_student.get('student_id')}")
                    print(f"DEBUG: Violation recorded in database, UID tracked: {rfid_last_violation_uid}")
                
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
                        # Determine strike number based on previous violations + 1 (the one we just inserted)
                        # Cap at 3
                        strike_num = previous_violation_count + 1
                        strike_num = max(1, min(3, strike_num))
                        print(f"DEBUG: Strike number calculated: {strike_num} (previous violations: {previous_violation_count})")
                        
                        try:
                            from src.config import get_student_violations as _get_v_list
                        except Exception:
                            _get_v_list = None
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
                        
                        # Read logo and convert to base64
                        import base64
                        logo_base64 = None
                        logo_path = os.path.join(app.root_path, 'static', 'images', 'dress_logo.png')
                        if os.path.exists(logo_path):
                            try:
                                with open(logo_path, 'rb') as logo_file:
                                    logo_base64 = base64.b64encode(logo_file.read()).decode('utf-8')
                            except Exception as logo_err:
                                print(f"DEBUG: Could not read logo: {logo_err}")
                        
                        # Generate email body using HTML template
                        html_body = generate_violation_email_body(
                            student_name=student_name,
                            violation_datetime=dt_str,
                            strike_num=strike_num,
                            offense_line=offense_line,
                            violation_history=violation_text,
                            image_cid=image_cid,
                            logo_base64=logo_base64
                        )
                        
                        # Create plain text fallback
                        image_attachment_text = "\n\nPROOF OF VIOLATION\nA proof image is attached to this email.\n" if image_cid else ""
                        plain_text_body = f"""DRESS CODE VIOLATION NOTIFICATION

Dear {student_name},

This is to inform you that the DRESS (Dress-code Recognition Surveillance System) detected a dress code violation on your part on {dt_str}.

Please remember that following the university dress code is part of maintaining discipline and professionalism. We ask you to comply with the proper uniform prescribed by the University, as stated in the Student Handbook, on your next visit.

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

def detection_worker():
    """Background thread that processes frames for detection without blocking camera feed"""
    global detection_queue, latest_detections, latest_detection_frame, detection_results_lock, detection_enabled, test_mode, test_mode_lock
    
    while True:
        try:
            # Get frame from queue (with timeout to allow checking if thread should continue)
            try:
                frame_data = detection_queue.get(timeout=0.1)
                if frame_data is None:  # Shutdown signal
                    break
                
                frame, detection_enabled_flag = frame_data
                
                # Note: Camera can work anytime, but detections/violations are only processed during scheduled hours
                # This allows camera feed to work but prevents violations from being recorded outside schedule
                # Exception: Test mode works outside scheduled hours
                if detection_enabled_flag and frame is not None:
                    # Check test mode first - test mode works outside scheduled hours
                    with test_mode_lock:
                        test_mode_active = test_mode
                    
                    # Only process detections if system is scheduled to be active OR test mode is active
                    if not test_mode_active and not is_system_scheduled_active():
                        # Skip detection processing if outside scheduled hours (but camera feed still works)
                        detection_queue.task_done()
                        continue
                    # Perform detection on the frame
                    detections = detect_persons_frame_with_dress(frame.copy())
                    
                    # Store latest detection results
                    with detection_results_lock:
                        latest_detections = detections
                        latest_detection_frame = frame.copy()
                    
                    # Attempt to record violation using the frame
                    _maybe_record_violation(frame.copy(), detections, None)
                
                detection_queue.task_done()
            except queue.Empty:
                # No frame to process, continue
                continue
        except Exception as e:
            print(f"Error in detection worker: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)

def generate_frames():
    """Generate video frames for streaming with smooth async detection"""
    global camera, detection_enabled, current_frame, frame_lock, detection_queue
    global latest_detections, latest_detection_frame, detection_results_lock, compliant_monitor_frame_counter
    
    # Use time-based frame rate control for smoother playback
    target_fps = 30.0
    frame_time = 1.0 / target_fps
    
    while True:
        frame_start_time = time.time()
        
        if camera is not None:
            success, frame = camera.read()
            if success:
                # Update current frame (minimize lock time)
                with frame_lock:
                    current_frame = frame.copy()
                
                # Get state variables quickly (minimize lock time)
                with test_mode_lock:
                    test_mode_active = test_mode
                
                with rfid_lock:
                    _present = rfid_present
                    _student_set = (rfid_last_student is not None)
                    _compliant = rfid_current_uid_compliant
                    _detection_enabled = detection_enabled
                    rfid_detection_enabled = _detection_enabled and _present and _student_set and not _compliant
                
                # In test mode, detection always runs regardless of RFID
                detection_enabled_for_frame = rfid_detection_enabled or test_mode_active
                
                # Even when compliant, periodically check for violations (every 30 frames = ~1 second at 30fps)
                # This prevents the system from hanging when transitioning from compliant to non-compliant
                should_monitor_compliant = False
                if _compliant and _present and _student_set and not test_mode_active:
                    compliant_monitor_frame_counter += 1
                    if compliant_monitor_frame_counter >= 30:  # Check every ~1 second
                        should_monitor_compliant = True
                        compliant_monitor_frame_counter = 0
                elif not _compliant:
                    # Reset counter when not compliant
                    compliant_monitor_frame_counter = 0
                
                # Add frame to detection queue (non-blocking, drop old frames if queue is full)
                if (detection_enabled_for_frame or should_monitor_compliant) and detection_queue is not None:
                    try:
                        detection_queue.put_nowait((frame.copy(), True))
                    except queue.Full:
                        pass
                elif not detection_enabled_for_frame and not should_monitor_compliant:
                    # Clear detection results when detection is disabled (but not during compliant monitoring)
                    with detection_results_lock:
                        latest_detections = None
                        latest_detection_frame = None
                
                # Get latest detection results (non-blocking)
                detections_to_draw = None
                with detection_results_lock:
                    if latest_detections is not None:
                        detections_to_draw = latest_detections
                
                # Draw detections on current frame if available
                if detection_enabled_for_frame and detections_to_draw is not None:
                    frame = draw_detections_frame(frame, detections_to_draw)
                
                # Encode frame as JPEG with optimized quality for faster encoding
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]  # Slightly lower quality for faster encoding
                ret, buffer = cv2.imencode('.jpg', frame, encode_params)
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
        
        # Adaptive sleep to maintain target FPS (accounts for processing time)
        elapsed = time.time() - frame_start_time
        sleep_time = max(0, frame_time - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

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


def schedule_rfid_checker():
    """Background thread that periodically checks schedule and updates RFID enabled status"""
    # Wait for app to be fully initialized
    time.sleep(5)
    
    while True:
        try:
            update_rfid_enabled_based_on_schedule()
        except Exception as e:
            print(f"Error in schedule RFID checker: {e}")
        
        # Check every 10 seconds
        time.sleep(10)


def auto_sync_to_aiven():
    """Background thread that periodically syncs local database to Aiven (backup)."""
    # Wait for app to be fully initialized
    time.sleep(15)
    
    from src.config import is_aiven_available
    global auto_sync_enabled, auto_sync_lock
    
    last_sync_time = 0
    sync_interval = 300  # Sync every 5 minutes (300 seconds)
    
    while True:
        try:
            # Check if auto-sync is enabled
            with auto_sync_lock:
                sync_enabled = auto_sync_enabled
            
            if not sync_enabled:
                # Auto-sync is disabled, wait and check again
                time.sleep(60)  # Check every minute if it's been re-enabled
                continue
            
            # Check if Aiven is available
            aiven_available = is_aiven_available(force_check=False)
            if not aiven_available:
                # Aiven is not available, log and wait
                print(f"DEBUG: Auto-sync skipped - Aiven not available (will retry in 60s)")
                time.sleep(60)
                continue
            
            current_time = time.time()
            time_since_last_sync = current_time - last_sync_time
            
            # Check if enough time has passed since last sync
            if time_since_last_sync >= sync_interval:
                print("🔄 Syncing local database to Aiven (backup)...")
                
                try:
                    # Import sync functions
                    import sys
                    from pathlib import Path
                    sys.path.insert(0, str(Path(__file__).parent))
                    
                    # Use mysql.connector for sync (different from pymysql used in app)
                    import mysql.connector
                    import os
                    
                    # Get connections
                    local_conn = mysql.connector.connect(
                        host=os.getenv('LOCAL_DB_HOST', 'localhost'),
                        port=int(os.getenv('LOCAL_DB_PORT', '3306')),
                        user=os.getenv('LOCAL_DB_USER', 'root'),
                        password=os.getenv('LOCAL_DB_PASSWORD', 'root'),
                        database=os.getenv('LOCAL_DB_NAME', 'dress')
                    )
                    
                    aiven_host = os.getenv('DB_HOST', '')
                    aiven_port = int(os.getenv('DB_PORT', '3306'))
                    aiven_user = os.getenv('DB_USER', '')
                    aiven_password = os.getenv('DB_PASSWORD', '')
                    aiven_database = os.getenv('DB_NAME', 'dress')
                    
                    is_aiven = 'aivencloud.com' in aiven_host.lower()
                    ssl_disabled = os.getenv('DB_SSL_DISABLED', 'false').lower() in {'1', 'true', 'yes', 'on'}
                    ssl_required = os.getenv('DB_SSL_REQUIRED', 'true' if is_aiven else 'false').lower() in {'1', 'true', 'yes', 'on'}
                    ssl_ca = os.getenv('DB_SSL_CA', 'certs/ca.pem' if is_aiven else None)
                    
                    aiven_params = {
                        'host': aiven_host,
                        'port': aiven_port,
                        'user': aiven_user,
                        'password': aiven_password,
                        'database': aiven_database
                    }
                    
                    if not ssl_disabled and (ssl_required or ssl_ca):
                        if ssl_ca and os.path.exists(ssl_ca):
                            aiven_params['ssl_ca'] = ssl_ca
                        if not ssl_disabled:
                            aiven_params['ssl_disabled'] = False
                    
                    aiven_conn = mysql.connector.connect(**aiven_params)
                    
                    # Simple sync: just sync data (assume schema is already synced)
                    local_cursor = local_conn.cursor()
                    aiven_cursor = aiven_conn.cursor()
                    
                    # Get all tables
                    local_cursor.execute("SHOW TABLES")
                    tables = [row[0] for row in local_cursor.fetchall()]
                    
                    # Disable foreign key checks
                    aiven_cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
                    
                    synced_count = 0
                    for table in tables:
                        try:
                            # Get data from local
                            local_cursor.execute(f"SELECT * FROM `{table}`")
                            rows = local_cursor.fetchall()
                            
                            if not rows:
                                continue
                            
                            # Get column names
                            local_cursor.execute(f"DESCRIBE `{table}`")
                            columns = [col[0] for col in local_cursor.fetchall()]
                            
                            # Clear Aiven table
                            aiven_cursor.execute(f"TRUNCATE TABLE `{table}`")
                            
                            # Insert data
                            placeholders = ', '.join(['%s'] * len(columns))
                            columns_str = ', '.join([f"`{col}`" for col in columns])
                            insert_query = f"INSERT INTO `{table}` ({columns_str}) VALUES ({placeholders})"
                            
                            aiven_cursor.executemany(insert_query, rows)
                            aiven_conn.commit()
                            synced_count += 1
                        except Exception as e:
                            print(f"  ⚠ Error syncing table {table}: {e}")
                    
                    # Re-enable foreign key checks
                    aiven_cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
                    aiven_conn.commit()
                    
                    local_conn.close()
                    aiven_conn.close()
                    
                    last_sync_time = current_time
                    print(f"✅ Auto-sync to Aiven (backup) completed! Synced {synced_count} tables.")
                
                except Exception as e:
                    print(f"⚠️ Auto-sync to Aiven failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                # Aiven is available but not enough time has passed
                remaining = sync_interval - time_since_last_sync
                print(f"DEBUG: Auto-sync waiting - {remaining:.0f}s until next sync (Aiven available)")
            
        except Exception as e:
            print(f"Error in auto-sync to Aiven checker: {e}")
            import traceback
            traceback.print_exc()
        
        # Check every 60 seconds
        time.sleep(60)

def followup_email_scheduler():
    """Background thread that periodically checks for and sends follow-up emails for unresolved violations."""
    # Wait for app to be fully initialized
    time.sleep(10)
    
    while True:
        try:
            # Check once per day for violations that need follow-up emails
            # This ensures we catch violations that are exactly 3 days old
            # The followup_sent flag prevents duplicate emails even if the app restarts
            with app.app_context():
                try:
                    # Call the follow-up email endpoint internally
                    from routes.violations import send_followup_emails
                    with app.test_request_context():
                        result = send_followup_emails()
                        if result and hasattr(result, 'get_json'):
                            data = result.get_json()
                            if data and data.get('sent', 0) > 0:
                                print(f"✓ Follow-up email scheduler: Sent {data.get('sent')} follow-up emails")
                            elif data:
                                print(f"✓ Follow-up email scheduler: No emails to send (all violations already processed)")
                except Exception as e:
                    print(f"✗ Error in follow-up email scheduler: {e}")
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            print(f"✗ Error in follow-up email scheduler loop: {e}")
        
        # Sleep for 24 hours (86400 seconds) before checking again
        # This ensures we check once per day for violations that are 3+ days old
        time.sleep(86400)


if __name__ == '__main__':
    print("Starting Flask app for person detection with Bot-SORT tracking...")
    print("Camera will auto-start when the app launches")
    print("RFID monitoring will start automatically")
    print("Detection will only work when RFID card is present")
    print("Make sure you have installed: pip install ultralytics opencv-python flask pillow scipy pyscard")
    
    # Load auto-sync state from database on startup
    try:
        conn = get_connection() if get_connection else None
        if conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT setting_value FROM settings WHERE setting_key = 'auto_sync_enabled'"
                )
                result = cur.fetchone()
                if result and result.get('setting_value'):
                    enabled = result['setting_value'].lower() in ('1', 'true', 'yes', 'on')
                    with auto_sync_lock:
                        auto_sync_enabled = enabled
                    print(f"✓ Auto-sync state loaded from database: {'enabled' if enabled else 'disabled'}")
                else:
                    # Default to enabled if not set, and save to database
                    with auto_sync_lock:
                        auto_sync_enabled = True
                    with conn.cursor() as save_cur:
                        save_cur.execute(
                            """
                            INSERT INTO settings (setting_key, setting_value)
                            VALUES ('auto_sync_enabled', '1')
                            ON DUPLICATE KEY UPDATE setting_value = '1'
                            """
                        )
                        conn.commit()
                    print("✓ Auto-sync state initialized to enabled (default)")
    except Exception as e:
        print(f"⚠ Warning: Could not load auto-sync state from database: {e}")
        # Default to enabled on error
        with auto_sync_lock:
            auto_sync_enabled = True
    
    # Start follow-up email scheduler in background
    try:
        followup_thread = threading.Thread(target=followup_email_scheduler, daemon=True)
        followup_thread.start()
        print("✓ Follow-up email scheduler started (checks daily for violations 3+ days old)")
    except Exception as e:
        print(f"✗ Warning: Could not start follow-up email scheduler: {e}")
    
    # Start schedule RFID checker in background
    try:
        schedule_checker_thread = threading.Thread(target=schedule_rfid_checker, daemon=True)
        schedule_checker_thread.start()
        print("✓ Schedule RFID checker started (checks every 10 seconds to enable/disable RFID based on schedule)")
    except Exception as e:
        print(f"✗ Warning: Could not start schedule RFID checker: {e}")
    
    # Start auto-sync checker in background (Local → Aiven backup)
    try:
        auto_sync_thread = threading.Thread(target=auto_sync_to_aiven, daemon=True)
        auto_sync_thread.start()
        sync_status = "enabled" if auto_sync_enabled else "disabled"
        print(f"✓ Auto-sync to Aiven (backup) started - Status: {sync_status} (syncs every 5 minutes when Aiven is available and enabled)")
    except Exception as e:
        print(f"✗ Warning: Could not start auto-sync to Aiven checker: {e}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
