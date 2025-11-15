"""
Camera and detection routes for DRESS application.
Handles camera control, video feed, and detection toggling.
"""

from flask import Blueprint, request, jsonify, Response, session
import base64
import cv2

camera_bp = Blueprint('camera', __name__)


@camera_bp.route('/video_feed')
def video_feed():
    """Video streaming route"""
    # Import here to avoid circular imports
    from app import generate_frames
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@camera_bp.route('/start_camera', methods=['POST'])
def start_camera():
    """Start webcam and RFID monitoring (or return status if already running)"""
    # Import here to avoid circular imports
    from app import camera, selected_camera_id, rfid_event_queue, rfid_enabled, initialize_rfid, start_rfid_monitoring, set_rfid_enabled, RFID_AVAILABLE, initialize_camera
    
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
            import app as app_module
            app_module.selected_camera_id = camera_id
            # Use initialize_camera which now handles detection queue setup
            if initialize_camera():
                # Initialize and start RFID monitoring when camera starts
                if RFID_AVAILABLE:
                    try:
                        # Initialize RFID if not already done
                        if rfid_event_queue is None:
                            initialize_rfid()
                        else:
                            start_rfid_monitoring()
                        
                        # Enable RFID processing
                        app_module.rfid_enabled = True
                        set_rfid_enabled(True)  # Enable RFID polling
                        print(f"DEBUG: RFID enabled set to True, rfid_enabled: {app_module.rfid_enabled}")
                        print("RFID monitoring started with camera")
                    except Exception as e:
                        print(f"Warning: Could not start RFID monitoring: {e}")
                
                return jsonify({'success': True, 'message': f'Camera {camera_id} and RFID monitoring started successfully'})
            else:
                import app as app_module
                app_module.camera = None
                return jsonify({'success': False, 'message': f'Failed to open camera {camera_id}'}), 500
        else:
            return jsonify({'success': True, 'message': 'Camera already running'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error starting camera: {str(e)}'}), 500


@camera_bp.route('/change_camera', methods=['POST'])
def change_camera():
    """Change to a different camera and enable RFID monitoring"""
    # Import here to avoid circular imports
    from app import camera, detection_enabled, selected_camera_id, rfid_enabled, rfid_event_queue, initialize_rfid, start_rfid_monitoring, set_rfid_enabled, RFID_AVAILABLE, initialize_camera
    import app as app_module
    
    try:
        data = request.get_json()
        camera_id = data.get('camera_id', 0)
        
        # Stop current camera
        if camera is not None:
            camera.release()
            app_module.camera = None
        
        # Start new camera using initialize_camera which handles detection queue setup
        app_module.selected_camera_id = camera_id  # Update global selected camera ID
        if initialize_camera():
            app_module.detection_enabled = False  # Reset detection when changing camera
            
            # Enable RFID monitoring when camera is started via switching
            if RFID_AVAILABLE:
                try:
                    if rfid_event_queue is None:
                        initialize_rfid()
                    else:
                        start_rfid_monitoring()
                    
                    # Enable RFID processing
                    app_module.rfid_enabled = True
                    set_rfid_enabled(True)  # Enable RFID polling
                    print(f"DEBUG: RFID enabled via camera switch, rfid_enabled: {app_module.rfid_enabled}")
                    print("RFID monitoring started with camera switch")
                except Exception as e:
                    print(f"Warning: Could not start RFID monitoring during camera switch: {e}")
            
            return jsonify({'success': True, 'message': f'Switched to camera {camera_id} and enabled RFID monitoring'})
        else:
            app_module.camera = None
            return jsonify({'success': False, 'message': f'Failed to open camera {camera_id}'}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error changing camera: {str(e)}'}), 500


@camera_bp.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop webcam and RFID monitoring"""
    # Import here to avoid circular imports
    from app import camera, detection_enabled, rfid_event_queue, rfid_enabled, rfid_lock, rfid_present, rfid_last_student, rfid_consecutive_non_compliant, rfid_last_compliance_status, rfid_current_uid_violated, rfid_last_uid, RFID_AVAILABLE, set_rfid_enabled, stop_rfid_monitoring, unsubscribe_from_rfid_events
    import app as app_module
    
    try:
        if camera is not None:
            camera.release()
            app_module.camera = None
            app_module.detection_enabled = False  # Also disable detection when camera stops
            
            # Stop RFID monitoring when camera stops
            if RFID_AVAILABLE:
                try:
                    # Disable RFID processing first
                    app_module.rfid_enabled = False
                    set_rfid_enabled(False)  # Disable RFID polling
                    print(f"DEBUG: RFID enabled set to False, rfid_enabled: {app_module.rfid_enabled}")
                    
                    stop_rfid_monitoring()
                    
                    # Unsubscribe from RFID events to completely stop monitoring
                    if rfid_event_queue is not None:
                        unsubscribe_from_rfid_events(rfid_event_queue)
                        app_module.rfid_event_queue = None
                    
                    print("RFID monitoring completely stopped with camera")
                    
                    # Reset RFID state
                    with rfid_lock:
                        app_module.rfid_present = False
                        app_module.rfid_last_student = None
                        app_module.rfid_consecutive_non_compliant = 0
                        app_module.rfid_last_compliance_status = None
                        app_module.rfid_current_uid_violated = False
                        app_module.rfid_last_uid = None
                        
                except Exception as e:
                    print(f"Warning: Could not stop RFID monitoring: {e}")
            
            return jsonify({'success': True, 'message': 'Camera and RFID monitoring stopped successfully'})
        else:
            return jsonify({'success': True, 'message': 'Camera was not running'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error stopping camera: {str(e)}'}), 500


@camera_bp.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    """Toggle person detection on/off (only works if RFID card is present)"""
    # Import here to avoid circular imports
    from app import detection_enabled, rfid_present, rfid_lock
    import app as app_module
    
    try:
        with rfid_lock:
            if not rfid_present:
                return jsonify({'success': False, 'message': 'RFID card must be present to enable detection'}), 400
            
            app_module.detection_enabled = not detection_enabled
            status = "enabled" if app_module.detection_enabled else "disabled"
            return jsonify({'success': True, 'detection_enabled': app_module.detection_enabled, 'message': f'Detection {status}'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error toggling detection: {str(e)}'}), 500


@camera_bp.route('/capture_frame', methods=['POST'])
def capture_frame():
    """Capture current frame and return detection results with tracking"""
    # Import here to avoid circular imports
    from app import current_frame, frame_lock, detection_enabled, test_mode, test_mode_lock, rfid_lock, rfid_present, rfid_last_student, rfid_current_uid_violated, detect_persons_frame_with_dress, draw_detections_frame, _maybe_record_violation
    import app as app_module
    
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


@camera_bp.route('/get_cameras', methods=['GET'])
def get_cameras():
    """Get list of available cameras"""
    import platform
    import cv2
    
    try:
        cameras = []
        
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


@camera_bp.route('/reset_tracker', methods=['POST'])
def reset_tracker():
    """Reset the tracker to clear all tracking IDs"""
    # Import here to avoid circular imports
    from src.botsort_tracker import BotSORT
    import app as app_module
    
    try:
        app_module.tracker = BotSORT()
        return jsonify({'success': True, 'message': 'Tracker reset successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error resetting tracker: {str(e)}'}), 500

