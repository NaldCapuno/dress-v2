"""
RFID routes for DRESS application.
Handles RFID status and reading endpoints.
"""

from flask import Blueprint, request, jsonify
from src.debug_utils import debug_print, debug_rfid

rfid_bp = Blueprint('rfid', __name__)


@rfid_bp.route('/rfid/status', methods=['GET'])
def rfid_status():
    """Get RFID scanner status"""
    # Import here to avoid circular imports
    from app import rfid_last_uid, rfid_present, detection_enabled, rfid_lock, rfid_last_student, rfid_enabled, camera, RFID_AVAILABLE, get_rfid_status, test_mode, test_mode_lock
    
    try:
        # Check if test mode is active - if so, RFID is disabled
        with test_mode_lock:
            test_mode_active = test_mode
        
        with rfid_lock:
            camera_active = camera is not None and camera.isOpened()
            debug_rfid(f"RFID status check - rfid_enabled: {rfid_enabled}, rfid_present: {rfid_present}, test_mode: {test_mode_active}")
            
            # If RFID is disabled or test mode is active, return inactive status
            if not rfid_enabled or not camera_active or test_mode_active:
                status = {
                    'available': RFID_AVAILABLE,
                    'present': False,
                    'last_uid': None,
                    'detection_enabled': False,
                    'student': None,
                    'enabled': False,
                    'camera_active': camera_active,
                }
                debug_rfid("RFID disabled or camera inactive, returning inactive status")
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
                    'last_time': status.get('last_time', 0),  # Include timestamp from scanner
                })
                debug_rfid(f"RFID enabled, returning status: {status}")
        return jsonify({'success': True, 'status': status})
    except Exception as e:
        debug_rfid(f"RFID status error: {e}")
        return jsonify({'success': False, 'message': f'Error getting RFID status: {str(e)}'}), 500


@rfid_bp.route('/rfid/read', methods=['POST'])
def rfid_read():
    """Read RFID card data once"""
    # Import here to avoid circular imports
    from app import get_connection, get_rfid_uid, find_student_by_rfid, insert_rfid_log
    
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

