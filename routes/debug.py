"""
Debug and test routes for DRESS application.
Handles debugging endpoints and test mode toggling.
"""

from flask import Blueprint, request, jsonify
from flask_mail import Message
import time

debug_bp = Blueprint('debug', __name__)


@debug_bp.route('/api/database/status', methods=['GET'])
def database_status():
    """Get current database connection status"""
    try:
        from src.config import get_current_database, is_aiven_available, has_pending_sync
        
        current_db = get_current_database()
        aiven_available = is_aiven_available()
        pending_sync = has_pending_sync()
        
        return jsonify({
            'success': True,
            'current_db': current_db or 'unknown',
            'aiven_available': aiven_available,
            'pending_sync': pending_sync
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@debug_bp.route('/debug_rfid')
def debug_rfid():
    """Debug endpoint to check RFID status"""
    # Import here to avoid circular imports
    from app import rfid_lock, rfid_present, rfid_last_uid, rfid_event_queue, detection_enabled, rfid_last_student, RFID_AVAILABLE, get_rfid_status
    
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


@debug_bp.route('/debug_state')
def debug_state():
    """Debug endpoint to check current system state"""
    # Import here to avoid circular imports
    from app import rfid_lock, rfid_present, rfid_last_uid, rfid_last_student, rfid_consecutive_non_compliant, rfid_last_compliance_status, rfid_current_uid_violated, detection_enabled, current_frame, RESULT_FOLDER
    import os
    
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


@debug_bp.route('/test_email', methods=['POST'])
def test_email():
    """Test email sending functionality"""
    # Import here to avoid circular imports
    from flask import current_app
    from flask_mail import Mail
    from app import app, mail
    
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


@debug_bp.route('/toggle_test_mode', methods=['POST'])
def toggle_test_mode():
    """Toggle test mode on/off"""
    # Import here to avoid circular imports
    from app import test_mode, detection_enabled, test_mode_lock, rfid_lock, rfid_present, rfid_enabled, set_rfid_enabled
    import app as app_module
    
    try:
        data = request.get_json()
        test_mode_enabled = data.get('test_mode', False)
        
        with test_mode_lock:
            app_module.test_mode = test_mode_enabled
            
        if test_mode_enabled:
            # In test mode, always enable detection and disable RFID
            app_module.detection_enabled = True
            # Disable RFID processing during test mode
            if set_rfid_enabled:
                set_rfid_enabled(False)
            app_module.rfid_enabled = False
            return jsonify({'success': True, 'test_mode': True, 'message': 'Test mode activated - Detection always enabled, RFID disabled'})
        else:
            # Exit test mode, return to RFID-based detection
            # Re-enable RFID if it was enabled before
            if set_rfid_enabled:
                set_rfid_enabled(True)
            app_module.rfid_enabled = True
            with rfid_lock:
                app_module.detection_enabled = rfid_present
            return jsonify({'success': True, 'test_mode': False, 'message': 'Test mode deactivated - Detection requires RFID card'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error toggling test mode: {str(e)}'}), 500

