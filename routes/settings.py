"""
Settings routes for DRESS application.
Handles system settings including schedule configuration.
"""

from flask import Blueprint, request, jsonify, session
import json
from datetime import datetime

settings_bp = Blueprint('settings', __name__)


@settings_bp.route('/api/settings/schedule', methods=['GET'])
def get_schedule():
    """Get the system schedule settings"""
    # Import here to avoid circular imports
    from app import get_connection
    
    try:
        # Check if user is authenticated and has security role
        admin = session.get('admin') or {}
        role = str(admin.get('role') or '').lower()
        if role != 'security':
            return jsonify({'success': False, 'message': 'Unauthorized'}), 403
        
        conn = get_connection() if get_connection else None
        if conn is None:
            return jsonify({'success': False, 'message': 'Database not configured'}), 500
        
        with conn.cursor() as cur:
            cur.execute(
                "SELECT setting_value FROM settings WHERE setting_key = 'system_schedule'"
            )
            result = cur.fetchone()
            
            if result and result.get('setting_value'):
                schedule = json.loads(result['setting_value'])
                return jsonify({'success': True, 'schedule': schedule})
            else:
                return jsonify({'success': True, 'schedule': []})
                
    except Exception as e:
        print(f"Error getting schedule: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@settings_bp.route('/api/settings/schedule', methods=['POST'])
def save_schedule():
    """Save the system schedule settings"""
    # Import here to avoid circular imports
    from app import get_connection
    
    try:
        # Check if user is authenticated and has security role
        admin = session.get('admin') or {}
        role = str(admin.get('role') or '').lower()
        if role != 'security':
            return jsonify({'success': False, 'message': 'Unauthorized'}), 403
        
        data = request.get_json()
        schedule = data.get('schedule', [])
        
        # Validate schedule entries
        for entry in schedule:
            if not entry.get('day') or not entry.get('start_time') or not entry.get('end_time'):
                return jsonify({'success': False, 'message': 'All fields are required for each schedule entry'}), 400
            
            if entry['start_time'] >= entry['end_time']:
                return jsonify({'success': False, 'message': 'Start time must be before end time'}), 400
        
        conn = get_connection() if get_connection else None
        if conn is None:
            return jsonify({'success': False, 'message': 'Database not configured'}), 500
        
        schedule_json = json.dumps(schedule)
        
        # Check if column needs to be altered (for existing databases)
        try:
            with conn.cursor() as cur:
                # Check current column type
                cur.execute("""
                    SELECT COLUMN_TYPE 
                    FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_SCHEMA = DATABASE() 
                    AND TABLE_NAME = 'settings' 
                    AND COLUMN_NAME = 'setting_value'
                """)
                result = cur.fetchone()
                
                if result and 'varchar' in result.get('COLUMN_TYPE', '').lower():
                    # Column is still VARCHAR, need to alter it to TEXT
                    print("Altering setting_value column from VARCHAR to TEXT...")
                    cur.execute("ALTER TABLE settings MODIFY COLUMN setting_value TEXT NOT NULL")
                    conn.commit()
                    print("Column altered successfully")
        except Exception as alter_error:
            print(f"Note: Could not alter column (may already be TEXT or error): {alter_error}")
        
        with conn.cursor() as cur:
            # Use INSERT ... ON DUPLICATE KEY UPDATE to handle both insert and update
            cur.execute(
                """
                INSERT INTO settings (setting_key, setting_value)
                VALUES ('system_schedule', %s)
                ON DUPLICATE KEY UPDATE setting_value = %s
                """,
                (schedule_json, schedule_json)
            )
            conn.commit()
        
        return jsonify({'success': True, 'message': 'Schedule saved successfully'})
        
    except Exception as e:
        print(f"Error saving schedule: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@settings_bp.route('/api/settings/schedule/check', methods=['GET'])
def check_schedule():
    """Check if the system should be active based on current time and schedule"""
    # Import here to avoid circular imports
    from app import get_connection
    
    try:
        conn = get_connection() if get_connection else None
        if conn is None:
            # If DB not configured, allow system to run
            return jsonify({'success': True, 'active': True, 'reason': 'Database not configured'})
        
        with conn.cursor() as cur:
            cur.execute(
                "SELECT setting_value FROM settings WHERE setting_key = 'system_schedule'"
            )
            result = cur.fetchone()
            
            if not result or not result.get('setting_value'):
                # No schedule set, system is always active
                return jsonify({'success': True, 'active': True, 'reason': 'No schedule configured'})
            
            schedule = json.loads(result['setting_value'])
            
            if not schedule or len(schedule) == 0:
                # Empty schedule, system is always active
                return jsonify({'success': True, 'active': True, 'reason': 'No schedule entries'})
            
            # Get current day and time
            now = datetime.now()
            current_day = now.strftime('%A')  # e.g., 'Monday'
            current_time = now.strftime('%H:%M')  # e.g., '14:30'
            
            # Check if current time falls within any schedule entry for today
            for entry in schedule:
                if entry['day'] == current_day:
                    if entry['start_time'] <= current_time < entry['end_time']:
                        return jsonify({
                            'success': True,
                            'active': True,
                            'reason': f"Within schedule: {entry['day']} {entry['start_time']}-{entry['end_time']}"
                        })
            
            # Not within any schedule
            return jsonify({
                'success': True,
                'active': False,
                'reason': f'Outside scheduled hours ({current_day} {current_time})'
            })
            
    except Exception as e:
        print(f"Error checking schedule: {e}")
        # On error, allow system to run (fail open)
        return jsonify({'success': True, 'active': True, 'reason': f'Error checking schedule: {str(e)}'})


@settings_bp.route('/api/settings/auto-sync', methods=['GET'])
def get_auto_sync_status():
    """Get the auto-sync enabled status from database"""
    # Import here to avoid circular imports
    from app import get_connection
    import app
    
    try:
        # Check if user is authenticated and has security role
        admin = session.get('admin') or {}
        role = str(admin.get('role') or '').lower()
        if role != 'security':
            return jsonify({'success': False, 'message': 'Unauthorized'}), 403
        
        conn = get_connection() if get_connection else None
        if conn is None:
            # If DB not configured, return default (enabled)
            return jsonify({'success': True, 'enabled': True})
        
        # Try to get from database first
        with conn.cursor() as cur:
            cur.execute(
                "SELECT setting_value FROM settings WHERE setting_key = 'auto_sync_enabled'"
            )
            result = cur.fetchone()
            
            if result and result.get('setting_value'):
                # Parse the stored value
                enabled = result['setting_value'].lower() in ('1', 'true', 'yes', 'on')
            else:
                # Default to enabled if not set
                enabled = True
                # Save default to database
                cur.execute(
                    """
                    INSERT INTO settings (setting_key, setting_value)
                    VALUES ('auto_sync_enabled', '1')
                    ON DUPLICATE KEY UPDATE setting_value = '1'
                    """,
                )
                conn.commit()
        
        # Update global variable to match database
        with app.auto_sync_lock:
            app.auto_sync_enabled = enabled
        
        return jsonify({'success': True, 'enabled': enabled})
        
    except Exception as e:
        print(f"Error getting auto-sync status: {e}")
        import traceback
        traceback.print_exc()
        # Return default on error
        return jsonify({'success': True, 'enabled': True})


@settings_bp.route('/api/settings/auto-sync', methods=['POST'])
def toggle_auto_sync():
    """Toggle auto-sync enabled/disabled and save to database"""
    # Import here to avoid circular imports
    from app import get_connection
    import app
    
    try:
        # Check if user is authenticated and has security role
        admin = session.get('admin') or {}
        role = str(admin.get('role') or '').lower()
        if role != 'security':
            return jsonify({'success': False, 'message': 'Unauthorized'}), 403
        
        data = request.get_json()
        enabled = data.get('enabled', True)
        
        # Save to database
        conn = get_connection() if get_connection else None
        if conn is None:
            return jsonify({'success': False, 'message': 'Database not configured'}), 500
        
        # Store as '1' for enabled, '0' for disabled
        setting_value = '1' if enabled else '0'
        
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO settings (setting_key, setting_value)
                VALUES ('auto_sync_enabled', %s)
                ON DUPLICATE KEY UPDATE setting_value = %s
                """,
                (setting_value, setting_value)
            )
            conn.commit()
        
        # Update global variable
        with app.auto_sync_lock:
            app.auto_sync_enabled = enabled
        
        status = "enabled" if enabled else "disabled"
        print(f"Auto-sync {status} by user {admin.get('username', 'unknown')} (saved to database)")
        
        return jsonify({
            'success': True, 
            'enabled': enabled,
            'message': f'Auto-sync {status}'
        })
        
    except Exception as e:
        print(f"Error toggling auto-sync: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

