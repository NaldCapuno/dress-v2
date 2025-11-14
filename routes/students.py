"""
Student management routes for DRESS application.
Handles all student-related endpoints for dean role.
"""

from flask import Blueprint, request, jsonify, session

students_bp = Blueprint('students', __name__)


@students_bp.route('/dean/college', methods=['GET'])
def dean_college():
    """Return the dean's college."""
    admin = session.get('admin') or {}
    college = admin.get('college')
    if not college:
        return jsonify({'success': False, 'error': 'College not found'}), 404
    return jsonify({'success': True, 'college': college})


@students_bp.route('/dean/students', methods=['GET'])
def dean_students():
    """Return all students for the dean's college."""
    # Import here to avoid circular imports
    from app import get_connection
    
    try:
        admin = session.get('admin') or {}
        role = str(admin.get('role') or '').lower()
        if role != 'dean':
            return jsonify({'success': False, 'error': 'Unauthorized'}), 403
        
        college = admin.get('college')
        if not college:
            return jsonify({'success': True, 'students': []})
        
        conn = get_connection() if get_connection else None
        if conn is None:
            return jsonify({'success': False, 'error': 'Database not configured'}), 500
        
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT student_id, rfid_uid, name, gender, year_level, program, college, email, created_at
                    FROM students
                    WHERE college = %s
                    ORDER BY name ASC
                    """,
                    (college,)
                )
                students = cur.fetchall() or []
                return jsonify({'success': True, 'students': students})
        finally:
            conn.close()
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@students_bp.route('/dean/students/<student_id>', methods=['GET'])
def dean_get_student(student_id):
    """Get a single student by ID."""
    # Import here to avoid circular imports
    from app import get_connection
    
    try:
        admin = session.get('admin') or {}
        role = str(admin.get('role') or '').lower()
        if role != 'dean':
            return jsonify({'success': False, 'error': 'Unauthorized'}), 403
        
        college = admin.get('college')
        if not college:
            return jsonify({'success': False, 'error': 'College not found'}), 404
        
        conn = get_connection() if get_connection else None
        if conn is None:
            return jsonify({'success': False, 'error': 'Database not configured'}), 500
        
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT student_id, rfid_uid, name, gender, year_level, program, college, email, created_at
                    FROM students
                    WHERE student_id = %s AND college = %s
                    LIMIT 1
                    """,
                    (student_id, college)
                )
                student = cur.fetchone()
                if not student:
                    return jsonify({'success': False, 'error': 'Student not found'}), 404
                return jsonify({'success': True, 'student': student})
        finally:
            conn.close()
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@students_bp.route('/dean/students/<student_id>', methods=['PUT'])
def dean_update_student(student_id):
    """Update a student."""
    # Import here to avoid circular imports
    from app import get_connection
    
    try:
        admin = session.get('admin') or {}
        role = str(admin.get('role') or '').lower()
        if role != 'dean':
            return jsonify({'success': False, 'error': 'Unauthorized'}), 403
        
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        college = admin.get('college')
        if not college:
            return jsonify({'success': False, 'error': 'College not found'}), 404
        
        # Ensure the student belongs to the dean's college
        conn = get_connection() if get_connection else None
        if conn is None:
            return jsonify({'success': False, 'error': 'Database not configured'}), 500
        
        try:
            with conn.cursor() as cur:
                # Check if student exists and belongs to dean's college
                cur.execute(
                    "SELECT student_id FROM students WHERE student_id = %s AND college = %s",
                    (student_id, college)
                )
                if not cur.fetchone():
                    return jsonify({'success': False, 'error': 'Student not found or access denied'}), 404
                
                # Check if RFID UID is being changed and if it conflicts with another student
                if data.get('rfid_uid'):
                    cur.execute(
                        "SELECT student_id FROM students WHERE rfid_uid = %s AND student_id != %s",
                        (data['rfid_uid'], student_id)
                    )
                    if cur.fetchone():
                        return jsonify({'success': False, 'error': 'RFID UID already exists for another student'}), 400
                
                # Update the student (student_id and college cannot be changed)
                update_fields = []
                update_values = []
                
                if 'rfid_uid' in data:
                    update_fields.append('rfid_uid = %s')
                    update_values.append(data['rfid_uid'])
                if 'name' in data:
                    update_fields.append('name = %s')
                    update_values.append(data['name'])
                if 'email' in data:
                    update_fields.append('email = %s')
                    update_values.append(data['email'])
                if 'gender' in data:
                    update_fields.append('gender = %s')
                    update_values.append(data['gender'])
                if 'year_level' in data:
                    update_fields.append('year_level = %s')
                    update_values.append(int(data['year_level']))
                if 'program' in data:
                    update_fields.append('program = %s')
                    update_values.append(data['program'])
                
                if not update_fields:
                    return jsonify({'success': False, 'error': 'No fields to update'}), 400
                
                update_values.append(student_id)
                update_values.append(college)
                
                cur.execute(
                    f"""
                    UPDATE students
                    SET {', '.join(update_fields)}
                    WHERE student_id = %s AND college = %s
                    """,
                    update_values
                )
                conn.commit()
                return jsonify({'success': True, 'message': 'Student updated successfully'})
        except Exception as e:
            conn.rollback()
            error_msg = str(e)
            if 'Duplicate entry' in error_msg:
                if 'rfid_uid' in error_msg:
                    return jsonify({'success': False, 'error': 'RFID UID already exists'}), 400
            return jsonify({'success': False, 'error': f'Database error: {error_msg}'}), 500
        finally:
            conn.close()
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@students_bp.route('/dean/students/<student_id>', methods=['DELETE'])
def dean_delete_student(student_id):
    """Delete a student."""
    # Import here to avoid circular imports
    from app import get_connection
    
    try:
        admin = session.get('admin') or {}
        role = str(admin.get('role') or '').lower()
        if role != 'dean':
            return jsonify({'success': False, 'error': 'Unauthorized'}), 403
        
        college = admin.get('college')
        if not college:
            return jsonify({'success': False, 'error': 'College not found'}), 404
        
        conn = get_connection() if get_connection else None
        if conn is None:
            return jsonify({'success': False, 'error': 'Database not configured'}), 500
        
        try:
            with conn.cursor() as cur:
                # Check if student exists and belongs to dean's college
                cur.execute(
                    "SELECT student_id FROM students WHERE student_id = %s AND college = %s",
                    (student_id, college)
                )
                if not cur.fetchone():
                    return jsonify({'success': False, 'error': 'Student not found or access denied'}), 404
                
                # Delete the student
                cur.execute(
                    "DELETE FROM students WHERE student_id = %s AND college = %s",
                    (student_id, college)
                )
                conn.commit()
                return jsonify({'success': True, 'message': 'Student deleted successfully'})
        except Exception as e:
            conn.rollback()
            return jsonify({'success': False, 'error': f'Database error: {str(e)}'}), 500
        finally:
            conn.close()
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@students_bp.route('/dean/students/add', methods=['POST'])
def dean_add_student():
    """Add a new student to the database."""
    # Import here to avoid circular imports
    from app import get_connection
    
    try:
        admin = session.get('admin') or {}
        role = str(admin.get('role') or '').lower()
        if role != 'dean':
            return jsonify({'success': False, 'error': 'Unauthorized'}), 403
        
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['student_id', 'rfid_uid', 'name', 'email', 'gender', 'year_level', 'program', 'college']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'success': False, 'error': f'Missing required field: {field}'}), 400
        
        # Ensure the student is being added to the dean's college
        dean_college = admin.get('college')
        if dean_college and data.get('college') != dean_college:
            return jsonify({'success': False, 'error': 'You can only add students to your own college'}), 403
        
        # Use dean's college if not provided
        if not data.get('college') and dean_college:
            data['college'] = dean_college
        
        conn = get_connection() if get_connection else None
        if conn is None:
            return jsonify({'success': False, 'error': 'Database not configured'}), 500
        
        try:
            with conn.cursor() as cur:
                # Check if student_id already exists
                cur.execute("SELECT student_id FROM students WHERE student_id = %s", (data['student_id'],))
                if cur.fetchone():
                    return jsonify({'success': False, 'error': 'Student ID already exists'}), 400
                
                # Check if rfid_uid already exists
                cur.execute("SELECT rfid_uid FROM students WHERE rfid_uid = %s", (data['rfid_uid'],))
                if cur.fetchone():
                    return jsonify({'success': False, 'error': 'RFID UID already exists'}), 400
                
                # Insert the new student
                cur.execute(
                    """
                    INSERT INTO students (student_id, rfid_uid, name, gender, year_level, program, college, email)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        data['student_id'],
                        data['rfid_uid'],
                        data['name'],
                        data['gender'],
                        int(data['year_level']),
                        data['program'],
                        data['college'],
                        data['email']
                    )
                )
                conn.commit()
                return jsonify({'success': True, 'message': 'Student added successfully'})
        except Exception as e:
            conn.rollback()
            error_msg = str(e)
            if 'Duplicate entry' in error_msg:
                if 'student_id' in error_msg:
                    return jsonify({'success': False, 'error': 'Student ID already exists'}), 400
                elif 'rfid_uid' in error_msg:
                    return jsonify({'success': False, 'error': 'RFID UID already exists'}), 400
            return jsonify({'success': False, 'error': f'Database error: {error_msg}'}), 500
        finally:
            conn.close()
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

