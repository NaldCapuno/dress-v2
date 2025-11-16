"""
Dashboard routes for DRESS application.
Handles main dashboard, OSAS, Guidance, and Dean dashboards.
"""

from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for
import re

dashboards_bp = Blueprint('dashboards', __name__)


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


@dashboards_bp.route('/')
def index():
    """Main dashboard - redirect logged-in users to their appropriate dashboard"""
    admin = session.get('admin') or {}
    role = str(admin.get('role') or '').lower()
    
    # If not logged in, redirect to login
    if not admin or not role:
        return redirect(url_for('auth.login'))
    
    # Redirect logged-in users to their appropriate dashboard
    if role == 'security':
        return render_template('index.html')
    elif role == 'osas':
        return redirect(url_for('dashboards.osas_dashboard'))
    elif role == 'guidance':
        return redirect(url_for('dashboards.guidance_dashboard'))
    elif role == 'dean':
        return redirect(url_for('dashboards.dean_dashboard'))
    else:
        # Unknown role, redirect to login
        return redirect(url_for('auth.login'))


@dashboards_bp.route('/dashboard')
def dashboard():
    """Alias for the main dashboard; redirect logged-in users to their appropriate dashboard."""
    admin = session.get('admin') or {}
    role = str(admin.get('role') or '').lower()
    
    # If not logged in, redirect to login
    if not admin or not role:
        return redirect(url_for('auth.login'))
    
    # Redirect logged-in users to their appropriate dashboard
    if role == 'security':
        return render_template('index.html')
    elif role == 'osas':
        return redirect(url_for('dashboards.osas_dashboard'))
    elif role == 'guidance':
        return redirect(url_for('dashboards.guidance_dashboard'))
    elif role == 'dean':
        return redirect(url_for('dashboards.dean_dashboard'))
    else:
        # Unknown role, redirect to login
        return redirect(url_for('auth.login'))


@dashboards_bp.route('/osas', methods=['GET'])
def osas_dashboard():
    """OSAS dashboard - only accessible to admins with role 'osas'."""
    admin = session.get('admin') or {}
    role = str(admin.get('role') or '').lower()
    
    # If not logged in, redirect to login
    if not admin or not role:
        return redirect(url_for('auth.login'))
    
    # If wrong role, redirect to their appropriate dashboard
    if role != 'osas':
        if role == 'security':
            return redirect(url_for('dashboards.index'))
        elif role == 'guidance':
            return redirect(url_for('dashboards.guidance_dashboard'))
        elif role == 'dean':
            return redirect(url_for('dashboards.dean_dashboard'))
        else:
            return redirect(url_for('auth.login'))
    
    return render_template('osas_dashboard.html')


@dashboards_bp.route('/osas/colleges', methods=['GET'])
def osas_colleges():
    """Return distinct colleges for OSAS filtering."""
    from app import get_connection
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


@dashboards_bp.route('/osas/programs', methods=['GET'])
def osas_programs():
    """Return distinct programs for OSAS filtering (optionally filtered by college).
    Returns all enum values from the database schema, optionally filtered by college mapping."""
    from app import get_connection
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


@dashboards_bp.route('/guidance', methods=['GET'])
def guidance_dashboard():
    """Guidance dashboard - only accessible to admins with role 'guidance'."""
    admin = session.get('admin') or {}
    role = str(admin.get('role') or '').lower()
    
    # If not logged in, redirect to login
    if not admin or not role:
        return redirect(url_for('auth.login'))
    
    # If wrong role, redirect to their appropriate dashboard
    if role != 'guidance':
        if role == 'security':
            return redirect(url_for('dashboards.index'))
        elif role == 'osas':
            return redirect(url_for('dashboards.osas_dashboard'))
        elif role == 'dean':
            return redirect(url_for('dashboards.dean_dashboard'))
        else:
            return redirect(url_for('auth.login'))
    
    return render_template('guidance_dashboard.html')


@dashboards_bp.route('/guiance', methods=['GET'])
def guidance_alias():
    """Alias path for guidance (handles common misspelling)."""
    admin = session.get('admin') or {}
    role = str(admin.get('role') or '').lower()
    
    # If not logged in, redirect to login
    if not admin or not role:
        return redirect(url_for('auth.login'))
    
    # If wrong role, redirect to their appropriate dashboard
    if role != 'guidance':
        if role == 'security':
            return redirect(url_for('dashboards.index'))
        elif role == 'osas':
            return redirect(url_for('dashboards.osas_dashboard'))
        elif role == 'dean':
            return redirect(url_for('dashboards.dean_dashboard'))
        else:
            return redirect(url_for('auth.login'))
    
    return redirect(url_for('dashboards.guidance_dashboard'))


@dashboards_bp.route('/dean', methods=['GET'])
def dean_dashboard():
    """Dean dashboard - only accessible to admins with role 'dean'."""
    admin = session.get('admin') or {}
    role = str(admin.get('role') or '').lower()
    
    # If not logged in, redirect to login
    if not admin or not role:
        return redirect(url_for('auth.login'))
    
    # If wrong role, redirect to their appropriate dashboard
    if role != 'dean':
        if role == 'security':
            return redirect(url_for('dashboards.index'))
        elif role == 'osas':
            return redirect(url_for('dashboards.osas_dashboard'))
        elif role == 'guidance':
            return redirect(url_for('dashboards.guidance_dashboard'))
        else:
            return redirect(url_for('auth.login'))
    
    return render_template('dean_dashboard.html', college=admin.get('college'))


@dashboards_bp.route('/dean/programs', methods=['GET'])
def dean_programs():
    """Return distinct programs for the dean's college."""
    from app import get_connection
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

