"""
Violation routes for DRESS application.
Handles all violation-related endpoints for dean, osas, and guidance roles.
"""

from flask import Blueprint, request, jsonify, session, Response
from io import BytesIO
from datetime import datetime, timezone
import time
import re

violations_bp = Blueprint('violations', __name__)


def convert_timestamp_to_iso(timestamp):
    """Convert datetime object to ISO format string.
    MySQL datetime fields are naive (no timezone), so we format them as-is.
    The frontend will interpret them correctly if we ensure consistent format.
    """
    if not timestamp:
        return None
    if isinstance(timestamp, str):
        # If already a string, ensure it's in the right format
        # MySQL format: 'YYYY-MM-DD HH:MM:SS' -> convert to ISO: 'YYYY-MM-DDTHH:MM:SS'
        if 'T' not in timestamp and ' ' in timestamp:
            return timestamp.replace(' ', 'T')
        return timestamp
    if isinstance(timestamp, datetime):
        # Format as ISO string (YYYY-MM-DDTHH:MM:SS)
        # This preserves the exact time from the database without timezone conversion
        return timestamp.strftime('%Y-%m-%dT%H:%M:%S')
    return str(timestamp)


def convert_row_timestamps(row):
    """Convert timestamp fields in a row to ISO format strings."""
    if isinstance(row, dict) and 'timestamp' in row and row['timestamp']:
        row['timestamp'] = convert_timestamp_to_iso(row['timestamp'])
    return row


@violations_bp.route('/dean/violations', methods=['GET'])
def dean_get_violations():
    """List violations for dean review (defaults to cases forwarded to dean)."""
    # Import here to avoid circular imports
    from app import get_connection
    
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
        search = request.args.get('search', '').strip()
        sort_column = request.args.get('sort_column', 'timestamp')
        sort_direction = request.args.get('sort_direction', 'desc').upper()

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
            where.append("s.program = %s")
            params.append(program)
        if start_dt:
            where.append("v.timestamp >= %s")
            params.append(start_dt)
        if end_dt:
            where.append("v.timestamp <= %s")
            params.append(end_dt)
        if search:
            search_pattern = f"%{search}%"
            where.append("(s.name LIKE %s OR v.student_id LIKE %s OR s.program LIKE %s OR v.violation_type LIKE %s)")
            params.extend([search_pattern, search_pattern, search_pattern, search_pattern])
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

        # Validate and sanitize sort column
        allowed_sort_columns = {
            'violation_id': 'v.violation_id',
            'timestamp': 'v.timestamp',
            'violation_type': 'v.violation_type',
            'status': 'v.status',
            'name': 's.name',
            'student_id': 'v.student_id',
            'program': 's.program',
            'college': 's.college'
        }
        sort_column_sql = allowed_sort_columns.get(sort_column, 'v.timestamp')
        if sort_direction not in ('ASC', 'DESC'):
            sort_direction = 'DESC'
        order_by = f"{sort_column_sql} {sort_direction}"

        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) AS cnt FROM violations v LEFT JOIN students s ON v.student_id = s.student_id{where_sql}", params)
            total = (cur.fetchone() or {}).get('cnt', 0)

            cur.execute(
                f"{base_select}{where_sql} ORDER BY {order_by} LIMIT %s OFFSET %s",
                params + [page_size, offset]
            )
            rows = cur.fetchall() or []
            # Convert timestamps to ISO format strings for consistent display
            rows = [convert_row_timestamps(row) for row in rows]
        conn.close()
        return jsonify({'success': True, 'rows': rows, 'total': total})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@violations_bp.route('/dean/violation/<int:violation_id>/status', methods=['POST'])
def dean_update_violation_status(violation_id: int):
    """Dean can forward to guidance, set pending, or resolve."""
    # Import here to avoid circular imports
    from app import get_connection
    
    try:
        data = request.get_json(silent=True) or {}
        status = str(data.get('status') or '').strip().lower()
        print(f"Dean status update: violation_id={violation_id}, status={status}, data={data}")
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


@violations_bp.route('/dean/notifications', methods=['GET'])
def dean_notifications():
    """Recent violations for the dean's college, newest first."""
    # Import here to avoid circular imports
    from app import get_connection
    
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
            where.append("v.status = %s")
            params.append(status_filter)
        # Older than 3 days by default
        where.append("v.timestamp < NOW() - INTERVAL 3 DAY")
        where_sql = " WHERE " + " AND ".join(where)
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT v.violation_id, v.student_id, v.violation_type, v.timestamp, v.image_proof, v.status,
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

@violations_bp.route('/dean/analytics', methods=['GET'])
def dean_analytics():
    """Aggregate analytics for dean view (college-level)."""
    # Import here to avoid circular imports
    from app import get_connection
    
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
            cur.execute(
                f"SELECT v.status AS label, COUNT(*) AS cnt FROM violations v LEFT JOIN students s ON v.student_id=s.student_id{where_sql} GROUP BY v.status",
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


@violations_bp.route('/dean/alerts', methods=['GET'])
def dean_alerts():
    """Return alert info for the dean's college when any students have pending >3 days."""
    # Import here to avoid circular imports
    from app import get_connection, dean_alerts_cache
    
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
                WHERE v.status = 'pending' AND v.timestamp < NOW() - INTERVAL 3 DAY AND s.college = %s
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
                    WHERE v.status = 'pending' AND v.timestamp < NOW() - INTERVAL 3 DAY AND s.college = %s
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


@violations_bp.route('/dean/alerts/students', methods=['GET'])
def dean_alert_students():
    """Return distinct students with pending violations older than 3 days for the dean's college."""
    # Import here to avoid circular imports
    from app import get_connection
    
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
                WHERE v.status = 'pending' AND v.timestamp < NOW() - INTERVAL 3 DAY AND s.college = %s
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

@violations_bp.route('/dean/trend', methods=['GET'])
def dean_trend():
    # Import here to avoid circular imports
    from app import get_connection
    
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
@violations_bp.route('/osas/violations', methods=['GET'])
def osas_get_violations():
    """List violations for OSAS review (university-wide)."""
    # Import here to avoid circular imports
    from app import get_connection
    
    try:
        # Show all statuses by default; only filter if provided
        status_filter = request.args.get('status')
        start_dt = request.args.get('start')
        end_dt = request.args.get('end')
        academic_year = request.args.get('academic_year')
        semester = request.args.get('semester')
        college = request.args.get('college')
        program = request.args.get('program')
        search = request.args.get('search', '').strip()
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 50))
        offset = max(0, (page - 1) * page_size)
        sort_column = request.args.get('sort_column', 'timestamp')
        sort_direction = request.args.get('sort_direction', 'desc').upper()

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
        if search:
            search_pattern = f"%{search}%"
            where.append("(s.name LIKE %s OR v.student_id LIKE %s OR s.college LIKE %s OR s.program LIKE %s OR v.violation_type LIKE %s)")
            params.extend([search_pattern, search_pattern, search_pattern, search_pattern, search_pattern])
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

        # Validate and sanitize sort column
        allowed_sort_columns = {
            'violation_id': 'v.violation_id',
            'timestamp': 'v.timestamp',
            'violation_type': 'v.violation_type',
            'status': 'v.status',
            'name': 's.name',
            'student_id': 'v.student_id',
            'program': 's.program',
            'college': 's.college'
        }
        sort_column_sql = allowed_sort_columns.get(sort_column, 'v.timestamp')
        if sort_direction not in ('ASC', 'DESC'):
            sort_direction = 'DESC'
        order_by = f"{sort_column_sql} {sort_direction}"

        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) AS cnt FROM violations v LEFT JOIN students s ON v.student_id = s.student_id{where_sql}", params)
            total = (cur.fetchone() or {}).get('cnt', 0)

            cur.execute(
                f"{base_select}{where_sql} ORDER BY {order_by} LIMIT %s OFFSET %s",
                params + [page_size, offset]
            )
            rows = cur.fetchall() or []
            # Convert timestamps to ISO format strings for consistent display
            rows = [convert_row_timestamps(row) for row in rows]
        conn.close()
        return jsonify({'success': True, 'rows': rows, 'total': total})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ------------------ Guidance endpoints (university-wide counseling) ------------------
@violations_bp.route('/guidance/violations', methods=['GET'])
def guidance_get_violations():
    """List violations for Guidance review (university-wide)."""
    # Import here to avoid circular imports
    from app import get_connection
    
    try:
        status_filter = request.args.get('status')
        start_dt = request.args.get('start')
        end_dt = request.args.get('end')
        college = request.args.get('college')
        program = request.args.get('program')
        search = request.args.get('search', '').strip()
        academic_year = request.args.get('academic_year')
        semester = request.args.get('semester')
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 50))
        offset = max(0, (page - 1) * page_size)
        sort_column = request.args.get('sort_column', 'timestamp')
        sort_direction = request.args.get('sort_direction', 'desc').upper()

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
        if college:
            where.append("s.college = %s")
            params.append(college)
        if program:
            where.append("s.program = %s")
            params.append(program)
        if search:
            search_pattern = f"%{search}%"
            where.append("(s.name LIKE %s OR v.student_id LIKE %s OR s.college LIKE %s OR s.program LIKE %s OR v.violation_type LIKE %s)")
            params.extend([search_pattern, search_pattern, search_pattern, search_pattern, search_pattern])
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

        # Validate and sanitize sort column
        allowed_sort_columns = {
            'violation_id': 'v.violation_id',
            'timestamp': 'v.timestamp',
            'violation_type': 'v.violation_type',
            'status': 'v.status',
            'name': 's.name',
            'student_id': 'v.student_id',
            'program': 's.program',
            'college': 's.college'
        }
        sort_column_sql = allowed_sort_columns.get(sort_column, 'v.timestamp')
        if sort_direction not in ('ASC', 'DESC'):
            sort_direction = 'DESC'
        order_by = f"{sort_column_sql} {sort_direction}"

        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) AS cnt FROM violations v LEFT JOIN students s ON v.student_id = s.student_id{where_sql}", params)
            total = (cur.fetchone() or {}).get('cnt', 0)

            cur.execute(
                f"{base_select}{where_sql} ORDER BY {order_by} LIMIT %s OFFSET %s",
                params + [page_size, offset]
            )
            rows = cur.fetchall() or []
            # Convert timestamps to ISO format strings for consistent display
            rows = [convert_row_timestamps(row) for row in rows]
        conn.close()
        return jsonify({'success': True, 'rows': rows, 'total': total})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@violations_bp.route('/guidance/violation/<int:violation_id>/status', methods=['POST'])
def guidance_update_violation_status(violation_id: int):
    """Guidance can set pending or resolved."""
    # Import here to avoid circular imports
    from app import get_connection
    
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


@violations_bp.route('/guidance/analytics', methods=['GET'])
def guidance_analytics():
    """Aggregate analytics for Guidance view (university-wide)."""
    # Import here to avoid circular imports
    from app import get_connection
    
    try:
        start_dt = request.args.get('start')
        end_dt = request.args.get('end')
        college = request.args.get('college')
        program = request.args.get('program')
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


@violations_bp.route('/guidance/trend', methods=['GET'])
def guidance_trend():
    # Import here to avoid circular imports
    from app import get_connection
    
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

@violations_bp.route('/osas/violation/<int:violation_id>/status', methods=['POST'])
def osas_update_violation_status(violation_id: int):
    """OSAS can forward to dean, guidance, or resolve."""
    # Import here to avoid circular imports
    from app import get_connection
    
    try:
        data = request.get_json(silent=True) or {}
        status = str(data.get('status') or '').strip().lower()
        print(f"OSAS status update: violation_id={violation_id}, status={status}, data={data}")
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


@violations_bp.route('/osas/analytics', methods=['GET'])
def osas_analytics():
    """Aggregate analytics for OSAS view (university-wide)."""
    # Import here to avoid circular imports
    from app import get_connection
    
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


@violations_bp.route('/osas/trend', methods=['GET'])
def osas_trend():
    # Import here to avoid circular imports
    from app import get_connection
    
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


@violations_bp.route('/violation_log')
def violation_log():
    """Display recent violations with proof images"""
    # Import here to avoid circular imports
    from app import get_connection
    
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


@violations_bp.route('/violation_report')
def violation_report():
    """Generate comprehensive violation report"""
    # Import here to avoid circular imports
    from app import get_connection
    
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


@violations_bp.route('/dean/violations/pdf', methods=['GET'])
def dean_generate_pdf_report():
    """Generate PDF report of violations for the dean's college."""
    # Import here to avoid circular imports
    from app import get_connection, REPORTLAB_AVAILABLE, get_college_abbreviation
    
    if not REPORTLAB_AVAILABLE:
        return jsonify({'success': False, 'error': 'PDF generation library not available'}), 500
    
    try:
        # Import reportlab components
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        
        # Get filter parameters
        start_dt = request.args.get('start')
        end_dt = request.args.get('end')
        program = request.args.get('program')
        
        # Enforce dean can only generate reports for their own college (from session)
        # Ignore any college parameter from request to prevent unauthorized access
        admin = session.get('admin') or {}
        college = admin.get('college')
        
        if not college:
            return jsonify({'success': False, 'error': 'College not specified. Please ensure you are logged in as a dean.'}), 400
        
        conn = get_connection() if get_connection else None
        if conn is None:
            return jsonify({'success': False, 'error': 'DB not configured'}), 500
        
        # Build query with filters (same logic as violations endpoint)
        where = []
        params = []
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
        
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""
        
        # Get all violations (no pagination for PDF)
        query = (
            "SELECT v.violation_id, v.student_id, v.violation_type, v.timestamp, v.status, "
            "s.name, s.gender, s.program, s.college "
            "FROM violations v LEFT JOIN students s ON v.student_id = s.student_id"
            f"{where_sql} ORDER BY v.timestamp DESC"
        )
        
        with conn.cursor() as cur:
            cur.execute(query, params)
            violations = cur.fetchall() or []
            
            # Get summary statistics
            stats_query = (
                f"SELECT COUNT(*) AS total, "
                f"COUNT(DISTINCT v.student_id) AS unique_students, "
                f"COUNT(CASE WHEN v.status = 'resolved' THEN 1 END) AS resolved, "
                f"COUNT(CASE WHEN v.status != 'resolved' THEN 1 END) AS unresolved "
                f"FROM violations v LEFT JOIN students s ON v.student_id = s.student_id{where_sql}"
            )
            cur.execute(stats_query, params)
            stats = cur.fetchone() or {}
        
        conn.close()
        
        # Get college abbreviation
        college_abbr = get_college_abbreviation(college)
        
        # Generate PDF
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=A4, 
            topMargin=0.5*inch, 
            bottomMargin=0.5*inch,
            title=f"Violation Records Report - {college_abbr}",
            author=college_abbr,
            subject=f"Violation Records Report for {college_abbr}",
            creator=f"Dean Dashboard - {college_abbr}"
        )
        elements = []
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=14,
            textColor=colors.HexColor('#f25a04'),
            spaceAfter=12,
            alignment=1  # Center
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=11,
            textColor=colors.HexColor('#1e293b'),
            spaceAfter=8
        )
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=9
        )
        
        # Title
        elements.append(Paragraph("Violation Records Report", title_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # College and date info
        college_info = f"<b>College:</b> {college_abbr} ({college})"
        if program:
            college_info += f"<br/><b>Program:</b> {program}"
        if start_dt or end_dt:
            date_range = f"<b>Date Range:</b> "
            if start_dt:
                date_range += f"From {start_dt}"
            if end_dt:
                date_range += f" To {end_dt}"
            college_info += f"<br/>{date_range}"
        college_info += f"<br/><b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        elements.append(Paragraph(college_info, normal_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Summary Statistics
        elements.append(Paragraph("Summary Statistics", heading_style))
        stats_data = [
            ['Total Violations', str(stats.get('total', 0))],
            ['Unique Students', str(stats.get('unique_students', 0))],
            ['Resolved', str(stats.get('resolved', 0))],
            ['Unresolved', str(stats.get('unresolved', 0))]
        ]
        stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f3f4f6')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')])
        ]))
        elements.append(stats_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Violations Table
        elements.append(Paragraph("Violation Records", heading_style))
        
        if not violations:
            elements.append(Paragraph("No violations found for the selected criteria.", normal_style))
        else:
            # Table headers - use Paragraph for headers too
            table_data = [[
                Paragraph('Date', normal_style),
                Paragraph('Student ID', normal_style),
                Paragraph('Name', normal_style),
                Paragraph('Program', normal_style),
                Paragraph('Gender', normal_style),
                Paragraph('Non Compliant', normal_style),
                Paragraph('Status', normal_style)
            ]]
            
            # Table rows
            for v in violations:
                timestamp = v.get('timestamp', '')
                if timestamp:
                    try:
                        dt = datetime.strptime(str(timestamp), '%Y-%m-%d %H:%M:%S')
                        # Match table format: "Mon, 25 Oct 2025 08:00 AM"
                        # JavaScript getDay(): 0=Sun, 1=Mon, ..., 6=Sat
                        # Python weekday(): 0=Mon, 1=Tue, ..., 6=Sun
                        # Convert Python weekday to JavaScript getDay: (weekday + 1) % 7
                        weekdays = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
                        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        weekday = weekdays[(dt.weekday() + 1) % 7]
                        day = dt.strftime('%d')
                        month = months[dt.month - 1]
                        year = dt.year
                        hour_12 = dt.hour % 12 or 12
                        minute = dt.strftime('%M')
                        am_pm = 'AM' if dt.hour < 12 else 'PM'
                        formatted_date = f"{weekday}, {day} {month} {year} {hour_12}:{minute} {am_pm}"
                    except:
                        formatted_date = str(timestamp)[:16]
                else:
                    formatted_date = ''
                
                # Clean violation type - remove any status info that might be embedded
                violation_type = str(v.get('violation_type', '')).strip()
                status = str(v.get('status', '')).strip()
                
                # If violation_type contains status info (like "non-compliant: shoepending")
                # Extract and clean it
                if violation_type:
                    # Check for status patterns at the end of violation_type
                    # Patterns to look for (longest first to avoid partial matches)
                    status_patterns = [
                        '_pending', '_resolved', 'pending', 'resolved'
                    ]
                    
                    violation_type_lower = violation_type.lower()
                    extracted_status = None
                    
                    # Try to find and extract status from violation_type
                    for pattern in status_patterns:
                        pattern_lower = pattern.lower()
                        # Check if violation_type ends with this pattern
                        if violation_type_lower.endswith(pattern_lower):
                            # Extract the status
                            extracted_status = pattern.replace('_', ' ').strip()
                            # Remove the pattern from violation_type
                            violation_type = violation_type[:-len(pattern)].strip()
                            break
                    
                    # Use extracted status if original status is empty
                    if not status and extracted_status:
                        status = extracted_status
                    
                    # Clean up any double spaces, underscores, or trailing separators
                    violation_type = violation_type.rstrip(':_-').strip()
                    violation_type = violation_type.replace('_', ' ').strip()
                    violation_type = ' '.join(violation_type.split())
                    
                    # Remove "Non-compliant:" or "non-compliant:" prefix
                    if violation_type.lower().startswith('non-compliant:'):
                        violation_type = violation_type.split(':', 1)[1].strip() if ':' in violation_type else violation_type
                    elif violation_type.lower().startswith('non compliant'):
                        violation_type = violation_type.replace('non compliant', '').replace(':', '').strip()
                    
                    # Format violation type nicely - capitalize first letter
                    if violation_type:
                        violation_type = violation_type.title()
                
                # Format status nicely
                if status:
                    status = status.replace('_', ' ').strip().title()
                else:
                    status = 'N/A'
                
                # Clean program name - remove "Bachelor of Science in" prefix
                program_name = str(v.get('program', '')).strip()
                if program_name:
                    # Remove common prefixes
                    prefixes_to_remove = [
                        'Bachelor of Science in ',
                        'Bachelor of Arts in ',
                        'Bachelor of Science in',
                        'Bachelor of Arts in',
                        'Bachelor of Science',
                        'Bachelor of Arts'
                    ]
                    for prefix in prefixes_to_remove:
                        if program_name.startswith(prefix):
                            program_name = program_name[len(prefix):].strip()
                            break
                
                # Format gender - capitalize first letter
                gender = str(v.get('gender', '')).strip()
                if gender:
                    gender = gender.capitalize()
                else:
                    gender = 'N/A'
                
                # Use Paragraph objects for better text wrapping
                table_data.append([
                    Paragraph(formatted_date, normal_style),
                    Paragraph(str(v.get('student_id', '')), normal_style),
                    Paragraph(str(v.get('name', ''))[:30], normal_style),
                    Paragraph(program_name[:35] if program_name else 'N/A', normal_style),
                    Paragraph(gender[:8], normal_style),
                    Paragraph(violation_type[:35] if violation_type else 'N/A', normal_style),
                    Paragraph(status[:20] if status else 'N/A', normal_style)
                ])
            
            # Create table with adjusted column widths - total width ~7.2 inches (fits A4 with margins)
            # Date, Student ID, Name, Program, Gender, Non Compliant, Status
            violation_table = Table(table_data, colWidths=[1.0*inch, 0.9*inch, 1.0*inch, 1.1*inch, 0.7*inch, 1.2*inch, 0.9*inch])
            violation_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f25a04')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 7),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('TOPPADDING', (0, 0), (-1, 0), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')]),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 3),
                ('RIGHTPADDING', (0, 0), (-1, -1), 3),
                ('TOPPADDING', (0, 1), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
            ]))
            elements.append(violation_table)
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        
        # Generate filename using abbreviation
        filename = f"violation_report_{college_abbr}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filename = re.sub(r'[^\w\-_\.]', '_', filename)  # Sanitize filename
        
        return Response(
            buffer.getvalue(),
            mimetype='application/pdf',
            headers={
                'Content-Disposition': f'attachment; filename="{filename}"'
            }
        )
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@violations_bp.route('/osas/violations/pdf', methods=['GET'])
def osas_generate_pdf_report():
    """Generate PDF report of ALL violations for OSAS (university-wide, filtered by date range)."""
    # Import here to avoid circular imports
    from app import get_connection, REPORTLAB_AVAILABLE, get_college_abbreviation
    
    if not REPORTLAB_AVAILABLE:
        return jsonify({'success': False, 'error': 'PDF generation library not available'}), 500
    
    try:
        # Import reportlab components
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        
        # Get filter parameters
        start_dt = request.args.get('start')
        end_dt = request.args.get('end')
        college = request.args.get('college')  # Optional filter
        program = request.args.get('program')  # Optional filter
        
        conn = get_connection() if get_connection else None
        if conn is None:
            return jsonify({'success': False, 'error': 'DB not configured'}), 500
        
        # Build query with filters - NO college restriction (OSAS sees all)
        where = []
        params = []
        
        # Optional filters (not required)
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
        
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""
        
        # Get all violations (no pagination for PDF)
        query = (
            "SELECT v.violation_id, v.student_id, v.violation_type, v.timestamp, v.status, "
            "s.name, s.gender, s.program, s.college "
            "FROM violations v LEFT JOIN students s ON v.student_id = s.student_id"
            f"{where_sql} ORDER BY v.timestamp DESC"
        )
        
        with conn.cursor() as cur:
            cur.execute(query, params)
            violations = cur.fetchall() or []
            
            # Get summary statistics
            stats_query = (
                f"SELECT COUNT(*) AS total, "
                f"COUNT(DISTINCT v.student_id) AS unique_students, "
                f"COUNT(CASE WHEN v.status = 'resolved' THEN 1 END) AS resolved, "
                f"COUNT(CASE WHEN v.status != 'resolved' THEN 1 END) AS unresolved "
                f"FROM violations v LEFT JOIN students s ON v.student_id = s.student_id{where_sql}"
            )
            cur.execute(stats_query, params)
            stats = cur.fetchone() or {}
        
        conn.close()
        
        # Generate PDF
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=A4, 
            topMargin=0.5*inch, 
            bottomMargin=0.5*inch,
            title="OSAS Violation Records Report",
            author="OSAS Dashboard",
            subject="University-Wide Violation Records Report",
            creator="OSAS Dashboard"
        )
        elements = []
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=14,
            textColor=colors.HexColor('#f25a04'),
            spaceAfter=12,
            alignment=1  # Center
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=11,
            textColor=colors.HexColor('#1e293b'),
            spaceAfter=8
        )
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=9
        )
        
        # Title
        elements.append(Paragraph("OSAS Violation Records Report", title_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Report info
        report_info = "<b>Scope:</b> All Colleges (University-Wide)"
        if college:
            college_abbr = get_college_abbreviation(college)
            report_info += f"<br/><b>Filtered by College:</b> {college_abbr} ({college})"
        if program:
            report_info += f"<br/><b>Filtered by Program:</b> {program}"
        if start_dt or end_dt:
            date_range = f"<b>Date Range:</b> "
            if start_dt:
                date_range += f"From {start_dt}"
            if end_dt:
                date_range += f" To {end_dt}"
            report_info += f"<br/>{date_range}"
        report_info += f"<br/><b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        elements.append(Paragraph(report_info, normal_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Summary Statistics
        elements.append(Paragraph("Summary Statistics", heading_style))
        stats_data = [
            ['Total Violations', str(stats.get('total', 0))],
            ['Unique Students', str(stats.get('unique_students', 0))],
            ['Resolved', str(stats.get('resolved', 0))],
            ['Unresolved', str(stats.get('unresolved', 0))]
        ]
        stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f3f4f6')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')])
        ]))
        elements.append(stats_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Violations Table
        elements.append(Paragraph("Violation Records", heading_style))
        
        if not violations:
            elements.append(Paragraph("No violations found for the selected criteria.", normal_style))
        else:
            # Table headers - use Paragraph for headers too
            table_data = [[
                Paragraph('Date', normal_style),
                Paragraph('Student ID', normal_style),
                Paragraph('Name', normal_style),
                Paragraph('College', normal_style),
                Paragraph('Program', normal_style),
                Paragraph('Gender', normal_style),
                Paragraph('Non Compliant', normal_style),
                Paragraph('Status', normal_style)
            ]]
            
            # Table rows
            for v in violations:
                timestamp = v.get('timestamp', '')
                if timestamp:
                    try:
                        dt = datetime.strptime(str(timestamp), '%Y-%m-%d %H:%M:%S')
                        # Match table format: "Mon, 25 Oct 2025 08:00 AM"
                        # JavaScript getDay(): 0=Sun, 1=Mon, ..., 6=Sat
                        # Python weekday(): 0=Mon, 1=Tue, ..., 6=Sun
                        # Convert Python weekday to JavaScript getDay: (weekday + 1) % 7
                        weekdays = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
                        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        weekday = weekdays[(dt.weekday() + 1) % 7]
                        day = dt.strftime('%d')
                        month = months[dt.month - 1]
                        year = dt.year
                        hour_12 = dt.hour % 12 or 12
                        minute = dt.strftime('%M')
                        am_pm = 'AM' if dt.hour < 12 else 'PM'
                        formatted_date = f"{weekday}, {day} {month} {year} {hour_12}:{minute} {am_pm}"
                    except:
                        formatted_date = str(timestamp)[:16]
                else:
                    formatted_date = ''
                
                # Clean violation type - remove any status info that might be embedded
                violation_type = str(v.get('violation_type', '')).strip()
                status = str(v.get('status', '')).strip()
                
                # If violation_type contains status info (like "non-compliant: shoepending")
                # Extract and clean it
                if violation_type:
                    # Check for status patterns at the end of violation_type
                    # Patterns to look for (longest first to avoid partial matches)
                    status_patterns = [
                        '_pending', '_resolved', 'pending', 'resolved'
                    ]
                    
                    violation_type_lower = violation_type.lower()
                    extracted_status = None
                    
                    # Try to find and extract status from violation_type
                    for pattern in status_patterns:
                        pattern_lower = pattern.lower()
                        # Check if violation_type ends with this pattern
                        if violation_type_lower.endswith(pattern_lower):
                            # Extract the status
                            extracted_status = pattern.replace('_', ' ').strip()
                            # Remove the pattern from violation_type
                            violation_type = violation_type[:-len(pattern)].strip()
                            break
                    
                    # Use extracted status if original status is empty
                    if not status and extracted_status:
                        status = extracted_status
                    
                    # Clean up any double spaces, underscores, or trailing separators
                    violation_type = violation_type.rstrip(':_-').strip()
                    violation_type = violation_type.replace('_', ' ').strip()
                    violation_type = ' '.join(violation_type.split())
                    
                    # Remove "Non-compliant:" or "non-compliant:" prefix
                    if violation_type.lower().startswith('non-compliant:'):
                        violation_type = violation_type.split(':', 1)[1].strip() if ':' in violation_type else violation_type
                    elif violation_type.lower().startswith('non compliant'):
                        violation_type = violation_type.replace('non compliant', '').replace(':', '').strip()
                    
                    # Format violation type nicely - capitalize first letter
                    if violation_type:
                        violation_type = violation_type.title()
                
                # Format status nicely
                if status:
                    status = status.replace('_', ' ').strip().title()
                else:
                    status = 'N/A'
                
                # Clean program name - remove "Bachelor of Science in" prefix
                program_name = str(v.get('program', '')).strip()
                if program_name:
                    # Remove common prefixes
                    prefixes_to_remove = [
                        'Bachelor of Science in ',
                        'Bachelor of Arts in ',
                        'Bachelor of Science in',
                        'Bachelor of Arts in',
                        'Bachelor of Science',
                        'Bachelor of Arts'
                    ]
                    for prefix in prefixes_to_remove:
                        if program_name.startswith(prefix):
                            program_name = program_name[len(prefix):].strip()
                            break
                
                # Format gender - capitalize first letter
                gender = str(v.get('gender', '')).strip()
                if gender:
                    gender = gender.capitalize()
                else:
                    gender = 'N/A'
                
                # Use Paragraph objects for better text wrapping
                # Get college abbreviation for display
                student_college = str(v.get('college', ''))
                college_display = get_college_abbreviation(student_college) if student_college else 'N/A'
                
                table_data.append([
                    Paragraph(formatted_date, normal_style),
                    Paragraph(str(v.get('student_id', '')), normal_style),
                    Paragraph(str(v.get('name', ''))[:30], normal_style),
                    Paragraph(college_display[:10], normal_style),  # Abbreviation is shorter
                    Paragraph(program_name[:30] if program_name else 'N/A', normal_style),
                    Paragraph(gender[:8], normal_style),
                    Paragraph(violation_type[:30] if violation_type else 'N/A', normal_style),
                    Paragraph(status[:20] if status else 'N/A', normal_style)
                ])
            
            # Create table with adjusted column widths - total width ~7.2 inches (fits A4 with margins)
            # Date, Student ID, Name, College, Program, Gender, Non Compliant, Status
            violation_table = Table(table_data, colWidths=[0.95*inch, 0.85*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.7*inch, 0.9*inch, 0.75*inch])
            violation_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f25a04')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 7),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('TOPPADDING', (0, 0), (-1, 0), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')]),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 3),
                ('RIGHTPADDING', (0, 0), (-1, -1), 3),
                ('TOPPADDING', (0, 1), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
            ]))
            elements.append(violation_table)
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        
        # Generate filename
        filename = f"osas_violation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        if start_dt and end_dt:
            filename = f"osas_violation_report_{start_dt}_to_{end_dt}.pdf"
        filename = re.sub(r'[^\w\-_\.]', '_', filename)  # Sanitize filename
        
        return Response(
            buffer.getvalue(),
            mimetype='application/pdf',
            headers={
                'Content-Disposition': f'attachment; filename="{filename}"'
            }
        )
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@violations_bp.route('/send-followup-emails', methods=['POST'])
def send_followup_emails():
    """Send follow-up emails for violations that are still pending after 3 days."""
    # Import here to avoid circular imports
    from app import get_connection, app, mail
    from flask_mail import Message
    from src.email_templates import generate_followup_email_body
    from src.config import find_student_by_id
    import os
    
    try:
        conn = get_connection() if get_connection else None
        if conn is None:
            return jsonify({'success': False, 'error': 'DB not configured'}), 500
        
        # Ensure followup_sent column exists (add it if it doesn't)
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    SELECT COUNT(*) as col_exists
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = DATABASE()
                    AND TABLE_NAME = 'violations'
                    AND COLUMN_NAME = 'followup_sent'
                """)
                col_exists = (cur.fetchone() or {}).get('col_exists', 0)
                if col_exists == 0:
                    cur.execute("ALTER TABLE violations ADD COLUMN followup_sent TINYINT(1) DEFAULT 0")
                    conn.commit()
                    print("✓ Added followup_sent column to violations table")
            except Exception as e:
                print(f"Note: Could not check/add followup_sent column: {e}")
                # Continue anyway - the query will handle missing column gracefully
        
        # Find violations that are pending, 3+ days old, and haven't had follow-up sent yet
        # Excludes violations where followup_sent = 1 (already sent)
        with conn.cursor() as cur:
            try:
                # Query with followup_sent column - only selects violations where followup_sent IS NULL or 0
                # This excludes violations where followup_sent = 1 (already sent)
                cur.execute(
                    """
                    SELECT v.violation_id, v.student_id, v.violation_type, v.timestamp, v.image_proof, v.status,
                           s.name, s.email, s.gender
                    FROM violations v
                    LEFT JOIN students s ON v.student_id = s.student_id
                    WHERE v.status = 'pending' 
                      AND v.timestamp <= NOW() - INTERVAL 3 DAY
                      AND (v.followup_sent IS NULL OR v.followup_sent = 0)
                    ORDER BY v.timestamp ASC
                    """
                )
                violations = cur.fetchall() or []
                print(f"DEBUG: Found {len(violations)} violations needing follow-up emails")
            except Exception as e:
                # If column doesn't exist, check if it's a column error
                error_str = str(e).lower()
                if 'followup_sent' in error_str or 'unknown column' in error_str:
                    # Column doesn't exist - try to add it and retry
                    try:
                        cur.execute("ALTER TABLE violations ADD COLUMN followup_sent TINYINT(1) DEFAULT 0")
                        conn.commit()
                        print("✓ Added followup_sent column to violations table")
                        # Retry the query
                        cur.execute(
                            """
                            SELECT v.violation_id, v.student_id, v.violation_type, v.timestamp, v.image_proof, v.status,
                                   s.name, s.email, s.gender
                            FROM violations v
                            LEFT JOIN students s ON v.student_id = s.student_id
                            WHERE v.status = 'pending' 
                              AND v.timestamp <= NOW() - INTERVAL 3 DAY
                              AND (v.followup_sent IS NULL OR v.followup_sent = 0)
                            ORDER BY v.timestamp ASC
                            """
                        )
                        violations = cur.fetchall() or []
                    except Exception as add_err:
                        print(f"Error adding followup_sent column: {add_err}")
                        violations = []
                else:
                    # Other error - log and return empty
                    print(f"Error querying violations for follow-up emails: {e}")
                    violations = []
        
        if not violations:
            print("DEBUG: No violations found that need follow-up emails (all are either resolved, less than 3 days old, or already sent)")
            return jsonify({'success': True, 'message': 'No violations require follow-up emails', 'sent': 0})
        
        sent_count = 0
        errors = []
        
        for violation in violations:
            try:
                violation_id = violation.get('violation_id')
                student_id = violation.get('student_id')
                student_name = violation.get('name', 'Student')
                student_email = violation.get('email')
                violation_timestamp = violation.get('timestamp')
                violation_type = violation.get('violation_type', '')
                image_proof = violation.get('image_proof')
                
                if not student_email:
                    errors.append(f"Violation {violation_id}: No email for student {student_id}")
                    continue
                
                # Mark violation as being processed BEFORE sending email to prevent duplicates
                # Use atomic UPDATE to mark it immediately
                try:
                    with conn.cursor() as mark_cur:
                        # First, verify the current state
                        mark_cur.execute(
                            "SELECT followup_sent FROM violations WHERE violation_id = %s",
                            (violation_id,)
                        )
                        current_state = mark_cur.fetchone()
                        if current_state:
                            current_followup_sent = current_state.get('followup_sent')
                            if current_followup_sent == 1:
                                print(f"⚠ Violation {violation_id} already has followup_sent=1, skipping duplicate")
                                continue
                        
                        # Try to atomically mark as sent (only if still 0/NULL)
                        mark_cur.execute(
                            """
                            UPDATE violations 
                            SET followup_sent = 1 
                            WHERE violation_id = %s 
                              AND (followup_sent IS NULL OR followup_sent = 0)
                            """,
                            (violation_id,)
                        )
                        rows_updated = mark_cur.rowcount
                        conn.commit()
                        
                        # If no rows were updated, this violation was already processed by another instance
                        if rows_updated == 0:
                            print(f"⚠ Violation {violation_id} already processed (followup_sent was already 1), skipping duplicate")
                            continue
                        
                        print(f"✓ Marked violation {violation_id} as followup_sent=1 before sending email")
                except Exception as mark_err:
                    print(f"Warning: Could not mark violation {violation_id} as processed: {mark_err}")
                    import traceback
                    traceback.print_exc()
                    # Don't continue - skip this violation to prevent duplicates
                    continue
                
                # Get student info to determine strike count
                student = find_student_by_id(student_id) if find_student_by_id else None
                if not student:
                    errors.append(f"Violation {violation_id}: Student not found")
                    continue
                
                # Get violation history for this student
                violation_lines = []
                try:
                    from src.config import get_student_violations as _get_v_list
                    if _get_v_list and student_id:
                        vlist = _get_v_list(student_id) or []
                        for v in vlist:
                            timestamp = v.get('timestamp')
                            if timestamp:
                                try:
                                    if isinstance(timestamp, str):
                                        dt = datetime.strptime(str(timestamp), '%Y-%m-%d %H:%M:%S')
                                    else:
                                        dt = timestamp
                                    tstr = dt.strftime('%a, %d %b %Y %I:%M %p')
                                except:
                                    tstr = str(timestamp)
                            else:
                                tstr = 'Unknown date'
                            vtype = str(v.get('violation_type') or '')
                            violation_lines.append(f"{tstr} – {vtype}")
                except:
                    pass
                violation_text = '\n'.join(violation_lines) if violation_lines else 'No history available'
                
                # Count strikes (pending violations count as strikes)
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT COUNT(*) as strike_count
                        FROM violations
                        WHERE student_id = %s AND status != 'resolved'
                        ORDER BY timestamp ASC
                        """,
                        (student_id,)
                    )
                    strike_result = cur.fetchone() or {}
                    strike_count = min(int(strike_result.get('strike_count', 1)), 3)
                
                # Format dates
                if violation_timestamp:
                    try:
                        if isinstance(violation_timestamp, str):
                            dt = datetime.strptime(str(violation_timestamp), '%Y-%m-%d %H:%M:%S')
                        else:
                            dt = violation_timestamp
                        dt_str = dt.strftime('%a, %d %b %Y %I:%M %p')
                        first_notice_date = dt_str  # Use violation date as first notice date
                    except:
                        dt_str = str(violation_timestamp)[:19]
                        first_notice_date = dt_str
                else:
                    dt_str = 'Unknown date'
                    first_notice_date = 'Unknown date'
                
                # Determine offense line
                if strike_count == 1:
                    offense_line = '1st Offense'
                    subject = 'Dress Code Violation Follow-Up - 1st Offense (Warning)'
                elif strike_count == 2:
                    offense_line = '2nd Offense'
                    subject = 'Dress Code Violation Follow-Up - 2nd Offense (5-day Suspension)'
                else:
                    offense_line = '3rd Offense'
                    subject = 'Dress Code Violation Follow-Up - 3rd Offense (Up to 1 month Suspension)'
                
                # Prepare image attachment
                image_cid = None
                proof_path = None
                if image_proof:
                    proof_path = os.path.join('results', 'violations', image_proof)
                    if os.path.exists(proof_path):
                        image_cid = f"violation_proof_{violation_id}"
                
                # Generate email body
                html_body = generate_followup_email_body(
                    student_name=student_name,
                    first_notice_date=first_notice_date,
                    violation_datetime=dt_str,
                    strike_num=strike_count,
                    offense_line=offense_line,
                    violation_history=violation_text,
                    image_cid=image_cid,
                    logo_base64=None,
                    logo_cid=None
                )
                
                # Create plain text fallback
                image_attachment_text = "\n\nPROOF OF VIOLATION\nA proof image is attached to this email.\n" if image_cid else ""
                plain_text_body = f"""DRESS CODE VIOLATION FOLLOW-UP NOTICE

Dear {student_name},

This is a follow-up to the DRESS (Dress-code Recognition Surveillance System) notification sent on {first_notice_date}. Our records show that the dress code violation detected on {dt_str} has not yet been addressed.

Following the university dress code is an important part of maintaining discipline and professionalism. We remind you to comply with the proper uniform prescribed by the University, as stated in the Student Handbook, on your next visit.

VIOLATION DETAILS
Current Strike Count: {strike_count} of 3
Your Recorded Offense: {offense_line}

Previously Recorded Violations:
{violation_text}{image_attachment_text}

UNIVERSITY GUIDELINES
• 1st Offense – Warning
• 2nd Offense – 5-day suspension
• 3rd Offense – 2-week to 1-month suspension

ACTION REQUIRED
Please report to the Guidance Office as soon as possible to settle this matter. Continued failure to respond may affect the sanction applied to your case.

Thank you for your immediate attention.

Respectfully,
DRESS Monitoring Team
Palawan State University

This is an automated notification. Please do not reply to this email."""
                
                # Create and send email
                with app.app_context():
                    msg = Message(
                        subject=subject,
                        recipients=[student_email],
                        html=html_body,
                        body=plain_text_body,
                        sender=app.config.get('MAIL_DEFAULT_SENDER', app.config.get('MAIL_USERNAME'))
                    )
                    
                    # Attach proof image if available
                    if image_cid and proof_path and os.path.exists(proof_path):
                        try:
                            with open(proof_path, 'rb') as img_file:
                                msg.attach(
                                    filename=image_proof,
                                    content_type='image/jpeg',
                                    data=img_file.read(),
                                    disposition='inline',
                                    headers={'Content-ID': f'<{image_cid}>'}
                                )
                        except Exception as attach_err:
                            print(f"Warning: Could not attach image for violation {violation_id}: {attach_err}")
                    
                    
                    mail.send(msg)
                    
                    # Violation was already marked as sent before sending email
                    # This prevents race conditions and duplicate sends
                    sent_count += 1
                    print(f"✓ Follow-up email sent to {student_email} for violation {violation_id}")
            
            except Exception as e:
                error_msg = f"Violation {violation.get('violation_id', 'unknown')}: {str(e)}"
                errors.append(error_msg)
                print(f"✗ Error sending follow-up email: {error_msg}")
                import traceback
                print(traceback.format_exc())
        
        conn.close()
        
        result = {
            'success': True,
            'sent': sent_count,
            'total': len(violations),
            'errors': errors if errors else None
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

