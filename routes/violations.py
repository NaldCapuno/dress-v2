"""
Violation routes for DRESS application.
Handles all violation-related endpoints for dean, osas, and guidance roles.
"""

from flask import Blueprint, request, jsonify, session
import time

violations_bp = Blueprint('violations', __name__)


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


@violations_bp.route('/dean/violation/<int:violation_id>/status', methods=['POST'])
def dean_update_violation_status(violation_id: int):
    """Dean can forward to guidance, set pending, or resolve."""
    # Import here to avoid circular imports
    from app import get_connection
    
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
@violations_bp.route('/guidance/violations', methods=['GET'])
def guidance_get_violations():
    """List violations for Guidance review (university-wide)."""
    # Import here to avoid circular imports
    from app import get_connection
    
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

