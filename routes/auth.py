"""
Authentication routes for DRESS application.
Handles login and logout functionality.
"""

from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for
from werkzeug.security import check_password_hash

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login"""
    # Import here to avoid circular imports
    from app import get_connection
    
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


@auth_bp.route('/logout', methods=['POST'])
def logout():
    """Clear session and log out the current user."""
    try:
        session.clear()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

