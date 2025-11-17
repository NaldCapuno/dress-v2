"""
Authentication routes for DRESS application.
Handles login and logout functionality.
"""

from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from flask_mail import Message
import random
import string
import threading
from datetime import datetime, timedelta

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login"""
    # Import here to avoid circular imports
    from app import get_connection
    
    if request.method == 'GET':
        # If already logged in, redirect to appropriate dashboard
        admin = session.get('admin') or {}
        role = str(admin.get('role') or '').lower()
        
        if admin and role:
            if role == 'security':
                return redirect(url_for('dashboards.index'))
            elif role == 'osas':
                return redirect(url_for('dashboards.osas_dashboard'))
            elif role == 'guidance':
                return redirect(url_for('dashboards.guidance_dashboard'))
            elif role == 'dean':
                return redirect(url_for('dashboards.dean_dashboard'))
        
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


@auth_bp.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    """Handle forgot password request - send reset code to email"""
    from app import get_connection, app, mail
    from src.email_templates import generate_password_reset_email_body
    
    if request.method == 'GET':
        return render_template('forgot_password.html')
    
    # POST: JSON { username } - can be username or email
    try:
        data = request.get_json(force=True, silent=True) or {}
        identifier = (data.get('username') or '').strip()
        
        if not identifier:
            return jsonify({'success': False, 'error': 'Username or email is required.'}), 400
        
        if get_connection is None:
            return jsonify({'success': False, 'error': 'Database not configured.'}), 500
        
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                # Check if identifier is an email (contains @) or username
                # Try to find by username or email
                cur.execute(
                    """
                    SELECT username, email
                    FROM admins
                    WHERE username = %s OR email = %s
                    LIMIT 1
                    """,
                    (identifier, identifier)
                )
                admin = cur.fetchone()
            
            if not admin:
                # Don't reveal if username/email exists for security
                return jsonify({'success': True, 'message': 'If the username or email exists, a reset code has been sent to the associated email address.'})
            
            email = admin.get('email')
            username = admin.get('username')
            
            # Generate 6-digit reset code
            reset_code = ''.join(random.choices(string.digits, k=6))
            
            # Set expiration to 15 minutes from now
            expires_at = datetime.now() + timedelta(minutes=15)
            
            # Store reset code directly in admins table
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE admins
                    SET reset_code = %s, reset_code_expires_at = %s
                    WHERE username = %s
                    """,
                    (reset_code, expires_at, username)
                )
            
            # Send email with reset code - asynchronously
            def send_email_async():
                """Send email in background thread to avoid blocking the response"""
                try:
                    # Read logo and convert to base64
                    import os
                    import base64
                    logo_base64 = None
                    logo_path = os.path.join(app.root_path, 'static', 'images', 'dress_logo.png')
                    if os.path.exists(logo_path):
                        try:
                            with open(logo_path, 'rb') as logo_file:
                                logo_base64 = base64.b64encode(logo_file.read()).decode('utf-8')
                        except Exception as logo_err:
                            print(f"Warning: Could not read logo: {logo_err}")
                    
                    html_body = generate_password_reset_email_body(username, reset_code, include_username=False, logo_base64=logo_base64)
                    
                    # Build plain text body
                    plain_text_body = f"""PASSWORD RESET REQUEST

Hello {username},

We received a request to reset your password for your DRESS admin account. Use the code below to reset your password:

Your Password Reset Code: {reset_code}

This code will expire in 15 minutes. If you did not request this password reset, please ignore this email or contact the system administrator.

Security Notice:
Never share this code with anyone. DRESS staff will never ask for your password reset code.

Respectfully,
DRESS System
Palawan State University

This is an automated notification. Please do not reply to this email.
"""
                    
                    msg = Message(
                        subject='DRESS Password Reset Code',
                        recipients=[email],
                        html=html_body,
                        body=plain_text_body,
                        sender=app.config.get('MAIL_DEFAULT_SENDER', app.config.get('MAIL_USERNAME'))
                    )
                    
                    with app.app_context():
                        mail.send(msg)
                    print(f"✓ Password reset email sent successfully to {email}")
                except Exception as email_error:
                    print(f"✗ Error sending password reset email: {email_error}")
                    import traceback
                    print(traceback.format_exc())
            
            # Start email sending in background thread
            email_thread = threading.Thread(target=send_email_async, daemon=True)
            email_thread.start()
            
            # Return immediately - email is being sent in background
            return jsonify({
                'success': True,
                'message': 'If the username or email exists, a reset code has been sent to the associated email address.'
            })
        finally:
            conn.close()
    except Exception as e:
        print(f"Error in forgot_password: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@auth_bp.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    """Handle password reset - verify code and update password"""
    from app import get_connection
    
    if request.method == 'GET':
        return render_template('reset_password.html')
    
    # POST: JSON { username, reset_code, new_password }
    try:
        data = request.get_json(force=True, silent=True) or {}
        username = (data.get('username') or '').strip()
        reset_code = (data.get('reset_code') or '').strip()
        new_password = data.get('new_password') or ''
        
        if not username or not reset_code or not new_password:
            return jsonify({'success': False, 'error': 'Username, reset code, and new password are required.'}), 400
        
        if len(new_password) < 6:
            return jsonify({'success': False, 'error': 'Password must be at least 6 characters long.'}), 400
        
        if get_connection is None:
            return jsonify({'success': False, 'error': 'Database not configured.'}), 500
        
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                # Verify reset code and get expiration
                cur.execute(
                    """
                    SELECT username, reset_code_expires_at
                    FROM admins
                    WHERE username = %s AND reset_code = %s
                    LIMIT 1
                    """,
                    (username, reset_code)
                )
                admin_record = cur.fetchone()
            
            if not admin_record:
                return jsonify({'success': False, 'error': 'Invalid reset code.'}), 400
            
            expires_at = admin_record.get('reset_code_expires_at')
            # Handle datetime object (from PyMySQL) or string formats
            if isinstance(expires_at, datetime):
                pass  # Already a datetime object
            elif isinstance(expires_at, str):
                try:
                    expires_at = datetime.strptime(expires_at, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    try:
                        expires_at = datetime.strptime(expires_at, '%Y-%m-%d %H:%M:%S.%f')
                    except ValueError:
                        expires_at = datetime.now() - timedelta(minutes=1)  # Expire if can't parse
            elif expires_at is None:
                return jsonify({'success': False, 'error': 'Reset code has expired. Please request a new one.'}), 400
            else:
                expires_at = datetime.now() - timedelta(minutes=1)  # Expire if unknown type
            
            # Check if code has expired
            if datetime.now() > expires_at:
                return jsonify({'success': False, 'error': 'Reset code has expired. Please request a new one.'}), 400
            
            # Update password and clear reset code
            password_hash = generate_password_hash(new_password)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE admins
                    SET password_hash = %s, reset_code = NULL, reset_code_expires_at = NULL
                    WHERE username = %s
                    """,
                    (password_hash, username)
                )
            
            return jsonify({
                'success': True,
                'message': 'Password has been reset successfully. You can now login with your new password.'
            })
        finally:
            conn.close()
    except Exception as e:
        print(f"Error in reset_password: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

