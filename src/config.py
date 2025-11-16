"""
Database connection configuration aligned with the provided DRESS database module.

Uses PyMySQL and environment variables with sensible defaults:
- LOCAL_DB_HOST=localhost, LOCAL_DB_PORT=3306, LOCAL_DB_USER=root, LOCAL_DB_PASSWORD=root, LOCAL_DB_NAME=dress

This module only establishes a connection and exposes get_connection().
It does not execute any queries.

Uses local database as primary. Aiven database is used only as backup (synced periodically).
"""

import os
import pymysql
import threading
import time
from typing import Any, Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, skip loading .env file
    pass

# Global state for database connection management
_db_state = {
    'current_db': None,  # 'aiven' or 'local'
    'aiven_available': None,  # True/False/None (unknown)
    'last_aiven_check': 0,
    'sync_pending': False,  # True if we need to sync when Aiven comes back
    'lock': threading.Lock()
}

# Cache for connection parameters
_aiven_params = None
_local_params = None


def _get_aiven_params():
    """Get Aiven database connection parameters."""
    global _aiven_params
    if _aiven_params is None:
        host = os.getenv('DB_HOST', '')
        if 'aivencloud.com' not in host.lower() and host:
            # If DB_HOST is set but not Aiven, use it as Aiven anyway
            pass
        
        _aiven_params = {
            'host': os.getenv('DB_HOST', ''),
            'port': int(os.getenv('DB_PORT', '3306')),
            'user': os.getenv('DB_USER', ''),
            'password': os.getenv('DB_PASSWORD', ''),
            'database': os.getenv('DB_NAME', 'dress'),
            'is_aiven': True
        }
    return _aiven_params


def _get_local_params():
    """Get local database connection parameters."""
    global _local_params
    if _local_params is None:
        _local_params = {
            'host': os.getenv('LOCAL_DB_HOST', 'localhost'),
            'port': int(os.getenv('LOCAL_DB_PORT', '3306')),
            'user': os.getenv('LOCAL_DB_USER', 'root'),
            'password': os.getenv('LOCAL_DB_PASSWORD', 'root'),
            'database': os.getenv('LOCAL_DB_NAME', 'dress'),
            'is_aiven': False
        }
    return _local_params


def _test_connection(params, timeout=3):
    """Test if a database connection can be established."""
    try:
        connection_params = {
            'host': params['host'],
            'port': params['port'],
            'user': params['user'],
            'password': params['password'],
            'database': params['database'],
            'cursorclass': pymysql.cursors.DictCursor,
            'connect_timeout': timeout,
        }
        
        # Add SSL for Aiven
        if params.get('is_aiven'):
            is_aiven = 'aivencloud.com' in params['host'].lower()
            ssl_disabled = os.getenv('DB_SSL_DISABLED', 'false').lower() in {'1', 'true', 'yes', 'on'}
            ssl_required = os.getenv('DB_SSL_REQUIRED', 'true' if is_aiven else 'false').lower() in {'1', 'true', 'yes', 'on'}
            ssl_ca = os.getenv('DB_SSL_CA', 'certs/ca.pem' if is_aiven else None)
            
            if not ssl_disabled and (ssl_required or ssl_ca):
                if ssl_ca and os.path.exists(ssl_ca):
                    connection_params['ssl'] = {'ca': ssl_ca}
                elif is_aiven:
                    import ssl
                    connection_params['ssl'] = {'ssl_disabled': False}
        
        conn = pymysql.connect(**connection_params)
        conn.close()
        return True
    except Exception:
        return False


def _create_connection(params):
    """Create a database connection with the given parameters."""
    connection_params = {
        'host': params['host'],
        'port': params['port'],
        'user': params['user'],
        'password': params['password'],
        'database': params['database'],
        'cursorclass': pymysql.cursors.DictCursor,
        'autocommit': True,
    }
    
    # Add SSL for Aiven
    if params.get('is_aiven'):
        is_aiven = 'aivencloud.com' in params['host'].lower()
        ssl_disabled = os.getenv('DB_SSL_DISABLED', 'false').lower() in {'1', 'true', 'yes', 'on'}
        ssl_required = os.getenv('DB_SSL_REQUIRED', 'true' if is_aiven else 'false').lower() in {'1', 'true', 'yes', 'on'}
        ssl_ca = os.getenv('DB_SSL_CA', 'certs/ca.pem' if is_aiven else None)
        
        if not ssl_disabled and (ssl_required or ssl_ca):
            if ssl_ca and os.path.exists(ssl_ca):
                connection_params['ssl'] = {'ca': ssl_ca}
            elif is_aiven:
                import ssl
                connection_params['ssl'] = {'ssl_disabled': False}
    
    return pymysql.connect(**connection_params)


def get_connection(force_aiven=False) -> Any:
    """
    Open and return a new PyMySQL connection to the local database (primary).
    
    Local database is used as primary for all operations.
    Aiven database is used only as backup (synced periodically in background).
    
    Args:
        force_aiven: If True, force connection to Aiven database (for backup operations only)
    
    Returns:
        PyMySQL connection object
    """
    global _db_state
    
    with _db_state['lock']:
        # If forcing Aiven (for backup operations), use Aiven
        if force_aiven:
            aiven_params = _get_aiven_params()
            if not aiven_params['host'] or not aiven_params['user'] or not aiven_params['password']:
                raise ValueError("Aiven database not configured. Set DB_HOST, DB_USER, DB_PASSWORD in .env")
            _db_state['current_db'] = 'aiven'
            return _create_connection(aiven_params)
        
        # Always use local as primary
        local_params = _get_local_params()
        _db_state['current_db'] = 'local'
        return _create_connection(local_params)


def get_current_database():
    """Get the currently active database. Always returns 'local' (primary)."""
    with _db_state['lock']:
        # Always return 'local' since it's the primary database
        return 'local'


def is_aiven_available(force_check=False):
    """Check if Aiven database is currently available.
    
    Args:
        force_check: If True, force a fresh check even if cached value exists
    """
    with _db_state['lock']:
        current_time = time.time()
        # Force check if requested, or if no cached value, or if cache is stale
        should_check = (
            force_check or
            _db_state['aiven_available'] is None or
            (current_time - _db_state['last_aiven_check']) > 30
        )
        
        if should_check:
            aiven_params = _get_aiven_params()
            _db_state['aiven_available'] = _test_connection(aiven_params, timeout=3)
            _db_state['last_aiven_check'] = time.time()
        return _db_state['aiven_available']


def has_pending_sync():
    """Check if there's a pending sync when Aiven comes back online."""
    with _db_state['lock']:
        return _db_state['sync_pending']


def clear_pending_sync():
    """Clear the pending sync flag."""
    with _db_state['lock']:
        _db_state['sync_pending'] = False


# ------------------ Database helpers (reusable across the app) ------------------
def find_student_by_rfid(rfid_uid: str):
    """Return student dict for given RFID UID, or None."""
    if not rfid_uid:
        return None
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT student_id, rfid_uid, name, gender, year_level, program, college, email
                FROM students
                WHERE rfid_uid = %s
                LIMIT 1
                """,
                (rfid_uid,)
            )
            return cur.fetchone()
    finally:
        conn.close()

def find_student_by_id(student_id: str):
    """Return student dict for given student_id, or None."""
    if not student_id:
        return None
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT student_id, rfid_uid, name, gender, year_level, program, college, email
                FROM students
                WHERE student_id = %s
                LIMIT 1
                """,
                (student_id,)
            )
            return cur.fetchone()
    finally:
        conn.close()


def get_student_violation_count(student_id: str) -> int:
    """Return total count of violations for a given student_id."""
    if not student_id:
        return 0
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) AS cnt
                FROM violations
                WHERE student_id = %s
                """,
                (student_id,)
            )
            row = cur.fetchone() or {}
            return int(row.get('cnt') or 0)
    except Exception:
        return 0
    finally:
        conn.close()


def get_student_violations(student_id: str):
    """Return list of violations for a given student_id (timestamp and type)."""
    if not student_id:
        return []
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT violation_id, violation_type, timestamp, status
                FROM violations
                WHERE student_id = %s
                ORDER BY timestamp ASC
                """,
                (student_id,)
            )
            return cur.fetchall() or []
    except Exception:
        return []
    finally:
        conn.close()



def insert_rfid_log(rfid_uid: str, student_id, status: str) -> bool:
    """Insert RFID scan log into rfid_logs. status in {'valid','unregistered'}."""
    if not rfid_uid:
        return False
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rfid_logs (student_id, rfid_uid, status)
                VALUES (%s, %s, %s)
                """,
                (student_id, rfid_uid, status)
            )
            return True
    except Exception:
        return False
    finally:
        conn.close()


def insert_violation(student_id, violation_type: str, image_proof_rel_path: str | None):
    """Insert a violation and return the new violation_id (or None on failure)."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO violations (student_id, violation_type, image_proof)
                VALUES (%s, %s, %s)
                """,
                (student_id, violation_type, image_proof_rel_path)
            )
            return cur.lastrowid
    except Exception:
        return None
    finally:
        conn.close()


def has_student_violation_today(student_id: str) -> bool:
    """Return True if the student already has a violation recorded today."""
    if not student_id:
        return False
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) AS cnt
                FROM violations
                WHERE student_id = %s
                  AND DATE(timestamp) = CURRENT_DATE
                """,
                (student_id,)
            )
            row = cur.fetchone() or {}
            return int(row.get('cnt') or 0) > 0
    except Exception:
        return False
    finally:
        conn.close()


