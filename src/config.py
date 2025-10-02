"""
Database connection configuration aligned with the provided DRESS database module.

Uses PyMySQL and environment variables with sensible defaults:
- DB_HOST=localhost, DB_PORT=3306, DB_USER=root, DB_PASSWORD=root, DB_NAME=dress

This module only establishes a connection and exposes get_connection().
It does not execute any queries.
"""

import os
import pymysql
from typing import Any


def get_connection() -> Any:
    """Open and return a new PyMySQL connection to the 'dress' database."""
    host = os.getenv('DB_HOST', 'dress-dress-b72e.k.aivencloud.com')
    port = int(os.getenv('DB_PORT', '22870'))
    user = os.getenv('DB_USER', 'avnadmin')
    password = os.getenv('DB_PASSWORD', '')  # Don't use default password - must be set via env var
    database = os.getenv('DB_NAME', 'dress')
    
    if not password:
        raise ValueError("DB_PASSWORD environment variable is not set. Please set it with your Aiven database password.")
    
    # SSL configuration for Aiven (or other cloud databases)
    # Aiven requires SSL connections, so enable by default if using Aiven host
    is_aiven = 'aivencloud.com' in host.lower()
    ssl_disabled = os.getenv('DB_SSL_DISABLED', 'false').lower() in {'1', 'true', 'yes', 'on'}
    ssl_required = os.getenv('DB_SSL_REQUIRED', 'true' if is_aiven else 'false').lower() in {'1', 'true', 'yes', 'on'}
    ssl_ca = os.getenv('DB_SSL_CA', 'certs/ca.pem' if is_aiven else None)
    
    connection_params = {
        'host': host,
        'port': port,
        'user': user,
        'password': password,
        'database': database,
        'cursorclass': pymysql.cursors.DictCursor,
        'autocommit': True,
    }
    
    # Add SSL configuration if required (Aiven typically requires SSL)
    if not ssl_disabled and (ssl_required or ssl_ca):
        if ssl_ca and os.path.exists(ssl_ca):
            # Use custom CA certificate
            connection_params['ssl'] = {'ca': ssl_ca}
        elif is_aiven:
            # For Aiven, try default SSL context if CA file not found
            import ssl
            connection_params['ssl'] = {'ssl_disabled': False}

    return pymysql.connect(**connection_params)


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


