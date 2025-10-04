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
    host = os.getenv('DB_HOST', 'localhost')
    port = int(os.getenv('DB_PORT', '3306'))
    user = os.getenv('DB_USER', 'root')
    password = os.getenv('DB_PASSWORD', 'root')
    database = os.getenv('DB_NAME', 'dress')

    return pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )


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
                SELECT student_id, rfid_uid, name, gender, year_level, course, college
                FROM students
                WHERE rfid_uid = %s
                LIMIT 1
                """,
                (rfid_uid,)
            )
            return cur.fetchone()
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


def insert_violation(student_id, violation_type: str, image_proof_rel_path: str | None, recorded_by: int | None = None):
    """Insert a violation and return the new violation_id (or None on failure)."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO violations (student_id, recorded_by, violation_type, image_proof)
                VALUES (%s, %s, %s, %s)
                """,
                (student_id, recorded_by, violation_type, image_proof_rel_path)
            )
            return cur.lastrowid
    except Exception:
        return None
    finally:
        conn.close()


