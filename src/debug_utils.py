"""
Debug utility module for centralized debug logging.
Allows enabling/disabling debug output globally.
"""

import os
from datetime import datetime

# Check if debug mode is enabled via environment variable
DEBUG_ENABLED = os.getenv('DEBUG', 'false').lower() in ('true', '1', 'yes', 'on')


def debug_print(message: str, prefix: str = "DEBUG"):
    """
    Print debug message if debug mode is enabled.
    
    Args:
        message: The debug message to print
        prefix: Optional prefix for the message (default: "DEBUG")
    """
    if DEBUG_ENABLED:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {prefix}: {message}")


def debug_rfid(message: str):
    """Debug print specifically for RFID operations."""
    debug_print(message, "DEBUG [RFID]")


def debug_violation(message: str):
    """Debug print specifically for violation operations."""
    debug_print(message, "DEBUG [VIOLATION]")


def debug_email(message: str):
    """Debug print specifically for email operations."""
    debug_print(message, "DEBUG [EMAIL]")


def debug_compliance(message: str):
    """Debug print specifically for compliance operations."""
    debug_print(message, "DEBUG [COMPLIANCE]")


def debug_database(message: str):
    """Debug print specifically for database operations."""
    debug_print(message, "DEBUG [DATABASE]")


def debug_camera(message: str):
    """Debug print specifically for camera operations."""
    debug_print(message, "DEBUG [CAMERA]")


def debug_sync(message: str):
    """Debug print specifically for sync operations."""
    debug_print(message, "DEBUG [SYNC]")

