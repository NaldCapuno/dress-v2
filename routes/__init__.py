"""
Routes package for DRESS application.
Organizes all Flask routes into separate modules using Blueprints.
"""

from routes.auth import auth_bp
from routes.dashboards import dashboards_bp
from routes.violations import violations_bp
from routes.files import files_bp
from routes.camera import camera_bp
from routes.rfid import rfid_bp
from routes.debug import debug_bp
from routes.students import students_bp

__all__ = [
    'auth_bp',
    'dashboards_bp',
    'violations_bp',
    'files_bp',
    'camera_bp',
    'rfid_bp',
    'debug_bp',
    'students_bp',
]
