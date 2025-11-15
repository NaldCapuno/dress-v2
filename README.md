# DRESS: Dress-code Recognition Surveillance System

A web application for automated dress code monitoring and violation tracking using computer vision and RFID technology.

## Features

- **Real-time Dress Code Detection**: Automated detection of dress code violations using YOLOv8 and custom models
- **RFID Integration**: Student identification via RFID cards
- **Role-based Dashboards**: Separate interfaces for Deans, OSAS, and Guidance counselors
- **Violation Management**: Track, filter, and manage dress code violations
- **PDF Reports**: Generate violation reports with analytics
- **Student Database**: Manage student information and records

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up database**:
   - Create MySQL database
   - Import schema: `database/dress_clean.sql`
   - (Optional) Import sample data: `database/dummy_data.sql`

3. **Configure database**:
   - Update database connection in `src/config.py`

4. **Create admin account**:
   ```bash
   python scripts/create_admin.py
   ```

5. **Run the application**:
   ```bash
   python app.py
   ```

6. **Access the application**:
   - Open browser: `http://localhost:5000`
   - Login with admin credentials

## User Roles

- **Security**: Monitor real-time violations via camera feed
- **Dean**: View and manage violations for their college
- **OSAS**: University-wide violation oversight and analytics
- **Guidance**: Counseling and student support for violations

## Requirements

- Python 3.8+
- MySQL database
- Webcam (for real-time detection)
- RFID reader (optional, for student identification)

## Project Structure

```
dress-v2/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── LICENSE                     # License file
├── README.md                   # This file
│
├── routes/                     # Route blueprints
│   ├── __init__.py            # Blueprint initialization
│   ├── auth.py                # Authentication routes
│   ├── violations.py          # Violation management routes
│   ├── dashboards.py          # Dashboard routes
│   ├── camera.py              # Camera and detection routes
│   ├── files.py               # File upload and serving
│   ├── rfid.py                # RFID scanner routes
│   ├── students.py            # Student management routes
│   └── debug.py               # Debug utilities
│
├── templates/                  # HTML templates
│   ├── login.html             # Login page
│   ├── index.html             # Main security dashboard
│   ├── dean_dashboard.html    # Dean dashboard
│   ├── osas_dashboard.html    # OSAS dashboard
│   └── guidance_dashboard.html # Guidance dashboard
│
├── static/                     # Static assets
│   ├── css/
│   │   ├── style.css          # Main stylesheet
│   │   └── table-styles.css   # Table-specific styles
│   ├── js/
│   │   ├── shared-table.js    # Shared table functionality
│   │   ├── table-pagination.js # Table pagination
│   │   └── table-utils.js     # Table utilities
│   └── images/
│       ├── login_bg.png       # Login background
│       └── login_logo.png     # Login logo
│
├── database/                   # Database files
│   ├── dress_clean.sql        # Clean database schema
│   └── dummy_data.sql         # Sample data for testing
│
├── models/                     # ML model files
│   ├── best.pt                # Custom dress code detection model
│   └── yolov8n.pt             # YOLOv8 person detection model
│
├── src/                        # Core modules
│   ├── config.py              # Database configuration
│   ├── rfid_scanner.py        # RFID scanner module
│   ├── botsort_tracker.py     # Bot-SORT tracking implementation
│   └── email_templates.py     # Email template helpers
│
├── scripts/                    # Utility scripts
│   └── create_admin.py        # Admin account creation script
│
├── uploads/                    # Uploaded files (auto-created)
├── results/                    # Processed images (auto-created)
│   └── violations/            # Violation images
└── __pycache__/               # Python cache (auto-generated)
```

## License

MIT License
