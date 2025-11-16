# DRESS: Dress-code Recognition Surveillance System

A web application for automated dress code monitoring and violation tracking using computer vision and RFID technology.

## Features

- **Real-time Dress Code Detection**: Automated detection of dress code violations using YOLOv8 and custom models
- **RFID Integration**: Student identification via RFID cards with automatic violation tracking
- **Schedule System**: Configure active hours and days for RFID and detection monitoring
- **Role-based Dashboards**: Separate interfaces for Security, Deans, OSAS, and Guidance counselors
- **Violation Management**: Track, filter, and manage dress code violations with strike system
- **PDF Reports**: Generate violation reports with analytics and statistics
- **Student Database**: Manage student information and records
- **Email Notifications**: Automatic email notifications for violations and follow-up reminders
- **Database Backup**: Automatic backup sync to Aiven cloud database (optional)
- **Test Mode**: Test detections outside scheduled hours for system validation

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd dress-v2
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```env
   # Local Database (Primary - Required)
   LOCAL_DB_HOST=localhost
   LOCAL_DB_PORT=3306
   LOCAL_DB_USER=root
   LOCAL_DB_PASSWORD=your_password
   LOCAL_DB_NAME=dress

   # Aiven Database (Backup - Optional)
   DB_HOST=your-aiven-host.aivencloud.com
   DB_PORT=22870
   DB_USER=avnadmin
   DB_PASSWORD=your_password
   DB_NAME=dress
   DB_SSL_CA=certs/ca.pem
   DB_SSL_REQUIRED=true

   # Email Configuration (Required for email notifications)
   MAIL_SERVER=smtp.gmail.com
   MAIL_PORT=587
   MAIL_USE_TLS=True
   MAIL_USERNAME=your_email@gmail.com
   MAIL_PASSWORD=your_app_password
   ```

4. **Set up database**:
   - Create MySQL database
   - Import schema: `database/dress_clean.sql`
   - (Optional) Import sample data: `database/dummy_data.sql`

5. **Create admin account**:
   ```bash
   python scripts/create_admin.py
   ```
   The script will prompt for:
   - Username
   - Password
   - Role (security, dean, osas, guidance)
   - College (if role is dean)

6. **Run the application**:
   ```bash
   python app.py
   ```

7. **Access the application**:
   - Open browser: `http://localhost:5000`
   - Login with admin credentials

## Configuration

### Schedule Settings
Configure when the system should be active:
- Access via Security dashboard → Settings icon
- Set active days (Monday-Sunday)
- Set active time range (start and end time)
- RFID and detection only work during scheduled hours (unless test mode is enabled)

### Database Configuration
- **Primary Database**: Local MySQL (always used for all operations)
- **Backup Database**: Aiven (optional, synced automatically every 5 minutes when available)
- See [DATABASE_SYNC_NOTES.md](https://github.com/NaldCapuno/dress-v2/blob/main/DATABASE_SYNC_NOTES.md) for detailed database sync information

### Email Configuration
- Configure SMTP settings in `.env` file
- System sends:
  - Initial violation notifications
  - Follow-up emails (after 3 days for unresolved violations)
- Follow-up emails are sent automatically via background scheduler

## User Roles

- **Security**: Monitor real-time violations via camera feed, manage schedule settings, test mode
- **Dean**: View and manage violations for their college, generate reports
- **OSAS**: University-wide violation oversight, analytics, and reporting
- **Guidance**: Counseling and student support for violations, student management

## Key Features

### Schedule System
- Configure active hours and days for monitoring
- RFID and detection automatically enabled/disabled based on schedule
- Test mode allows testing outside scheduled hours
- Visual indicators show when system is outside scheduled hours

### RFID Integration
- Automatic student identification via RFID cards
- Detection only enabled when valid RFID card is present
- Violation tracking linked to student records
- Strike system (1st, 2nd, 3rd offense tracking)

### Email Notifications
- **Initial Notification**: Sent immediately when violation is detected
- **Follow-up Notification**: Sent automatically after 3 days if violation is still pending
- Includes violation details, strike count, and proof images
- Duplicate prevention ensures emails are only sent once

### Database Backup
- Automatic backup to Aiven cloud database (if configured)
- Syncs every 5 minutes when Aiven is available
- System continues normally even if backup is unavailable
- See `DATABASE_SYNC_NOTES.md` for details

## Requirements

- Python 3.8+
- MySQL database (local, required)
- Webcam (for real-time detection)
- RFID reader (optional, for student identification)
- Aiven account (optional, for cloud backup)

## Project Structure

```
dress-v2/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── LICENSE                     # License file
├── README.md                   # This file
├── DATABASE_SYNC_NOTES.md      # Database sync documentation
├── .env                        # Environment variables (create this)
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
│   ├── settings.py            # Settings management routes
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
│   ├── create_admin.py        # Admin account creation script
│   └── sync_database.py       # Manual database sync script
│
├── uploads/                    # Uploaded files (auto-created)
├── results/                    # Processed images (auto-created)
│   └── violations/            # Violation images
└── __pycache__/               # Python cache (auto-generated)
```

## Background Services

The application runs several background threads:

1. **Schedule Checker**: Manages RFID enabled/disabled based on schedule (checks every 10 seconds)
2. **Follow-up Email Scheduler**: Sends follow-up emails for 3+ day old violations (checks daily)
3. **Database Backup Sync**: Syncs local database to Aiven backup (every 5 minutes when available)
4. **Detection Worker**: Processes frames for dress code detection asynchronously
5. **RFID Event Handler**: Handles RFID card detection and student lookup

## Manual Database Sync

To manually sync databases, use the sync script:
```bash
python scripts/sync_database.py
```

Options:
- Sync Local → Aiven (backup)
- Sync Aiven → Local (restore)
- Sync specific tables
- View sync status

## Troubleshooting

### Camera Not Working
- Check if camera is connected and not in use by another application
- Verify camera permissions
- Try different camera IDs in settings

### RFID Not Detecting
- Verify RFID reader is connected
- Check if system is within scheduled hours (or enable test mode)
- Ensure RFID cards are properly registered in database

### Emails Not Sending
- Verify SMTP settings in `.env` file
- Check email credentials (use app password for Gmail)
- Check logs for email errors

### Database Connection Issues
- Verify database credentials in `.env` file
- Ensure MySQL service is running
- Check database exists and schema is imported

### Follow-up Emails Not Sending
- Check if violations are 3+ days old
- Verify violations have `status = 'pending'`
- Check logs for scheduler errors
- Ensure `followup_sent` column exists in violations table

## License

[MIT License](https://github.com/NaldCapuno/dress-v2/blob/main/LICENSE)
