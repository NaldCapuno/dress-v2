# DRESS System - Comprehensive Technical Notes

## System Overview

**DRESS (Dress-code Recognition Surveillance System)** is a Flask-based web application that uses computer vision (YOLOv8) and RFID technology to automatically monitor and track dress code violations in an educational institution.

---

## Architecture

### Technology Stack
- **Backend**: Flask (Python web framework)
- **Database**: MySQL (Local primary, Aiven cloud backup)
- **Computer Vision**: YOLOv8 (Ultralytics) for person detection + custom model for dress code detection
- **Tracking**: Bot-SORT algorithm for person tracking
- **Frontend**: HTML/CSS/JavaScript (vanilla, no framework)
- **Email**: Flask-Mail for SMTP notifications
- **PDF Generation**: ReportLab (optional)

### Core Components

1. **Main Application** (`app.py`)
   - Flask app initialization
   - Global state management
   - Background thread orchestration
   - Detection pipeline coordination

2. **Route Blueprints** (`routes/`)
   - Modular route organization
   - Separation of concerns by feature

3. **Core Modules** (`src/`)
   - Database configuration
   - RFID scanner integration
   - Bot-SORT tracker
   - Email templates

4. **ML Models** (`models/`)
   - `yolov8n.pt`: Person detection
   - `best.pt`: Custom dress code detection

---

## Database Architecture

### Primary Database: Local MySQL
- **Purpose**: All application operations
- **Connection**: Always used for reads/writes
- **Performance**: Fast, no network dependency
- **Configuration**: `LOCAL_DB_*` environment variables

### Backup Database: Aiven (Cloud)
- **Purpose**: Periodic backup only
- **Sync Direction**: Local → Aiven (one-way)
- **Sync Frequency**: Every 5 minutes (when available)
- **Availability**: Optional, system works without it
- **Configuration**: `DB_*` environment variables

### Key Tables

1. **`admins`**
   - User accounts with role-based access
   - Roles: `security`, `dean`, `osas`, `guidance`
   - Password hashing for security

2. **`students`**
   - Student information
   - RFID UID mapping
   - College, program, year level

3. **`violations`**
   - Dress code violation records
   - Links to students
   - Status tracking (pending, resolved, etc.)
   - Strike count (1st, 2nd, 3rd offense)
   - Follow-up email flag

4. **`rfid_logs`**
   - RFID card scan history
   - Tracks valid/unregistered cards

5. **`settings`**
   - Key-value configuration storage
   - Schedule settings
   - Auto-sync enabled/disabled flag

6. **`email_outbox`**
   - Email queue for offline operation
   - Stores emails that need to be sent
   - Status: `pending`, `sending`, `sent`, `failed`
   - Tracks attempt count, last attempt time, and error messages
   - Links to violations via `violation_id` foreign key

---

## Detection System

### Two-Stage Detection Pipeline

#### Stage 1: Person Detection
- **Model**: YOLOv8n (person class only)
- **Purpose**: Detect and track people in frame
- **Tracking**: Bot-SORT algorithm assigns track IDs
- **Output**: Bounding boxes with track IDs

#### Stage 2: Dress Code Detection
- **Model**: Custom `best.pt` model
- **Input**: Cropped person regions from Stage 1
- **Processing**: Only processes track_id == 1 (primary tracked person)
- **Output**: Dress code compliance status per item

### Detection Items
The system checks for:
- Upper body clothing (shirt, polo, etc.)
- Lower body clothing (pants, skirt, etc.)
- Footwear (shoes)
- Gender-specific requirements

### Compliance Statuses
- **COMPLIANT**: All required items present
- **PARTIALLY COMPLIANT**: Some items missing
- **NON-COMPLIANT**: Critical items missing
- **NO_DETECTION**: Person not detected or out of frame

### Violation Recording Logic

**Requirements for Recording:**
1. Valid RFID card must be present
2. Student must be registered in database
3. Requires 3 consecutive violation detections
4. Only records once per RFID scan session
5. Resets counter when status changes to COMPLIANT
6. Does NOT reset on temporary NO_DETECTION

**Violation States:**
- `rfid_consecutive_non_compliant`: Counter for violations
- `rfid_consecutive_compliant`: Counter for compliance
- `rfid_current_uid_violated`: Flag if violation recorded
- `rfid_current_uid_compliant`: Flag if compliant detected

**Asynchronous Recording:**
- Violation recording happens in a background thread (`threading.Thread`)
- Prevents blocking the detection worker
- Email queuing is non-blocking (only queues, never sends synchronously)
- System continues operating even if violation recording fails

---

## RFID Integration

### RFID Scanner Module (`src/rfid_scanner.py`)
- **Hardware**: USB RFID reader
- **Protocol**: Serial communication
- **Event System**: Queue-based event handling

### RFID Workflow

1. **Card Detection**
   - RFID reader detects card
   - UID extracted and queued
   - Event handler processes in background

2. **Student Lookup**
   - UID matched against `students` table
   - Student information loaded
   - RFID log entry created

3. **Detection Control**
   - Detection enabled only when valid card present
   - Disabled if student already has violation today
   - Disabled if compliant status detected
   - Reset when new card detected

4. **Violation Linking**
   - Violations linked to student via RFID UID
   - Strike count calculated automatically
   - Email notifications sent to student

### RFID States
- `rfid_present`: Card currently detected
- `rfid_last_uid`: Last detected UID
- `rfid_last_student`: Student info for current card
- `rfid_enabled`: System-level enable/disable flag

---

## Schedule System

### Purpose
Controls when RFID and detection systems are active.

### Configuration
- **Days**: Monday through Sunday (toggle on/off)
- **Time Range**: Start time and end time
- **Storage**: `settings` table with JSON format

### Behavior
- **Within Schedule**: RFID and detection active
- **Outside Schedule**: RFID and detection disabled
- **Test Mode**: Overrides schedule (for testing)

### Implementation
- Background thread checks schedule every 10 seconds
- Updates `rfid_enabled` flag based on schedule
- Visual indicators in UI show schedule status

---

## Background Threads

### 1. Schedule Checker (`schedule_rfid_checker()`)
- **Frequency**: Every 10 seconds
- **Purpose**: Enable/disable RFID based on schedule
- **Location**: `app.py`

### 2. RFID Event Handler (`rfid_event_handler()`)
- **Type**: Event-driven (processes queue)
- **Purpose**: Handle RFID card detections
- **Actions**: Student lookup, detection control, logging

### 3. Detection Worker (`detection_worker()`)
- **Type**: Queue-based processing
- **Purpose**: Process video frames for detection
- **Input**: Frame queue from camera feed
- **Output**: Detection results stored in shared state

### 4. Email Outbox Worker (`email_outbox_worker()`)
- **Frequency**: Checks every 15 seconds
- **Purpose**: Process queued emails and retry failed sends
- **Batch Size**: Processes up to 5 emails per cycle
- **Retry Delay**: 10 seconds for failed emails
- **Status Management**: Updates email status in `email_outbox` table
- **Error Handling**: Logs errors and marks emails as failed
- **Location**: `app.py` line ~306

### 5. Follow-up Email Scheduler (`followup_email_scheduler()`)
- **Frequency**: Daily (24 hours)
- **Purpose**: Send follow-up emails for old violations
- **Criteria**: Violations 3+ days old, status='pending', followup_sent=0
- **Duplicate Prevention**: Sets `followup_sent=1` before sending

### 6. Auto-Sync Thread (`auto_sync_to_aiven()`)
- **Frequency**: Checks every 60 seconds, syncs every 5 minutes
- **Purpose**: Backup local database to Aiven
- **Conditions**: Only syncs when Aiven available and auto-sync enabled
- **Debug Logging**: Comprehensive logging for troubleshooting

### 7. Violation Recording Thread
- **Type**: Spawned per violation (background thread)
- **Purpose**: Record violation asynchronously without blocking detection
- **Actions**: Database insert, email queuing
- **Location**: `app.py` `_maybe_record_violation()` function

---

## User Roles & Dashboards

### Security Role
- **Dashboard**: `index.html` (main security dashboard)
- **Features**:
  - Real-time camera feed with detection overlay
  - RFID status monitoring
  - Schedule configuration
  - Test mode toggle
  - Auto-sync control
  - System status indicators

### Dean Role
- **Dashboard**: `dean_dashboard.html`
- **Features**:
  - View violations for their college only
  - Filter and search violations
  - Update violation status
  - Generate PDF reports
  - Analytics and statistics

### OSAS Role
- **Dashboard**: `osas_dashboard.html`
- **Features**:
  - University-wide violation oversight
  - All colleges visible
  - Advanced analytics
  - Report generation
  - System-wide statistics

### Guidance Role
- **Dashboard**: `guidance_dashboard.html`
- **Features**:
  - Student counseling support
  - Violation management
  - Student information access
  - Support tools

---

## Email System

### Email Queuing & Offline Support

The system includes a robust email queuing mechanism that ensures emails are sent even when the system is offline:

#### Email Outbox Table (`email_outbox`)
- Stores all emails that need to be sent
- Tracks status: `pending`, `sending`, `sent`, `failed`
- Records attempt count and last error for debugging
- Links to violations via `violation_id` foreign key

#### Email Queuing Process
1. **Violation Detected**: Email details are queued in `email_outbox` table
2. **Asynchronous Queuing**: Violation recording happens in background thread (non-blocking)
3. **Background Worker**: `email_outbox_worker()` processes queued emails every 15 seconds
4. **Retry Logic**: Failed emails are retried after 10 seconds
5. **Automatic Recovery**: When connectivity returns, all queued emails are sent automatically

#### Email Outbox Worker (`email_outbox_worker()`)
- **Frequency**: Checks every 15 seconds
- **Batch Size**: Processes up to 5 emails per cycle
- **Retry Delay**: 10 seconds for failed emails
- **Status Management**: Updates email status (`pending` → `sending` → `sent`/`failed`)
- **Error Handling**: Logs errors and marks emails as failed with error message

### Email Types

#### 1. Initial Violation Notification
- **Trigger**: When violation is recorded
- **Process**: Queued in `email_outbox` table (not sent immediately)
- **Recipient**: Student email from database
- **Content**: Violation details, strike count, proof image
- **Template**: `generate_violation_email_body()`
- **Offline Behavior**: Queued and sent when connectivity returns

#### 2. Follow-up Notification
- **Trigger**: Automatic (3+ days after violation, if still pending)
- **Purpose**: Remind student of unresolved violation
- **Duplicate Prevention**: `followup_sent` flag in database
- **Scheduler**: Background thread runs daily

### Email Configuration
- **SMTP**: Configured via `.env` file
- **Library**: Flask-Mail
- **Templates**: HTML email templates in `src/email_templates.py`
- **Offline Support**: Emails queued when offline, sent automatically when online

### Email Outbox Functions (`src/config.py`)
- `enqueue_email_outbox()`: Queue email for sending
- `get_due_email_outbox_entries()`: Get emails ready to send (pending or failed after retry delay)
- `mark_email_outbox_attempting()`: Mark email as being sent
- `mark_email_outbox_sent()`: Mark email as successfully sent
- `mark_email_outbox_failed()`: Mark email as failed with error message

---

## API Endpoints

### Authentication (`routes/auth.py`)
- `POST /login` - User login
- `POST /logout` - User logout

### Camera (`routes/camera.py`)
- `GET /video_feed` - Video stream with detections
- `POST /camera/start` - Start camera
- `POST /camera/stop` - Stop camera
- `GET /camera/status` - Camera status

### RFID (`routes/rfid.py`)
- `GET /rfid/status` - RFID status and student info
- `POST /rfid/enable` - Enable RFID
- `POST /rfid/disable` - Disable RFID

### Violations (`routes/violations.py`)
- `GET /dean/violations` - Dean violations (college-filtered)
- `GET /osas/violations` - OSAS violations (all)
- `GET /guidance/violations` - Guidance violations
- `POST /violations/<id>/update` - Update violation status
- `GET /violations/<id>/report` - Generate PDF report
- `POST /violations/followup` - Send follow-up emails

### Students (`routes/students.py`)
- `GET /students` - List students
- `POST /students` - Create student
- `PUT /students/<id>` - Update student
- `DELETE /students/<id>` - Delete student

### Settings (`routes/settings.py`)
- `GET /api/settings/schedule` - Get schedule
- `POST /api/settings/schedule` - Update schedule
- `GET /api/settings/schedule/check` - Check if currently active
- `GET /api/settings/auto-sync` - Get auto-sync status
- `POST /api/settings/auto-sync` - Toggle auto-sync

### Files (`routes/files.py`)
- `GET /uploads/<filename>` - Serve uploaded files
- `GET /results/<filename>` - Serve result images
- `GET /violations/<filename>` - Serve violation images

---

## File Structure

### Uploads & Results
- **`uploads/`**: User-uploaded images
- **`results/`**: Processed detection images
- **`results/violations/`**: Violation proof images

### Static Assets
- **`static/css/`**: Stylesheets
- **`static/js/`**: JavaScript modules
- **`static/images/`**: UI images

### Templates
- **`templates/`**: HTML templates for each dashboard
- Jinja2 templating for dynamic content

---

## Configuration

### Environment Variables (`.env`)

#### Required
```env
# Local Database (Primary)
LOCAL_DB_HOST=localhost
LOCAL_DB_PORT=3306
LOCAL_DB_USER=root
LOCAL_DB_PASSWORD=your_password
LOCAL_DB_NAME=dress

# Email Configuration
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USE_TLS=True
MAIL_USERNAME=your_email@gmail.com
MAIL_PASSWORD=your_app_password
```

#### Optional
```env
# Aiven Database (Backup)
DB_HOST=your-aiven-host.aivencloud.com
DB_PORT=22870
DB_USER=avnadmin
DB_PASSWORD=your_password
DB_NAME=dress
DB_SSL_CA=certs/ca.pem
DB_SSL_REQUIRED=true
DB_SSL_DISABLED=false
```

---

## Key Algorithms & Logic

### Bot-SORT Tracking
- **Purpose**: Track people across frames
- **Implementation**: `src/botsort_tracker.py`
- **Output**: Consistent track IDs for same person
- **Usage**: Only track_id == 1 is processed for dress detection

### Violation Recording Algorithm
1. Check if RFID student present
2. Check if already violated today → skip
3. Check detection results for worst status
4. Increment violation counter if non-compliant
5. Increment compliance counter if compliant
6. Record violation after 3 consecutive violations
7. Reset counters when status changes

### Auto-Sync Algorithm
1. Check if auto-sync enabled (global flag)
2. Check if Aiven available (cached 30 seconds)
3. Check if 5 minutes passed since last sync
4. Connect to both databases
5. Sync all tables (truncate + insert)
6. Update last sync time
7. Log results

---

## Security Features

### Authentication
- Session-based authentication
- Password hashing (Werkzeug)
- Role-based access control (RBAC)

### Authorization
- Route-level role checks
- College-level filtering for deans
- Security role required for system settings

### Data Protection
- SQL injection prevention (parameterized queries)
- File upload validation
- Secure password storage

---

## Performance Optimizations

### Async Detection Processing
- Detection runs in background thread
- Camera feed not blocked by detection
- Queue-based frame processing
- Latest results cached for display

### Asynchronous Violation Recording
- Violation recording happens in separate thread
- Detection worker never blocks on database operations
- Email queuing is non-blocking (only queues, never sends synchronously)
- System remains responsive even during violation recording

### Database Optimization
- Connection pooling via `get_connection()`
- Local database for all operations (fast)
- Backup sync doesn't block operations

### Caching
- Aiven availability cached (30 seconds)
- Detection results cached
- Dean alerts cached per college

---

## Error Handling

### Database Errors
- Graceful degradation if Aiven unavailable
- Local database errors logged and handled
- Transaction rollback on sync failures

### Detection Errors
- Continues processing if frame fails
- Logs errors without crashing
- Fallback to previous detection results

### RFID Errors
- Handles unregistered cards gracefully
- Continues operation if RFID unavailable
- Logs all RFID events

### Network/Email Errors
- **Offline Operation**: System works fully offline
- **Email Queuing**: Emails queued when offline, sent when online
- **Retry Logic**: Failed emails retried automatically after 10 seconds
- **Error Logging**: Email errors stored in `email_outbox.last_error`
- **Non-Blocking**: Email failures don't block violation recording

### Frontend Resilience
- **Network Detection**: Detects online/offline status
- **Timeout Handling**: All fetch requests have 7-second timeout
- **Request Deduplication**: Prevents concurrent identical requests
- **Error Recovery**: Automatically retries failed requests
- **Video Feed Recovery**: Auto-restarts video feed on errors or network reconnection
- **Localhost Endpoints**: Local endpoints (`/rfid/status`, `/api/settings/schedule/check`) work offline

---

## Debugging & Logging

### Debug Messages
- RFID status checks
- Detection processing
- Violation recording
- Auto-sync operations
- Schedule checks

### Log Files
- `rfid_scanner.log`: RFID scanner events
- Console output: Application logs

### Debug Endpoints
- `routes/debug.py`: Debug utilities
- System state inspection
- Manual trigger endpoints

---

## Testing Features

### Test Mode
- Overrides schedule restrictions
- Allows testing outside scheduled hours
- Toggle in Security dashboard
- Visual indicator when active

### Manual Sync
- `scripts/sync_database.py`: Interactive sync tool
- Choose sync direction
- Select tables to sync
- View sync status

---

## Deployment Considerations

### Requirements
- Python 3.8+
- MySQL database
- Webcam (for detection)
- RFID reader (optional)
- Internet connection (for Aiven backup, optional)

### Startup Sequence
1. Load environment variables
2. Initialize database connections
3. Load ML models
4. Start background threads
5. Initialize Flask app
6. Register blueprints
7. Start web server

### Background Threads Startup
- Schedule checker: Immediate
- RFID handler: Immediate
- Detection worker: Immediate
- Follow-up email: 10 second delay
- Auto-sync: 15 second delay

---

## Known Limitations

1. **Single Person Tracking**: Only track_id == 1 is processed for dress detection
2. **Schedule Dependency**: Detection disabled outside scheduled hours (unless test mode)
3. **RFID Required**: Violations only recorded when valid RFID card present
4. **3-Strike Rule**: Requires 3 consecutive violations before recording
5. **Aiven Dependency**: Backup sync requires Aiven availability (optional, system works without it)

## Offline Operation

### What Works Offline
- ✅ Violation detection and recording
- ✅ RFID scanning and student identification
- ✅ Camera feed display
- ✅ Dashboard updates (localhost endpoints)
- ✅ Database operations (local database)
- ✅ Email queuing (emails stored for later sending)

### What Gets Queued
- ⏳ Email notifications (queued in `email_outbox` table)
- ⏳ Cloud database sync (resumes when online)

### Automatic Recovery
- Email worker checks every 15 seconds for queued emails
- Retries failed emails after 10 seconds
- Sends all queued emails when connectivity returns
- Frontend automatically detects network reconnection and updates UI

---

## Future Enhancements (Potential)

1. Multi-person tracking and detection
2. Real-time notifications (WebSocket)
3. Mobile app integration
4. Advanced analytics dashboard
5. Machine learning model retraining pipeline
6. Multi-camera support
7. Cloud deployment options

---

## Maintenance Notes

### Regular Tasks
- Monitor auto-sync logs
- Check email delivery
- Review violation records
- Update student database
- Backup local database

### Troubleshooting
- Check logs for errors
- Verify schedule configuration
- Test RFID reader connection
- Verify camera access
- Check email SMTP settings

---

## Version Information

- **System**: DRESS v2
- **Python**: 3.8+
- **Flask**: Latest
- **YOLOv8**: Ultralytics
- **Database**: MySQL 8.0+

---

*Last Updated: Based on current codebase analysis*
*Documentation covers all major system components and workflows*




