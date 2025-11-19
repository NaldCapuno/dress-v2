# DRESS System Deployment Guide

## Overview
This guide explains how to deploy the DRESS (Dress-code Recognition Surveillance System) in a university setting, including single and multi-entry point configurations, remote camera setups, and infrastructure requirements.

---

## System Architecture

### Core Components
1. **Flask Application Server** - Main processing engine
2. **MySQL Database** - Local database (primary) + optional cloud backup
3. **Camera** - USB webcam or IP/network camera
4. **RFID Reader** - USB-connected RFID scanner
5. **Email System** - SMTP server for violation notifications

---

## Deployment Options

### Option 1: Single Entry Point (Simple Setup)
**Architecture:**
```
[Single Computer at Entrance]
├── Flask App (runs 24/7)
├── MySQL Database (local)
├── USB Camera
├── USB RFID Reader
└── Internet connection (for email)
```

**Requirements:**
- One dedicated computer/server
- Camera and RFID reader connected via USB
- Stable power supply (UPS recommended)
- Internet connection (optional - system works offline)

**Use Case:** Single main entrance monitoring

---

### Option 2: Multiple Entry Points (Distributed)
**Architecture:**
```
[Main Server Computer]
├── Flask App Instance 1
├── MySQL Database (shared)
└── Connects to multiple entry points

[Entry Point 1 Computer]
├── Camera
└── RFID Reader

[Entry Point 2 Computer]
├── Camera
└── RFID Reader
```

**Requirements:**
- One main server computer
- Multiple entry point computers (one per entrance)
- Network connection between all computers
- Shared database (network MySQL or cloud sync)

**Use Case:** Multiple entrances, centralized management

---

### Option 3: Centralized Server with Remote Cameras (Recommended)
**Architecture:**
```
[Main Server Computer - Server Room]
├── Flask App (runs 24/7)
├── MySQL Database (local)
├── RFID Reader Connection (see options below)
└── Network connection to cameras

[IP Camera at Entrance 1]
└── Streams video over network (RTSP/HTTP)

[IP Camera at Entrance 2]
└── Streams video over network (RTSP/HTTP)

[RFID Reader at Entrance]
└── Connected via USB extension or network solution
```

**Requirements:**
- One main server computer (can be in server room/closet)
- IP/Network cameras at each entrance
- Network infrastructure (LAN)
- RFID reader connection solution (see RFID Connection Options below)

**Use Case:** Professional setup with centralized management

**Benefits:**
- Server secured in server room
- Cameras at entrances (no exposed computer)
- Multiple cameras can connect to one server
- Easier maintenance and monitoring

---

## Remote Camera Configuration

### Supported Camera Types

#### 1. USB Webcam (Local)
- Direct USB connection to server computer
- Camera ID: `0, 1, 2...` (selected in dashboard)
- **Code:** `cv2.VideoCapture(0)`

#### 2. IP/Network Camera (Remote)
- Network-connected camera
- Streams video over RTSP or HTTP
- **Code:** `cv2.VideoCapture('rtsp://192.168.1.100:554/stream')` or `cv2.VideoCapture('http://192.168.1.100/video')`

### Setting Up Remote Camera

1. **Configure IP Camera:**
   - Set static IP address (e.g., `192.168.1.100`)
   - Enable RTSP/HTTP streaming
   - Note the stream URL (e.g., `rtsp://192.168.1.100:554/stream`)

2. **Update Camera Selection:**
   - In the dashboard, select camera by URL instead of ID
   - Or modify `app.py` to use camera URL directly

3. **Network Requirements:**
   - Server must be able to reach camera IP address
   - Ensure firewall allows camera stream ports
   - Test connection: `ping 192.168.1.100`

### Example Camera URLs
- RTSP: `rtsp://username:password@192.168.1.100:554/stream`
- HTTP: `http://192.168.1.100/video`
- MJPEG: `http://192.168.1.100:8080/video`

---

## RFID Scanner Connection Options

The system uses **PC/SC-compatible USB RFID readers** (e.g., ACR122U). When the server is in a closet/room away from the entrance, you have several connection options:

### Option 1: USB Extension Cable (Simple, Limited Distance)
**Setup:**
- Use active USB extension cable (up to 5 meters / 16 feet)
- Connect RFID reader at entrance to server via USB cable
- **Limitations:** USB 2.0 max length ~5 meters, USB 3.0 ~3 meters
- **Cost:** Low
- **Best For:** Short distances (< 5 meters)

**Architecture:**
```
[Server Room]
└── USB Extension Cable (5m max)
    └── [Entrance]
        └── RFID Reader (ACR122U)
```

### Option 2: USB over Ethernet/IP Extender (Recommended for Medium Distance)
**Setup:**
- USB-to-Ethernet extender device (sender + receiver)
- Connect RFID reader to extender at entrance
- Connect extender to server via Ethernet cable (up to 100 meters / 328 feet)
- **Limitations:** Requires power at entrance for extender device
- **Cost:** Medium
- **Best For:** Medium distances (5-100 meters)

**Architecture:**
```
[Server Room]
└── USB over Ethernet Receiver
    └── Ethernet Cable (up to 100m)
        └── [Entrance]
            └── USB over Ethernet Sender
                └── RFID Reader (ACR122U)
```

**Popular Products:**
- StarTech USB 2.0 over Ethernet Extender
- IOGEAR USB 2.0 Extender over Cat5e/6
- ATEN USB Extender over IP

### Option 3: Network-Enabled RFID Reader (Best for Long Distance)
**Setup:**
- Use network-enabled RFID reader (if available)
- Reader connects to network via Ethernet/WiFi
- Server communicates with reader over network
- **Limitations:** Requires compatible network-enabled reader (may need code modifications)
- **Cost:** High
- **Best For:** Long distances, multiple entrances

**Architecture:**
```
[Server Room]
└── Network Connection
    └── [Entrance]
        └── Network RFID Reader
            └── Ethernet/WiFi Connection
```

**Note:** Current system uses PC/SC standard (USB). Network-enabled readers may require code modifications to use TCP/IP instead of PC/SC.

### Option 4: Intermediate Computer at Entrance (Alternative)
**Setup:**
- Small computer/Raspberry Pi at entrance
- RFID reader connected to entrance computer
- Entrance computer communicates with main server over network
- **Limitations:** Requires additional computer, network communication layer
- **Cost:** Medium-High
- **Best For:** Multiple entrances, complex setups

**Architecture:**
```
[Server Room]
└── Main Server
    └── Network Connection
        └── [Entrance]
            └── Entrance Computer (Raspberry Pi)
                └── RFID Reader (ACR122U)
```

**Note:** This would require modifying the system to support network-based RFID communication or running a lightweight RFID service on the entrance computer.

### Option 5: USB Hub with Active Extension (Multiple Readers)
**Setup:**
- USB hub at entrance
- Multiple RFID readers connected to hub
- Hub connected to server via active USB extension
- **Limitations:** USB hub adds complexity, distance still limited
- **Cost:** Medium
- **Best For:** Multiple readers at same entrance

**Architecture:**
```
[Server Room]
└── Active USB Extension Cable
    └── [Entrance]
        └── USB Hub
            ├── RFID Reader 1
            └── RFID Reader 2
```

### Recommended Solution

**For Most Deployments:** Use **Option 2 (USB over Ethernet Extender)**
- Works with existing PC/SC-compatible readers
- No code changes required
- Supports distances up to 100 meters
- Reliable and cost-effective
- Easy to set up and maintain

**For Very Long Distances:** Consider **Option 3 (Network-Enabled Reader)** or **Option 4 (Intermediate Computer)**
- Requires code modifications or additional infrastructure
- Better for complex multi-entrance setups

### Current System Compatibility

The system uses:
- **Library:** `pyscard` (PC/SC standard)
- **Reader Type:** PC/SC-compatible USB readers (e.g., ACR122U)
- **Connection:** Direct USB via PC/SC interface

**Compatible with:**
- ✅ Option 1: USB Extension Cable
- ✅ Option 2: USB over Ethernet Extender (appears as local USB device)
- ⚠️ Option 3: Network Reader (may require code changes)
- ⚠️ Option 4: Intermediate Computer (requires network communication layer)
- ✅ Option 5: USB Hub with Extension

---

## Infrastructure Requirements

### Hardware
- **Server Computer:**
  - CPU: Multi-core processor (recommended)
  - RAM: 8GB+ (16GB recommended for multiple cameras)
  - Storage: SSD recommended for database
  - OS: Windows 10/11 or Linux
  - Network: Ethernet connection (WiFi not recommended for server)

- **Camera:**
  - USB webcam OR IP/network camera
  - Minimum 720p resolution
  - Good lighting conditions at entrance

- **RFID Reader:**
  - USB-connected RFID scanner
  - Compatible with PC/SC standard

- **Power:**
  - UPS (Uninterruptible Power Supply) recommended
  - Prevents data loss during power outages

### Software
- Python 3.8+
- MySQL 8.0+
- Required Python packages (see `requirements.txt`)
- Web browser for admin dashboard

### Network
- **Local Network:** Required for remote cameras and multi-entry setups
- **Internet:** Optional but recommended for:
  - Email notifications (queued when offline, sent when online)
  - Cloud database backup (Aiven)
  - Remote admin access (if configured)

---

## Database Configuration

### Primary Database (Local)
- **Type:** MySQL (local installation)
- **Purpose:** All operations (violations, students, RFID logs)
- **Location:** Same computer as Flask app
- **Benefits:** Fast, works offline, no latency

### Backup Database (Cloud - Optional)
- **Type:** Aiven MySQL or similar cloud database
- **Purpose:** Backup/backup sync
- **Sync:** Periodic sync from local to cloud (every 5 minutes)
- **Benefits:** Data redundancy, disaster recovery

### Email Outbox Queue
- **Table:** `email_outbox`
- **Purpose:** Queue emails when offline or when sending fails
- **Behavior:** 
  - Emails queued when violation detected (always queued, never sent synchronously)
  - Background worker checks every 15 seconds
  - Processes up to 5 emails per cycle
  - Retries failed emails after 10 seconds
  - Automatically sends when connectivity returns
- **Status Tracking:** `pending`, `sending`, `sent`, `failed`
- **Error Handling:** Stores error messages in `last_error` field for debugging

---

## System Behavior When Offline

### What Still Works
- ✅ Violation detection and recording
- ✅ RFID scanning and student identification
- ✅ Camera feed display
- ✅ Dashboard updates (localhost endpoints)
- ✅ Database operations (local database)

### What Gets Queued
- ⏳ Email notifications (queued in `email_outbox` table)
- ⏳ Cloud database sync (resumes when online)

### Automatic Recovery
- **Email Outbox Worker:** Checks every 15 seconds for queued emails
- **Retry Logic:** Failed emails retried after 10 seconds
- **Batch Processing:** Processes up to 5 emails per cycle to avoid SMTP overload
- **Status Management:** Tracks email status and attempt count
- **Error Logging:** Stores error messages for troubleshooting
- **Frontend Recovery:** Automatically detects network reconnection and updates UI
- **Video Feed Recovery:** Auto-restarts video feed on errors or network reconnection

---

## Security Considerations

### Server Location
- **Recommended:** Secure server room/closet
- **Benefits:** 
  - Physical security
  - Climate control
  - Centralized management
  - No exposed equipment at entrance

### Network Security
- Use local network (not public internet) for camera streams
- Firewall rules to restrict access
- VPN for remote admin access (if needed)

### Access Control
- Admin dashboard requires login
- Role-based access (security, OSAS, dean, guidance)
- Session management

---

## Monitoring & Maintenance

### System Monitoring
- Check Flask app is running (auto-restart on crash recommended)
- Monitor database disk space
- Check email queue for stuck emails
- Review violation logs

### Maintenance Tasks
- Regular database backups
- Clear old violation images (if storage limited)
- Update Python packages periodically
- Check camera/RFID hardware health

### Troubleshooting
- **Server hangs:** Check violation recording (now async, shouldn't happen)
- **Emails not sending:** 
  - Check `email_outbox` table: `SELECT * FROM email_outbox WHERE status != 'sent'`
  - Verify SMTP settings in `.env` file
  - Check `last_error` field for error messages
  - Ensure email outbox worker is running (check logs)
- **Camera not working:** Verify camera connection/URL
- **RFID not detecting:** Check USB connection and drivers
- **Dashboard freezing:** Check browser console for errors, verify network connectivity
- **Offline operation issues:** Verify localhost endpoints work (`/rfid/status`, `/api/settings/schedule/check`)

---

## Deployment Checklist

### Initial Setup
- [ ] Install Python and dependencies
- [ ] Install and configure MySQL
- [ ] Run database schema (`database/dress_clean.sql`)
- [ ] Configure `.env` file (database, email settings)
- [ ] Connect camera (USB or IP)
- [ ] Connect RFID reader
- [ ] Test camera feed
- [ ] Test RFID scanning
- [ ] Configure email settings
- [ ] Set up system schedule
- [ ] Import student data

### Production Deployment
- [ ] Set up UPS for power protection
- [ ] Configure auto-start on boot (Windows Task Scheduler or systemd)
- [ ] Set up database backups (daily recommended)
- [ ] Configure cloud backup sync (optional)
- [ ] Test offline functionality
- [ ] Train admin users
- [ ] Document camera/RFID locations

---

## Performance Considerations

### Single Camera
- **CPU Usage:** Moderate (~20-30% on modern CPU)
- **Memory:** ~2-4GB RAM
- **Network:** Minimal (local only)

### Multiple Cameras
- **CPU Usage:** Scales linearly (~20-30% per camera)
- **Memory:** ~2-4GB per camera
- **Network:** Depends on camera stream quality

### Recommendations
- Use IP cameras for multiple entry points (better scalability)
- Consider dedicated GPU for AI inference (optional, for faster processing)
- SSD storage for database (faster violation recording)

---

## Support & Documentation

- **System Notes:** See `docs/SYSTEM_NOTES.md`
- **Debug Guide:** See `docs/DEBUG_README.md`
- **Database Schema:** See `database/dress_clean.sql`
- **Email Templates:** See `src/email_templates.py`

---

## Notes

- System is designed to work offline - violations are recorded even without internet
- Email notifications are queued and sent automatically when connectivity returns
- Localhost endpoints (`/rfid/status`, `/api/settings/schedule/check`) work without internet
- Database operations use local MySQL (fast, reliable, works offline)
- Cloud backup is optional but recommended for data redundancy

