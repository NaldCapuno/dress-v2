# Database Sync System - How It Works

## Overview
The system uses **Local (MySQL)** as the **primary database** for all operations, with **Aiven (cloud)** serving as a **backup** that is synced periodically.

---

## 1. Database Strategy

### Primary Database: Local
- **All operations** use the local MySQL database
- **No fallback needed** - local database is always used
- **Fast and reliable** - no network dependency for normal operations

### Backup Database: Aiven
- **Used only for backup** - synced periodically from local
- **Automatic sync** - happens in background every 5 minutes (when available)
- **Offline-capable** - system works normally even if Aiven is unavailable

### Connection Logic (`src/config.py`)
1. When `get_connection()` is called:
   - Always connects to **Local database** (primary)
   - Aiven is only used for backup sync operations
   - No automatic fallback needed

### Key Functions
- `get_connection()` - Always returns connection to local database (primary)
- `get_current_database()` - Always returns 'local'
- `is_aiven_available()` - Checks if Aiven is reachable (for backup sync)
- `auto_sync_to_aiven()` - Syncs local → Aiven periodically

### Configuration
Set in `.env` file:
```env
# Aiven Database (Backup Only)
DB_HOST=your-aiven-host.aivencloud.com
DB_PORT=22870
DB_USER=avnadmin
DB_PASSWORD=your-password
DB_NAME=dress

# Local Database (Primary)
LOCAL_DB_HOST=localhost
LOCAL_DB_PORT=3306
LOCAL_DB_USER=root
LOCAL_DB_PASSWORD=root
LOCAL_DB_NAME=dress
```

---

## 2. Backup Sync System

### Sync Direction: Local → Aiven (Backup)
**When**: Periodically when Aiven is available

**How It Works**:
1. Background thread checks if Aiven is available (cached for 30 seconds)
2. Every 5 minutes (300 seconds):
   - If Aiven is available → syncs all data from Local → Aiven
   - If Aiven is unavailable → skips sync, continues checking
3. Sync happens automatically in background
4. Normal operations continue on Local database regardless of sync status

**Function**: `auto_sync_to_aiven()` in `app.py` (line ~1951)

**Timing**:
- Checks Aiven availability every iteration
- Syncs every 5 minutes (300 seconds) when Aiven is available
- Waits 15 seconds after app startup before first check

---

## 3. Sync Process Details

### What Gets Synced
- **All tables** in the database
- **All data** (rows) in each table
- Tables are synced in dependency order (parent tables first)

### Sync Method
1. Disables foreign key checks temporarily
2. Truncates destination tables (Aiven)
3. Inserts all data from source (Local)
4. Re-enables foreign key checks
5. Commits transaction

### Important Notes
- **Schema is NOT synced** - assumes both databases have same structure
- **Data is overwritten** - Aiven tables are cleared before sync
- **Foreign keys are handled** - automatically managed during sync
- **Local is source of truth** - Aiven is always overwritten with Local data

---

## 4. Timing Configuration

### Connection Check Cache
- **Location**: `src/config.py` line ~184
- **Value**: 30 seconds
- **Purpose**: Caches Aiven availability check to reduce connection attempts

### Local → Aiven Sync
- **Function**: `auto_sync_to_aiven()` in `app.py` (line ~1951)
- **Initial Delay**: 15 seconds after app startup
- **Check Interval**: Continuous (checks every iteration)
- **Sync Interval**: 300 seconds (5 minutes) when Aiven is available

### Frontend Status Checks
- **Location**: `templates/index.html`
- **Interval**: 5000ms (5 seconds)
- **Purpose**: Updates UI indicators for database status

---

## 5. Scenarios

### Scenario 1: Normal Operation (Aiven Available)
1. System starts → uses Local database (primary)
2. Background thread checks Aiven availability
3. Every 5 minutes → syncs Local → Aiven (backup)
4. UI shows: "Local (Backup: Aiven)" (green)
5. All operations use Local database

### Scenario 2: Aiven Unavailable
1. System starts → uses Local database (primary)
2. Background thread detects Aiven unavailable
3. Sync is skipped (Aiven not available)
4. UI shows: "Local (Backup: Offline)" (yellow/orange)
5. All operations continue normally on Local database
6. Sync resumes automatically when Aiven becomes available

### Scenario 3: Aiven Comes Back Online
1. System is using Local database (primary)
2. Background thread detects Aiven available
3. Next sync cycle (within 5 minutes) → syncs Local → Aiven
4. UI updates to: "Local (Backup: Aiven)" (green)
5. Continues periodic sync every 5 minutes

### Scenario 4: Extended Aiven Downtime
1. System uses Local database for extended period
2. Many changes accumulate in Local
3. When Aiven comes back:
   - Next sync cycle syncs all Local changes to Aiven
   - Aiven gets updated with all accumulated changes
   - Then continues normal periodic sync

---

## 6. UI Indicators

### Database Status Indicator
- **Location**: Status bar at top of page
- **Shows**:
  - 🟢 **Local (Backup: Aiven)** (green) - Using Local, Aiven backup available
  - 🟡 **Local (Backup: Offline)** (yellow/orange) - Using Local, Aiven backup unavailable
  - ⚪ **Unknown** (gray) - Status unclear

### Update Frequency
- Checks every 5 seconds
- Updates automatically

---

## 7. Background Threads

### Thread 1: Local → Aiven Backup Sync
- **Function**: `auto_sync_to_aiven()`
- **Started**: On app startup
- **Purpose**: Periodically syncs Local → Aiven (every 5 minutes when Aiven available)
- **Location**: `app.py` line ~1951

### Thread 2: Schedule Checker
- **Function**: `schedule_rfid_checker()`
- **Purpose**: Manages RFID enabled/disabled based on schedule
- **Location**: `app.py`

### Thread 3: Follow-up Email Scheduler
- **Function**: `followup_email_scheduler()`
- **Purpose**: Sends follow-up emails for violations that are 3+ days old and still pending
- **Location**: `app.py` line ~2077
- **Timing**: 
  - Waits 10 seconds after app startup
  - Checks immediately, then every 24 hours
  - Uses `followup_sent` flag in `violations` table to prevent duplicates
- **Details**:
  - Queries violations where `status = 'pending'`, `timestamp <= NOW() - INTERVAL 3 DAY`, and `followup_sent IS NULL OR followup_sent = 0`
  - Marks violations as `followup_sent = 1` before sending to prevent duplicates
  - Includes debug logging for troubleshooting

---

## 8. Follow-up Email System

### How It Works
1. Background scheduler checks for violations needing follow-up emails
2. Finds violations that are:
   - Status: `pending`
   - Age: 3+ days old (`timestamp <= NOW() - INTERVAL 3 DAY`)
   - Not yet sent: `followup_sent IS NULL OR followup_sent = 0`
3. For each violation:
   - Marks as `followup_sent = 1` BEFORE sending (prevents duplicates)
   - Sends email to student
   - Logs success/failure

### Duplicate Prevention
- **Database flag**: `followup_sent` column in `violations` table
- **Atomic update**: Uses `UPDATE ... WHERE followup_sent IS NULL OR followup_sent = 0` to prevent race conditions
- **Pre-check**: Verifies current state before marking to avoid unnecessary processing
- **App restart safe**: Even if app restarts, already-sent emails won't be resent

### Database Schema
- Column: `followup_sent TINYINT(1) DEFAULT 0`
- Values: `0` or `NULL` = not sent, `1` = sent
- Auto-created: Column is automatically added if it doesn't exist

### Function Location
- **Endpoint**: `routes/violations.py` - `send_followup_emails()` (line ~1732)
- **Scheduler**: `app.py` - `followup_email_scheduler()` (line ~2077)

---

## 9. Error Handling

### Connection Failures
- Local database failures: System cannot operate (local is primary)
- Aiven failures: System continues normally (Aiven is backup only)
- Logs warning messages for Aiven unavailability

### Sync Failures
- Logs error messages
- Continues checking for next sync opportunity
- Doesn't interrupt normal operation (Local is primary)

### Database Errors
- Handles foreign key constraints during sync
- Manages transaction rollbacks
- Continues with other tables if one fails

### Follow-up Email Errors
- Logs errors for individual violations
- Continues processing other violations
- Marks violations as sent only after successful email send

---

## 10. Best Practices

### For Aiven Free Plan
- Current timings are optimized for free tier
- 5-minute sync interval is reasonable
- 30-second cache reduces connection attempts
- Local primary strategy reduces Aiven usage

### For Production
- Consider increasing sync intervals if needed
- Monitor sync logs for errors
- Ensure both databases have same schema
- Regular backups of Local database recommended

### Manual Sync
- Use `scripts/sync_database.py` for manual syncs
- Interactive menu for choosing sync direction
- Can sync specific tables if needed

---

## 11. Troubleshooting

### Sync Not Happening
- Check if Aiven is actually available
- Verify Aiven credentials in `.env` file
- Check logs for error messages
- Verify sync thread is running (check app startup logs)

### Follow-up Emails Not Sending
- Check if violations are 3+ days old
- Verify violations have `status = 'pending'`
- Check if `followup_sent` is already `1` (already sent)
- Check logs for debug messages: `"DEBUG: Found X violations needing follow-up emails"`
- Verify email configuration is correct

### Data Conflicts
- Sync overwrites Aiven data with Local data
- Local is always source of truth
- Use manual sync script for selective syncs if needed

### Performance Issues
- Increase sync intervals if needed
- Reduce frontend check frequency
- Monitor database connection limits
- Local database performance is most important (primary)

---

## Summary

The system provides:
✅ **Local database as primary** - Fast, reliable, no network dependency
✅ **Aiven as backup** - Automatic periodic sync (every 5 minutes when available)
✅ **Offline-capable** - Works normally even if Aiven is unavailable
✅ **Visual indicators in UI** - Shows backup status
✅ **Seamless operation** - No interruption to normal operations
✅ **Error handling and recovery** - Continues operation even if backup sync fails
✅ **Follow-up email system** - Automatic emails for 3+ day old violations with duplicate prevention

All syncing happens automatically in the background - no manual intervention needed!
