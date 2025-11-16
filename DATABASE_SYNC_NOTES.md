# Database Sync System - How It Works

## Overview
The system automatically manages database connections between **Aiven (cloud)** and **Local (MySQL)** databases with bidirectional synchronization.

---

## 1. Automatic Fallback System

### How It Works
- **Primary**: Tries to connect to Aiven first
- **Fallback**: If Aiven is unreachable, automatically switches to Local database
- **Transparent**: No code changes needed - works automatically

### Connection Logic (`src/config.py`)
1. When `get_connection()` is called:
   - Checks if Aiven is available (cached for 30 seconds)
   - If Aiven available → connects to Aiven
   - If Aiven unavailable → connects to Local
   - Tracks which database is currently active

### Key Functions
- `get_connection()` - Main connection function with automatic fallback
- `get_current_database()` - Returns 'aiven' or 'local'
- `is_aiven_available()` - Checks if Aiven is reachable
- `has_pending_sync()` - Checks if sync is needed

### Configuration
Set in `.env` file:
```env
# Aiven Database (Primary)
DB_HOST=your-aiven-host.aivencloud.com
DB_PORT=22870
DB_USER=avnadmin
DB_PASSWORD=your-password
DB_NAME=dress

# Local Database (Fallback)
LOCAL_DB_HOST=localhost
LOCAL_DB_PORT=3306
LOCAL_DB_USER=root
LOCAL_DB_PASSWORD=root
LOCAL_DB_NAME=dress
```

---

## 2. Bidirectional Sync System

### Sync Direction 1: Local → Aiven
**When**: When Aiven comes back online after being offline

**How It Works**:
1. System detects Aiven is unavailable → switches to Local
2. `sync_pending` flag is set to `True`
3. Background thread checks every 60 seconds
4. When Aiven becomes available:
   - Syncs all data from Local → Aiven
   - Clears `sync_pending` flag
   - System switches to Aiven

**Function**: `auto_sync_to_aiven()` in `app.py` (line ~1911)

**Timing**:
- Checks every 60 seconds
- Syncs immediately when Aiven comes back online

---

### Sync Direction 2: Aiven → Local
**When**: When actively using Aiven database

**How It Works**:
1. System detects it's using Aiven
2. Background thread checks every 60 seconds
3. Every 5 minutes (300 seconds):
   - Syncs all data from Aiven → Local
   - Keeps local database updated

**Function**: `auto_sync_to_local()` in `app.py` (line ~2034)

**Timing**:
- Checks every 60 seconds
- Syncs every 5 minutes (300 seconds) when using Aiven

---

## 3. Sync Process Details

### What Gets Synced
- **All tables** in the database
- **All data** (rows) in each table
- Tables are synced in dependency order (parent tables first)

### Sync Method
1. Disables foreign key checks temporarily
2. Truncates destination tables
3. Inserts all data from source
4. Re-enables foreign key checks
5. Commits transaction

### Important Notes
- **Schema is NOT synced** - assumes both databases have same structure
- **Data is overwritten** - destination tables are cleared before sync
- **Foreign keys are handled** - automatically managed during sync

---

## 4. Timing Configuration

### Connection Check Cache
- **Location**: `src/config.py` line 174, 223
- **Value**: 30 seconds
- **Purpose**: Caches Aiven availability check to reduce connection attempts

### Local → Aiven Sync
- **Check Interval**: 60 seconds (`app.py` line 2031)
- **Sync Trigger**: Immediately when Aiven comes back online

### Aiven → Local Sync
- **Check Interval**: 60 seconds (`app.py` line 2155)
- **Sync Interval**: 300 seconds (5 minutes) (`app.py` line 2042)

### Frontend Status Checks
- **Location**: `templates/index.html` lines 721, 724, 728
- **Interval**: 5000ms (5 seconds)
- **Purpose**: Updates UI indicators

---

## 5. Scenarios

### Scenario 1: Normal Operation (Online)
1. System starts → tries Aiven
2. Aiven available → uses Aiven
3. Every 5 minutes → syncs Aiven → Local
4. UI shows: "Aiven" (green)

### Scenario 2: Going Offline
1. System detects Aiven unavailable
2. Automatically switches to Local
3. Sets `sync_pending = True`
4. UI shows: "Local" (yellow/orange)
5. All operations continue on Local

### Scenario 3: Coming Back Online
1. System detects Aiven available
2. Immediately syncs Local → Aiven
3. Switches to Aiven
4. Clears `sync_pending` flag
5. UI shows: "Aiven" (green)
6. Continues syncing Aiven → Local every 5 minutes

### Scenario 4: Extended Offline Period
1. System uses Local for extended period
2. Many changes accumulate in Local
3. When Aiven comes back:
   - All Local changes sync to Aiven
   - Aiven gets updated with all offline changes
   - Then continues normal bidirectional sync

---

## 6. UI Indicators

### Database Status Indicator
- **Location**: Status bar at top of page
- **Shows**:
  - 🟢 **Aiven** (green) - Using Aiven database
  - 🟡 **Local** (yellow/orange) - Using Local database
  - 🟡 **Local (Syncing...)** - Using Local, sync pending
  - ⚪ **Unknown** (gray) - Status unclear

### Update Frequency
- Checks every 5 seconds
- Updates automatically

---

## 7. Background Threads

### Thread 1: Local → Aiven Sync
- **Function**: `auto_sync_to_aiven()`
- **Started**: On app startup
- **Purpose**: Syncs Local → Aiven when Aiven comes back online

### Thread 2: Aiven → Local Sync
- **Function**: `auto_sync_to_local()`
- **Started**: On app startup
- **Purpose**: Syncs Aiven → Local periodically when using Aiven

### Thread 3: Schedule Checker
- **Function**: `schedule_rfid_checker()`
- **Purpose**: Manages RFID based on schedule

### Thread 4: Follow-up Email Scheduler
- **Function**: `followup_email_scheduler()`
- **Purpose**: Sends follow-up emails for violations

---

## 8. Error Handling

### Connection Failures
- Automatically falls back to Local
- Logs warning messages
- Continues operation seamlessly

### Sync Failures
- Logs error messages
- Continues checking for next sync opportunity
- Doesn't interrupt normal operation

### Database Errors
- Handles foreign key constraints
- Manages transaction rollbacks
- Continues with other tables if one fails

---

## 9. Best Practices

### For Aiven Free Plan
- Current timings are optimized for free tier
- 5-minute sync interval is reasonable
- 30-second cache reduces connection attempts

### For Production
- Consider increasing sync intervals if needed
- Monitor sync logs for errors
- Ensure both databases have same schema

### Manual Sync
- Use `scripts/sync_database.py` for manual syncs
- Interactive menu for choosing sync direction
- Can sync specific tables if needed

---

## 10. Troubleshooting

### Sync Not Happening
- Check if `sync_pending` flag is set
- Verify Aiven is actually available
- Check logs for error messages

### Data Conflicts
- Sync overwrites destination data
- Last sync wins (no merge logic)
- Use manual sync script for selective syncs

### Performance Issues
- Increase sync intervals if needed
- Reduce frontend check frequency
- Monitor database connection limits

---

## Summary

The system provides:
✅ Automatic fallback (Aiven → Local)
✅ Automatic sync when Aiven comes back online (Local → Aiven)
✅ Periodic sync when using Aiven (Aiven → Local)
✅ Visual indicators in UI
✅ Seamless operation without interruption
✅ Error handling and recovery

All syncing happens automatically in the background - no manual intervention needed!

