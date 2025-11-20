# Debug Utilities

This module provides centralized debug logging for the DRESS application.

## Usage

### Enabling Debug Mode

Set the `DEBUG` environment variable to enable debug output:

```bash
# Windows (Command Prompt)
set DEBUG=true

# Windows (PowerShell)
$env:DEBUG="true"

# Linux/Mac
export DEBUG=true
```

Or add to your `.env` file:
```
DEBUG=true
```

### Using Debug Functions

Import the debug functions you need:

```python
from src.debug_utils import debug_print, debug_rfid, debug_violation, debug_email
```

Available debug functions:
- `debug_print(message, prefix="DEBUG")` - General debug messages
- `debug_rfid(message)` - RFID operations
- `debug_violation(message)` - Violation detection and recording
- `debug_email(message)` - Email sending operations
- `debug_compliance(message)` - Compliance checking
- `debug_database(message)` - Database operations
- `debug_camera(message)` - Camera operations
- `debug_sync(message)` - Sync operations

### Example

```python
from src.debug_utils import debug_rfid

# This will only print if DEBUG=true
debug_rfid(f"Card detected: {uid}")
```

## Benefits

1. **Centralized Control**: Enable/disable all debug output with one environment variable
2. **Categorized Output**: Different prefixes for different subsystems
3. **Timestamps**: All debug messages include timestamps
4. **Clean Production**: No debug output in production unless explicitly enabled
5. **Easy Maintenance**: Update debug behavior in one place

