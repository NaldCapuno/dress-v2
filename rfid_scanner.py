#!/usr/bin/env python3
"""
Standalone RFID Scanner for ACR122U with Mifare Classic 1K
Clean helper functions approach with threading support
"""

import time
import threading
from typing import Optional, Dict, List, Any, Tuple
import logging
from datetime import datetime
from queue import Queue

# Configure logging
def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('rfid_scanner.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ------------------ RFID (ACR122U) Helpers ------------------
_rfid_io_lock = threading.Lock()

def get_rfid_uid(timeout_seconds=8):
    """Get RFID card UID with timeout"""
    try:
        from smartcard.System import readers
        from smartcard.util import toHexString
        from smartcard.Exceptions import CardConnectionException
    except Exception as import_error:
        return None, f"pyscard not installed or unavailable: {import_error}"

    try:
        with _rfid_io_lock:
            available_readers = readers()
            if not available_readers:
                return None, 'No PC/SC readers found. Ensure ACR122U driver is installed.'

            reader = None
            # Prefer ACR122U if multiple
            for r in available_readers:
                if 'ACR122' in str(r).upper() or 'ACR 122' in str(r).upper():
                    reader = r
                    break
            if reader is None:
                reader = available_readers[0]

            connection = reader.createConnection()
            connection.connect()  # T=1 or direct

            # ACR122U: Get Card UID command
            get_uid_apdu = [0xFF, 0xCA, 0x00, 0x00, 0x00]

            # Poll until card present or timeout
            start_time = time.time()
            last_error = None
            while time.time() - start_time < timeout_seconds:
                try:
                    data, sw1, sw2 = connection.transmit(get_uid_apdu)
                    if sw1 == 0x90 and sw2 == 0x00 and data:
                        uid_hex = ''.join(f"{b:02X}" for b in data)
                        return uid_hex, None
                    last_error = f"Unexpected status: {sw1:02X}{sw2:02X}"
                except CardConnectionException as e:
                    last_error = str(e)
                time.sleep(0.25)

            return None, last_error or 'Timeout waiting for card'
    except Exception as e:
        return None, f"RFID error: {e}"


def read_mifare_classic_block(block_number, key_hex='FFFFFFFFFFFF', key_type='A', key_slot=0x00):
    """Read a specific block from Mifare Classic card"""
    try:
        from smartcard.System import readers
    except Exception as import_error:
        return None, f"pyscard not installed or unavailable: {import_error}"

    if key_type.upper() not in ('A', 'B'):
        return None, 'key_type must be A or B'

    try:
        key_bytes = [int(key_hex[i:i+2], 16) for i in range(0, 12, 2)]
    except Exception:
        return None, 'key_hex must be 12 hex chars (e.g., FFFFFFFFFFFF)'

    with _rfid_io_lock:
        available_readers = readers()
        if not available_readers:
            return None, 'No PC/SC readers found.'

        reader = None
        for r in available_readers:
            if 'ACR122' in str(r).upper() or 'ACR 122' in str(r).upper():
                reader = r
                break
        if reader is None:
            reader = available_readers[0]

        conn = reader.createConnection()
        conn.connect()

        def transmit(apdu):
            data, sw1, sw2 = conn.transmit(apdu)
            if not (sw1 == 0x90 and sw2 == 0x00):
                raise RuntimeError(f"APDU failed SW={sw1:02X}{sw2:02X}")
            return data

        # Load key into volatile slot
        load_key_apdu = [0xFF, 0x82, 0x00, key_slot, 0x06] + key_bytes
        transmit(load_key_apdu)

        # Authenticate
        key_code = 0x60 if key_type.upper() == 'A' else 0x61
        auth_apdu = [0xFF, 0x86, 0x00, 0x00, 0x05, 0x01, 0x00, block_number & 0xFF, key_code, key_slot]
        transmit(auth_apdu)

        # Read 16 bytes
        read_apdu = [0xFF, 0xB0, 0x00, block_number & 0xFF, 0x10]
        data = transmit(read_apdu)
        return bytes(data), None


def read_mifare_classic_range(start_block, num_blocks, key_hex='FFFFFFFFFFFF', key_type='A'):
    """Read multiple consecutive blocks from Mifare Classic card"""
    collected = bytearray()
    for b in range(start_block, start_block + num_blocks):
        data, err = read_mifare_classic_block(b, key_hex=key_hex, key_type=key_type)
        if err:
            return None, f"Block {b}: {err}"
        collected.extend(data)
    return bytes(collected), None


def write_mifare_classic_block(block_number, data_bytes, key_hex='FFFFFFFFFFFF', key_type='A', key_slot=0x00):
    """Write data to a specific Mifare Classic block"""
    try:
        from smartcard.System import readers
    except Exception as import_error:
        return False, f"pyscard not installed or unavailable: {import_error}"

    if key_type.upper() not in ('A', 'B'):
        return False, 'key_type must be A or B'

    if len(data_bytes) != 16:
        return False, 'data_bytes must be exactly 16 bytes'

    try:
        key_bytes = [int(key_hex[i:i+2], 16) for i in range(0, 12, 2)]
    except Exception:
        return False, 'key_hex must be 12 hex chars (e.g., FFFFFFFFFFFF)'

    with _rfid_io_lock:
        available_readers = readers()
        if not available_readers:
            return False, 'No PC/SC readers found.'

        reader = None
        for r in available_readers:
            if 'ACR122' in str(r).upper() or 'ACR 122' in str(r).upper():
                reader = r
                break
        if reader is None:
            reader = available_readers[0]

        conn = reader.createConnection()
        conn.connect()

        def transmit(apdu):
            data, sw1, sw2 = conn.transmit(apdu)
            if not (sw1 == 0x90 and sw2 == 0x00):
                raise RuntimeError(f"APDU failed SW={sw1:02X}{sw2:02X}")
            return data

        # Load key into volatile slot
        load_key_apdu = [0xFF, 0x82, 0x00, key_slot, 0x06] + key_bytes
        transmit(load_key_apdu)

        # Authenticate
        key_code = 0x60 if key_type.upper() == 'A' else 0x61
        auth_apdu = [0xFF, 0x86, 0x00, 0x00, 0x05, 0x01, 0x00, block_number & 0xFF, key_code, key_slot]
        transmit(auth_apdu)

        # Write 16 bytes
        write_apdu = [0xFF, 0xD6, 0x00, block_number & 0xFF, 0x10] + list(data_bytes)
        transmit(write_apdu)
        return True, None


# Live RFID autoscan infrastructure
_rfid_thread = None
_rfid_thread_stop = threading.Event()
_rfid_subscribers = set()
_rfid_subscribers_lock = threading.Lock()
_rfid_last_uid = None
_rfid_last_time = 0.0
_rfid_presence_lock = threading.Lock()
_rfid_present = False

def _rfid_set_present(present: bool):
    global _rfid_present
    with _rfid_presence_lock:
        _rfid_present = bool(present)

def _rfid_is_present():
    with _rfid_presence_lock:
        return _rfid_present

def _publish_event(payload):
    with _rfid_subscribers_lock:
        for q in list(_rfid_subscribers):
            try:
                q.put_nowait(payload)
            except Exception:
                pass

def _rfid_poll_loop():
    """Background polling loop for RFID detection"""
    global _rfid_last_uid, _rfid_last_time
    debounce_seconds = 2.0
    while not _rfid_thread_stop.is_set():
        uid, err = get_rfid_uid(timeout_seconds=1)
        now = time.time()
        if uid:
            if uid != _rfid_last_uid or (now - _rfid_last_time) > debounce_seconds:
                _rfid_last_uid = uid
                _rfid_last_time = now
                _rfid_set_present(True)
                _publish_event({'type': 'uid', 'uid': uid, 'timestamp': now})
        else:
            # No UID read within this poll; mark not present
            _rfid_set_present(False)
        time.sleep(0.2)

def _ensure_rfid_thread_running():
    """Start RFID polling thread if not already running"""
    global _rfid_thread
    if _rfid_thread is None or not _rfid_thread.is_alive():
        _rfid_thread_stop.clear()
        _rfid_thread = threading.Thread(target=_rfid_poll_loop, name='rfid-poll', daemon=True)
        _rfid_thread.start()

def start_rfid_monitoring():
    """Start RFID monitoring in background"""
    _ensure_rfid_thread_running()
    logger.info("RFID monitoring started")

def stop_rfid_monitoring():
    """Stop RFID monitoring"""
    global _rfid_thread
    _rfid_thread_stop.set()
    if _rfid_thread and _rfid_thread.is_alive():
        _rfid_thread.join(timeout=2)
    logger.info("RFID monitoring stopped")

def subscribe_to_rfid_events() -> Queue:
    """Subscribe to RFID events, returns a queue for receiving events"""
    q = Queue()
    with _rfid_subscribers_lock:
        _rfid_subscribers.add(q)
    return q

def unsubscribe_from_rfid_events(q: Queue):
    """Unsubscribe from RFID events"""
    with _rfid_subscribers_lock:
        _rfid_subscribers.discard(q)

def get_rfid_status() -> Dict[str, Any]:
    """Get current RFID status"""
    return {
        'present': _rfid_is_present(),
        'last_uid': _rfid_last_uid,
        'last_time': _rfid_last_time,
        'monitoring': _rfid_thread is not None and _rfid_thread.is_alive(),
        'subscribers': len(_rfid_subscribers)
    }


class ACR122UScanner:
    """ACR122U RFID Scanner with helper functions"""
    
    def __init__(self):
        self.card_history = []
        self.monitoring = False
        
    def initialize(self) -> bool:
        """Initialize scanner and check for readers"""
        try:
            uid, error = get_rfid_uid(timeout_seconds=1)
            if error and "No PC/SC readers found" in error:
                logger.error("No RFID readers found")
                return False
            logger.info("RFID scanner initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing RFID scanner: {e}")
            return False
    
    def read_card(self) -> Optional[Dict[str, Any]]:
        """Read current card data"""
        uid, error = get_rfid_uid(timeout_seconds=2)
        if uid:
            card_data = {
                'uid': uid,
                'card_type': 'Mifare Classic 1K',
                'timestamp': time.time(),
                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'reader_name': 'ACR122U'
            }
            
            # Add to history
            self.card_history.append(card_data.copy())
            if len(self.card_history) > 50:
                self.card_history.pop(0)
            
            return card_data
        return None
    
    def read_block(self, block_number: int, key_hex: str = 'FFFFFFFFFFFF') -> Optional[str]:
        """Read a specific Mifare block"""
        data, error = read_mifare_classic_block(block_number, key_hex=key_hex)
        if data:
            return data.hex().upper()
        logger.warning(f"Failed to read block {block_number}: {error}")
        return None
    
    def write_block(self, block_number: int, data_str: str, key_hex: str = 'FFFFFFFFFFFF') -> bool:
        """Write data to a specific Mifare block"""
        # Convert string to bytes and pad/truncate to 16 bytes
        data_bytes = data_str.encode('utf-8')
        if len(data_bytes) > 16:
            data_bytes = data_bytes[:16]
        else:
            data_bytes = data_bytes.ljust(16, b'\x00')
        
        success, error = write_mifare_classic_block(block_number, data_bytes, key_hex=key_hex)
        if success:
            logger.info(f"Successfully wrote to block {block_number}")
            return True
        else:
            logger.error(f"Failed to write block {block_number}: {error}")
            return False
    
    def start_monitoring(self):
        """Start background monitoring"""
        if not self.monitoring:
            start_rfid_monitoring()
            self.monitoring = True
            logger.info("Started RFID monitoring")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        if self.monitoring:
            stop_rfid_monitoring()
            self.monitoring = False
            logger.info("Stopped RFID monitoring")
    
    def get_status(self) -> Dict[str, Any]:
        """Get scanner status"""
        rfid_status = get_rfid_status()
        return {
            'available': True,
            'initialized': True,
            'monitoring': self.monitoring,
            'history_count': len(self.card_history),
            'rfid_status': rfid_status
        }
    
    def get_card_history(self) -> List[Dict[str, Any]]:
        """Get card detection history"""
        return self.card_history.copy()
    
    def clear_history(self):
        """Clear card detection history"""
        self.card_history.clear()
        logger.info("Card history cleared")


def card_detected_callback(card_data):
    """Default callback function for card detection"""
    print(f"\n🎯 CARD DETECTED!")
    print(f"   UID: {card_data.get('uid', 'N/A')}")
    print(f"   Type: {card_data.get('card_type', 'N/A')}")
    print(f"   Time: {card_data.get('datetime', 'N/A')}")
    print(f"   Reader: {card_data.get('reader_name', 'N/A')}")
    print("-" * 50)


def main():
    """Main function for standalone RFID scanner"""
    import sys
    
    # Check for debug flag
    debug_mode = '--debug' in sys.argv or '-d' in sys.argv
    
    # Setup logging with debug mode
    global logger
    logger = setup_logging(debug=debug_mode)
    
    print("=" * 60)
    print("🔍 ACR122U RFID Scanner - Standalone")
    if debug_mode:
        print("🐛 Debug mode enabled")
    print("=" * 60)
    
    # Initialize scanner
    scanner = ACR122UScanner()
    
    if not scanner.initialize():
        print("❌ Failed to initialize RFID scanner")
        print("Make sure:")
        print("  1. ACR122U reader is connected")
        print("  2. Required drivers are installed")
        print("  3. RFID library is installed: pip install pyscard")
        return
    
    print("✅ RFID scanner initialized successfully")
    
    try:
        print("\n" + "=" * 60)
        print("🎮 RFID Scanner Commands:")
        print("=" * 60)
        print("1. 'scan'     - Start continuous scanning")
        print("2. 'stop'     - Stop scanning")
        print("3. 'read'     - Read card once")
        print("4. 'block'    - Read specific Mifare block")
        print("5. 'write'    - Write data to card")
        print("6. 'status'   - Show scanner status")
        print("7. 'history'  - Show card detection history")
        print("8. 'clear'    - Clear history")
        print("9. 'quit'     - Exit scanner")
        print("=" * 60)
        
        # Subscribe to RFID events for monitoring
        event_queue = subscribe_to_rfid_events()
        
        while True:
            try:
                command = input("\n🔧 Enter command: ").strip().lower()
                
                if command == 'quit' or command == 'exit':
                    break
                elif command == 'scan':
                    print("🔄 Starting continuous scanning...")
                    scanner.start_monitoring()
                    
                    # Simple monitoring loop
                    print("📡 Monitoring for cards... (Press Ctrl+C to stop)")
                    try:
                        while True:
                            try:
                                event = event_queue.get(timeout=1.0)
                                if event['type'] == 'uid':
                                    card_data = {
                                        'uid': event['uid'],
                                        'card_type': 'Mifare Classic 1K',
                                        'datetime': datetime.fromtimestamp(event['timestamp']).strftime('%Y-%m-%d %H:%M:%S'),
                                        'reader_name': 'ACR122U'
                                    }
                                    card_detected_callback(card_data)
                            except:
                                # Timeout or empty queue, continue
                                pass
                    except KeyboardInterrupt:
                        print("\n⏹️ Stopping scan...")
                    
                    scanner.stop_monitoring()
                    
                elif command == 'stop':
                    print("⏹️ Stopping scanning...")
                    scanner.stop_monitoring()
                elif command == 'read':
                    print("📖 Reading card...")
                    card_data = scanner.read_card()
                    if card_data:
                        card_detected_callback(card_data)
                    else:
                        print("❌ No card detected")
                elif command == 'block':
                    try:
                        block_num = int(input("📖 Enter block number (0-63): ").strip())
                        if 0 <= block_num <= 63:
                            print(f"📖 Reading block {block_num}...")
                            block_data = scanner.read_block(block_num)
                            if block_data:
                                print(f"✅ Block {block_num} data: {block_data}")
                            else:
                                print(f"❌ Failed to read block {block_num}")
                        else:
                            print("❌ Block number must be between 0 and 63")
                    except ValueError:
                        print("❌ Please enter a valid block number")
                elif command == 'write':
                    data = input("📝 Enter data to write (max 16 chars): ").strip()
                    if data:
                        try:
                            block_num = int(input("📝 Enter block number (4-63): ").strip())
                            if 4 <= block_num <= 63:
                                print(f"✍️ Writing data to block {block_num}...")
                                success = scanner.write_block(block_num, data)
                                if success:
                                    print("✅ Data written successfully!")
                                else:
                                    print("❌ Failed to write data")
                            else:
                                print("❌ Block number must be between 4 and 63")
                        except ValueError:
                            print("❌ Please enter a valid block number")
                    else:
                        print("❌ No data provided")
                elif command == 'status':
                    status = scanner.get_status()
                    print("\n📊 Scanner Status:")
                    print(f"   Available: {status['available']}")
                    print(f"   Initialized: {status['initialized']}")
                    print(f"   Monitoring: {status['monitoring']}")
                    print(f"   History Count: {status['history_count']}")
                    rfid_status = status['rfid_status']
                    print(f"   Card Present: {rfid_status['present']}")
                    print(f"   Last UID: {rfid_status['last_uid']}")
                elif command == 'history':
                    history = scanner.get_card_history()
                    if history:
                        print(f"\n📚 Card History ({len(history)} entries):")
                        for i, card in enumerate(history[-10:], 1):  # Show last 10
                            print(f"   {i}. {card['datetime']} - {card['uid']} ({card['card_type']})")
                    else:
                        print("📚 No card history available")
                elif command == 'clear':
                    scanner.clear_history()
                    print("🗑️ History cleared")
                else:
                    print("❌ Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                print("\n\n⚠️ Interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    finally:
        print("\n🧹 Cleaning up...")
        scanner.stop_monitoring()
        unsubscribe_from_rfid_events(event_queue)
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()