"""
Database synchronization script for syncing between local and Aiven databases.

Usage:
    python scripts/sync_database.py --direction local-to-aiven [--data] [--tables table1,table2]
    python scripts/sync_database.py --direction aiven-to-local [--data] [--tables table1,table2]
    python scripts/sync_database.py --direction local-to-aiven --schema-only
    python scripts/sync_database.py --direction aiven-to-local --schema-only
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import mysql.connector


def get_local_connection():
    """Get connection to local database. Uses LOCAL_DB_* env vars or defaults."""
    return mysql.connector.connect(
        host=os.getenv('LOCAL_DB_HOST', 'localhost'),
        port=int(os.getenv('LOCAL_DB_PORT', '3306')),
        user=os.getenv('LOCAL_DB_USER', 'root'),
        password=os.getenv('LOCAL_DB_PASSWORD', 'root'),
        database=os.getenv('LOCAL_DB_NAME', 'dress')
    )


def get_aiven_connection():
    """Get connection to Aiven database using environment variables."""
    host = os.getenv('DB_HOST')
    port = int(os.getenv('DB_PORT', '3306'))
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWORD')
    database = os.getenv('DB_NAME', 'dress')
    
    if not all([host, user, password]):
        raise ValueError("Aiven connection requires DB_HOST, DB_USER, and DB_PASSWORD in .env file")
    
    is_aiven = 'aivencloud.com' in host.lower()
    ssl_disabled = os.getenv('DB_SSL_DISABLED', 'false').lower() in {'1', 'true', 'yes', 'on'}
    ssl_required = os.getenv('DB_SSL_REQUIRED', 'true' if is_aiven else 'false').lower() in {'1', 'true', 'yes', 'on'}
    ssl_ca = os.getenv('DB_SSL_CA', 'certs/ca.pem' if is_aiven else None)
    
    connection_params = {
        'host': host,
        'port': port,
        'user': user,
        'password': password,
        'database': database
    }
    
    if not ssl_disabled and (ssl_required or ssl_ca):
        if ssl_ca and os.path.exists(ssl_ca):
            connection_params['ssl_ca'] = ssl_ca
        if not ssl_disabled:
            connection_params['ssl_disabled'] = False
    
    return mysql.connector.connect(**connection_params)


def get_source_connection(direction):
    """Get connection to source database based on direction."""
    if direction == 'local-to-aiven':
        return get_local_connection()
    else:
        return get_aiven_connection()


def get_dest_connection(direction):
    """Get connection to destination database based on direction."""
    if direction == 'local-to-aiven':
        return get_aiven_connection()
    else:
        return get_local_connection()


def get_source_info(direction):
    """Get source database connection info for display."""
    if direction == 'local-to-aiven':
        return {
            'host': os.getenv('LOCAL_DB_HOST', 'localhost'),
            'port': os.getenv('LOCAL_DB_PORT', '3306'),
            'name': 'Local'
        }
    else:
        return {
            'host': os.getenv('DB_HOST', 'unknown'),
            'port': os.getenv('DB_PORT', 'unknown'),
            'name': 'Aiven'
        }


def get_dest_info(direction):
    """Get destination database connection info for display."""
    if direction == 'local-to-aiven':
        return {
            'host': os.getenv('DB_HOST', 'unknown'),
            'port': os.getenv('DB_PORT', 'unknown'),
            'name': 'Aiven'
        }
    else:
        return {
            'host': os.getenv('LOCAL_DB_HOST', 'localhost'),
            'port': os.getenv('LOCAL_DB_PORT', '3306'),
            'name': 'Local'
        }


def sync_schema(source_conn, dest_conn, direction):
    """Sync database schema (structure only) from source to destination."""
    source_info = get_source_info(direction)
    dest_info = get_dest_info(direction)
    print(f"\n📋 Syncing schema from {source_info['name']} to {dest_info['name']}...")
    
    source_cursor = source_conn.cursor()
    dest_cursor = dest_conn.cursor()
    
    try:
        # Get all tables from source
        source_cursor.execute("SHOW TABLES")
        tables = [row[0] for row in source_cursor.fetchall()]
        
        print(f"Found {len(tables)} tables: {', '.join(tables)}")
        
        # Determine correct order based on foreign key dependencies
        # For dropping: child tables first, then parent tables (reverse order)
        # For creating: parent tables first, then child tables (normal order)
        try:
            ordered_tables = get_table_dependency_order(source_cursor, tables)
            print(f"Syncing tables in dependency order:")
            print(f"  Order: {' → '.join(ordered_tables)}")
        except Exception as e:
            print(f"⚠️  Could not determine dependency order: {e}")
            print(f"Using original order: {', '.join(tables)}")
            ordered_tables = tables
        
        # Disable foreign key checks for dropping tables
        dest_cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
        
        # Step 1: Drop all tables in reverse dependency order (children first)
        print(f"\n  Dropping existing tables...")
        for table in reversed(ordered_tables):
            dest_cursor.execute(f"DROP TABLE IF EXISTS `{table}`")
        
        # Step 2: Create tables in dependency order (parents first)
        print(f"  Creating tables...")
        for table in ordered_tables:
            print(f"  → Syncing table: {table}")
            
            # Get CREATE TABLE statement
            source_cursor.execute(f"SHOW CREATE TABLE `{table}`")
            create_stmt = source_cursor.fetchone()[1]
            
            # Normalize CREATE statement: replace double quotes with backticks for MySQL compatibility
            # This handles cases where source database uses ANSI_QUOTES mode or was exported from other DBs
            # MySQL uses backticks for identifiers, not double quotes
            # Note: CREATE TABLE statements typically don't contain string literals with double quotes,
            # so replacing all double quotes with backticks is safe here
            normalized_stmt = create_stmt.replace('"', '`')
            
            # Create table in destination
            dest_cursor.execute(normalized_stmt)
            dest_conn.commit()
            
            print(f"    ✓ Table '{table}' synced")
        
        # Re-enable foreign key checks
        dest_cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
        dest_conn.commit()
        
        print(f"\n✅ Schema sync completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error syncing schema: {e}")
        # Re-enable foreign key checks even on error
        try:
            dest_cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
            dest_conn.commit()
        except:
            pass
        dest_conn.rollback()
        return False
    finally:
        source_cursor.close()
        dest_cursor.close()


def get_table_dependency_order(cursor, tables):
    """Determine the correct order to sync tables based on foreign key relationships.
    Returns tables in order: parent tables first, then child tables."""
    if not tables:
        return []
    
    # Get foreign key relationships
    # Build safe IN clause with escaped table names
    table_list = ','.join([f"'{t.replace(chr(39), chr(39)*2)}'" for t in tables])
    query = f"""
        SELECT 
            TABLE_NAME,
            REFERENCED_TABLE_NAME
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE TABLE_SCHEMA = DATABASE()
        AND REFERENCED_TABLE_NAME IS NOT NULL
        AND TABLE_NAME IN ({table_list})
    """
    
    cursor.execute(query)
    
    fk_relations = {}
    for row in cursor.fetchall():
        child = row[0]
        parent = row[1]
        if child not in fk_relations:
            fk_relations[child] = []
        fk_relations[child].append(parent)
    
    # Topological sort: parent tables first
    ordered = []
    remaining = set(tables)
    
    while remaining:
        # Find tables with no dependencies (or all dependencies already processed)
        ready = [t for t in remaining if t not in fk_relations or 
                 all(p in ordered for p in fk_relations.get(t, []))]
        
        if not ready:
            # Circular dependency or missing table - just add remaining tables
            ordered.extend(remaining)
            break
        
        ordered.extend(ready)
        remaining -= set(ready)
    
    return ordered


def sync_data(source_conn, dest_conn, tables=None, direction=None):
    """Sync data from source to destination. If tables is None, sync all tables."""
    if direction:
        source_info = get_source_info(direction)
        dest_info = get_dest_info(direction)
        print(f"\n📊 Syncing data from {source_info['name']} to {dest_info['name']}...")
    else:
        print(f"\n📊 Syncing data...")
    
    source_cursor = source_conn.cursor()
    dest_cursor = dest_conn.cursor()
    
    try:
        # Get tables to sync
        if tables:
            tables_to_sync = [t.strip() for t in tables.split(',')]
        else:
            source_cursor.execute("SHOW TABLES")
            tables_to_sync = [row[0] for row in source_cursor.fetchall()]
        
        # Determine correct order based on foreign key dependencies
        # Parent tables first, then child tables
        try:
            ordered_tables = get_table_dependency_order(dest_cursor, tables_to_sync)
            print(f"Syncing data for {len(ordered_tables)} tables in dependency order:")
            print(f"  Order: {' → '.join(ordered_tables)}")
        except Exception as e:
            print(f"⚠ Could not determine dependency order: {e}")
            print(f"Using original order: {', '.join(tables_to_sync)}")
            ordered_tables = tables_to_sync
        
        # Disable foreign key checks temporarily to allow truncating in any order
        dest_cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
        
        for table in ordered_tables:
            print(f"  → Syncing data for table: {table}")
            
            # Check if table exists in destination
            dest_cursor.execute(f"SHOW TABLES LIKE '{table}'")
            if not dest_cursor.fetchone():
                print(f"    ⚠ Table '{table}' does not exist in destination, skipping...")
                continue
            
            # Get all data from source
            source_cursor.execute(f"SELECT * FROM `{table}`")
            rows = source_cursor.fetchall()
            
            if not rows:
                print(f"    ℹ Table '{table}' is empty in source")
                # Still truncate to clear destination table
                dest_cursor.execute(f"TRUNCATE TABLE `{table}`")
                dest_conn.commit()
                continue
            
            # Get column names
            source_cursor.execute(f"DESCRIBE `{table}`")
            columns = [col[0] for col in source_cursor.fetchall()]
            
            # Clear destination table (now safe with FK checks disabled)
            dest_cursor.execute(f"TRUNCATE TABLE `{table}`")
            
            # Insert data
            placeholders = ', '.join(['%s'] * len(columns))
            columns_str = ', '.join([f"`{col}`" for col in columns])
            insert_query = f"INSERT INTO `{table}` ({columns_str}) VALUES ({placeholders})"
            
            dest_cursor.executemany(insert_query, rows)
            dest_conn.commit()
            
            print(f"    ✓ Synced {len(rows)} rows for table '{table}'")
        
        # Re-enable foreign key checks
        dest_cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
        dest_conn.commit()
        
        print(f"\n✅ Data sync completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error syncing data: {e}")
        # Re-enable foreign key checks even on error
        try:
            dest_cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
            dest_conn.commit()
        except:
            pass
        dest_conn.rollback()
        return False
    finally:
        source_cursor.close()
        dest_cursor.close()


def get_available_tables(conn):
    """Get list of available tables from database."""
    cursor = conn.cursor()
    try:
        cursor.execute("SHOW TABLES")
        tables = [row[0] for row in cursor.fetchall()]
        return tables
    finally:
        cursor.close()


def main():
    print("=" * 60)
    print("DATABASE SYNC TOOL")
    print("=" * 60)
    print()
    
    # Step 1: Choose sync direction
    print("Choose sync direction:")
    print("  1. Local → Aiven (push local database to Aiven)")
    print("  2. Aiven → Local (pull Aiven database to local)")
    print()
    
    while True:
        direction_choice = input("Enter choice (1 or 2): ").strip()
        if direction_choice == '1':
            direction = 'local-to-aiven'
            break
        elif direction_choice == '2':
            direction = 'aiven-to-local'
            break
        else:
            print("❌ Invalid choice. Please enter 1 or 2.")
    
    # Step 2: Choose what to sync
    print()
    print("What would you like to sync?")
    print("  1. Schema only (table structure, no data)")
    print("  2. Schema + All Data (structure and all table data)")
    print("  3. Schema + Specific Tables (structure and selected table data)")
    print()
    
    while True:
        sync_choice = input("Enter choice (1, 2, or 3): ").strip()
        if sync_choice in ['1', '2', '3']:
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")
    
    schema_only = (sync_choice == '1')
    sync_data_flag = (sync_choice in ['2', '3'])
    tables = None
    
    # Step 3: If specific tables, let user choose
    if sync_choice == '3':
        print()
        print("Connecting to source database to get table list...")
        try:
            source_conn = get_source_connection(direction)
            available_tables = get_available_tables(source_conn)
            source_conn.close()
            
            print()
            print("Available tables:")
            for i, table in enumerate(available_tables, 1):
                print(f"  {i}. {table}")
            print()
            
            while True:
                table_input = input("Enter table numbers (comma-separated, e.g., 1,2,3) or 'all': ").strip()
                
                if table_input.lower() == 'all':
                    tables = None
                    break
                
                try:
                    indices = [int(x.strip()) for x in table_input.split(',')]
                    selected_tables = [available_tables[i-1] for i in indices if 1 <= i <= len(available_tables)]
                    
                    if not selected_tables:
                        print("❌ No valid tables selected. Please try again.")
                        continue
                    
                    tables = ','.join(selected_tables)
                    print(f"Selected tables: {tables}")
                    break
                except (ValueError, IndexError):
                    print("❌ Invalid input. Please enter numbers separated by commas.")
        except Exception as e:
            print(f"❌ Error getting table list: {e}")
            print("Will sync all tables instead.")
            tables = None
    
    # Display sync information
    source_info = get_source_info(direction)
    dest_info = get_dest_info(direction)
    
    print()
    print("=" * 60)
    print("SYNC SUMMARY")
    print("=" * 60)
    print(f"Source: {source_info['name']} ({source_info['host']}:{source_info['port']})")
    print(f"Destination: {dest_info['name']} ({dest_info['host']}:{dest_info['port']})")
    print(f"Mode: {'Schema only' if schema_only else 'Schema + Data'}")
    if tables:
        print(f"Tables: {tables}")
    elif sync_data_flag:
        print("Tables: All tables")
    print("=" * 60)
    
    # Confirm before proceeding
    print()
    response = input("⚠️  This will modify the destination database. Continue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("❌ Sync cancelled")
        sys.exit(0)
    
    # Get connections
    try:
        print("\n🔌 Connecting to databases...")
        source_conn = get_source_connection(direction)
        dest_conn = get_dest_connection(direction)
        
        # Verify connections are correct by checking host info
        source_info = get_source_info(direction)
        dest_info = get_dest_info(direction)
        
        # Double-check connections by querying database names
        source_cursor = source_conn.cursor()
        dest_cursor = dest_conn.cursor()
        source_cursor.execute("SELECT DATABASE()")
        source_db = source_cursor.fetchone()[0]
        dest_cursor.execute("SELECT DATABASE()")
        dest_db = dest_cursor.fetchone()[0]
        source_cursor.close()
        dest_cursor.close()
        
        print(f"✅ Connected successfully")
        print(f"📤 Source: {source_info['name']} ({source_info['host']}:{source_info['port']}) - Database: {source_db}")
        print(f"📥 Destination: {dest_info['name']} ({dest_info['host']}:{dest_info['port']}) - Database: {dest_db}")
        print(f"🔄 Sync Direction: {source_info['name']} → {dest_info['name']}")
        
        # Warn if auto-sync might interfere
        if direction == 'aiven-to-local':
            print(f"\n⚠️  WARNING: Auto-sync runs every 5 minutes and syncs Local → Aiven.")
            print(f"   If auto-sync runs after this sync, it will overwrite Aiven with Local data.")
            print(f"   Consider temporarily disabling auto-sync if you want to keep Aiven data.")
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        sys.exit(1)
    
    try:
        # Always sync schema first
        print(f"\n{'='*60}")
        print(f"Starting schema sync: {source_info['name']} → {dest_info['name']}")
        print(f"{'='*60}")
        if not sync_schema(source_conn, dest_conn, direction):
            print("\n❌ Schema sync failed. Aborting.")
            sys.exit(1)
        
        # Sync data if requested
        if sync_data_flag:
            print(f"\n{'='*60}")
            print(f"Starting data sync: {source_info['name']} → {dest_info['name']}")
            print(f"{'='*60}")
            if not sync_data(source_conn, dest_conn, tables, direction):
                print("\n❌ Data sync failed.")
                sys.exit(1)
        
        print("\n" + "=" * 60)
        print("✅ SYNC COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n❌ Sync interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
    finally:
        source_conn.close()
        dest_conn.close()
        print("\n🔒 Database connections closed")


if __name__ == "__main__":
    main()

