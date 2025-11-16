import mysql.connector
from mysql.connector import Error
from werkzeug.security import generate_password_hash
import getpass
import re
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, skip loading .env file
    pass

# Allowed roles and colleges from your schema
ALLOWED_ROLES = ['security', 'osas', 'dean', 'guidance']
ALLOWED_COLLEGES = [
    'College of Sciences',
    'College of Engineering',
    'College of Architecture and Design',
    'College of Arts and Humanities',
    'College of Business and Accountancy',
    'College of Criminal Justice Education',
    'College of Hospitality Management and Tourism',
    'College of Nursing and Health Sciences',
    'College of Teacher Education'
]

def create_admin():
    print("=== Create New Admin Account ===\n")

    # MySQL connection setup - use local database from .env file (primary)
    # All local database settings should be configured in .env file
    default_host = os.getenv('LOCAL_DB_HOST', 'localhost')
    default_port = os.getenv('LOCAL_DB_PORT', '3306')
    default_user = os.getenv('LOCAL_DB_USER')
    default_password = os.getenv('LOCAL_DB_PASSWORD')
    default_database = os.getenv('LOCAL_DB_NAME', 'dress')
    
    # Validate required settings
    if not default_user or not default_password:
        print("⚠ ERROR: LOCAL_DB_USER and LOCAL_DB_PASSWORD must be set in .env file.")
        print("Please configure your local database credentials in .env file first.")
        return
    
    host = input(f"MySQL Host (default: {default_host}): ") or default_host
    port_input = input(f"MySQL Port (default: {default_port}): ") or default_port
    try:
        port = int(port_input)
    except ValueError:
        print(f"Invalid port number, using default {default_port}")
        port = int(default_port)
    user = input(f"MySQL Username (default: {default_user}): ") or default_user
    
    # For password, show default only if it's not empty (but don't display the actual password)
    if default_password:
        password_prompt = "MySQL Password (default: [from .env]): "
    else:
        password_prompt = "MySQL Password: "
    password_db = getpass.getpass(password_prompt) or default_password
    database = input(f"Database name (default: {default_database}): ") or default_database

    # Admin details
    username = input("Enter new admin username: ").strip()
    while not username:
        username = input("Username cannot be empty. Enter new admin username: ").strip()

    # Email input with validation
    email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    while True:
        email = input("Enter email address: ").strip()
        if not email:
            print("❌ Email cannot be empty.")
            continue
        if len(email) > 45:
            print("❌ Email address is too long (max 45 characters).")
            continue
        if not email_pattern.match(email):
            print("❌ Invalid email format. Please enter a valid email address.")
            continue
        break

    # Password with confirmation
    while True:
        password = getpass.getpass("Enter password: ")
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            print("❌ Passwords do not match. Try again.")
        elif not password:
            print("❌ Password cannot be empty.")
        else:
            break

    # Role input
    print(f"\nAvailable roles: {', '.join(ALLOWED_ROLES)}")
    role = input("Enter role: ").strip().lower()
    while role not in ALLOWED_ROLES:
        print("❌ Invalid role. Choose from:", ALLOWED_ROLES)
        role = input("Enter role: ").strip().lower()

    # Require college if role is dean
    college = None
    if role == "dean":
        print("\nSelect a college for this dean:")
        for i, col in enumerate(ALLOWED_COLLEGES, 1):
            print(f" {i}. {col}")
        while True:
            try:
                choice = int(input("Enter number of college: "))
                if 1 <= choice <= len(ALLOWED_COLLEGES):
                    college = ALLOWED_COLLEGES[choice - 1]
                    break
                else:
                    print(f"❌ Invalid choice. Enter a number between 1 and {len(ALLOWED_COLLEGES)}")
            except ValueError:
                print("❌ Invalid input. Please enter a number.")

    # Generate hash using Werkzeug (PBKDF2-HMAC-SHA256)
    password_hash = generate_password_hash(password)

    try:
        print("\nConnecting to database...")
        # Check if using Aiven (requires SSL)
        is_aiven = 'aivencloud.com' in host.lower()
        connection_params = {
            'host': host,
            'port': port,
            'user': user,
            'password': password_db,
            'database': database
        }
        
        # Add SSL for Aiven connections (or if configured via environment variables)
        ssl_disabled = os.getenv('DB_SSL_DISABLED', 'false').lower() in {'1', 'true', 'yes', 'on'}
        ssl_required = os.getenv('DB_SSL_REQUIRED', 'true' if is_aiven else 'false').lower() in {'1', 'true', 'yes', 'on'}
        ssl_ca_env = os.getenv('DB_SSL_CA', None)
        
        if is_aiven or ssl_required:
            # Try environment variable first, then default location
            if ssl_ca_env and os.path.exists(ssl_ca_env):
                ssl_ca = ssl_ca_env
            else:
                ssl_ca = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'certs', 'ca.pem')
            
            if not ssl_disabled and ssl_ca and os.path.exists(ssl_ca):
                connection_params['ssl_ca'] = ssl_ca
            if not ssl_disabled:
                connection_params['ssl_disabled'] = False
        
        connection = mysql.connector.connect(**connection_params)

        if connection.is_connected():
            cursor = connection.cursor()
            insert_query = """
                INSERT INTO admins (username, password_hash, role, college, email)
                VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query, (username, password_hash, role, college, email))
            connection.commit()
            print(f"\n✅ Admin '{username}' created successfully with role '{role}'.")
            print(f"📧 Email: {email}")
            if college:
                print(f"🏫 Assigned to: {college}")

    except Error as e:
        print(f"\n❌ MySQL Error: {e}")

    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()
            print("🔒 Database connection closed.")

if __name__ == "__main__":
    create_admin()
