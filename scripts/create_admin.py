import mysql.connector
from mysql.connector import Error
from werkzeug.security import generate_password_hash
import getpass
import re

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

    # MySQL connection setup
    host = input("MySQL Host (default: localhost): ") or "localhost"
    user = input("MySQL Username (default: root): ") or "root"
    password_db = getpass.getpass("MySQL Password (default: root): ") or "root"
    database = input("Database name (default: dress): ") or "dress"

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
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password_db,
            database=database
        )

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
