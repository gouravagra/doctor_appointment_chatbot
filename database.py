# database.py
import mysql.connector
import bcrypt
import streamlit as st
from datetime import datetime, time, timedelta
import os 
from dotenv import load_dotenv
load_dotenv()

db_host = os.getenv("DB_HOST")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_name = os.getenv("DB_NAME")
db_port = os.getenv("DB_PORT")

# Database connection setup
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name,
            port=db_port
        )
        return connection
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        st.error(f"Error: {err}")
        return None

# Create tables if they don't exist
def create_tables():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(255) NOT NULL UNIQUE,
        password VARCHAR(255) NOT NULL
    );
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chat_history (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(255) NOT NULL,
        session_id VARCHAR(255) NOT NULL,
        user_message TEXT,
        ai_message TEXT,
        message_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (username) REFERENCES users(username)
    );
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS appointment_slots (
        id INT AUTO_INCREMENT PRIMARY KEY,
        appointment_date DATE NOT NULL,
        slot_time TIME NOT NULL,
        is_booked BOOLEAN DEFAULT FALSE,
        booked_by VARCHAR(255),
        FOREIGN KEY (booked_by) REFERENCES users(username)
    );
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_feedback (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(255) NOT NULL,
        feedback TEXT,
        feedback_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (username) REFERENCES users(username)
    );
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_personal_information (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(255) NOT NULL,
        name VARCHAR(255),
        birth_date DATE,
        reason_for_appointment TEXT,
        FOREIGN KEY (username) REFERENCES users(username)
    );
    ''')

    conn.commit()
    cursor.close()
    conn.close()

create_tables()

# Hashing password function
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# Verify password function
def verify_password(stored_password, provided_password):
    return bcrypt.checkpw(provided_password.encode(), stored_password.encode())

# User registration function
def register_user(username, password):
    if not username:
        st.warning("Username cannot be empty.")
        return

    hashed_password = hash_password(password)

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM users WHERE username=%s', (username,))
    existing_user = cursor.fetchone()

    if existing_user:
        st.warning("Username already exists. Please choose a different username.")
        return

    cursor.execute('INSERT INTO users (username, password) VALUES (%s, %s)', (username, hashed_password))
    conn.commit()
    st.success("You have successfully registered")

    cursor.close()
    conn.close()

# User login function
def login_user(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('SELECT password FROM users WHERE username=%s', (username,))
    result = cursor.fetchone()

    cursor.close()
    conn.close()

    if result and verify_password(result[0], password):
        return True
    return False

def insert_personal_information(username, name, birth_date, reason_for_appointment):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute('''
        INSERT INTO user_personal_information (username, name, birth_date, reason_for_appointment)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        name=%s, birth_date=%s, reason_for_appointment=%s
        ''', (username, name, birth_date, reason_for_appointment, name, birth_date, reason_for_appointment))
        conn.commit()
    except mysql.connector.Error as err:
        st.error(f"Database error: {err}")
    finally:
        cursor.close()
        conn.close()

# Fetch user's chat sessions
def fetch_user_chat_sessions(username):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('SELECT DISTINCT session_id FROM chat_history WHERE username=%s', (username,))
    sessions = [row[0] for row in cursor.fetchall()]

    cursor.close()
    conn.close()

    return sessions

# Fetch chat history for a session
def fetch_chat_history(username, session_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('SELECT user_message, ai_message FROM chat_history WHERE username=%s AND session_id=%s ORDER BY message_time', (username, session_id))
    history = cursor.fetchall()

    cursor.close()
    conn.close()

    return history

# Save chat history to database
def save_chat_history(username, session_id, user_message, ai_message):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute('INSERT INTO chat_history (username, session_id, user_message, ai_message) VALUES (%s, %s, %s, %s)', (username, session_id, user_message, ai_message))
        conn.commit()
    except mysql.connector.Error as err:
        st.error(f"Database error: {err}")
    finally:
        cursor.close()
        conn.close()
def fetch_available_slots(date):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Clinic time settings
    start_time = time(9, 0)   # 09:00
    end_time = time(17, 0)    # 17:00
    interval = timedelta(minutes=30)

    # Define lunch break time
    lunch_start = time(13, 0)  # 13:00
    lunch_end = time(14, 0)    # 14:00

    available_slots = []
    current_time = datetime.combine(date, start_time)

    # Query to fetch all booked slots for the given date
    cursor.execute('''
        SELECT slot_time FROM appointment_slots 
        WHERE appointment_date=%s
    ''', (date,))
    booked_slots = set(slot[0] for slot in cursor.fetchall())

    while current_time.time() < end_time:
        slot_time = current_time.time()

        # Skip lunch break slots
        if lunch_start <= slot_time < lunch_end:
            current_time += interval
            continue

        # Check if the slot is not booked
        if slot_time not in booked_slots:
            available_slots.append(slot_time)
        
        current_time += interval

    cursor.close()
    conn.close()
    return available_slots

def book_appointment(username, appointment_date, slot_time_str):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Convert slot_time_str to a time object
    slot_time = datetime.strptime(slot_time_str, "%H:%M").time()

    # Define lunch break time
    lunch_start = time(13, 0)  # 13:00
    lunch_end = time(14, 0)    # 14:00

    try:
        # Check if the slot is within the lunch break
        if lunch_start <= slot_time < lunch_end:
            return "No appointments can be booked during lunch time from 13:00 to 14:00."

        # Check if the slot is already booked
        cursor.execute('''
            SELECT is_booked 
            FROM appointment_slots 
            WHERE appointment_date=%s AND slot_time=%s
        ''', (appointment_date, slot_time))
        result = cursor.fetchone()
        if result and result[0]:  # If the slot is already booked
            return "Slot already booked."

        # Insert a new record for the booked slot if not already booked
        cursor.execute('''
            INSERT INTO appointment_slots (appointment_date, slot_time, is_booked, booked_by)
            VALUES (%s, %s, True, %s)
        ''', (appointment_date, slot_time, username))
        conn.commit()

        return f"appointment booked successfully"

    except mysql.connector.Error as err:
        conn.rollback()  # Rollback in case of error
        return f"Database error: {err}"
    
    finally:
        cursor.close()
        conn.close()

def collect_feedback(username):
    feedback = st.text_area("Please provide your feedback before logging out:", key="feedback_input")
    if st.button("Submit Feedback"):
        if feedback.strip():
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('INSERT INTO user_feedback (username, feedback) VALUES (%s, %s)', (username, feedback))
            conn.commit()
            cursor.close()
            conn.close()
            st.success("Thank you for your feedback!")
            st.session_state["feedback_collected"] = True
        else:
            st.warning("Feedback cannot be empty.")
