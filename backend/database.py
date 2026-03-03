import sqlite3
from datetime import datetime

DB_NAME = "road_safety.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS road_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            latitude REAL,
            longitude REAL,
            friction REAL,
            risk_level TEXT,
            timestamp TEXT
        )
    """)

    conn.commit()
    conn.close()


def insert_record(latitude, longitude, friction, risk_level):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO road_data (latitude, longitude, friction, risk_level, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (latitude, longitude, friction, risk_level, datetime.now().isoformat()))

    conn.commit()
    conn.close()

def get_all_records():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT latitude, longitude, friction, risk_level FROM road_data")
    rows = cursor.fetchall()

    conn.close()
    return rows
