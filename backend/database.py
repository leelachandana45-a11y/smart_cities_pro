import sqlite3
from datetime import datetime
import json
from typing import Optional, Dict

DB_NAME = "road_safety.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Create main road_data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS road_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            latitude REAL,
            longitude REAL,
            friction REAL,
            risk_level TEXT,
            timestamp TEXT,
            weather_used BOOLEAN DEFAULT 0,
            weather_data TEXT
        )
    """)
    
    # Create weather_history table for optional detailed weather tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weather_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            road_data_id INTEGER,
            temperature REAL,
            humidity REAL,
            rainfall REAL,
            wind_speed REAL,
            timestamp TEXT,
            FOREIGN KEY(road_data_id) REFERENCES road_data(id)
        )
    """)

    conn.commit()
    conn.close()


def insert_record(
    latitude: float, 
    longitude: float, 
    friction: float, 
    risk_level: str,
    weather_data: Optional[Dict] = None,
    weather_used: bool = False
):
    """
    Insert a road condition record with optional weather data.
    
    Args:
        latitude: GPS latitude
        longitude: GPS longitude
        friction: Friction coefficient
        risk_level: Risk level (HIGH, MEDIUM, LOW)
        weather_data: Dict with temperature, humidity, rainfall, wind_speed
        weather_used: Whether weather data was used in prediction
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Convert weather_data dict to JSON string
    weather_json = None
    if weather_data:
        weather_json = json.dumps(weather_data)

    cursor.execute("""
        INSERT INTO road_data (latitude, longitude, friction, risk_level, timestamp, weather_used, weather_data)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (latitude, longitude, friction, risk_level, datetime.now().isoformat(), weather_used, weather_json))

    road_data_id = cursor.lastrowid

    # Insert detailed weather data if available
    if weather_data:
        cursor.execute("""
            INSERT INTO weather_history (road_data_id, temperature, humidity, rainfall, wind_speed, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            road_data_id,
            weather_data.get("temperature", 0),
            weather_data.get("humidity", 0),
            weather_data.get("rainfall", 0),
            weather_data.get("wind_speed", 0),
            datetime.now().isoformat()
        ))

    conn.commit()
    conn.close()


def get_all_records():
    """Fetch all recorded road condition measurements"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT latitude, longitude, friction, risk_level, timestamp, weather_used, weather_data 
        FROM road_data
        ORDER BY timestamp DESC
    """)
    rows = cursor.fetchall()

    conn.close()
    return rows


def get_records_with_weather():
    """Fetch only records that include weather data"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT latitude, longitude, friction, risk_level, timestamp, weather_data
        FROM road_data
        WHERE weather_used = 1 AND weather_data IS NOT NULL
        ORDER BY timestamp DESC
    """)
    rows = cursor.fetchall()

    conn.close()
    return rows


def get_records_by_risk_level(risk_level: str):
    """Fetch records filtered by risk level"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT latitude, longitude, friction, risk_level, timestamp, weather_used
        FROM road_data
        WHERE risk_level = ?
        ORDER BY timestamp DESC
    """, (risk_level,))
    rows = cursor.fetchall()

    conn.close()
    return rows


def get_weather_statistics():
    """Get aggregated weather statistics from stored data"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT 
            AVG(temperature) as avg_temp,
            AVG(humidity) as avg_humidity,
            AVG(rainfall) as avg_rainfall,
            AVG(wind_speed) as avg_wind,
            MAX(temperature) as max_temp,
            MIN(temperature) as min_temp
        FROM weather_history
    """)
    stats = cursor.fetchone()

    conn.close()
    return {
        "avg_temperature": stats[0],
        "avg_humidity": stats[1],
        "avg_rainfall": stats[2],
        "avg_wind_speed": stats[3],
        "max_temperature": stats[4],
        "min_temperature": stats[5]
    } if stats[0] is not None else {}
