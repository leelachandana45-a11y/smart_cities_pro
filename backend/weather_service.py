import requests
import os
from typing import Dict, Optional

# OpenWeather API Configuration
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "YOUR_API_KEY_HERE")
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"


def get_weather(city: str, state: Optional[str] = None, country: Optional[str] = None) -> Dict:
    """
    Fetch real-time weather data from OpenWeather API.
    
    Args:
        city: City name
        state: State/Province (optional)
        country: Country code (optional)
    
    Returns:
        Dictionary with weather features:
        {
            'temperature': float (Celsius),
            'humidity': float (percentage),
            'rainfall': float (mm, 0 if no rain),
            'wind_speed': float (m/s),
            'success': bool
        }
    """
    try:
        # Build location string
        location = city
        if state:
            location = f"{city},{state}"
        if country:
            location = f"{location},{country}"
        
        params = {
            "q": location,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric"  # Get temperature in Celsius
        }
        
        response = requests.get(OPENWEATHER_BASE_URL, params=params, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract weather data
        main = data.get("main", {})
        wind = data.get("wind", {})
        rain = data.get("rain", {})
        
        weather_data = {
            "temperature": float(main.get("temp", 0.0)),
            "humidity": float(main.get("humidity", 0.0)),
            "rainfall": float(rain.get("1h", 0.0)),  # Rainfall in last hour
            "wind_speed": float(wind.get("speed", 0.0)),
            "success": True
        }
        
        return weather_data
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return {
            "temperature": 0.0,
            "humidity": 0.0,
            "rainfall": 0.0,
            "wind_speed": 0.0,
            "success": False
        }


def get_weather_by_coordinates(latitude: float, longitude: float) -> Dict:
    """
    Fetch real-time weather data using GPS coordinates.
    
    Args:
        latitude: GPS latitude
        longitude: GPS longitude
    
    Returns:
        Dictionary with weather features (same format as get_weather)
    """
    try:
        params = {
            "lat": latitude,
            "lon": longitude,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric"
        }
        
        response = requests.get(OPENWEATHER_BASE_URL, params=params, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        
        main = data.get("main", {})
        wind = data.get("wind", {})
        rain = data.get("rain", {})
        
        weather_data = {
            "temperature": float(main.get("temp", 0.0)),
            "humidity": float(main.get("humidity", 0.0)),
            "rainfall": float(rain.get("1h", 0.0)),
            "wind_speed": float(wind.get("speed", 0.0)),
            "success": True
        }
        
        return weather_data
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return {
            "temperature": 0.0,
            "humidity": 0.0,
            "rainfall": 0.0,
            "wind_speed": 0.0,
            "success": False
        }


def normalize_weather_features(temperature: float, humidity: float, rainfall: float, wind_speed: float) -> list:
    """
    Normalize weather features to [0, 1] range for neural network input.
    
    Args:
        temperature: Temperature in Celsius
        humidity: Humidity percentage (0-100)
        rainfall: Rainfall in mm
        wind_speed: Wind speed in m/s
    
    Returns:
        List of normalized features
    """
    # Normalize temperature to [-40, 50] -> [0, 1]
    norm_temp = (temperature + 40) / 90
    norm_temp = max(0.0, min(1.0, norm_temp))  # Clamp to [0, 1]
    
    # Normalize humidity (already percentage 0-100)
    norm_humidity = humidity / 100.0
    norm_humidity = max(0.0, min(1.0, norm_humidity))
    
    # Normalize rainfall (0-100mm range)
    norm_rainfall = rainfall / 100.0
    norm_rainfall = max(0.0, min(1.0, norm_rainfall))
    
    # Normalize wind speed (0-30 m/s range)
    norm_wind = wind_speed / 30.0
    norm_wind = max(0.0, min(1.0, norm_wind))
    
    return [norm_temp, norm_humidity, norm_rainfall, norm_wind]
