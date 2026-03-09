from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import Optional

from .inference_engine import predict_friction, predict_friction_image_only
from .database import init_db, insert_record, get_all_records

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database when app starts
init_db()


@app.get("/")
def home():
    return {"message": "API is running"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    city: Optional[str] = Form(None),
    use_weather: bool = Form(False)
):
    """
    Predict road friction coefficient from image.
    
    Args:
        file: Road image file
        latitude: GPS latitude for weather lookup (optional)
        longitude: GPS longitude for weather lookup (optional)
        city: City name for weather lookup (optional)
        use_weather: Whether to include weather data in prediction (default: False)
    
    Returns:
        JSON with friction prediction and optional weather data
    """
    image_bytes = await file.read()
    
    # Get prediction
    prediction = predict_friction(
        image_bytes,
        use_weather=use_weather,
        city=city,
        latitude=latitude,
        longitude=longitude
    )
    
    friction = prediction["friction"]
    
    # Risk logic based on friction coefficient
    if friction < 0.3:
        risk = "HIGH"
    elif friction < 0.6:
        risk = "MEDIUM"
    else:
        risk = "LOW"
    
    # Save to database if GPS provided
    if latitude is not None and longitude is not None:
        insert_record(
            latitude, 
            longitude, 
            friction, 
            risk,
            weather_data=prediction.get("weather_data"),
            weather_used=prediction.get("weather_used", False)
        )
    
    # Build response
    response = {
        "friction": friction,
        "risk_level": risk,
        "model_type": "weather-aware" if prediction.get("weather_used") else "image-only"
    }
    
    # Include weather data if available
    if prediction.get("weather_data"):
        response["weather"] = prediction["weather_data"]
    
    if "error" in prediction:
        response["note"] = prediction["error"]
    
    return response


@app.post("/predict-weather-aware")
async def predict_weather_aware(
    file: UploadFile = File(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    city: Optional[str] = Form(None)
):
    """
    Predict road friction coefficient using both image and weather data.
    Convenience endpoint that always uses weather data.
    
    Args:
        file: Road image file
        latitude: GPS latitude for weather lookup
        longitude: GPS longitude for weather lookup
        city: City name for weather lookup (fallback if coordinates not provided)
    
    Returns:
        JSON with friction prediction including weather contribution
    """
    image_bytes = await file.read()
    prediction = predict_friction(
        image_bytes,
        use_weather=True,
        city=city,
        latitude=latitude,
        longitude=longitude
    )
    
    friction = prediction["friction"]
    
    # Risk logic
    if friction < 0.3:
        risk = "HIGH"
    elif friction < 0.6:
        risk = "MEDIUM"
    else:
        risk = "LOW"
    
    # Save to database if GPS provided
    if latitude is not None and longitude is not None:
        insert_record(
            latitude,
            longitude,
            friction,
            risk,
            weather_data=prediction.get("weather_data"),
            weather_used=prediction.get("weather_used", False)
        )
    
    return {
        "friction": friction,
        "risk_level": risk,
        "weather": prediction.get("weather_data"),
        "model_type": "weather-aware",
        "note": prediction.get("error") if not prediction.get("weather_used") else None
    }


@app.post("/predict-image-only")
async def predict_image_only(
    file: UploadFile = File(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None)
):
    """
    Predict road friction coefficient using image data only.
    Uses the original CNN model without weather features.
    
    Args:
        file: Road image file
        latitude: GPS latitude (for database storage only)
        longitude: GPS longitude (for database storage only)
    
    Returns:
        JSON with friction prediction
    """
    image_bytes = await file.read()
    friction = predict_friction_image_only(image_bytes)
    
    # Risk logic
    if friction < 0.3:
        risk = "HIGH"
    elif friction < 0.6:
        risk = "MEDIUM"
    else:
        risk = "LOW"
    
    # Save to database if GPS provided
    if latitude is not None and longitude is not None:
        insert_record(latitude, longitude, friction, risk)
    
    return {
        "friction": float(friction),
        "risk_level": risk,
        "model_type": "image-only"
    }


@app.get("/all-data")
def fetch_data():
    """Fetch all recorded road condition measurements"""
    records = get_all_records()

    return [
        {
            "latitude": record[0],
            "longitude": record[1],
            "friction": record[2],
            "risk_level": record[3],
            "timestamp": record[4],
            "weather_used": record[5] if len(record) > 5 else False
        }
        for record in records
    ]


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "smart_cities_road_condition_api",
        "features": ["image_only_prediction", "weather_aware_prediction", "gps_tracking", "database_storage"]
    }