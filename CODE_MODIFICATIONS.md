# Code Modifications Summary

## Overview of Changes

This document provides a detailed breakdown of all code modifications made to integrate weather data into the smart_cities_pro road condition model.

---

## 1. NEW FILE: `/backend/weather_service.py`

**Purpose**: Fetches real-time weather data from OpenWeather API

### Key Functions:

#### `get_weather(city: str, state: str = None, country: str = None) -> Dict`
- Fetches weather by city name
- Returns: temperature, humidity, rainfall, wind_speed
- API endpoint: `https://api.openweathermap.org/data/2.5/weather`

```python
Example:
weather = get_weather("New York")
# Output: {"temperature": 15.2, "humidity": 65.0, "rainfall": 0.0, "wind_speed": 5.3, "success": True}
```

#### `get_weather_by_coordinates(latitude: float, longitude: float) -> Dict`
- Fetches weather using GPS coordinates
- Same output format as `get_weather()`

#### `normalize_weather_features(...) -> list`
- Normalizes raw weather values to [0, 1] range
- Temperature: [-40, 50]°C → [0, 1]
- Humidity: [0, 100]% → [0, 1]
- Rainfall: [0, 100]mm → [0, 1]
- Wind Speed: [0, 30]m/s → [0, 1]

---

## 2. MODIFIED FILE: `/main.py` (Training Script)

### Added Classes:

#### `SnowCSVDatasetWithWeather(Dataset)`
```python
class SnowCSVDatasetWithWeather(Dataset):
    def __init__(self, csv_file, root_dir, weather_file=None)
    
    # Methods:
    - _generate_synthetic_weather()  # Creates realistic weather patterns
    - _normalize_weather(...)        # Normalizes features
    - __getitem__(idx) -> {"image": tensor, "weather": tensor, "label": tensor}
```

#### `MultiHeadCNNWithWeather(nn.Module)`
```python
class MultiHeadCNNWithWeather(nn.Module):
    def __init__(self, weather_input_size=4)
    
    # Branches:
    - image_features: CNN (Conv2d layers)
    - weather_network: Dense (32 → 16 units)
    - combined_network: Fusion network (64 → 32 → 1)
    
    # Forward pass:
    - Takes image (batch_size, 3, H, W) and weather (batch_size, 4)
    - Returns friction prediction (batch_size, 1)
```

### Changes to Training Loop:

**Original (preserved)**:
```python
for epoch in range(epochs):
    for images, labels in train_loader:  # Single input branch
        outputs = model(images)           # Image-only
```

**Added (new)**:
```python
for epoch in range(epochs):
    for batch in train_weather_loader:    # Structured batch
        images = batch["image"]
        weather = batch["weather"]
        labels = batch["label"]
        
        outputs = weather_model(images, weather)  # Both inputs
```

### Output Changes:
- Saves TWO models: 
  - `snow_model_csv.pth` (original)
  - `snow_model_weather.pth` (new)
- Generates comparison plot: `loss_curve_comparison.pdf`

---

## 3. MODIFIED FILE: `/backend/model_loader.py`

### Class Changes:

#### Original `MultiHeadCNN` - PRESERVED
```python
class MultiHeadCNN(nn.Module):
    # No changes - maintains backward compatibility
```

#### New `MultiHeadCNNWithWeather`
```python
class MultiHeadCNNWithWeather(nn.Module):
    def __init__(self, weather_input_size=4):
        # Image CNN branch (128 features)
        self.image_features = nn.Sequential(...)
        
        # Weather dense branch (16 features)
        self.weather_network = nn.Sequential(...)
        
        # Fusion network (144 → 64 → 32 → 1)
        self.combined_network = nn.Sequential(...)
    
    def forward(self, image, weather):
        # Processes both inputs and concatenates
```

#### Updated `load_model()` Function
```python
# Before:
def load_model():
    model = MultiHeadCNN()
    model.load_state_dict(torch.load("snow_model_csv.pth", ...))
    return model

# After:
def load_model(use_weather=False):
    if use_weather:
        model = MultiHeadCNNWithWeather()
        model.load_state_dict(torch.load("snow_model_weather.pth", ...))
    else:
        model = MultiHeadCNN()
        model.load_state_dict(torch.load("snow_model_csv.pth", ...))
    model.eval()
    return model
```

---

## 4. MODIFIED FILE: `/backend/inference_engine.py`

### Function Signature Changes:

#### `predict_friction()` - Enhanced
```python
# Before:
def predict_friction(image_bytes):
    # Simple image input only
    return output.item()

# After:
def predict_friction(
    image_bytes, 
    use_weather=False, 
    city=None, 
    latitude=None, 
    longitude=None
):
    # Returns detailed dict with:
    # - friction: prediction
    # - weather_used: boolean
    # - weather_data: dict
    # - error: optional error message
```

### New Function:
```python
def predict_friction_image_only(image_bytes):
    """Legacy wrapper for backward compatibility"""
    result = predict_friction(image_bytes, use_weather=False)
    return result["friction"]
```

### Workflow:

**Without Weather** (use_weather=False):
```python
image_tensor = transform(image).unsqueeze(0)
output = model_image_only(image_tensor)
return {"friction": output.item(), "weather_used": False}
```

**With Weather** (use_weather=True):
```python
# Get weather data
if latitude and longitude:
    weather_dict = get_weather_by_coordinates(lat, lon)
else:
    weather_dict = get_weather(city)

# Normalize weather
weather_tensor = normalize_weather_features(...)

# Predict with both inputs
output = model_with_weather(image_tensor, weather_tensor)
return {
    "friction": output.item(),
    "weather_used": True,
    "weather_data": weather_dict
}
```

---

## 5. MODIFIED FILE: `/backend/snow_api.py`

### Endpoint Changes:

#### Enhanced POST `/predict`
```python
# Before:
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    latitude: float = Form(None),
    longitude: float = Form(None)
)

# After:
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    city: Optional[str] = Form(None),
    use_weather: bool = Form(False)  # NEW PARAMETER
)
```

**New Response Format**:
```json
{
    "friction": 0.75,
    "risk_level": "LOW",
    "model_type": "weather-aware",      // NEW FIELD
    "weather": {                         // NEW FIELD (if use_weather=True)
        "temperature": 15.2,
        "humidity": 65.0,
        "rainfall": 0.0,
        "wind_speed": 5.3
    },
    "note": "error_message_if_any"      // NEW FIELD
}
```

### New Endpoints:

#### POST `/predict-weather-aware`
```python
@app.post("/predict-weather-aware")
async def predict_weather_aware(
    file: UploadFile,
    latitude: Optional[float],
    longitude: Optional[float],
    city: Optional[str]
)
# Always uses weather data, fails gracefully if unavailable
```

#### POST `/predict-image-only`
```python
@app.post("/predict-image-only")
async def predict_image_only(
    file: UploadFile,
    latitude: Optional[float],
    longitude: Optional[float]
)
# Uses original CNN only (backward compatible)
```

#### GET `/health`
```python
@app.get("/health")
def health_check()
# Returns service status and available features
```

### Enhanced GET `/all-data`
```python
# Before response:
{
    "latitude": ...,
    "longitude": ...,
    "friction": ...,
    "risk_level": ...
}

# After response (includes new field):
{
    "latitude": ...,
    "longitude": ...,
    "friction": ...,
    "risk_level": ...,
    "timestamp": ...,
    "weather_used": ...  // NEW FIELD
}
```

---

## 6. MODIFIED FILE: `/backend/database.py`

### Schema Changes:

#### Updated `road_data` Table
```sql
-- Before:
CREATE TABLE road_data (
    id INTEGER PRIMARY KEY,
    latitude REAL,
    longitude REAL,
    friction REAL,
    risk_level TEXT,
    timestamp TEXT
)

-- After:
CREATE TABLE road_data (
    id INTEGER PRIMARY KEY,
    latitude REAL,
    longitude REAL,
    friction REAL,
    risk_level TEXT,
    timestamp TEXT,
    weather_used BOOLEAN DEFAULT 0,    -- NEW
    weather_data TEXT                  -- NEW (JSON)
)
```

#### New Table: `weather_history`
```sql
CREATE TABLE weather_history (
    id INTEGER PRIMARY KEY,
    road_data_id INTEGER,              -- FK reference
    temperature REAL,
    humidity REAL,
    rainfall REAL,
    wind_speed REAL,
    timestamp TEXT,
    FOREIGN KEY(road_data_id) REFERENCES road_data(id)
)
```

### Function Changes:

#### `insert_record()` - Enhanced
```python
# Before:
def insert_record(latitude, longitude, friction, risk_level):
    # Simple 4 parameters

# After:
def insert_record(
    latitude: float,
    longitude: float,
    friction: float,
    risk_level: str,
    weather_data: Optional[Dict] = None,   # NEW
    weather_used: bool = False             # NEW
):
    # Stores weather JSON and creates weather_history record
```

### New Functions:

```python
def get_records_with_weather():
    # Returns only weather-enhanced predictions

def get_records_by_risk_level(risk_level: str):
    # Filters by HIGH/MEDIUM/LOW

def get_weather_statistics():
    # Returns aggregated weather metrics:
    # - avg/max/min temperature
    # - avg humidity, rainfall, wind_speed
```

---

## 7. MODIFIED FILE: `/backend/requirements.txt`

```ini
# Before:
fastapi
uvicorn
torch
torchvision
pillow

# After:
fastapi
uvicorn
torch
torchvision
pillow
requests                  # NEW - for OpenWeather API
pandas                    # Explicit (was implicit)
opencv-python            # Explicit (was implicit via cv2)
scikit-learn             # Explicit (was implicit)
matplotlib               # Explicit (was implicit)
```

---

## Integration Points Summary

### Data Flow with Weather:

```
HTTP Request (image + city/coordinates)
    ↓
snow_api.py (/predict endpoint)
    ↓
inference_engine.py (predict_friction)
    ├─ Load image
    ├─ Call weather_service.get_weather()
    ├─ Normalize weather features
    └─ Load model_with_weather
        ├─ image → CNN → 128 features
        ├─ weather → Dense → 16 features
        └─ Concatenate → Final output
    ↓
Database (road_data + weather_history)
    ↓
JSON Response (friction + weather data)
```

### Backward Compatibility Maintained:

✅ Original training script still works
✅ Original model (`snow_model_csv.pth`) still loads
✅ Legacy `/predict` endpoint with `use_weather=False`
✅ New `/predict-image-only` endpoint for explicit image-only mode
✅ `predict_friction_image_only()` function preserved

---

## Key Design Decisions

1. **Two Separate Models**: 
   - Original CNN for backward compatibility
   - New weather-aware CNN for enhanced predictions
   - Both can coexist and be A/B tested

2. **Graceful Degradation**:
   - If weather API fails, falls back to image-only prediction
   - No crashes on missing weather data

3. **Synthetic Training Data**:
   - Generates realistic weather patterns for training
   - Can be replaced with real historical data

4. **Normalized Features**:
   - All weather features normalized to [0, 1]
   - Prevents dominance of large-value features
   - Improves neural network training stability

5. **JSON Weather Storage**:
   - Flexible schema in database
   - Easy to extend with additional weather metrics
   - Human-readable in database inspection

---

## Testing Checklist

- [ ] Train models: `python main.py`
- [ ] Load image-only model: `load_model(use_weather=False)`
- [ ] Load weather-aware model: `load_model(use_weather=True)`
- [ ] Test image-only prediction: `predict_friction(bytes, use_weather=False)`
- [ ] Test weather-integrated prediction: `predict_friction(bytes, use_weather=True, city="NYC")`
- [ ] Test API `/predict` endpoint
- [ ] Test API `/predict-weather-aware` endpoint
- [ ] Test API `/predict-image-only` endpoint
- [ ] Verify database schema changes
- [ ] Check weather data storage in database
- [ ] Test with OpenWeather API (requires key)
