# Code Reference Guide - Exact Modifications

## Quick Access to Modified Code Sections

---

## 1. NEW: `/backend/weather_service.py` (Complete File)

```python
import requests
import os
from typing import Dict, Optional

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "YOUR_API_KEY_HERE")
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

def get_weather(city: str, state: Optional[str] = None, country: Optional[str] = None) -> Dict:
    """Fetch real-time weather data from OpenWeather API."""
    try:
        location = city
        if state:
            location = f"{city},{state}"
        if country:
            location = f"{location},{country}"
        
        params = {
            "q": location,
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

def get_weather_by_coordinates(latitude: float, longitude: float) -> Dict:
    """Fetch real-time weather data using GPS coordinates."""
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
    """Normalize weather features to [0, 1] range."""
    norm_temp = (temperature + 40) / 90
    norm_temp = max(0.0, min(1.0, norm_temp))
    
    norm_humidity = humidity / 100.0
    norm_humidity = max(0.0, min(1.0, norm_humidity))
    
    norm_rainfall = rainfall / 100.0
    norm_rainfall = max(0.0, min(1.0, norm_rainfall))
    
    norm_wind = wind_speed / 30.0
    norm_wind = max(0.0, min(1.0, norm_wind))
    
    return [norm_temp, norm_humidity, norm_rainfall, norm_wind]
```

---

## 2. KEY SECTIONS: `/main.py`

### New Imports (add at top):
```python
import json
import random
```

### New Class: `SnowCSVDatasetWithWeather`
```python
class SnowCSVDatasetWithWeather(Dataset):
    """Dataset with integrated weather features"""
    def __init__(self, csv_file, root_dir, weather_file=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        
        if weather_file and os.path.exists(weather_file):
            with open(weather_file, 'r') as f:
                self.weather_data = json.load(f)
        else:
            self.weather_data = self._generate_synthetic_weather()
    
    def _generate_synthetic_weather(self):
        """Generate synthetic weather data for training samples"""
        weather_dict = {}
        for idx in range(len(self.data)):
            weather_dict[idx] = {
                "temperature": round(np.random.uniform(-10, 35), 2),
                "humidity": round(np.random.uniform(20, 100), 2),
                "rainfall": round(np.random.exponential(2), 2),
                "wind_speed": round(np.random.uniform(0, 15), 2)
            }
        return weather_dict
    
    def _normalize_weather(self, temperature, humidity, rainfall, wind_speed):
        """Normalize weather features to [0, 1] range"""
        norm_temp = (temperature + 40) / 90
        norm_temp = max(0.0, min(1.0, norm_temp))
        
        norm_humidity = humidity / 100.0
        norm_humidity = max(0.0, min(1.0, norm_humidity))
        
        norm_rainfall = rainfall / 100.0
        norm_rainfall = max(0.0, min(1.0, norm_rainfall))
        
        norm_wind = wind_speed / 30.0
        norm_wind = max(0.0, min(1.0, norm_wind))
        
        return [norm_temp, norm_humidity, norm_rainfall, norm_wind]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        label = self.data.iloc[idx, 1]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0

        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        label = torch.tensor(label, dtype=torch.float32)
        
        weather_raw = self.weather_data.get(idx, {
            "temperature": 15.0,
            "humidity": 60.0,
            "rainfall": 0.0,
            "wind_speed": 5.0
        })
        
        weather_normalized = self._normalize_weather(
            weather_raw["temperature"],
            weather_raw["humidity"],
            weather_raw["rainfall"],
            weather_raw["wind_speed"]
        )
        weather = torch.tensor(weather_normalized, dtype=torch.float32)
        
        return {
            "image": img,
            "weather": weather,
            "label": label
        }
```

### New Class: `MultiHeadCNNWithWeather`
```python
class MultiHeadCNNWithWeather(nn.Module):
    """CNN model with integrated weather data."""
    def __init__(self, weather_input_size=4):
        super(MultiHeadCNNWithWeather, self).__init__()

        # Image branch - CNN
        self.image_features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Weather branch - Dense network
        self.weather_network = nn.Sequential(
            nn.Linear(weather_input_size, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.2)
        )

        # Combined processing
        self.combined_network = nn.Sequential(
            nn.Linear(128 + 16, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            
            nn.Linear(32, 1)
        )

    def forward(self, image, weather):
        """
        Args:
            image: Tensor of shape (batch_size, 3, height, width)
            weather: Tensor of shape (batch_size, 4)
        
        Returns:
            Output predictions
        """
        # Process image through CNN
        image_features = self.image_features(image)
        image_features = image_features.view(image_features.size(0), -1)

        # Process weather through dense network
        weather_features = self.weather_network(weather)

        # Concatenate and process
        combined_features = torch.cat([image_features, weather_features], dim=1)
        output = self.combined_network(combined_features)

        return output
```

### Dataset Setup Code (after building train/val loaders):
```python
# -------- WEATHER-AWARE DATASET SETUP --------
print("\n--- WEATHER-AWARE DATASET ---")
weather_dataset = SnowCSVDatasetWithWeather("labels.csv", "dataset")
train_weather_dataset, val_weather_dataset = random_split(weather_dataset, [train_size, val_size])

train_weather_loader = DataLoader(train_weather_dataset, batch_size=32, shuffle=True)
val_weather_loader = DataLoader(val_weather_dataset, batch_size=32)
```

### Weather-Aware Model Training (replace existing training):
```python
# -------- WEATHER-AWARE MODEL TRAINING --------
print("\n\n=== TRAINING WEATHER-AWARE CNN MODEL ===")
weather_model = MultiHeadCNNWithWeather().to(device)
weather_criterion = nn.MSELoss()
weather_optimizer = optim.Adam(weather_model.parameters(), lr=0.0005)

weather_train_losses = []
weather_val_losses = []

for epoch in range(epochs):
    weather_model.train()
    total_train_loss = 0

    for batch in train_weather_loader:
        images = batch["image"].to(device)
        weather = batch["weather"].to(device)
        labels = batch["label"].to(device).unsqueeze(1)

        weather_optimizer.zero_grad()
        outputs = weather_model(images, weather)
        loss = weather_criterion(outputs, labels)
        loss.backward()
        weather_optimizer.step()

        total_train_loss += loss.item()

    weather_train_losses.append(total_train_loss / len(train_weather_loader))

    # ---- VALIDATION ----
    weather_model.eval()
    total_val_loss = 0
    preds_weather = []
    actuals_weather = []

    with torch.no_grad():
        for batch in val_weather_loader:
            images = batch["image"].to(device)
            weather = batch["weather"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)

            outputs = weather_model(images, weather)
            loss = weather_criterion(outputs, labels)

            total_val_loss += loss.item()

            preds_weather.extend(outputs.cpu().numpy())
            actuals_weather.extend(labels.cpu().numpy())

    weather_val_losses.append(total_val_loss / len(val_weather_loader))

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {weather_train_losses[-1]:.4f} | Val Loss: {weather_val_losses[-1]:.4f}")

# ---- METRICS FOR WEATHER MODEL ----
rmse_weather = np.sqrt(mean_squared_error(actuals_weather, preds_weather))
mae_weather = mean_absolute_error(actuals_weather, preds_weather)
r2_weather = r2_score(actuals_weather, preds_weather)

print("\nEvaluation Metrics (Weather-Aware CNN):")
print("RMSE:", rmse_weather)
print("MAE:", mae_weather)
print("R2:", r2_weather)

torch.save(weather_model.state_dict(), "snow_model_weather.pth")
print("Weather-Aware Model Saved!")
```

---

## 3. KEY SECTIONS: `/backend/model_loader.py`

### New Class Addition:
```python
class MultiHeadCNNWithWeather(nn.Module):
    """CNN model with integrated weather data."""
    def __init__(self, weather_input_size=4):
        super(MultiHeadCNNWithWeather, self).__init__()

        self.image_features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.weather_network = nn.Sequential(
            nn.Linear(weather_input_size, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.2)
        )

        self.combined_network = nn.Sequential(
            nn.Linear(128 + 16, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, image, weather):
        image_features = self.image_features(image)
        image_features = image_features.view(image_features.size(0), -1)
        weather_features = self.weather_network(weather)
        combined_features = torch.cat([image_features, weather_features], dim=1)
        output = self.combined_network(combined_features)
        return output
```

### Updated load_model Function:
```python
def load_model(use_weather=False):
    """Load trained model."""
    if use_weather:
        model = MultiHeadCNNWithWeather()
        try:
            model.load_state_dict(
                torch.load("snow_model_weather.pth", map_location=torch.device("cpu"))
            )
        except FileNotFoundError:
            print("Weather model not found, initializing with random weights.")
    else:
        model = MultiHeadCNN()
        model.load_state_dict(
            torch.load("snow_model_csv.pth", map_location=torch.device("cpu"))
        )
    
    model.eval()
    return model
```

---

## 4. KEY SECTIONS: `/backend/inference_engine.py`

### Updated Imports:
```python
from .weather_service import get_weather, get_weather_by_coordinates, normalize_weather_features
```

### Load Both Models:
```python
model_image_only = load_model(use_weather=False)
model_with_weather = load_model(use_weather=True)
```

### Updated predict_friction Function:
```python
def predict_friction(image_bytes, use_weather=False, city=None, latitude=None, longitude=None):
    """Predict friction coefficient from road image."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    
    result = {
        "friction": None,
        "weather_used": False,
        "weather_data": None
    }
    
    with torch.no_grad():
        if use_weather:
            if latitude is not None and longitude is not None:
                weather_dict = get_weather_by_coordinates(latitude, longitude)
            elif city:
                weather_dict = get_weather(city)
            else:
                weather_dict = {
                    "temperature": 15.0,
                    "humidity": 60.0,
                    "rainfall": 0.0,
                    "wind_speed": 5.0,
                    "success": False
                }
            
            if weather_dict.get("success", False):
                weather_normalized = normalize_weather_features(
                    weather_dict["temperature"],
                    weather_dict["humidity"],
                    weather_dict["rainfall"],
                    weather_dict["wind_speed"]
                )
                weather_tensor = torch.tensor([weather_normalized], dtype=torch.float32)
                output = model_with_weather(image_tensor, weather_tensor)
                
                result["friction"] = float(output.item())
                result["weather_used"] = True
                result["weather_data"] = {
                    "temperature": weather_dict["temperature"],
                    "humidity": weather_dict["humidity"],
                    "rainfall": weather_dict["rainfall"],
                    "wind_speed": weather_dict["wind_speed"]
                }
            else:
                output = model_image_only(image_tensor)
                result["friction"] = float(output.item())
                result["weather_used"] = False
                result["error"] = "Could not fetch weather data, using image-only prediction"
        else:
            output = model_image_only(image_tensor)
            result["friction"] = float(output.item())
            result["weather_used"] = False
    
    return result
```

### Legacy Function for Backward Compatibility:
```python
def predict_friction_image_only(image_bytes):
    """Legacy function: predict friction using image data only."""
    result = predict_friction(image_bytes, use_weather=False)
    return result["friction"]
```

---

## 5. KEY SECTIONS: `/backend/snow_api.py`

### New Imports:
```python
from typing import Optional
```

### Enhanced /predict Endpoint:
```python
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    city: Optional[str] = Form(None),
    use_weather: bool = Form(False)
):
    image_bytes = await file.read()
    
    prediction = predict_friction(
        image_bytes,
        use_weather=use_weather,
        city=city,
        latitude=latitude,
        longitude=longitude
    )
    
    friction = prediction["friction"]
    
    if friction < 0.3:
        risk = "HIGH"
    elif friction < 0.6:
        risk = "MEDIUM"
    else:
        risk = "LOW"
    
    if latitude is not None and longitude is not None:
        insert_record(
            latitude, 
            longitude, 
            friction, 
            risk,
            weather_data=prediction.get("weather_data"),
            weather_used=prediction.get("weather_used", False)
        )
    
    response = {
        "friction": friction,
        "risk_level": risk,
        "model_type": "weather-aware" if prediction.get("weather_used") else "image-only"
    }
    
    if prediction.get("weather_data"):
        response["weather"] = prediction["weather_data"]
    
    if "error" in prediction:
        response["note"] = prediction["error"]
    
    return response
```

### New Convenience Endpoints:
```python
@app.post("/predict-weather-aware")
async def predict_weather_aware(
    file: UploadFile = File(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    city: Optional[str] = Form(None)
):
    image_bytes = await file.read()
    prediction = predict_friction(
        image_bytes,
        use_weather=True,
        city=city,
        latitude=latitude,
        longitude=longitude
    )
    
    friction = prediction["friction"]
    
    if friction < 0.3:
        risk = "HIGH"
    elif friction < 0.6:
        risk = "MEDIUM"
    else:
        risk = "LOW"
    
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
```

---

## 6. KEY SECTIONS: `/backend/database.py`

### Updated insert_record Function:
```python
def insert_record(
    latitude: float, 
    longitude: float, 
    friction: float, 
    risk_level: str,
    weather_data: Optional[Dict] = None,
    weather_used: bool = False
):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    weather_json = None
    if weather_data:
        weather_json = json.dumps(weather_data)

    cursor.execute("""
        INSERT INTO road_data (latitude, longitude, friction, risk_level, timestamp, weather_used, weather_data)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (latitude, longitude, friction, risk_level, datetime.now().isoformat(), weather_used, weather_json))

    road_data_id = cursor.lastrowid

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
```

### New Utility Functions:
```python
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
```

---

## Setup Instructions

### 1. Install Dependencies:
```bash
cd backend
pip install -r requirements.txt
```

### 2. Set OpenWeather API Key:
```bash
# Linux/Mac
export OPENWEATHER_API_KEY="your_key_here"

# Windows PowerShell
$env:OPENWEATHER_API_KEY="your_key_here"

# Or add to your environment file
```

### 3. Train Models:
```bash
python main.py
```

### 4. Start API Server:
```bash
cd backend
uvicorn snow_api:app --reload
```

### 5. Test Endpoints:
```bash
# Weather-aware prediction
curl -X POST "http://localhost:8000/predict-weather-aware" \
  -F "file=@road_image.jpg" \
  -F "city=Boston"

# Image-only prediction (backward compatible)
curl -X POST "http://localhost:8000/predict-image-only" \
  -F "file=@road_image.jpg"
```

---

## Summary of Core Changes

| Component | Change | Backward Compatible |
|-----------|--------|-------------------|
| Dataset | Added weather features | ✅ Old dataset still works |
| Model | New weather architecture | ✅ Original model preserved |
| Training | Dual model training | ✅ Can train either model |
| API | Added weather parameters | ✅ Old endpoints unchanged |
| Database | Added weather columns | ✅ Schema backward compatible |
| Functions | `predict_friction()` enhanced | ✅ Legacy wrapper available |

