import torch
import torchvision.transforms as transforms
from PIL import Image
import io

from .model_loader import load_model
from .weather_service import get_weather, get_weather_by_coordinates, normalize_weather_features

# Load models
model_image_only = load_model(use_weather=False)
model_with_weather = load_model(use_weather=True)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


def predict_friction(image_bytes, use_weather=False, city=None, latitude=None, longitude=None):
    """
    Predict friction coefficient from road image, optionally including weather data.
    
    Args:
        image_bytes: Image file bytes
        use_weather: Whether to include weather data in prediction
        city: City name for weather lookup (used if use_weather=True and coordinates not provided)
        latitude: GPS latitude for weather lookup
        longitude: GPS longitude for weather lookup
    
    Returns:
        Dictionary with prediction results
    """
    # Process image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    
    result = {
        "friction": None,
        "weather_used": False,
        "weather_data": None
    }
    
    with torch.no_grad():
        if use_weather:
            # Get weather data
            if latitude is not None and longitude is not None:
                weather_dict = get_weather_by_coordinates(latitude, longitude)
            elif city:
                weather_dict = get_weather(city)
            else:
                # Default fallback
                weather_dict = {
                    "temperature": 15.0,
                    "humidity": 60.0,
                    "rainfall": 0.0,
                    "wind_speed": 5.0,
                    "success": False
                }
            
            if weather_dict.get("success", False):
                # Normalize weather features
                weather_normalized = normalize_weather_features(
                    weather_dict["temperature"],
                    weather_dict["humidity"],
                    weather_dict["rainfall"],
                    weather_dict["wind_speed"]
                )
                weather_tensor = torch.tensor([weather_normalized], dtype=torch.float32)
                
                # Run prediction with weather-aware model
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
                # Fall back to image-only prediction
                output = model_image_only(image_tensor)
                result["friction"] = float(output.item())
                result["weather_used"] = False
                result["error"] = "Could not fetch weather data, using image-only prediction"
        else:
            # Image-only prediction
            output = model_image_only(image_tensor)
            result["friction"] = float(output.item())
            result["weather_used"] = False
    
    return result


def predict_friction_image_only(image_bytes):
    """
    Legacy function: predict friction using image data only.
    Maintains backward compatibility.
    
    Args:
        image_bytes: Image file bytes
    
    Returns:
        Friction coefficient as float
    """
    result = predict_friction(image_bytes, use_weather=False)
    return result["friction"]