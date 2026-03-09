import torch
import torch.nn as nn


class MultiHeadCNN(nn.Module):
    def __init__(self):
        super(MultiHeadCNN, self).__init__()

        self.features = nn.Sequential(
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

            nn.AdaptiveAvgPool2d((1,1))
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


class MultiHeadCNNWithWeather(nn.Module):
    """
    CNN model with integrated weather data.
    Processes image through CNN and weather features through dense network,
    then concatenates and combines outputs.
    """
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
        # Image features: 128 dims, Weather features: 16 dims
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
            weather: Tensor of shape (batch_size, 4) with normalized weather features
        
        Returns:
            Output predictions
        """
        # Process image through CNN
        image_features = self.image_features(image)
        image_features = image_features.view(image_features.size(0), -1)  # Flatten

        # Process weather through dense network
        weather_features = self.weather_network(weather)

        # Concatenate image and weather features
        combined_features = torch.cat([image_features, weather_features], dim=1)

        # Final prediction
        output = self.combined_network(combined_features)

        return output


def load_model(use_weather=False):
    """
    Load trained model.
    
    Args:
        use_weather: If True, loads weather-aware model. If False, loads original CNN model.
    
    Returns:
        Loaded model in eval mode
    """
    if use_weather:
        model = MultiHeadCNNWithWeather()
        # Try to load weather model if available
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