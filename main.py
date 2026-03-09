import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import pandas as pd
import cv2
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import json
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------- CUSTOM DATASET USING CSV --------
class SnowCSVdataset(Dataset):
    """Original dataset without weather features"""
    def __init__(self, csv_file, root_dir):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        label = self.data.iloc[idx, 1]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0

        img = torch.tensor(img, dtype=torch.float32).permute(2,0,1)
        label = torch.tensor(label, dtype=torch.float32)

        return img, label


class SnowCSVDatasetWithWeather(Dataset):
    """Dataset with integrated weather features"""
    def __init__(self, csv_file, root_dir, weather_file=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        
        # Load or generate weather data
        if weather_file and os.path.exists(weather_file):
            with open(weather_file, 'r') as f:
                self.weather_data = json.load(f)
        else:
            # Generate synthetic weather data for training
            self.weather_data = self._generate_synthetic_weather()
    
    def _generate_synthetic_weather(self):
        """Generate synthetic weather data for training samples"""
        weather_dict = {}
        for idx in range(len(self.data)):
            # Simulate realistic weather patterns
            weather_dict[idx] = {
                "temperature": round(np.random.uniform(-10, 35), 2),  # -10 to 35°C
                "humidity": round(np.random.uniform(20, 100), 2),      # 20-100%
                "rainfall": round(np.random.exponential(2), 2),        # mm, exponential distribution
                "wind_speed": round(np.random.uniform(0, 15), 2)       # 0-15 m/s
            }
        return weather_dict
    
    def _normalize_weather(self, temperature, humidity, rainfall, wind_speed):
        """Normalize weather features to [0, 1] range"""
        norm_temp = (temperature + 40) / 90  # [-40, 50] -> [0, 1]
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
        # Load image
        img_path = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        label = self.data.iloc[idx, 1]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0

        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        label = torch.tensor(label, dtype=torch.float32)
        
        # Get weather data
        weather_raw = self.weather_data.get(idx, {
            "temperature": 15.0,
            "humidity": 60.0,
            "rainfall": 0.0,
            "wind_speed": 5.0
        })
        
        # Normalize weather features
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


dataset = SnowCSVdataset("labels.csv", "dataset")
print("Total Images:", len(dataset))

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# -------- WEATHER-AWARE DATASET SETUP --------
print("\n--- WEATHER-AWARE DATASET ---")
weather_dataset = SnowCSVDatasetWithWeather("labels.csv", "dataset")
train_weather_dataset, val_weather_dataset = random_split(weather_dataset, [train_size, val_size])

train_weather_loader = DataLoader(train_weather_dataset, batch_size=32, shuffle=True)
val_weather_loader = DataLoader(val_weather_dataset, batch_size=32)

# -------- MODEL --------
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
    
    Architecture:
    - Image branch: CNN -> 128 features
    - Weather branch: Dense network -> 16 features
    - Combined: Concatenate -> Dense layers -> 1 output
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
            Output predictions of shape (batch_size, 1)
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


model = MultiHeadCNN().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

epochs = 15
train_losses = []
val_losses = []

print("\n=== TRAINING ORIGINAL CNN MODEL (IMAGE ONLY) ===")
for epoch in range(epochs):
    model.train()
    total_train_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    train_losses.append(total_train_loss / len(train_loader))

    # ---- VALIDATION ----
    model.eval()
    total_val_loss = 0
    preds = []
    actuals = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_val_loss += loss.item()

            preds.extend(outputs.cpu().numpy())
            actuals.extend(labels.cpu().numpy())

    val_losses.append(total_val_loss / len(val_loader))

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")

# ---- METRICS ----
rmse = np.sqrt(mean_squared_error(actuals, preds))
mae = mean_absolute_error(actuals, preds)
r2 = r2_score(actuals, preds)

print("\nEvaluation Metrics (Original CNN):")
print("RMSE:", rmse)
print("MAE:", mae)
print("R2:", r2)

torch.save(model.state_dict(), "snow_model_csv.pth")
print("Original Model Saved!")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.legend()
plt.title("CNN Only: Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

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

plt.subplot(1, 2, 2)
plt.plot(weather_train_losses, label="Train Loss")
plt.plot(weather_val_losses, label="Validation Loss")
plt.legend()
plt.title("CNN + Weather: Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.tight_layout()
plt.savefig("loss_curve_comparison.pdf")
print("Loss curves saved to loss_curve_comparison.pdf")