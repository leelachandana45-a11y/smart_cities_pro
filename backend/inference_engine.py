import torch
import torchvision.transforms as transforms
from PIL import Image
import io

from .model_loader import load_model

# Load model once
model = load_model()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


def predict_friction(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)

    return output.item()