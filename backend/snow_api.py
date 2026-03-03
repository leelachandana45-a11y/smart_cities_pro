from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from .inference_engine import predict_friction
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
    latitude: float = Form(None),
    longitude: float = Form(None)
):
    image_bytes = await file.read()
    friction = predict_friction(image_bytes)

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
        "risk_level": risk
    }


@app.get("/all-data")
def fetch_data():
    records = get_all_records()

    return [
        {
            "latitude": record[0],
            "longitude": record[1],
            "friction": record[2],
            "risk_level": record[3]
        }
        for record in records
    ]