import os
import numpy as np
import joblib
import pandas as pd
import traceback
import smtplib
from email.message import EmailMessage
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dotenv import load_dotenv  
load_dotenv()                  


# Feature names used in the model
FEATURE_NAMES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
]

app = FastAPI()

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "models", "model.pkl")
try:
    model = joblib.load(model_path)
    print(f"✅ Model loaded from: {model_path}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# Function to send email alerts
def send_alert_email(to_email: str, subject: str, body: str):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = os.getenv("EMAIL_ADDRESS")
    sender_password = os.getenv("EMAIL_PASSWORD")  # Use secure env variable

    if not sender_email or not sender_password:
        print("⚠️ Email credentials are not set in environment variables.")
        return

    msg = EmailMessage()
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print("✅ Alert email sent successfully")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

# Request body schema
class InputData(BaseModel):
    product_type: str           # 'L', 'M', 'H'
    air_temperature: float      # °C
    process_temperature: float  # °C
    rotational_speed: float     # rpm
    torque: float               # Nm
    tool_wear: float            # min
    twf: int
    hdf: int
    pwf: int
    osf: int
    rnf: int

# Encode product type
def encode_product_type(pt: str) -> int:
    mapping = {"l": 0, "m": 1, "h": 2}
    return mapping.get(pt.lower(), -1)

@app.post("/predict")
def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    pt_feature = encode_product_type(data.product_type)
    if pt_feature == -1:
        raise HTTPException(status_code=400, detail="Invalid product_type. Use L, M, or H.")

    try:
        # Convert °C to Kelvin
        air_temp_K = data.air_temperature + 273.15
        process_temp_K = data.process_temperature + 273.15

        input_dict = {
            "Type": pt_feature,
            "Air temperature [K]": air_temp_K,
            "Process temperature [K]": process_temp_K,
            "Rotational speed [rpm]": data.rotational_speed,
            "Torque [Nm]": data.torque,
            "Tool wear [min]": data.tool_wear,
            "TWF": data.twf,
            "HDF": data.hdf,
            "PWF": data.pwf,
            "OSF": data.osf,
            "RNF": data.rnf
        }
        input_df = pd.DataFrame([input_dict], columns=FEATURE_NAMES)

        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0] if hasattr(model, "predict_proba") else None

        # Send alert if failure probability is high
        if proba is not None and proba[1] > 0.5:
            subject = "⚠️ Predictive Maintenance Alert: High Failure Probability"
            body = (
                f"Machine failure probability is {proba[1]*100:.2f}%.\n"
                f"Please inspect the machine urgently.\n"
                f"Input parameters:\n{input_dict}"
            )
            send_alert_email("ssheikh@netfotech.in", subject, body)

        return {
            "prediction": int(prediction),
            "normal_probability": float(proba[0]) if proba is not None else None,
            "failure_probability": float(proba[1]) if proba is not None else None
        }

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
