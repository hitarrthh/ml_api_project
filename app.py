# app.py

import logging
import logging_loki
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from schemas import HousingInput, PredictionOutput

# --- NEW Logging Setup ---
# 1. Get credentials from environment
LOKI_HOST = os.environ.get("LOKI_HOST")
LOKI_USERNAME = os.environ.get("LOKI_USERNAME")
LOKI_API_KEY = os.environ.get("LOKI_API_KEY")

# 2. Create the Loki handler
if LOKI_HOST and LOKI_USERNAME and LOKI_API_KEY:
    handler = logging_loki.LokiHandler(
        url=f"{LOKI_HOST}/loki/api/v1/push",
        auth=(LOKI_USERNAME, LOKI_API_KEY),
        version="1",
    )
    # Add a tag to identify our app in Grafana
    handler.setFormatter(
        logging_loki.LokiFormatter(
            {"app": "my-housing-api"},
        )
    )
else:
    print("WARNING: Grafana Loki credentials not set. Logging to console only.")
    handler = logging.StreamHandler() # Fallback to console

# 3. Get the root logger and add our handler
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# --- End Logging Setup ---


# --- Model Loading ---
MODEL_PATH = "models/model_pipeline.joblib"
pipeline = None
scaler = None
model = None
features = None

try:
    pipeline = joblib.load(MODEL_PATH)
    scaler = pipeline['scaler']
    model = pipeline['model']
    features = pipeline['features']
    logger.info(f"Model pipeline loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading model: {e}", exc_info=True)
    pipeline = None

# --- FastAPI App ---
app = FastAPI(
    title="California Housing Price Predictor API",
    description="An API to predict housing prices using a pre-trained model.",
    version="1.0.0"
)

@app.on_event("startup")
def startup_event():
    global pipeline, scaler, model, features
    if pipeline is None:
        try:
            pipeline = joblib.load(MODEL_PATH)
            scaler = pipeline['scaler']
            model = pipeline['model']
            features = pipeline['features']
            logger.info("Model pipeline re-loaded at startup.")
        except Exception as e:
            logger.critical(f"Failed to load model on startup: {e}")
    
@app.get("/", tags=["Health Check"])
def read_root():
    """Health check endpoint to ensure the API is running."""
    logger.info("Health check endpoint was hit.")
    return {"status": "ok", "message": "API is running"}

@app.post("/predict", tags=["Prediction"], response_model=PredictionOutput)
def predict_price(data: HousingInput):
    """
    Predict the median house value based on input features.
    
    - **MedInc**: Median income in block group
    - **HouseAge**: Median house age in block group
    - **AveRooms**: Average number of rooms
    - **AveBedrms**: Average number of bedrooms
    - **Population**: Block group population
    - **AveOccup**: Average number of household members
    - **Latitude**: House block latitude
    - **Longitude**: House block longitude
    """
    if pipeline is None:
        logger.error("Prediction attempted but model is not loaded.")
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model.")
        
    try:
        # Log the input data (using Pydantic 2.x model_dump)
        logger.info(
            f"Received prediction request",
            extra={"tags": {"input": data.model_dump()}}
        )
        
        # 1. Convert input data to DataFrame
        input_df = pd.DataFrame([data.model_dump()])
        input_df = input_df[features]
        
        # 2. Scale the input data
        input_scaled = scaler.transform(input_df)
        
        # 3. Make prediction
        prediction = model.predict(input_scaled)
        
        # 4. Format output
        predicted_price = prediction[0]
        response = {"predicted_price": predicted_price}
        
        # Log the response
        logger.info(
            f"Prediction successful: {predicted_price:.2f}",
            extra={"tags": {"output": response}}
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)