# app.py

import logging
import logging_loki # Import the loki handler library
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from schemas import HousingInput, PredictionOutput

# --- NEW Logging Setup ---
# 1. Get credentials from environment variables set in Render
LOKI_HOST = os.environ.get("LOKI_HOST")
LOKI_USERNAME = os.environ.get("LOKI_USERNAME")
LOKI_API_KEY = os.environ.get("LOKI_API_KEY")

# 2. Define default tags to be added to every log message
default_tags = {"app": "my-housing-api", "environment": "production"}

# 3. Create the Loki handler if credentials are found
if LOKI_HOST and LOKI_USERNAME and LOKI_API_KEY:
    handler = logging_loki.LokiHandler(
        url=f"{LOKI_HOST}/loki/api/v1/push", # Standard Loki push endpoint
        tags=default_tags, # Pass default tags directly here
        auth=(LOKI_USERNAME, LOKI_API_KEY), # Use Basic Auth
        version="1", # Loki API version
    )
    print(f"Loki logging enabled, sending logs to {LOKI_HOST}")
else:
    # If credentials aren't set, fall back to standard console logging
    print("WARNING: Grafana Loki credentials not set in environment variables.")
    print("Logging will only go to console/Render logs.")
    handler = logging.StreamHandler()
    # Add a standard formatter for console logs
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

# 4. Get the root logger configure it
logger = logging.getLogger()

# Clear existing handlers if any (important for Uvicorn reload)
if logger.hasHandlers():
    logger.handlers.clear()

logger.addHandler(handler) # Add our Loki (or console) handler
logger.setLevel(logging.INFO) # Set the minimum log level

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
except FileNotFoundError:
    logger.error(f"Model file not found at {MODEL_PATH}. App will start but /predict will fail.")
    pipeline = None # Ensure pipeline is None if loading fails
except Exception as e:
    logger.error(f"Error loading model from {MODEL_PATH}: {e}", exc_info=True)
    pipeline = None

# --- FastAPI App ---
app = FastAPI(
    title="California Housing Price Predictor API",
    description="An API to predict housing prices using a pre-trained model.",
    version="1.0.0"
)

# Optional: Re-attempt loading model on startup if it failed initially
@app.on_event("startup")
def startup_event():
    global pipeline, scaler, model, features
    if pipeline is None and os.path.exists(MODEL_PATH):
        try:
            pipeline = joblib.load(MODEL_PATH)
            scaler = pipeline['scaler']
            model = pipeline['model']
            features = pipeline['features']
            logger.info("Model pipeline re-loaded successfully at startup.")
        except Exception as e:
            logger.critical(f"Failed to load model on startup from {MODEL_PATH}: {e}", exc_info=True)
            pipeline = None # Ensure it stays None on failure

@app.get("/", tags=["Health Check"])
def read_root():
    """Health check endpoint to ensure the API is running."""
    # This log message will include the default_tags
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
    # Check if model loaded successfully
    if pipeline is None:
        logger.error("Prediction attempted but model is not loaded.")
        raise HTTPException(status_code=503, detail="Model not loaded. Check application logs.")

    try:
        # Log the input data, adding extra context-specific tags
        # Pydantic's model_dump() is preferred over .dict() in v2+
        logger.info(
            "Received prediction request",
            extra={'tags': {"endpoint": "/predict", "action": "request"}}
        )

        # 1. Convert input data to DataFrame, ensuring feature order
        input_df = pd.DataFrame([data.model_dump()])
        # Ensure columns match the training order using the loaded features list
        try:
            input_df = input_df[features]
        except KeyError as e:
            logger.error(f"Input data missing expected feature: {e}", exc_info=True)
            raise HTTPException(status_code=422, detail=f"Missing feature in input: {e}")


        # 2. Scale the input data using the loaded scaler
        input_scaled = scaler.transform(input_df)

        # 3. Make prediction using the loaded model
        prediction = model.predict(input_scaled)

        # 4. Format output
        predicted_price = float(prediction[0]) # Ensure it's a standard float
        response = {"predicted_price": predicted_price}

        # Log the successful prediction result, adding extra tags
        logger.info(
            f"Prediction successful: {predicted_price:.2f}",
            extra={'tags': {"endpoint": "/predict", "action": "response", "predicted_value": round(predicted_price, 2)}}
        )

        return response

    except Exception as e:
        # Log any unexpected errors during prediction, adding error tags
        logger.error(f"Error during prediction: {e}", exc_info=True, extra={'tags': {"endpoint": "/predict", "action": "error"}})
        # Re-raise as an HTTP exception for the client
        raise HTTPException(status_code=500, detail=f"Internal Server Error during prediction.")

# --- Run the app (for local development) ---
if __name__ == "__main__":
    import uvicorn
    # Use port 8000 for local dev, Render uses its own internal port mapping
    uvicorn.run(app, host="0.0.0.0", port=8000)