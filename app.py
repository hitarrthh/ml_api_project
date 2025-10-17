import logging
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from schemas import HousingInput, PredictionOutput  
import os


os.makedirs("logs", exist_ok=True)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("logs/api.log")
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
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
    logger.error(f"Error loading model: {e}")
    pipeline = None

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
def predict_price(data: HousingInput): # Updated input schema
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
        # Log the input data
        logger.info(f"Received prediction request: {data.dict()}")
        

        input_df = pd.DataFrame([data.dict()])
        input_df = input_df[features] # Ensure columns are in the same order as training
        

        input_scaled = scaler.transform(input_df)
        

        prediction = model.predict(input_scaled)
        

        predicted_price = prediction[0]
        
        response = {
            "predicted_price": predicted_price
        }
        

        logger.info(f"Prediction successful: {response}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)