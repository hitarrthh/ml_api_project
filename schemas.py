from pydantic import BaseModel
from typing import List, Optional
class HousingInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "MedInc": 3.8716,
                "HouseAge": 25.0,
                "AveRooms": 6.284,
                "AveBedrms": 1.08,
                "Population": 1200.0,
                "AveOccup": 2.54,
                "Latitude": 34.23,
                "Longitude": -118.49
            }
        }

class PredictionOutput(BaseModel):
    predicted_price: float