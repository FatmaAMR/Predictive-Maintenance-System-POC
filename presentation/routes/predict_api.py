from fastapi import APIRouter
import pandas as pd
from presentation.schemas import Features, PredictionOutput
from business.services.anomaly_predictor import AnomalyPredictor

data_router = APIRouter() 
model = AnomalyPredictor()

@data_router.post("/predict", response_model=PredictionOutput)
def predict(features: Features):
    df = pd.DataFrame([features.dict(by_alias=True)])
    
    expected_cols = [
    "Product ID", "Type", "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
    "TWF", "HDF", "PWF", "OSF", "RNF"
    ]
    
    df = df[expected_cols]
    insight = model.generate_insight(df)
    return insight
