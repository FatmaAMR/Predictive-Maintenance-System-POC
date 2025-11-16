from pydantic import BaseModel, Field

class Features(BaseModel):
    UID: int = Field(..., alias="UID")
    Product_ID: str = Field(..., alias="Product ID")
    Type: str
    Air_temperature: float = Field(..., alias="Air temperature [K]")
    Process_temperature: float = Field(..., alias="Process temperature [K]")
    Rotational_speed: int = Field(..., alias="Rotational speed [rpm]")
    Torque: float = Field(..., alias="Torque [Nm]")
    Tool_wear: int = Field(..., alias="Tool wear [min]")
    
    TWF: int
    HDF: int
    PWF: int
    OSF: int
    RNF: int

    class Config:
        allow_population_by_field_name = True
        populate_by_name = True  # for FastAPI so field names or aliases both work


class PredictionOutput(BaseModel):
    failure_probability: float
    predicted_RUL: float
    risk_level: str
    action: str
