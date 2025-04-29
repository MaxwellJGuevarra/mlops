from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd

# Define input format
class ModelInput(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: int
    total_rooms: int
    total_bedrooms: float
    population: int
    households: int
    median_income: float

app = FastAPI()

# Set MLflow Tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Load model
model_name = "lab_8_best_model"

# Model URI
model_uri = f"models:/{model_name}/latest"

# Load model
model = mlflow.sklearn.load_model(model_uri)

@app.post("/predict")
def predict(input_data: ModelInput):
    input_df = pd.DataFrame([input_data.dict()])
    prediction = model.predict(input_df)
    return {"prediction": prediction.tolist()}

