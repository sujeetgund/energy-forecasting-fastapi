import boto3
import joblib
import os
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from datetime import datetime

# S3 Bucket name and file paths
S3_BUCKET_NAME = 'energy-consumption-s3'
MODEL_KEY = 'artifacts/random_forest_model.pkl'
SCALER_KEY = 'artifacts/scaler.pkl'
LOCAL_ARTIFACTS_DIR = "artifacts/"

# Ensure the local artifacts directory exists
os.makedirs(LOCAL_ARTIFACTS_DIR, exist_ok=True)

# S3 Client
s3 = boto3.client('s3')

# Helper function to download artifacts from S3
def download_from_s3(s3_key, local_path):
    s3.download_file(S3_BUCKET_NAME, s3_key, local_path)

# Download the model and scaler from S3
download_from_s3(MODEL_KEY, os.path.join(LOCAL_ARTIFACTS_DIR, "random_forest_model.pkl"))
download_from_s3(SCALER_KEY, os.path.join(LOCAL_ARTIFACTS_DIR, "scaler.pkl"))

# Load the pre-trained model and scaler
model = joblib.load(os.path.join(LOCAL_ARTIFACTS_DIR, "random_forest_model.pkl"))
scaler = joblib.load(os.path.join(LOCAL_ARTIFACTS_DIR, "scaler.pkl"))

# Population Density dictionary
population_density_dict = {
    "States_Punjab": 551, "States_Haryana": 573, "States_Rajasthan": 201, 
    "States_Delhi": 11297, "States_UP": 828, "States_Uttarakhand": 189, 
    "States_HP": 123, "States_J&K": 297, "States_Chandigarh": 350, 
    "States_Chhattisgarh": 189, "States_Gujarat": 308, "States_MP": 236, 
    "States_Maharashtra": 365, "States_Goa": 394, "States_DNH": 970,
    "States_Andhra Pradesh": 303, "States_Telangana": 312, "States_Karnataka": 319, 
    "States_Kerala": 859, "States_Tamil Nadu": 555, "States_Pondy": 2598, 
    "States_Bihar": 1106, "States_Jharkhand": 414, "States_Odisha": 269, 
    "States_West Bengal": 1028, "States_Sikkim": 86, "States_Arunachal Pradesh": 17, 
    "States_Assam": 398, "States_Manipur": 122, "States_Meghalaya": 132, 
    "States_Mizoram": 52, "States_Nagaland": 119, "States_Tripura": 350
}

# List of states used for one-hot encoding
state_list = [
    'States_Arunachal Pradesh', 'States_Assam',
       'States_Bihar', 'States_Chandigarh', 'States_Chhattisgarh',
       'States_DNH', 'States_Delhi', 'States_Goa', 'States_Gujarat',
       'States_HP', 'States_Haryana', 'States_J&K', 'States_Jharkhand',
       'States_Karnataka', 'States_Kerala', 'States_MP', 'States_Maharashtra',
       'States_Manipur', 'States_Meghalaya', 'States_Mizoram',
       'States_Nagaland', 'States_Odisha', 'States_Pondy', 'States_Punjab',
       'States_Rajasthan', 'States_Sikkim', 'States_Tamil Nadu',
       'States_Telangana', 'States_Tripura', 'States_UP', 'States_Uttarakhand',
       'States_West Bengal'
]

# Define the input data schema using Pydantic
class UsagePredictionInput(BaseModel):
    state: str
    date: str  # Date in YYYY-MM-DD format

app = FastAPI()

# Helper function to prepare features for prediction
def prepare_input_data(input_data: UsagePredictionInput):
    # Convert input date to datetime
    date_obj = datetime.strptime(input_data.date, "%Y-%m-%d")
    
    # Extract features from the date
    day_of_week = date_obj.weekday()  # 0 = Monday, 6 = Sunday
    week_of_month = (date_obj.day - 1) // 7 + 1  # Calculate week of the month (1-4)
    month = date_obj.month  # Month (1-12)
    
    # Extract PopulationDensity from the dictionary
    population_density = population_density_dict.get(f"States_{input_data.state}", 0)
    
    # Create one-hot encoded state
    state_features = [1 if state == f"States_{input_data.state}" else 0 for state in state_list]
    
    # Generate lag features using historic dataset
    lag_features = [0] * 3
    
    # Create a DataFrame for the features
    features = pd.DataFrame([{
        'DayOfWeek': day_of_week,
        'WeekOfMonth': week_of_month,
        'Month': month,
        'PopulationDensity': population_density,
        'lag_1': lag_features[0],
        'lag_2': lag_features[1],
        'lag_3': lag_features[2],
        **dict(zip(state_list, state_features))
    }])
    
    # Load the pre-trained scaler
    scaler = joblib.load('artifacts/scaler.pkl')
    
    # Apply scaling only to the columns that were scaled during training
    cols_to_scale = ['PopulationDensity', 'lag_1', 'lag_2', 'lag_3']
    features[cols_to_scale] = scaler.transform(features[cols_to_scale])
    
    # Ensure the column order matches exactly with X_train used during training
    all_columns = ['DayOfWeek', 'WeekOfMonth', 'Month', 'PopulationDensity', 
                   'lag_1', 'lag_2', 'lag_3'] + state_list
    features = features[all_columns]
    
    return features



# Prediction endpoint
@app.post("/predict")
def predict_usage(input_data: UsagePredictionInput):
    # Prepare the input features
    scaled_features = prepare_input_data(input_data)
    
    # Make prediction
    prediction = model.predict(scaled_features)
    
    # Return the prediction
    return {"predicted_usage": prediction[0]}

@app.get("/")
def read_root():
    return {"message": "FastAPI app running on EC2"}