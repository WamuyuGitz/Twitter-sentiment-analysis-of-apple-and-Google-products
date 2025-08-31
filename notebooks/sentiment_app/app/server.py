from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import joblib
import numpy as np
import os
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
PIPELINE_PATH = os.path.join(BASE_DIR, "pipeline.pkl")  # name is fine

model = joblib.load(PIPELINE_PATH)

# Define possible classes
class_names = model.classes_

# FastAPI app
app = FastAPI()

# Request body schema
class TweetRequest(BaseModel):
    tweets: list[str]  # Expecting a list of raw text tweets


@app.get("/")
def read_root():
    return {'message': 'Twitter Sentiments Model API'}

@app.post("/predict")
def predict(request: TweetRequest):
    try:
        df = pd.DataFrame(request.tweets, columns=["text"])
        predictions = model.predict(df)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# @app.post("/predict")
# def predict(request: TweetRequest):
#     if model is None:
#         return JSONResponse(
#             status_code=500,
#             content={"error": "Model not loaded. Check server logs."}
#         )
#     try:
#         # Convert list of tweets to DataFrame (matching your pipeline)
#         df = pd.DataFrame(request.tweets, columns=["text"])
        
#         # Predict sentiments
#         predictions = model.predict(df)
        
#         # Map each tweet to its prediction
#         results = [{"tweet": tweet, "prediction": pred} for tweet, pred in zip(request.tweets, predictions)]
        
#         return {"predictions": results}
    
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})
