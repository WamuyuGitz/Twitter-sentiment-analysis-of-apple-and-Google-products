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

# @app.post("/predict")
# def predict(request: TweetRequest):
#     """
#     Predict sentiments for a list of tweets.
#     """
#     # Pass raw text directly to the pipeline (TF-IDF will handle preprocessing)
#     predictions = model.predict(request.tweets)

#     # Map predictions to class labels
#     predicted_labels = [class_names[p] for p in predictions]

#     return {"tweets": request.tweets, "predictions": predicted_labels}

@app.post("/predict")
def predict(request: TweetRequest):
    try:
        df = pd.DataFrame(request.tweets, columns=["text"])
        predictions = model.predict(df)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})