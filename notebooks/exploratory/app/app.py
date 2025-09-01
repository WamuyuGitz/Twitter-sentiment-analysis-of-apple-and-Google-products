from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

# --------------------
#  Load the pipeline
# --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pipeline_path = os.path.join(BASE_DIR, "pipeline.pkl")
pipeline = joblib.load(pipeline_path)

# --------------------
#  FastAPI App
# --------------------
app = FastAPI(title="Tweet Sentiment API")

# --------------------
#  Request Body Schema
# --------------------
class TweetRequest(BaseModel):
    tweets: list[str]

# --------------------
#  Routes
# --------------------
@app.get("/")
def home():
    return {"message": "Welcome to the Tweet Sentiment API"}

@app.post("/predict")
def predict_sentiment(request: TweetRequest):
    predictions = pipeline.predict(request.tweets)
    results = [
        {"tweet": tweet, "sentiment": sentiment}
        for tweet, sentiment in zip(request.tweets, predictions)
    ]
    return {"results": results}

