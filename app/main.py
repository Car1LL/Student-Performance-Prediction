from fastapi import FastAPI
import pandas as pd
import src.preprocessing as preprocessing
from app.model import model, threshold, MINOR_CLASS
from app.schemas import Student

app = FastAPI(title="Student Performance Prediction API")

@app.get("/")
def root():
    return {"message": "API is running"}

@app.post("/predict")
def predict(data: Student):

    df = pd.DataFrame([data.model_dump()])

    probs = model.predict_proba(df)

    minor_probs = probs[:, MINOR_CLASS]

    y_pred = probs.argmax(axis=1)
    y_pred[minor_probs > threshold] = MINOR_CLASS

    prediction = int(y_pred[0])

    map_prediction = {
        0: "C",
        1: "B",
        2: "A"
    }

    return {
        "prediction": map_prediction[prediction]
    }
    

    