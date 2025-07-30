import numpy as np
import pandas as pd
from flask import Flask, request
from ETL import handler
from ML_pipeline import train_handler, inference_handler

train_data = None
app=Flask(__name__)

@app.route("/")
def home_endpoint():
  return "Hello World!"

@app.route("/preprocess", methods=["POST"])
def preprocess():
    global train_data
    if request.method == "POST":
        input_files = request.get_json()
        train_data = handler(input_files)
    return train_data

@app.route("/train", methods=["POST"])
def train_model():
    global train_data
    model = None
    if request.method == "POST":
        if not train_data:
            train_data = request.json["train_data"]
        model = train_handler(train_data)
    return model

@app.route("/predict", methods=["POST"])
def predict():
    predictions = None
    if request.method == "POST":
        input_data = request.json["input_data"]
        input_df = pd.DataFrame(np.array(input_data['data']), columns=input_data['columns'])
        predictions = inference_handler(input_df)
        print(f"Predictions: {predictions}")
    return predictions.tolist()


if __name__ == "__main__":
    app.run(host="localhost", port=5050)
