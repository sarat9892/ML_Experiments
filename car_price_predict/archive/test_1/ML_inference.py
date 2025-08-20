import argparse
from datetime import datetime
import json
import os

import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


def load_data(event):
    filename = os.listdir(event["store"])
    dataframe = pd.read_csv(os.path.join(event["store"], filename[0]))

    return dataframe

def load_model(model_uri):
    return mlflow.sklearn.load_model(model_uri)

def inference(x_new):
    model_name = "LinRegress_API_Test"
    model_version = "latest"

    # Load the model from the Model Registry
    model_uri = f"models:/{model_name}/{model_version}"
    model = load_model(model_uri)

    # Generate a new dataset for prediction and predict
    y_hat_new = model.predict(x_new)
    return y_hat_new

def inference_handler(event):
    pred = inference(event)

    return pred

def handler(event):
    df = load_data(event)

    df.drop(labels=["saledate"], axis=1, inplace=True)

    # replace low frequency categorical variables with 'other'

    numerical_data = event["metadata"]["numerical"]
    categorical_data = event["metadata"]["nominal"]

    pipeline = create_pipeline(numerical_data, categorical_data)

    x = df.drop(labels=[event["metadata"]["target"][0]], axis=1)
    y = df[event["metadata"]["target"]]

    datasets = {}

    (datasets["x_train"], datasets["x_test"],
     datasets["y_train"], datasets["y_test"]) = train_test_split(x, y, test_size=0.25)

    train_event = {}

    training_run(train_event, datasets, pipeline)


if __name__ == "__main__":
    default_event = "event/event.txt"

    parser = argparse.ArgumentParser()
    parser.add_argument("--event", "-e",
                        default=default_event,
                        help="event data file path")
    args = parser.parse_args()

    with open(args.event, "r") as file_handler:
        event_text = file_handler.read()

    event_json = json.loads(str(event_text))

    _ = handler(event_json)
