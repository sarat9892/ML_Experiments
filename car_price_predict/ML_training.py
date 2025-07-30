import argparse
from datetime import datetime
import json
import os

import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def load_data(event):
    filename = os.listdir(event["store"])
    dataframe = pd.read_csv(os.path.join(event["store"], filename[0]))

    return dataframe

def create_pipeline(numeric_cols, categorical_cols, model):
    numeric_preprocessor = Pipeline(
        steps=[
            ('polynomial_features', PolynomialFeatures(degree=1)),
            ("scaler", StandardScaler()),
        ]
    )
    onehot_encoder = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5))
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numerical", numeric_preprocessor, numeric_cols),
            ("categorical", onehot_encoder, categorical_cols),
        ]
    )
    return make_pipeline(preprocessor, model)

def training_run(train_event, datasets, pipeline):
    mlflow.set_experiment(train_event["expt_name"])

    with mlflow.start_run(run_name=f"LinReg_{datetime.now().strftime("%Y%m%d_%H%M%S")}"):
        pipeline.fit(datasets["x_train"], datasets["y_train"])
        r_sq = pipeline.score(datasets["x_train"], datasets["y_train"])

        y_hat = pipeline.predict(datasets["x_test"])
        r_squared, rmse, mae = eval_metrics(datasets["y_test"], y_hat)

        mlflow.log_metric("R Squared-Training", r_sq)
        mlflow.log_metric("R Squared-Prediction", r_squared)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)

        signature = infer_signature(datasets["x_train"], y_hat)
        mlflow.sklearn.log_model(pipeline, "model", input_example=datasets["x_train"],
                                 registered_model_name=train_event["model_name"], signature=signature)

        mlflow.set_tags(context_tags)

        output = f"Run complete: R2-Train={r_sq:.2f}, R2-Pred={r_squared:.2f}, MAE={mae:.2f}, RMSE={rmse:.2f}"
        print(output)

        return ""

def eval_metrics(y, y_hat):
    r_squared = r2_score(y, y_hat)
    rmse = np.sqrt(mean_squared_error(y, y_hat))
    mae = mean_absolute_error(y, y_hat)
    return r_squared, rmse, mae

def handler(event):
    df = load_data(event)
    df.drop(labels=["saledate"], axis=1, inplace=True)

    numerical_data = event["metadata"]["numerical"]
    categorical_data = event["metadata"]["nominal"]
    ml_model = LinearRegression()

    # Basic pre-train pre-processing
    df.dropna(subset=numerical_data, how='any', inplace=True)

    pipeline = create_pipeline(numerical_data, categorical_data, ml_model)

    x = df.drop(labels=[event["metadata"]["target"][0]], axis=1)
    y = df[event["metadata"]["target"]]

    datasets = dict()
    (datasets["x_train"], datasets["x_test"],
     datasets["y_train"], datasets["y_test"]) = train_test_split(x, y, test_size=0.25)

    train_event = {"expt_name": "Linear_Regression",
                   "model_name": "LR_Poly1",
                   "metadata": {
            "model": "Linear Regression",
            "degree": "1",
            "onehot_encoding_min_freq": "5"
        }}
    training_run(train_event, datasets, pipeline)

    return ""


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
