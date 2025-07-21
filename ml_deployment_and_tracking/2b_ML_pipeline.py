import argparse

import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from datetime import datetime


def load_data(path):
    df = pd.read_csv(path)
    return df

def split_data(x, y, frac):
    return train_test_split(x, y, test_size=frac)

def preprocess():
    ...

def generate_features():
    ...

def create_pipeline():
    numerical = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                 'pH', 'sulphates', 'alcohol']
    # binary = ["red", "white"]
    # target = ["quality"]

    numeric_preprocessor = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("numerical", numeric_preprocessor, numerical),
        ]
    )

    return make_pipeline(preprocessor, LinearRegression())

def training_run(x_train, x_test, y_train, y_test):
    mlflow.set_experiment(f"LinearRegression_Expt_2")

    with mlflow.start_run(run_name=f"LinReg_{datetime.now().strftime("%Y%m%d_%H%M%S")}"):
        pipe = create_pipeline()
        pipe.fit(x_train, y_train)
        r_sq = pipe.score(x_train, y_train)

        y_hat = pipe.predict(x_test)
        r_squared, rmse, mae = eval_metrics(y_test, y_hat)

        mlflow.log_metric("R Squared-Training", r_sq)
        mlflow.log_metric("R Squared-Prediction", r_squared)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)

        # signature = infer_signature(x_train, y_hat)
        mlflow.sklearn.log_model(pipe, "model", input_example=x_train, registered_model_name="LinRegress_Test")

        output = f"Run complete: R2={r_squared:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}"

        print(output)

def load_model():
    ...

def inference(x_new):
    model_name = "LinRegress_Test"
    model_version = "latest"

    # Load the model from the Model Registry
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.sklearn.load_model(model_uri)

    # Generate a new dataset for prediction and predict
    y_hat_new = model.predict(x_new)
    return y_hat_new

def eval_metrics(y, y_hat):
    r_squared = r2_score(y, y_hat)
    rmse = np.sqrt(mean_squared_error(y, y_hat))
    mae = mean_absolute_error(y, y_hat)
    return r_squared, rmse, mae

def train_handler(data_file):

    df = load_data(data_file)

    x = df.drop(["quality"], axis=1)
    y = df["quality"]

    x_train, x_test, y_train, y_test = split_data(x, y, 0.25)

    training_run(x_train, x_test, y_train, y_test)

def inference_handler(event):

    pred = inference(event)

    print(f"Predictions: {pred}")

    return pred


if __name__ == "__main__":
    data_path = "data/processed/wine.csv"

    test_data1 = [6.3, 0.39, 0.16, 1.4, 0.08, 11, 23, 0.9955, 3.34, 0.56, 9.3, 1, 0]  # 5
    test_data2 = [10.4, 0.44, 0.42, 1.5, 0.145, 34, 48, 0.99832, 3.38, 0.86, 9.9, 1, 0]  # 3
    test_data3 = [6.4, 0.57, 0.12, 2.3, 0.12, 25, 36, 0.99519, 3.47, 0.71, 11.3, 1, 0]  # 7

    input_df = pd.DataFrame(np.array([test_data1, test_data2, test_data3]),
                            columns=["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                                     "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
                                     "pH", "sulphates", "alcohol", "red", "white"])

    training_event = {"train_data": data_path,
                      "type": "training"}

    inference_event = {"input_data": input_df,
                       "type": "inference"}

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p",
                         default=data_path,
                         help="Processed data path")
    args = parser.parse_args()

    train_handler(args.path)

    inference_handler(inference_event["input_data"])
