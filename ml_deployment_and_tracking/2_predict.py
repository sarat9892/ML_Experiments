import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from datetime import datetime


def load_data():
    ...

def split_data():
    ...

def preprocess():
    ...

def generate_features():
    ...

def train_model():
    ...

def save_model():
    ...

def load_model():
    ...

def inference():
    ...

def eval_metrics(y, y_hat):
    r_squared = r2_score(y, y_hat)
    rmse = np.sqrt(mean_squared_error(y, y_hat))
    mae = mean_absolute_error(y, y_hat)
    return r_squared, rmse, mae

def handler(data_file):

    wine = pd.read_csv(data_file)

    numerical = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                 'pH', 'sulphates', 'alcohol']
    binary = ["red", "white"]
    target = ["quality"]

    x = wine.drop(["quality"], axis=1)
    y = wine["quality"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    scaler = StandardScaler().fit(x_train)

    x_train_norm = scaler.transform(x_train)
    x_test_norm = scaler.transform(x_test)

    print("Experiment: ", datetime.now().strftime("%Y%m%d_%H%M%S"))
    mlflow.set_experiment(f"LinearRegression_{datetime.now().strftime("%Y%m%d_%H%M%S")}")

    with mlflow.start_run():
        model = LinearRegression()
        model.fit(x_train_norm, y_train)

        r_sq = model.score(x_train_norm, y_train)

        y_hat = model.predict(x_test_norm)

        r_squared, rmse, mae = eval_metrics(y_test, y_hat)

        signature = infer_signature(x_train_norm, y_hat)

        mlflow.log_metric("R Squared-Training", r_sq)
        mlflow.log_metric("R Squared-Prediction", r_squared)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)

        mlflow.sklearn.log_model(model, "model", signature=signature)

        print(f"Run complete: R2={r_squared:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")


if __name__ == "__main__":

    data_path = "data/processed/wine.csv"

    handler(data_path)