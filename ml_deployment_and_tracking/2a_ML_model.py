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


def load_data(path):
    df = pd.read_csv(path)
    return df

def split_data(x, y, frac):
    return train_test_split(x, y, test_size=frac)

def preprocess():
    ...

def generate_features(dataframe):
    scaler = StandardScaler().fit(dataframe)
    return scaler.transform(dataframe)

def training_run(x_train, x_test, y_train, y_test):
    mlflow.set_experiment(f"LinearRegression_Expt_1")

    with mlflow.start_run(run_name=f"LinReg_{datetime.now().strftime("%Y%m%d_%H%M%S")}"):
        model = LinearRegression()
        model.fit(x_train, y_train)
        r_sq = model.score(x_train, y_train)

        y_hat = model.predict(x_test)
        r_squared, rmse, mae = eval_metrics(y_test, y_hat)

        mlflow.log_metric("R Squared-Training", r_sq)
        mlflow.log_metric("R Squared-Prediction", r_squared)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)

        signature = infer_signature(x_train, y_hat)
        mlflow.sklearn.log_model(model, "model", signature=signature)

        print(f"Run complete: R2={r_squared:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

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

    df = load_data(data_file)

    x = df.drop(["quality"], axis=1)
    y = df["quality"]

    x_train, x_test, y_train, y_test = split_data(x, y, 0.25)

    training_run(x_train, x_test, y_train, y_test)

    numerical = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                 'pH', 'sulphates', 'alcohol']
    binary = ["red", "white"]
    target = ["quality"]

    x_train_norm = generate_features(x_train)
    x_test_norm = generate_features(x_test)

    training_run(x_train_norm, x_test_norm, y_train, y_test)


if __name__ == "__main__":

    data_path = "data/processed/wine.csv"
    handler(data_path)