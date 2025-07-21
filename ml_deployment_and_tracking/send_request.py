import requests
import pandas as pd
import numpy as np

data_path = "data/processed/wine.csv"

test_data1 = [6.3, 0.39, 0.16, 1.4, 0.08, 11, 23, 0.9955, 3.34, 0.56, 9.3, 1, 0]  # 5
test_data2 = [10.4, 0.44, 0.42, 1.5, 0.145, 34, 48, 0.99832, 3.38, 0.86, 9.9, 1, 0]  # 3
test_data3 = [6.4, 0.57, 0.12, 2.3, 0.12, 25, 36, 0.99519, 3.47, 0.71, 11.3, 1, 0]  # 7

input_df = pd.DataFrame(np.array([test_data1, test_data2, test_data3]),
                        columns=["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                                 "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
                                 "pH", "sulphates", "alcohol", "red", "white"])

input_data = {"data": [test_data1, test_data2, test_data3],
              "columns": ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                          "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
                          "pH", "sulphates", "alcohol", "red", "white"]}

training_event = {"train_data": data_path}

inference_event = {"input_data": input_data}

# training_response = requests.post("http://localhost:5050/train", json=training_event)
# print(training_response)

inference_response = requests.post("http://localhost:5050/predict", json=inference_event)
print(inference_response)

# print(inference_response.json())