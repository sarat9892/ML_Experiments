import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split


# Loading data
bank_marketing_csv = "data/bank-full.csv"
bank_data = pd.read_csv(bank_marketing_csv, sep=";")

# Checking for nulls
print(bank_data.isnull().sum())

# Classifying columns into feature types
nominal_features = ["job", "marital", "education", "contact", "poutcome"]
ordinal = ["education"]
binary_features = ["default", "housing", "loan"]
numerical_features = ["balance", "duration", "campaign", "pdays", "previous"]

nonscaled_bank_data = pd.DataFrame()

# Converting categorical binary strings to 0 and 1
nonscaled_bank_data["target"] = np.where(bank_data["y"] == "no", 0, 1)
nonscaled_bank_data["default"] = np.where(bank_data["default"] == "no", 0, 1)
nonscaled_bank_data["housing"] = np.where(bank_data["housing"] == "no", 0, 1)
nonscaled_bank_data["loan"] = np.where(bank_data["loan"] == "no", 0, 1)

# Creating Encoder objects
oe = OrdinalEncoder()
ohe = OneHotEncoder(sparse_output=False)

# One Hot encoding nominal data
nominal_data = bank_data[["job", "marital", "education", "contact", "poutcome", "education"]]
ohe.fit(nominal_data)

nominal_transformed = ohe.transform(nominal_data)
nominal_transformed_df = pd.DataFrame(nominal_transformed, columns=ohe.get_feature_names_out())

numerical_data = bank_data[numerical_features]

# Collecting all the columns together
nonscaled_bank_data = pd.concat([nonscaled_bank_data, nominal_transformed_df, numerical_data], axis=1)

# Splitting the data into train, validation and test sets before scaling
Y = nonscaled_bank_data[["target"]]
X = nonscaled_bank_data.drop(["target"], axis=1)

x_temp, x_test, y_temp, y_test = train_test_split(X, Y, test_size=0.20)
x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.15)

x_train.reset_index(inplace=True, drop=True)
x_test.reset_index(inplace=True, drop=True)
x_val.reset_index(inplace=True, drop=True)

y_train.reset_index(inplace=True, drop=True)
y_test.reset_index(inplace=True, drop=True)
y_val.reset_index(inplace=True, drop=True)

numerical_train_data = x_train[numerical_features]
numerical_test_data = x_test[numerical_features]
numerical_val_data = x_val[numerical_features]

# Scaling
scaler = StandardScaler().fit(numerical_train_data)

scaled_train_data = scaler.transform(numerical_train_data)
scaled_test_data = scaler.transform(numerical_test_data)
scaled_val_data = scaler.transform(numerical_val_data)

scaled_train_df = pd.DataFrame(scaled_train_data, columns=numerical_features)
scaled_test_df = pd.DataFrame(scaled_test_data, columns=numerical_features)
scaled_val_df = pd.DataFrame(scaled_val_data, columns=numerical_features)

x_train.drop(numerical_features, axis=1, inplace=True)
x_test.drop(numerical_features, axis=1, inplace=True)
x_val.drop(numerical_features, axis=1, inplace=True)

x_train = pd.concat([x_train, scaled_train_df], axis=1)
x_test = pd.concat([x_test, scaled_test_df], axis=1)
x_val = pd.concat([x_val, scaled_val_df], axis=1)

# Saving the preprocessed data
x_train.to_csv("data/processed/bank_train.csv")
x_test.to_csv("data/processed/bank_test.csv")
x_val.to_csv("data/processed/bank_val.csv")

y_train.to_csv("data/processed/bank_train_labels.csv")
y_test.to_csv("data/processed/bank_test_labels.csv")
y_val.to_csv("data/processed/bank_val_labels.csv")




