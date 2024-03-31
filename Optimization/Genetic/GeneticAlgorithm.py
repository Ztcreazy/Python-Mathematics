import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Read the heart data csv file and store it in a pandas dataframe
df = pd.read_csv("Optimization/Genetic/heart_data.csv")
print(df.head(10))

# Drop the Unnamed: 0 column from the dataframe
df = df.drop(columns=["Unnamed: 0"])

# Display the number of missing values in each column of the dataframe
print(df.isna().sum())

# Generate descriptive statistics of the dataframe
print(df.describe().T)

# Compute the pairwise correlation between the columns and the target variable heart.disease
print(df.corrwith(df["heart.disease"]))

# Split the dataframe into the feature matrix X and the target vector y
X = df.drop(columns=["heart.disease"])
y = df['heart.disease']

# Display the first 5 rows of the feature matrix X
print(X.head())