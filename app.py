import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st

# Load data
dataset = pd.read_csv('SD.csv')
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1:2].values

# Impute missing values
imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
X = imputer.fit_transform(X)
y = imputer.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Prediction function
def predict_salary(experience):
    prediction = regressor.predict([[experience]])
    return prediction[0][0]

# Streamlit UI
st.title("Salary Prediction App")
experience = st.number_input("Years of Experience", value=1.1)
predicted_salary = predict_salary(experience)
st.write(f"Predicted Salary will be: {predicted_salary}")
