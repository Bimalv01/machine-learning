import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import gradio as gi

# Load the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 1]

# Encoding Categorical variable
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Splitting the dataset into the Training set and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# Training the Multiple Linear Regression model on the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)


def predict_profit(RnD_Spend, Administration, Marketing_Spend, State):
    input_data = [[RnD_Spend, Administration, Marketing_Spend, State]]
    input_data_encoded = ct.transform(input_data)
    prediction = regressor.predict(input_data_encoded)
    return prediction[0]

# Create Gradio Interface
inputs = [
    gi.Number(label="R&D Spend"),
    gi.Number(label="Administration"),
    gi.Number(label="Marketing Spend"),
    gi.Dropdown(label="State", choices=["California", "Florida", "New York"])
]

output = gi.Textbox(label="Predicted Profit")

interface = gi.Interface(fn=predict_profit, inputs=inputs, outputs=output, title="Startup Profit Prediction")
interface.launch(share=True)