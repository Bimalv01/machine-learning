import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import streamlit as st

# Load the dataset
dataset = pd.DataFrame({
    'Position': ['Business Analyst', 'Junior Consultant', 'Senior Consultant', 'Manager', 'Country Manager',
                 'Region Manager', 'Partner', 'Senior Partner', 'C-level', 'CEO'],
    'Level': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [45000, 50000, 60000, 80000, 110000, 150000, 200000, 220000, 250000, 1000000]
})

# Extracting features and target variable
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Building the Polynomial regression model
poly_degree = 2
poly_regs = PolynomialFeatures(degree=poly_degree)
x_poly = poly_regs.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# Streamlit interface
st.title('Salary Prediction App')

# Sidebar for user input
st.sidebar.header('Select Level')
selected_level = st.sidebar.slider('Choose Level', min_value=1, max_value=10, value=1, step=1)

# Predictions
predicted_salary = lin_reg_2.predict(poly_regs.transform([[selected_level]]))

# Display predictions
st.write('### Predicted Salary:')
st.write(f"- For Level {selected_level}: ${predicted_salary[0]:,.2f}")

# Plotting the polynomial regression curve
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', label='Actual Data')
plt.plot(x, lin_reg_2.predict(poly_regs.fit_transform(x)), color='blue', label='Polynomial Regression')
plt.scatter(selected_level, predicted_salary, color='green', label='Predicted Salary')
plt.title('Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
st.pyplot(plt)
