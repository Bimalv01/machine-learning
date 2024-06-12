import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('logreg.csv')  # Replace 'your_dataset.csv' with your actual dataset filename

# Split dataset into features and target
X = df[['BloodPressure', 'Cholesterol']]
y = df['Healthy']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit app
st.title('Health Prediction App')

# Sidebar
st.sidebar.header('Input')

blood_pressure_input = st.sidebar.slider('Blood Pressure', min_value=X['BloodPressure'].min(), max_value=X['BloodPressure'].max(), value=X['BloodPressure'].min())
cholesterol_input = st.sidebar.slider('Cholesterol', min_value=X['Cholesterol'].min(), max_value=X['Cholesterol'].max(), value=X['Cholesterol'].min())

# Prediction
prediction = model.predict([[blood_pressure_input, cholesterol_input]])

st.subheader('Prediction')
if prediction[0] == 1:
    st.write('The person is healthy')
else:
    st.write('The person is not healthy')

# Display model accuracy
st.write('Model Accuracy:', accuracy_score(y_test, model.predict(X_test)))