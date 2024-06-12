import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd

# Load the dataset
def load_data():
    dataset = pd.read_csv('bmi.csv')  # replace with your dataset
    return dataset

def main():
    # Set title
    st.title("K-Nearest Neighbors (KNN) Algorithm")

    # Load the dataset
    dataset = load_data()

    # Separate features and target variable
    X = dataset[['Age', 'Height', 'Weight']].values
    y = dataset['BmiClass'].values

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Training the K-NN model on the Training set
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Predicting user input
    st.subheader("Predict bmi class")
    age = st.number_input("Enter Age")
    height = st.number_input("Enter Height")
    weight = st.number_input("Enter Weight")
    new_data = sc.transform([[age, height, weight]])  # replace with your feature inputs
    new_pred = classifier.predict(new_data)
    st.write("Prediction:", new_pred[0])

if __name__ == "__main__":
    main()
