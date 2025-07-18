

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("iris_model.pkl")

# Title
st.title("🌸 Iris Flower Classifier")
st.write("Enter the flower features to predict the species.")

# User input
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Create input data
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                          columns=["sepal length (cm)", "sepal width (cm)",
                                   "petal length (cm)", "petal width (cm)"])

# Prediction
prediction = model.predict(input_data)
proba = model.predict_proba(input_data)
species = ["Setosa", "Versicolor", "Virginica"]

# Output
st.subheader("🔍 Prediction:")
st.write(f"**Predicted Species:** {species[prediction[0]]}")

# Plot probability
st.subheader("📊 Prediction Probabilities:")
fig, ax = plt.subplots()
ax.bar(species, proba[0], color=["blue", "green", "orange"])
ax.set_ylabel("Probability")
st.pyplot(fig)
