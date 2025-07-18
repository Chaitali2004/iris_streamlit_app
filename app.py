import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# ------------------------------------------
# 🎯 Page Config
# ------------------------------------------
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered",
)

# ------------------------------------------
# 📦 Load Trained Model (with caching)
# ------------------------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("iris_model.pkl")
    except FileNotFoundError:
        st.error("❌ Model file not found. Please upload 'iris_model.pkl' to the app directory.")
        return None

model = load_model()
species_names = ["Setosa", "Versicolor", "Virginica"]

# ------------------------------------------
# 🖼️ App Header
# ------------------------------------------
st.title("🌸 Iris Flower Classifier")
st.markdown("Enter the flower's measurements to predict its species using a trained ML model.")

# ------------------------------------------
# 📥 User Inputs
# ------------------------------------------
with st.form("input_form"):
    st.subheader("🔧 Input Flower Features")
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

    submitted = st.form_submit_button("Predict")

# ------------------------------------------
# 🔍 Prediction
# ------------------------------------------
if submitted and model:
    input_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                            columns=["sepal length (cm)", "sepal width (cm)", 
                                     "petal length (cm)", "petal width (cm)"])

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    st.subheader("✅ Prediction Result")
    st.success(f"**Predicted Species:** {species_names[prediction]}")

    # --------------------------------------
    # 📊 Probability Visualization
    # --------------------------------------
    st.subheader("📊 Prediction Probabilities")
    fig, ax = plt.subplots()
    ax.bar(species_names, probabilities, color=["#4c72b0", "#55a868", "#c44e52"])
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    st.pyplot(fig)
