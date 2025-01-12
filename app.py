import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load pre-trained models and other necessary objects
models = {
    "SVM": joblib.load('svm_model.pkl'),
    "Random Forest": joblib.load('random_forest_model.pkl'),
    "Logistic Regression": joblib.load('logistic_regression_model.pkl'),
    "K-Nearest Neighbors": joblib.load('k-nearest_neighbors_model.pkl'),
    "Decision Tree": joblib.load('decision_tree_model.pkl'),
    "Gradient Boosting": joblib.load('gradient_boosting_model.pkl'),
    "Naive Bayes": joblib.load('naive_bayes_model.pkl')
}

scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# List of feature inputs expected by the model (excluding irrelevant features)
features = [
    "alpha", "delta", "u", "g", "r", "i", "z", "redshift"
]

st.set_page_config(
    page_title="SpectraSense",
    page_icon="🌌",
    layout="centered",
    initial_sidebar_state="auto"
)

bg = """<style>[data-testid="stAppViewContainer"]{
    background-image: url("https://images.pexels.com/photos/2150/sky-space-dark-galaxy.jpg");
    backgground-size: 10% 10%;
}
    </style>"""

st.markdown(bg, unsafe_allow_html=True)

# Streamlit App
st.title("Celestial Body(Star, Galaxy, Quasar) Classification")
st.write(
    "This app uses 7 pre-trained machine learning models to classify stars, "
    "galaxies, and quasars based on their properties."
    " You need to give the required parameters from the sidebar."
)

# User inputs for features
st.sidebar.header("Input Features")
inputs = {}
for feature in features:
    inputs[feature] = st.sidebar.number_input(f"{feature.capitalize()}:", value=0.0, format="%.5f")

# Create input DataFrame
input_df = pd.DataFrame([inputs])

# Scale inputs
scaled_inputs = scaler.transform(input_df)

# Predict and display results
st.subheader("Model Predictions")
results = []
for name, model in models.items():
    prediction = model.predict(scaled_inputs)
    class_label = label_encoder.inverse_transform(prediction)[0]
    results.append({"Model": name, "Predicted Class": class_label})

# Display predictions in a table
results_df = pd.DataFrame(results)
st.dataframe(results_df)

# Final aggregated prediction (majority vote)
final_prediction = results_df["Predicted Class"].mode()[0]
st.subheader("Final Prediction (Majority Vote)")
st.write(f"The majority vote predicts the object is a **{final_prediction}**.")
st.markdown("---")
st.header("Understanding U, G, R, I, Z Magnitudes")
st.write(
    """
    In astronomy, **U, G, R, I, Z** are photometric filters used to measure the brightness 
    of celestial objects in specific wavelength ranges. These filters are part of the Sloan Digital Sky Survey (SDSS) photometric system.
    """
)

st.markdown("### Photometric Filters")
st.write(
    """
    - **U (Ultraviolet):** Captures ultraviolet light (~300-400 nm). Sensitive to hot, young stars.
    - **G (Green):** Captures green light (~400-550 nm). Represents intermediate wavelengths visible to the human eye.
    - **R (Red):** Captures red light (~550-700 nm). Useful for studying older and cooler stars.
    - **I (Infrared):** Captures near-infrared light (~700-850 nm). Penetrates dust to reveal cooler stars.
    - **Z (Near-Infrared):** Captures longer near-infrared light (~850-1000 nm). Less affected by interstellar dust.
    """
)

st.markdown("### Magnitude Values")
st.write(
    """
    Magnitudes are a logarithmic measure of an object's brightness. Smaller values mean brighter objects:
    
    - **Bright Objects:** Negative or small magnitudes (e.g., the Sun or nearby stars).
    - **Faint Objects:** Larger magnitudes (e.g., distant galaxies or faint stars).
    
    A difference of 5 magnitudes corresponds to a brightness ratio of 100.
    """
)

st.markdown("### How Astronomers Use Magnitudes")
st.write(
    """
    By analyzing magnitudes across filters, astronomers can:
    - **Classify objects:** Stars, galaxies, and quasars have distinct magnitude patterns.
    - **Measure distances:** Using redshifts and magnitude comparisons.
    - **Study properties:** Temperature, composition, and age of celestial objects.
    """
)

st.markdown("### Example")
st.write(
    """
    - A **blue star** will have a smaller U magnitude (brighter in ultraviolet) compared to G or R.
    - A **red star** will have a smaller R or I magnitude (brighter in red/infrared) compared to U or G.
    - A **quasar** may show unusual brightness in U and G due to its energetic emissions.
    """
)

st.markdown("---")
st.write("Explore the predictions and learn more about the fascinating world of astronomy!")

