import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained pipeline
# -----------------------------
model = joblib.load("best_model.pkl")

# -----------------------------
# Page title
# -----------------------------
st.title("🎓 CGPA → Package Predictor")

st.write("Enter a student's CGPA to predict the expected placement package.")

# -----------------------------
# Single prediction (UI input)
# -----------------------------
cgpa = st.number_input("Enter CGPA", min_value=0.0, max_value=10.0, step=0.1)

if st.button("Predict Package"):
    input_df = pd.DataFrame({"cgpa": [cgpa]})
    prediction = model.predict(input_df)[0]

    st.success(f"💰 Predicted Package: {prediction:.2f} LPA")

# -----------------------------
# Batch prediction (API style)
# -----------------------------
st.subheader("📂 Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV with 'cgpa' column")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    predictions = model.predict(data)
    data["Predicted_Package"] = predictions

    st.write(data)

    csv = data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Predictions",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv",
    )