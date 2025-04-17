import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# ----- Page Config -----
st.set_page_config(page_title="Prompt Literacy Predictor", layout="centered")
st.markdown(
    """
    <style>
    body {
        background-image: url("https://i.imgur.com/nHfHkPW.png");
        background-size: 400px;
        background-repeat: repeat;
        background-attachment: fixed;
        background-color: #f3edf9;
    }

    .stApp {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 0 20px rgba(0,0,0,0.1);
    }

    h1, h2, h3 {
        color: #4b367c;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----- Title & Description -----
st.markdown("""
# ✨ Prompt Literacy Predictor  
Upload your eye-tracking data and get a quick prediction of the prompt literacy level (High / Medium / Low).
""")

st.markdown("""
This tool uses a machine learning model trained on eye-tracking features to evaluate users' prompt literacy based on their interaction during a creative task.
""")

st.markdown("---")

# ----- File Upload -----
uploaded_file = st.file_uploader("📤 Upload your CSV file with eye-tracking features", type=["csv"])

if uploaded_file:
    try:
        # Load data
        input_data = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
        st.write("Preview of uploaded data:", input_data.head())

        # Load model and scaler
        model = joblib.load("model/prompt_lit_model.pkl")
        scaler = joblib.load("model/scaler.pkl")

        # Scale data
        scaled_features = scaler.transform(input_data.drop(columns=["participant_id"], errors="ignore"))

        # Predict
        predictions = model.predict(scaled_features)
        input_data["Predicted Literacy Level"] = predictions

        st.markdown("### 🧠 Prediction Results")
        st.dataframe(input_data)

        # Download button
        csv = input_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download results as CSV",
            data=csv,
            file_name="predicted_literacy.csv",
            mime="text/csv",
        )

        st.markdown("---")
        st.markdown("✨ *Powered by Streamlit & Scikit-learn*")

    except Exception as e:
        st.error(f"⚠️ Something went wrong: {e}")
else:
    st.info("👈 Please upload a CSV file to start.")
