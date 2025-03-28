import streamlit as st
import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt

st.title("🩺 Cancer Prediction - Live Patient Evaluation")

# Step 1: Load trained model and features
model = st.session_state.get("selected_model")
columns = st.session_state.get("sampled_df").drop(columns=[st.session_state.target_column]).columns if "sampled_df" in st.session_state else []

if model is None or not columns.any():
    st.warning("⚠️ Please train a model first on the Modeling page.")
    st.stop()

# Step 2: Input Form for Patient Data
st.subheader("🧾 Enter Patient Information")
with st.form("patient_form"):
    user_input = {}
    for col in columns:
        user_input[col] = st.number_input(f"{col}", step=0.01, format="%.2f")
    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([user_input])

    # Step 3: Prediction
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]
    st.success(f"🔍 Prediction: **{pred}**")
    st.write(f"🧪 Prediction Probabilities: {dict(zip(model.classes_, prob))}")

    # Step 4: SHAP Explanation
    st.subheader("🔎 Model Explanation (SHAP)")
    with st.spinner("Calculating SHAP values..."):
        try:
            explainer = shap.Explainer(model, pd.DataFrame([user_input]))
        except:
            explainer = shap.KernelExplainer(model.predict_proba, pd.DataFrame([user_input]))

        shap_values = explainer(pd.DataFrame([user_input]))
        shap.plots.waterfall(shap.Explanation(
            values=shap_values.values[0],
            base_values=shap_values.base_values[0],
            data=pd.DataFrame([user_input]).iloc[0],
            feature_names=columns
        ), show=False)
        fig = plt.gcf()
        st.pyplot(fig)
