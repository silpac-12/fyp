import io
import pickle
import streamlit as st
import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import tempfile
import os


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

    if pred == 0:
        pred = "No Cancer"
    else:
        pred = "Cancer"

    st.success(f"🔍 Prediction: **{pred}**")
    st.subheader(f"🧪 Prediction Probabilities: {dict(zip(model.classes_, prob))}")

    # Step 4: SHAP Explanation
    st.subheader("🔎 Model Explanation (SHAP)")
    with st.spinner("Calculating SHAP values..."):
        try:
            explainer = shap.Explainer(model, pd.DataFrame([user_input]))
        except:
            explainer = shap.KernelExplainer(model.predict_proba, pd.DataFrame([user_input]))

        shap_values = explainer(pd.DataFrame([user_input]))
        # Detect SHAP shape
        shape = shap_values.values.shape
        num_dims = len(shape)
        st.write("SHAP values shape:", shap_values.values.shape)
        st.write("Base values shape:", shap_values.base_values.shape)

        # Shape check
        shape = shap_values.values.shape
        num_dims = len(shape)

        try:
            if num_dims == 3:
                # Multiclass: (1, n_features, n_classes)
                n_samples, n_features, n_classes = shape
                st.text(f"Multiclass prediction shape: {shape}")

                class_index = 1 if n_classes == 2 else st.selectbox("Select class to explain", range(n_classes),
                                                                    key="shap_class_index")

                vals = shap_values.values[0, :, class_index]
                base_val = shap_values.base_values[0, class_index]
            elif num_dims == 2:
                # Binary classification: (1, n_features)
                st.text(f"Binary prediction shape: {shape}")
                vals = shap_values.values[0]
                base_val = shap_values.base_values[0]
            else:
                raise ValueError(f"Unsupported SHAP shape: {shape}")

            # ✅ Construct single-instance Explanation
            explanation = shap.Explanation(
                values=vals,
                base_values=base_val,
                data=pd.DataFrame([user_input]).iloc[0],
                feature_names=columns
            )

            # ✅ Safe rendering
            shap.plots.waterfall(explanation, show=False)
            fig = plt.gcf()

            # Optional sanity check
            w, h = fig.get_size_inches()
            if w > 100 or h > 100:
                raise ValueError("Generated figure is too large.")

            st.pyplot(fig)
            plt.clf()

        except Exception as e:
            st.error(f"❌ Could not render SHAP waterfall plot:\n`{e}`")

        st.subheader("💾 Download Trained Model")

        if model is not None:
            shap_plot_path = os.path.join(tempfile.gettempdir(), "shap_waterfall.png")
            st.session_state.fig.savefig(shap_plot_path, bbox_inches="tight", dpi=150)
            buffer = io.BytesIO()
            pickle.dump(model, buffer)
            buffer.seek(0)
            st.download_button(
                label="📥 Download Model (.pkl)",
                data=buffer,
                file_name="trained_cancer_model.pkl",
                mime="application/octet-stream"
            )
        else:
            st.info("Model not available for download.")

