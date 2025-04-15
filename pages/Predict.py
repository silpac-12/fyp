import io
import pickle
import streamlit as st
import pandas as pd

from src.imputation import decode_categorical
from src.modeling import get_final_estimator

st.title("🩺 Cancer Prediction - Live Patient Evaluation")

# Step 1: Load trained model and features
model = st.session_state.get("selected_model")
columns = st.session_state.get("sampled_df").drop(columns=[st.session_state.target_column]).columns if "sampled_df" in st.session_state else []

if model is None or not columns.any():
    st.warning("⚠️ Please train a model first on the Modeling page.")
    st.stop()

# Step 2: Input Form for Patient Data
st.header(f"Model Used: {get_final_estimator(st.session_state.selected_model)}")
st.subheader("🧾 Enter Patient Information")

# Use a reference dataframe to infer dtypes (e.g., the sampled_df before encoding)
reference_df = st.session_state.sampled_decoded_df.drop(columns=[st.session_state.target_column])

with st.form("patient_form"):
    user_input = {}
    for col in columns:
        if reference_df[col].dtype == "object" or reference_df[col].dtype.name == "category":
            # Use selectbox for categorical columns
            unique_vals = reference_df[col].dropna().unique().tolist()
            user_input[col] = st.selectbox(f"{col}", options=unique_vals)
        else:
            # Use number_input for numeric columns
            user_input[col] = st.number_input(f"{col}", step=0.01, format="%.2f")
    submitted = st.form_submit_button("Predict")


if submitted:
    input_df = pd.DataFrame([user_input])
    input_df = decode_categorical(input_df, st.session_state.mappings)

    #st.write(input_df.read())

    # Step 3: Prediction
    # Step 3: Prediction
    pred_class = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]

    if st.session_state.target_column == "case_csPCa":
        # Label map for known classes
        label_map = {0: "No Cancer", 1: "Cancer"}

        # Safely convert prediction label
        pred_readable = label_map.get(pred_class, str(pred_class))
        st.success(f"🔍 Prediction: **{pred_readable}**")

        # Format probabilities
        readable_probs = {
            label_map.get(cls, str(cls)): round(p, 4)
            for cls, p in zip(model.classes_, prob)
        }
    else:

        # Dynamically get the label encoding mapping for the target column
        target_col = st.session_state.target_column
        target_mapping = st.session_state.mappings.get(target_col, {})

        # Reverse it: {0: "No Cancer", 1: "Cancer"} or {0: "Benign", 1: "Intermediate", 2: "Malignant"}
        label_map = {v: k for k, v in target_mapping.items()}

        # Safely convert prediction label
        pred_readable = label_map.get(pred_class, str(pred_class))
        st.success(f"🔍 Prediction: **{pred_readable}**")

        # Format probabilities
        readable_probs = {
            label_map.get(cls, str(cls)): round(p, 4)
            for cls, p in zip(model.classes_, prob)
        }

    # Highlight the class with the highest probability
    max_class = max(readable_probs, key=readable_probs.get)
    st.subheader("🧪 Prediction Probabilities:")
    for label, p in readable_probs.items():
        label_display = f"**{label}**" if label == max_class else label
        st.markdown(f"- {label_display}: `{p}`")


    st.subheader("💾 Download Trained Model")

    if model is not None:
        #shap_plot_path = os.path.join(tempfile.gettempdir(), "shap_waterfall.png")
        #st.session_state.fig.savefig(shap_plot_path, bbox_inches="tight", dpi=150)
        buffer = io.BytesIO()
        pickle.dump(model, buffer)
        buffer.seek(0)
        st.download_button(
            label="📥 Download Model (.pkl)",
            data=buffer,
            file_name="trained_cancer_model1.pkl",
            mime="application/octet-stream"
        )
    else:
        st.info("Model not available for download.")

